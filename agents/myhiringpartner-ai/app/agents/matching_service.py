from datetime import datetime, date
import json
import os
import sys
import uuid
from string import Template

from google.cloud import bigquery
from google.cloud.exceptions import NotFound

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
import google.auth

from ..agent import BaseAgent
from langchain_google_vertexai import VertexAI
from vertexai.language_models import TextEmbeddingModel
import logging

logger = logging.getLogger('Agent.MatchingService')

# Prompt templates
MATCH_PROMPT =  Template("""
You are an elite AI-recruitment analyst whose prime directive is to protect the hiring-team's time while achieving ≥ 95 % decision accuracy.

TODAY = "2025-07-22"

══════════  INPUT OBJECTS  ══════════
• JD_JSON      - structured job description                     $jd_json
• RESUME_JSON  - structured résumé data                          $resume_json
• TODAY        - reference date for “Present” end-dates

═══════════  CONFIG PARAMS  ═════════
NEAR_THRESHOLD_YRS         = 1.0      # gap ≤ 1 yr → Clarify
STALE_YRS_PRIMARY          = 3        # primary skill unused > 3 yrs → Stale
STALE_YRS_NICHE            = 4        # niche skill unused > 4 yrs → Clarify
MANDATORY_COVERAGE_GOAL    = 0.95     # < 85 % → hard review
FUZZY_TITLE_MIN_SIM        = 0.78     # Jaccard similarity to avoid false “Core-Mismatch”
ALLOWED_FOR_CLARIFICATION  = {
  "Git", "GitLab", "CI/CD", "Jenkins", "Docker", "Unit Testing",
  "Terraform", "SciPy", "NumPy", "SQL",
  "React Testing Library", "SyncFusion", "Omniscript",
  "Patch Management", "LinkedIn Outreach", "Visualforce"
}

════════  STAGE-1  (JD & CV DECODE)  ════════
1 **JD parsing**
    jd_core_function      - distilled role label
    jd_primary_skill      - single, load-bearing tech / domain
    mandatory_items_raw   - all *explicit* “must” items
    ▸ Partition mandatory_items_raw
        Tier-1 = jd_primary_skill + any item literally tagged “MANDATORY” /
                 or containing phrases {“must have”, “required”}
        Tier-2 = remaining mandatory items

2 **Résumé analysis**
    core_profile          - synthesise from most-recent titles + summary
    evidence_map          - {mandatory_item : evidence-string | null}
    last_used_years       - {skill : YEARS since last end-date}
    total_exp_years       - de-overlapped years, rounded 0.5
    empty_resume          = len(resume_json["experience"]) == 0

3 **Niche flag**
    is_niche_role = jd_core_function or jd_primary_skill in
                    {"Guidewire","COBOL","Murex","Vlocity","Hybris",
                     "TM1","Hyperion Essbase","Zuora","ABAP (only)",
                     "Medical-Device (IEC 62304)"}

════════  STAGE-2  (DECISION FUNNEL)  ════════
Apply rules in order; first hit wins.

**Rule 0  - Location Mismatch  → Rejected**
     if (JD does NOT allow remote AND does NOT allow relocation)
         AND (candidate city NOT in JD allowed cities)
         → Rejected
     else if (candidate country NOT in JD allowed countries)
         → Rejected

 **Rule 1  - Core Mismatch  → Rejected**
       if   JaccardSim(core_profile, jd_core_function) < FUZZY_TITLE_MIN_SIM
        OR  empty_resume == true.

 **Rule 2  - Over-qualified  → Rejected**
       if JD defines a *maximum* years / level AND total_exp_years > max.

 **Rule 3  - Niche Clarification  → Clarify**
       if is_niche_role
          AND jd_primary_skill in evidence_map
          AND ( missing_any_other_mandatory
                OR last_used_years[jd_primary_skill] > STALE_YRS_NICHE )

 **Rule 4  - Critical Gap  → Rejected**
       if (missing ≥ 1 Tier-1 item)
          OR (missing ≥ 3 Tier-2 items not in ALLOWED_FOR_CLARIFICATION).

 **Rule 5  - Near-Threshold  → Clarify**
       if ( exactly 1 Tier-2 item missing
            OR gap_in_required_years ≤ NEAR_THRESHOLD_YRS
            OR missing_item ∈ ALLOWED_FOR_CLARIFICATION ).

 **Rule 6  - Stale Experience  → Rejected**
       if last_used_years[jd_primary_skill] > STALE_YRS_PRIMARY
          AND is_niche_role == false.

 **Rule 7  - Plausible Omission  → Clarify**
       if strong domain alignment (≥ 85 % Tier-2 present)
          BUT 1 technical Tier-1/Tier-2 item absent.

 **Rule 8  - Strong Match  → Moving Forward**
       if   no Tier-1 gaps
        AND len(missing_items) == 0
        AND mandatory_coverage_ratio ≥ MANDATORY_COVERAGE_GOAL.

── **Post-Decision Sanity Check** ──  
IF outcome == "Rejected" AND (
        mandatory_coverage_ratio ≥ MANDATORY_COVERAGE_GOAL
     OR len(missing_items) ≤ 2 )
   THEN   outcome = "Requires Clarification"
          rule_flagged = "Post-Check Adjustment"

════════  STAGE-3  (JSON RESPONSE)  ════════
Return **only**:

{
  "outcome": "Moving Forward | Rejected | Requires Clarification",
  "reason": "<concise text or empty>",
  "rule_flagged": "Rule # - Name | Post-Check Adjustment",
  "is_niche_role": true|false,
  "core_function": "<JD label>",
  "core_profile": "<Résumé label>",
  "missing_items": [ … ],
  "stale_skills":  [ … ],
  "recency_summary": { "skill": "X years ago", … }
}

Notes  
• JaccardSim(a,b) = |tokensₐ∩tokens_b| ÷ |tokensₐutokens_b| after lower-casing & stop-word drop.  
• mandatory_coverage_ratio = (# mandatory_items with evidence) ÷ (total mandatory_items_raw).  
• ALWAYS prefer “Clarification” over “Rejection” when in doubt and coverage ≥ 85 %.

Respond **only** with the JSON - no extra prose, markdown, or line-break errors.
"""
)


def default_json_serializer(obj):
    """JSON serializer for objects not serializable by default json code"""
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    raise TypeError ("Type %s not serializable" % type(obj))

class MatchingService(BaseAgent):
    def __init__(self, model_name="gemini-2.0-flash-001", temperature=0.2, location="us-central1"):
        super().__init__("MatchingService", model_name=model_name)
        self.temperature = min(max(temperature, 0.0), 2.0)
        self.location = location

        # Configuration from central config object
        self.project_id = os.getenv('GOOGLE_CLOUD_PROJECT')
        self.dataset_id = 'myhiringpartner_ai'
        self.jd_table_name = 'job_details'
        self.resume_table_name = 'resumes'
        self.match_table_name = 'matches'


        self.credentials, _ = google.auth.default()
        self._initialize_model()
        self._initialize_bigquery()
        self._initialize_embedding_model()

    def _initialize_model(self):
        try:
            self.llm = VertexAI(
                model_name=self.model_name,
                temperature=self.temperature,
                max_output_tokens=4096,
                location=self.location
            )
            self.output_parser = StrOutputParser()
        except Exception as e:
            raise Exception(f"Error initializing model: {str(e)}")

    def _initialize_bigquery(self):
        try:
            self.bq_client = bigquery.Client(credentials=self.credentials)

            dataset_ref = self.bq_client.dataset(self.dataset_id)
            try:
                self.bq_client.get_dataset(dataset_ref)
            except NotFound:
                dataset = bigquery.Dataset(dataset_ref)
                dataset.location = self.location
                self.bq_client.create_dataset(dataset)

            self._ensure_match_table_exists()
        except Exception as e:
            raise Exception(f"Error initializing BigQuery: {str(e)}")

    def _initialize_embedding_model(self):
        try:
            self.embedding_model = TextEmbeddingModel.from_pretrained(
                "text-embedding-005")
        except Exception as e:
            raise Exception(f"Error initializing embedding model: {str(e)}")

    def run(self, **kwargs):
        if 'job_id' in kwargs:
            return self.workflow1_jd_to_resumes(**kwargs)
        elif 'candidate_id' in kwargs:
            return self.workflow2_resume_to_jds(**kwargs)
        else:
            raise ValueError("MatchingService.run requires either 'job_id' or 'candidate_id'")
            
    def _generate_and_store_embedding(self, text, table_name, id_field_name, id_value):
        try:
            embeddings = self._generate_embeddings(text=text)
            
            if not embeddings:
                logger.error(f"Failed to generate embeddings for {id_field_name}={id_value}")
                return None
                
            query = f"""
            UPDATE `{self.project_id}.{self.dataset_id}.{table_name}`
            SET embeddings = ARRAY{json.dumps(embeddings)}
            WHERE {id_field_name} = '{id_value}'
            """
            
            logger.debug(f"Executing query to update embeddings: {query[:100]}...")
            self.bq_client.query(query).result()
            logger.info(f"Successfully updated embeddings for {id_field_name}={id_value}")
            
            return embeddings
        except Exception as e:
            logger.error(f"Error in _generate_and_store_embedding: {str(e)}", exc_info=True)
            return None


    def _ensure_match_table_exists(self):
        try:
            table_id = f"{self.project_id}.{self.dataset_id}.{self.match_table_name}"
            table_ref = bigquery.TableReference.from_string(table_id)

            try:
                self.bq_client.get_table(table_ref)
            except NotFound:
                schema = [
                    bigquery.SchemaField(
                        "match_id", "STRING", mode="REQUIRED"),
                    bigquery.SchemaField("job_id", "STRING", mode="REQUIRED"),
                    bigquery.SchemaField(
                        "candidate_id", "STRING", mode="REQUIRED"),
                    bigquery.SchemaField(
                        "semantic_similarity_score", "FLOAT", mode="REQUIRED"),
                    bigquery.SchemaField("mandatory_requirements", "RECORD", fields=[
                        bigquery.SchemaField(
                            "skills_requirement_met", "BOOLEAN"),
                        bigquery.SchemaField(
                            "experience_requirement_met", "BOOLEAN"),
                        bigquery.SchemaField(
                            "domain_requirement_met", "BOOLEAN"),
                        bigquery.SchemaField(
                            "education_requirement_met", "BOOLEAN"),
                        bigquery.SchemaField(
                            "location_requirement_met", "BOOLEAN")
                    ]),
                    bigquery.SchemaField("match_scores", "RECORD", fields=[
                        bigquery.SchemaField("overall_match_score", "FLOAT"),
                        bigquery.SchemaField("skill_match_score", "FLOAT"),
                        bigquery.SchemaField(
                            "experience_match_score", "FLOAT"),
                        bigquery.SchemaField("domain_match_score", "FLOAT"),
                        bigquery.SchemaField("education_match_score", "FLOAT"),
                        bigquery.SchemaField("location_match_score", "FLOAT")
                    ]),
                    bigquery.SchemaField(
                        "missing_required_skills", "STRING", mode="REPEATED"),
                    bigquery.SchemaField(
                        "missing_preferred_skills", "STRING", mode="REPEATED"),
                    bigquery.SchemaField(
                        "key_strengths", "STRING", mode="REPEATED"),
                    bigquery.SchemaField(
                        "gaps_and_concerns", "STRING", mode="REPEATED"),
                    bigquery.SchemaField("screening_decision", "RECORD", fields=[
                        bigquery.SchemaField("status", "STRING"),
                        bigquery.SchemaField("explanation", "STRING"),
                        bigquery.SchemaField(
                            "clarification_questions", "STRING", mode="REPEATED"),
                        bigquery.SchemaField(
                            "interview_recommendation", "BOOLEAN"),
                        bigquery.SchemaField(
                            "suggested_interview_questions", "STRING", mode="REPEATED")
                    ]),
                    bigquery.SchemaField("match_status", "STRING"),
                    bigquery.SchemaField("rule_flagged", "STRING", mode="NULLABLE"),
                    bigquery.SchemaField("created_at", "TIMESTAMP"),
                    bigquery.SchemaField("updated_at", "TIMESTAMP")
                ]

                table = bigquery.Table(table_ref, schema=schema)
                self.bq_client.create_table(table)
        except Exception as e:
            raise Exception(f"Error ensuring match table exists: {str(e)}")

    def _cosine_similarity(self, embedding1, embedding2):
        if not embedding1 or not embedding2:
            return 0.0

        dot_product = sum(a * b for a, b in zip(embedding1, embedding2))
        magnitude1 = sum(a * a for a in embedding1) ** 0.5
        magnitude2 = sum(b * b for b in embedding2) ** 0.5

        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0

        return dot_product / (magnitude1 * magnitude2)

    def _fetch_jd_data(self, job_id):
        try:
            query = f"""
            SELECT * FROM `{self.project_id}.{self.dataset_id}.{self.jd_table_name}`
            WHERE job_id = @job_id
            LIMIT 1
            """

            job_config = bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ScalarQueryParameter("job_id", "STRING", job_id)
                ]
            )

            query_job = self.bq_client.query(query, job_config=job_config)
            results = list(query_job.result())

            if not results:
                return None

            return dict(results[0])
        except Exception as e:
            raise Exception(f"Error fetching JD data: {str(e)}")

    def _fetch_resume_data(self, candidate_id):
        try:
            query = f"""
            SELECT * FROM `{self.project_id}.{self.dataset_id}.{self.resume_table_name}`
            WHERE candidate_id = @candidate_id
            LIMIT 1
            """

            job_config = bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ScalarQueryParameter(
                        "candidate_id", "STRING", candidate_id)
                ]
            )

            query_job = self.bq_client.query(query, job_config=job_config)
            results = list(query_job.result())

            if not results:
                return None

            return dict(results[0])
        except Exception as e:
            raise Exception(f"Error fetching resume data: {str(e)}")

    def _fetch_all_resumes(self, filters=None):
        try:
            query = f"SELECT * FROM `{self.project_id}.{self.dataset_id}.{self.resume_table_name}`"

            if filters:
                where_clauses = []
                for key, value in filters.items():
                    if isinstance(value, list):
                        elements = ', '.join([f"'{v}'" for v in value])
                        where_clauses.append(
                            f"EXISTS(SELECT 1 FROM UNNEST({key}) AS x WHERE x IN ({elements}))")
                    elif isinstance(value, (int, float)):
                        where_clauses.append(f"{key} >= {value}")
                    else:
                        where_clauses.append(f"{key} = '{value}'")

                if where_clauses:
                    query += " WHERE " + " AND ".join(where_clauses)

            query_job = self.bq_client.query(query)
            return list(query_job.result())
        except Exception as e:
            raise Exception(f"Error fetching all resumes: {str(e)}")

    def _calculate_semantic_match(self, jd_embedding, resume_embedding):
        try:
            if not jd_embedding or not resume_embedding:
                logger.warning("Missing embedding data for semantic match calculation")
                return 0.0

            if isinstance(jd_embedding, str):
                try:
                    jd_embedding = json.loads(jd_embedding)
                except json.JSONDecodeError:
                    if jd_embedding.startswith("gs://"):
                        logger.warning("JD embedding is a GCS URI, this should be resolved before calling this method")
                        return 0.0
                    else:
                        logger.warning(f"Failed to parse JD embedding from string: {jd_embedding[:50]}...")
                        return 0.0

            if isinstance(resume_embedding, str):
                try:
                    resume_embedding = json.loads(resume_embedding)
                except json.JSONDecodeError:
                    if resume_embedding.startswith("gs://"):
                        logger.warning("Resume embedding is a GCS URI, this should be resolved before calling this method")
                        return 0.0
                    else:
                        logger.warning(f"Failed to parse resume embedding from string: {resume_embedding[:50]}...")
                        return 0.0

            if isinstance(jd_embedding, dict):
                if 'values' in jd_embedding:
                    jd_embedding = jd_embedding['values']
                elif 'embedding' in jd_embedding:
                    jd_embedding = jd_embedding['embedding']

            if isinstance(resume_embedding, dict):
                if 'values' in resume_embedding:
                    resume_embedding = resume_embedding['values']
                elif 'embedding' in resume_embedding:
                    resume_embedding = resume_embedding['embedding']

            if not isinstance(jd_embedding, list) or not isinstance(resume_embedding, list):
                logger.warning(f"Invalid embedding format after normalization: JD type={type(jd_embedding)}, Resume type={type(resume_embedding)}")
                if isinstance(jd_embedding, str) and jd_embedding.startswith("[") and jd_embedding.endswith("]"):
                    try:
                        jd_embedding = json.loads(jd_embedding)
                    except:
                        pass
                if isinstance(resume_embedding, str) and resume_embedding.startswith("[") and resume_embedding.endswith("]"):
                    try:
                        resume_embedding = json.loads(resume_embedding)
                    except:
                        pass

                if not isinstance(jd_embedding, list) or not isinstance(resume_embedding, list):
                    logger.warning("Failed to convert embeddings to list format after all attempts")
                    return 0.0

            if not all(isinstance(x, (int, float)) for x in jd_embedding) or \
               not all(isinstance(x, (int, float)) for x in resume_embedding):
                logger.warning("Embeddings contain non-numeric elements")
                return 0.0

            return self._cosine_similarity(jd_embedding, resume_embedding)
        except Exception as e:
            logger.error(f"Error calculating semantic match: {str(e)}", exc_info=True)
            return 0.0


    def _vector_search_query(self, query_embedding, limit=20):
        try:
            query = f"""
            WITH query_embedding AS (
                SELECT {json.dumps(query_embedding)} AS embedding
            )
            SELECT 
                r.*,
                1 - distance AS similarity_score
            FROM 
                VECTOR_SEARCH(
                    TABLE `{self.project_id}.{self.dataset_id}.{self.resume_table_name}`,
                    'embeddings',
                    (SELECT embedding FROM query_embedding),
                    distance_type => 'COSINE',
                    top_k => {limit}
            )
            ORDER BY similarity_score DESC
            """

            query_job = self.bq_client.query(query)
            results = list(query_job.result())
            logger.debug(f"Vector search query returned {len(results)} results")
            return results

        except Exception as e:
            logger.error(f"Error in vector search query: {str(e)}", exc_info=True)
            return []

    def _detailed_llm_match(self, jd_data, resume_data):
        try:
            jd_json = json.dumps(jd_data, indent=2, default=default_json_serializer)
            resume_json = json.dumps(resume_data, indent=2, default=default_json_serializer)
            
            # Render the Template into a plain string before invoking the LLM
            prompt_text = MATCH_PROMPT.substitute(
                jd_json=jd_json,
                resume_json=resume_json,
            )

            response = self.llm.invoke(prompt_text)

            if response.strip().startswith("```json"):
                response = response.strip()[7:-4]

            logger.debug(f"LLM match response: {response[:100]}...")
            return json.loads(response)

        except Exception as e:
            logger.error(f"Error performing detailed LLM match: {str(e)}", exc_info=True)
            raise Exception(f"Error performing detailed LLM match: {str(e)}")

    def _store_match_results(self, job_id, candidate_id, semantic_score, match_details):
        try:
            match_id = str(uuid.uuid4())
            created_at = datetime.utcnow().isoformat()

            row_to_insert = {
                "match_id": match_id,
                "job_id": job_id,
                "candidate_id": candidate_id,
                "semantic_similarity_score": semantic_score,
                "created_at": created_at,
                "updated_at": created_at,
                "match_status": "Pending Review",
            }

            # Extract rule_flagged from match_details (from LLM response)
            rule_flagged = match_details.get("rule_flagged", "")
            if rule_flagged is None:
                rule_flagged = ""
            row_to_insert["rule_flagged"] = rule_flagged

            # Map LLM fields to existing schema to avoid unknown-field errors
            # outcome -> screening_decision.status
            # reason  -> screening_decision.explanation
            outcome = match_details.get("outcome") or ""
            reason  = match_details.get("reason") or ""

            # Ensure correct types for BigQuery
            if not isinstance(outcome, str):
                outcome = str(outcome)
            if not isinstance(reason, str):
                reason = str(reason)

            # Helpers to coerce types safely
            def _as_bool(v):
                try:
                    if isinstance(v, bool) or v is None:
                        return v
                    if isinstance(v, (int, float)):
                        return bool(v)
                    if isinstance(v, str):
                        return v.strip().lower() in {"true", "1", "yes", "y"}
                except Exception:
                    return None
                return None

            def _as_float(v):
                try:
                    return float(v) if v is not None and v != "" else None
                except Exception:
                    return None

            def _as_str_list(v):
                if v is None:
                    return []
                if not isinstance(v, list):
                    v = [v]
                out = []
                for x in v:
                    if x is None:
                        continue
                    out.append(str(x))
                return out

            screening_decision = {
                "status": outcome,
                "explanation": reason,
                # Optional nested fields from LLM if present
                "clarification_questions": _as_str_list(match_details.get("clarification_questions")),
                "interview_recommendation": _as_bool(match_details.get("interview_recommendation")),
                "suggested_interview_questions": _as_str_list(match_details.get("suggested_interview_questions")),
            }
            row_to_insert["screening_decision"] = screening_decision

            # Map mandatory_requirements RECORD
            mand = match_details.get("mandatory_requirements", {}) or {}
            if isinstance(mand, dict):
                row_to_insert["mandatory_requirements"] = {
                    "skills_requirement_met": _as_bool(mand.get("skills_requirement_met")),
                    "experience_requirement_met": _as_bool(mand.get("experience_requirement_met")),
                    "domain_requirement_met": _as_bool(mand.get("domain_requirement_met")),
                    "education_requirement_met": _as_bool(mand.get("education_requirement_met")),
                    "location_requirement_met": _as_bool(mand.get("location_requirement_met")),
                }

            # Map match_scores RECORD
            scores = match_details.get("match_scores", {}) or {}
            if isinstance(scores, dict):
                row_to_insert["match_scores"] = {
                    "overall_match_score": _as_float(scores.get("overall_match_score")),
                    "skill_match_score": _as_float(scores.get("skill_match_score")),
                    "experience_match_score": _as_float(scores.get("experience_match_score")),
                    "domain_match_score": _as_float(scores.get("domain_match_score")),
                    "education_match_score": _as_float(scores.get("education_match_score")),
                    "location_match_score": _as_float(scores.get("location_match_score")),
                }

            # Map repeated string fields
            row_to_insert["missing_required_skills"] = _as_str_list(match_details.get("missing_required_skills"))
            row_to_insert["missing_preferred_skills"] = _as_str_list(match_details.get("missing_preferred_skills"))
            row_to_insert["key_strengths"] = _as_str_list(match_details.get("key_strengths"))
            row_to_insert["gaps_and_concerns"] = _as_str_list(match_details.get("gaps_and_concerns"))

            errors = self.bq_client.insert_rows_json(f"{self.project_id}.{self.dataset_id}.{self.match_table_name}", [row_to_insert])
            if errors:
                logger.error(f"Failed to insert match result into BigQuery: {errors}")
                raise Exception(f"Failed to insert match result into BigQuery: {errors}")
            
            logger.debug(f"Stored match result with match_id={match_id}")
            return match_id

        except Exception as e:
            logger.error(f"Error storing match results: {str(e)}", exc_info=True)
            return None

    def _check_for_existing_match(self, job_id, candidate_id):
        query = f"""
        SELECT match_id
        FROM `{self.project_id}.{self.dataset_id}.{self.match_table_name}`
        WHERE job_id = @job_id AND candidate_id = @candidate_id
        LIMIT 1
        """
        query_params = [
            bigquery.ScalarQueryParameter("job_id", "STRING", job_id),
            bigquery.ScalarQueryParameter(
                "candidate_id", "STRING", candidate_id),
        ]
        job_config = bigquery.QueryJobConfig(query_parameters=query_params)

        try:
            query_job = self.bq_client.query(query, job_config=job_config)
            results = list(query_job.result())
            logger.debug(f"Checked for existing match: job_id={job_id}, candidate_id={candidate_id}, found={len(results) > 0}")
            return len(results) > 0
        except Exception as e:
            logger.error(f"Error checking for existing match: {str(e)}", exc_info=True)
            return False

    def _vector_search_jds_query(self, query_embedding, limit=20):
        """Performs a vector search on the job postings table."""
        try:
            query = f"""
            SELECT
                base.job_id,
                distance
            FROM
                VECTOR_SEARCH(
                    TABLE `{self.project_id}.{self.dataset_id}.{self.jd_table_name}`,
                    'embeddings',
                    (
                        SELECT
                            ARRAY{json.dumps(query_embedding)} AS embeddings
                    ),
                    top_k => {limit},
                    distance_type => 'COSINE'
            )
            """
            logger.debug(f"Executing vector search query on JDs: {query[:150]}...")
            query_job = self.bq_client.query(query)
            results = [{"job_id": row["job_id"], "distance": row["distance"]} for row in query_job.result()]
            logger.debug(f"Vector search found {len(results)} potential JD matches.")
            return results
        except Exception as e:
            logger.error(f"Error performing vector search on JDs: {str(e)}", exc_info=True)
            return []

    def workflow2_resume_to_jds(self, candidate_id, metadata_filters=None, limit=10):
        """
        Workflow to match a single resume (by candidate_id) against all job descriptions.
        """
        try:
            logger.info(f"workflow2 start: candidate_id={candidate_id}, limit={limit}")
            logger.debug("Fetching resume data for candidate")
            resume_data = self._fetch_resume_data(candidate_id)
            if not resume_data:
                return {"error": f"Resume with candidate_id {candidate_id} not found."}

            # Get candidate name early
            candidate_name = resume_data.get("candidate_name", "Unknown Candidate")

            resume_embedding = resume_data.get("embeddings")
            if not resume_embedding:
                logger.info("Resume embedding not found, generating a new one.")
                resume_text = resume_data.get("resume_text", "")
                if not resume_text:
                    return {"error": "Resume text is empty, cannot generate embedding."}
                resume_embedding = self._generate_and_store_embedding(
                    resume_text, self.resume_table_name, "candidate_id", candidate_id
                )
                if not resume_embedding:
                    return {"error": "Failed to generate and store resume embedding."}

            # Rest of the method remains the same
            logger.debug("Running JD vector search for candidate")
            vector_results = self._vector_search_jds_query(resume_embedding, limit=limit)
            if not vector_results:
                logger.info("No potential matches found via vector search.")
                return {
                    "candidate_id": candidate_id,
                    "candidate_name": candidate_name,  # Added candidate name here
                    "matches": []
                }

            logger.debug(f"Vector search returned {len(vector_results)} results")
            match_results = []
            for match in vector_results:
                try:
                    job_id = match["job_id"]
                    logger.debug(f"Fetching JD data for job_id={job_id}")
                    jd_data = self._fetch_jd_data(job_id)
                    if not jd_data:
                        logger.warning(f"Could not fetch data for job_id {job_id}, skipping.")
                        continue

                    semantic_score = 1 - match["distance"] # Cosine distance to similarity
                    logger.info(f"LLM match: candidate_id={candidate_id}, job_id={job_id}, semantic_score={semantic_score:.4f}")
                    if self._check_for_existing_match(job_id, candidate_id):
                        logger.debug(f"Match already exists for candidate {candidate_id} and job {job_id}. Skipping.")
                        continue

                    logger.debug("Fetching full resume data for detailed match")
                    full_resume_data = self._fetch_resume_data(candidate_id)
                    if not full_resume_data:
                        logger.warning(f"Could not fetch full resume data for {candidate_id}")
                        continue

                    try:
                        logger.debug("Calling _detailed_llm_match")
                        match_details = self._detailed_llm_match(
                            jd_data, full_resume_data)
                        
                        logger.debug("Storing match results in BigQuery")
                        match_id = self._store_match_results(job_id, candidate_id, semantic_score, match_details)
                        if match_id:
                            result = {
                                "match_id": match_id,
                                "job_id": job_id,
                                "candidate_id": candidate_id,
                                "candidate_name": candidate_name,
                                "semantic_score": semantic_score,
                                "match_details": match_details
                            }
                            match_results.append(result)

                    except Exception as match_error:
                        logger.error(
                            f"Error processing match for candidate {candidate_id}: {str(match_error)}", exc_info=True)

                except Exception as e:
                    logger.error(f"Error processing match for job {match.get('job_id')}: {str(e)}", exc_info=True)

            logger.info(f"workflow2 complete: candidate_id={candidate_id}, processed_jobs={len(vector_results)}, matches={len(match_results)}")
            return {
                "candidate_id": candidate_id,
                "candidate_name": resume_data.get("candidate_name"),
                "total_jobs_processed": len(vector_results),
                "matches": match_results,
            }

        except Exception as e:
            logger.error(f"An unexpected error occurred in workflow2_resume_to_jds: {str(e)}", exc_info=True)
            return {
                "candidate_id": candidate_id,
                "error": str(e),
                "matches": [],
                "status": "Failed",
            }