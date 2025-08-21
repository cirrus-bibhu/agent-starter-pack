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
from typing import Optional

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
Return JSON with the following EXACT shape and key names.
All arrays must be arrays of strings unless stated. Use true/false for booleans and numbers for scores.
Do not return null; use empty string "" for optional strings and empty arrays [] when you have no items.

{
  "outcome": "Moving Forward" | "Rejected" | "Requires Clarification",
  "reason": "<concise text or empty string>",
  "rule_flagged": "Rule # - Name" | "Post-Check Adjustment" | "",

  "mandatory_requirements": {
    "skills_requirement_met": true|false,
    "experience_requirement_met": true|false,
    "domain_requirement_met": true|false,
    "education_requirement_met": true|false,
    "location_requirement_met": true|false
  },

  "match_scores": {
    "overall_match_score": <float between 0 and 1>,
    "skill_match_score": <float between 0 and 1>,
    "experience_match_score": <float between 0 and 1>,
    "domain_match_score": <float between 0 and 1>,
    "education_match_score": <float between 0 and 1>,
    "location_match_score": <float between 0 and 1>
  },

  "missing_required_skills": ["..."],
  "missing_preferred_skills": ["..."],
  "key_strengths": ["..."],
  "gaps_and_concerns": ["..."],

  "screening_decision": {
    "status": "Move Forward" | "Rejected" | "Requires Clarification",
    "explanation": "<short justification or empty string>",
    "clarification_questions": ["...", "..."]
  },

  "interview_recommendation": true|false,
  "suggested_interview_questions": ["...", "..."],

  "is_niche_role": true|false,
  "core_function": "<JD label>",
  "core_profile": "<Résumé label>",
  "missing_items": ["..."],
  "stale_skills": ["..."],
  "recency_summary": { "skill": "X years ago" }
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

    def _fetch_jd_data(self, job_id_or_linkedin_id: str):
        """Fetch a job_details row by internal job_id, with fallback to linkedin_job_id.

        Backward-compatible: existing callers pass internal job_id. If that fails and the
        input looks like a numeric LinkedIn ID, we try matching on linkedin_job_id.
        Returns a dict row or None.
        """
        try:
            # 1) Try by internal job_id
            query = f"""
            SELECT * FROM `{self.project_id}.{self.dataset_id}.{self.jd_table_name}`
            WHERE job_id = @id
            LIMIT 1
            """
            job_config = bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ScalarQueryParameter("id", "STRING", job_id_or_linkedin_id)
                ]
            )
            results = list(self.bq_client.query(query, job_config=job_config).result())
            if results:
                return dict(results[0])

            # 2) Fallback: if input seems like a LinkedIn numeric id, try linkedin_job_id
            if isinstance(job_id_or_linkedin_id, str) and job_id_or_linkedin_id.isdigit():
                query2 = f"""
                SELECT * FROM `{self.project_id}.{self.dataset_id}.{self.jd_table_name}`
                WHERE linkedin_job_id = @lid
                LIMIT 1
                """
                job_config2 = bigquery.QueryJobConfig(
                    query_parameters=[
                        bigquery.ScalarQueryParameter("lid", "STRING", job_id_or_linkedin_id)
                    ]
                )
                results2 = list(self.bq_client.query(query2, job_config=job_config2).result())
                if results2:
                    return dict(results2[0])

            return None
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
            # If a row exists, we'll update it and preserve match_id/created_at.
            # Otherwise, we'll insert a new row with a fresh match_id and timestamps.
            new_match_id = str(uuid.uuid4())
            match_status = "Pending Review"

            # Extract rule_flagged from match_details (from LLM response)
            rule_flagged = match_details.get("rule_flagged", "")
            if rule_flagged is None:
                rule_flagged = ""

            # Map LLM fields to existing schema to avoid unknown-field errors
            # Prefer nested screening_decision.*; fall back to legacy top-level fields for backward compatibility
            outcome = match_details.get("outcome") or ""
            reason  = match_details.get("reason") or ""
            screening_decision = match_details.get("screening_decision") or {}
            if not isinstance(screening_decision, dict):
                screening_decision = {}
            sd_status = screening_decision.get("status") or outcome
            sd_explanation = screening_decision.get("explanation") or reason
            sd_clar_qs = screening_decision.get("clarification_questions") or match_details.get("clarification_questions")

            # Ensure correct types for BigQuery
            if not isinstance(outcome, str):
                outcome = str(outcome)
            if not isinstance(reason, str):
                reason = str(reason)
            if not isinstance(sd_status, str):
                sd_status = str(sd_status)
            if not isinstance(sd_explanation, str):
                sd_explanation = str(sd_explanation)

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

            # Prepare all mapped values for MERGE parameters
            # screening_decision fields
            sc_status = sd_status
            sc_explanation = sd_explanation
            sc_clar_qs = _as_str_list(sd_clar_qs)
            sc_interview_reco = _as_bool(match_details.get("interview_recommendation"))
            sc_suggested_qs = _as_str_list(match_details.get("suggested_interview_questions"))

            # match_status should reflect the high-level outcome
            match_status = outcome or match_status

            # Map mandatory_requirements RECORD
            mand = match_details.get("mandatory_requirements", {}) or {}
            mand_skills = _as_bool(mand.get("skills_requirement_met")) if isinstance(mand, dict) else None
            mand_experience = _as_bool(mand.get("experience_requirement_met")) if isinstance(mand, dict) else None
            mand_domain = _as_bool(mand.get("domain_requirement_met")) if isinstance(mand, dict) else None
            mand_education = _as_bool(mand.get("education_requirement_met")) if isinstance(mand, dict) else None
            mand_location = _as_bool(mand.get("location_requirement_met")) if isinstance(mand, dict) else None

            # Map match_scores RECORD
            scores = match_details.get("match_scores", {}) or {}
            sc_overall = _as_float(scores.get("overall_match_score")) if isinstance(scores, dict) else None
            sc_skill = _as_float(scores.get("skill_match_score")) if isinstance(scores, dict) else None
            sc_experience = _as_float(scores.get("experience_match_score")) if isinstance(scores, dict) else None
            sc_domain = _as_float(scores.get("domain_match_score")) if isinstance(scores, dict) else None
            sc_education = _as_float(scores.get("education_match_score")) if isinstance(scores, dict) else None
            sc_location = _as_float(scores.get("location_match_score")) if isinstance(scores, dict) else None

            # Repeated string fields
            arr_missing_required = _as_str_list(match_details.get("missing_required_skills"))
            arr_missing_preferred = _as_str_list(match_details.get("missing_preferred_skills"))
            arr_key_strengths = _as_str_list(match_details.get("key_strengths"))
            arr_gaps_concerns = _as_str_list(match_details.get("gaps_and_concerns"))

            # Retrieve existing match_id if present to return a stable id on updates
            existing_id = None
            try:
                q = f"""
                SELECT match_id FROM `{self.project_id}.{self.dataset_id}.{self.match_table_name}`
                WHERE job_id = @job_id AND candidate_id = @candidate_id
                LIMIT 1
                """
                cfg = bigquery.QueryJobConfig(query_parameters=[
                    bigquery.ScalarQueryParameter("job_id", "STRING", job_id),
                    bigquery.ScalarQueryParameter("candidate_id", "STRING", candidate_id),
                ])
                res = list(self.bq_client.query(q, job_config=cfg).result())
                if res:
                    existing_id = res[0]["match_id"]
            except Exception:
                existing_id = None

            # Dynamically build MERGE to avoid failing when some columns are absent in live schema
            # Discover available top-level columns
            cols_query = f"""
            SELECT column_name
            FROM `{self.project_id}.{self.dataset_id}.INFORMATION_SCHEMA.COLUMNS`
            WHERE table_name = '{self.match_table_name}'
            """
            try:
                cols_res = list(self.bq_client.query(cols_query).result())
                available_cols = {row["column_name"] for row in cols_res}
            except Exception:
                # If discovery fails, assume minimal set
                available_cols = set()

            has_top_ir = "interview_recommendation" in available_cols
            has_top_siq = "suggested_interview_questions" in available_cols

            update_parts = [
                "semantic_similarity_score = @semantic_similarity_score",
                "mandatory_requirements = STRUCT(\n                @mand_skills AS skills_requirement_met,\n                @mand_experience AS experience_requirement_met,\n                @mand_domain AS domain_requirement_met,\n                @mand_education AS education_requirement_met,\n                @mand_location AS location_requirement_met\n              )",
                "match_scores = STRUCT(\n                @sc_overall AS overall_match_score,\n                @sc_skill AS skill_match_score,\n                @sc_experience AS experience_match_score,\n                @sc_domain AS domain_match_score,\n                @sc_education AS education_match_score,\n                @sc_location AS location_match_score\n              )",
                "missing_required_skills = @arr_missing_required",
                "missing_preferred_skills = @arr_missing_preferred",
                "key_strengths = @arr_key_strengths",
                "gaps_and_concerns = @arr_gaps_concerns",
                "screening_decision = STRUCT(\n                @sc_status AS status,\n                @sc_explanation AS explanation,\n                @sc_clar_qs AS clarification_questions,\n                @sc_interview_reco AS interview_recommendation,\n                @sc_suggested_qs AS suggested_interview_questions\n              )",
                "match_status = @match_status",
                "rule_flagged = @rule_flagged",
                "updated_at = CURRENT_TIMESTAMP()",
            ]

            if has_top_ir:
                update_parts.insert(-3, "interview_recommendation = @sc_interview_reco")
            if has_top_siq:
                update_parts.insert(-3 if has_top_ir else -3, "suggested_interview_questions = @sc_suggested_qs")

            insert_columns = [
                "match_id", "job_id", "candidate_id",
                "semantic_similarity_score", "mandatory_requirements", "match_scores",
                "missing_required_skills", "missing_preferred_skills", "key_strengths", "gaps_and_concerns",
                "screening_decision",
            ]
            if has_top_ir:
                insert_columns.append("interview_recommendation")
            if has_top_siq:
                insert_columns.append("suggested_interview_questions")
            insert_columns += ["match_status", "rule_flagged", "created_at", "updated_at"]

            insert_values = [
                "@match_id", "@job_id", "@candidate_id",
                "@semantic_similarity_score",
                "STRUCT(@mand_skills, @mand_experience, @mand_domain, @mand_education, @mand_location)",
                "STRUCT(@sc_overall, @sc_skill, @sc_experience, @sc_domain, @sc_education, @sc_location)",
                "@arr_missing_required", "@arr_missing_preferred", "@arr_key_strengths", "@arr_gaps_concerns",
                "STRUCT(@sc_status, @sc_explanation, @sc_clar_qs, @sc_interview_reco, @sc_suggested_qs)",
            ]
            if has_top_ir:
                insert_values.append("@sc_interview_reco")
            if has_top_siq:
                insert_values.append("@sc_suggested_qs")
            insert_values += ["@match_status", "@rule_flagged", "CURRENT_TIMESTAMP()", "CURRENT_TIMESTAMP()"]

            # Precompute SQL fragments to avoid backslashes inside f-string expressions
            update_sql = ',\n              '.join(update_parts)
            insert_cols_sql = ', '.join(insert_columns)
            insert_vals_sql = ', '.join(insert_values)

            merge_sql = f"""
            MERGE `{self.project_id}.{self.dataset_id}.{self.match_table_name}` T
            USING (SELECT @job_id AS job_id, @candidate_id AS candidate_id) S
            ON T.job_id = S.job_id AND T.candidate_id = S.candidate_id
            WHEN MATCHED THEN UPDATE SET
              {update_sql}
            WHEN NOT MATCHED THEN INSERT (
              {insert_cols_sql}
            ) VALUES (
              {insert_vals_sql}
            )
            """

            params = [
                bigquery.ScalarQueryParameter("job_id", "STRING", job_id),
                bigquery.ScalarQueryParameter("candidate_id", "STRING", candidate_id),
                bigquery.ScalarQueryParameter("match_id", "STRING", new_match_id),
                bigquery.ScalarQueryParameter("semantic_similarity_score", "FLOAT64", float(semantic_score)),
                bigquery.ScalarQueryParameter("mand_skills", "BOOL", mand_skills),
                bigquery.ScalarQueryParameter("mand_experience", "BOOL", mand_experience),
                bigquery.ScalarQueryParameter("mand_domain", "BOOL", mand_domain),
                bigquery.ScalarQueryParameter("mand_education", "BOOL", mand_education),
                bigquery.ScalarQueryParameter("mand_location", "BOOL", mand_location),
                bigquery.ScalarQueryParameter("sc_overall", "FLOAT64", sc_overall),
                bigquery.ScalarQueryParameter("sc_skill", "FLOAT64", sc_skill),
                bigquery.ScalarQueryParameter("sc_experience", "FLOAT64", sc_experience),
                bigquery.ScalarQueryParameter("sc_domain", "FLOAT64", sc_domain),
                bigquery.ScalarQueryParameter("sc_education", "FLOAT64", sc_education),
                bigquery.ScalarQueryParameter("sc_location", "FLOAT64", sc_location),
                bigquery.ArrayQueryParameter("arr_missing_required", "STRING", arr_missing_required),
                bigquery.ArrayQueryParameter("arr_missing_preferred", "STRING", arr_missing_preferred),
                bigquery.ArrayQueryParameter("arr_key_strengths", "STRING", arr_key_strengths),
                bigquery.ArrayQueryParameter("arr_gaps_concerns", "STRING", arr_gaps_concerns),
                bigquery.ScalarQueryParameter("sc_status", "STRING", sc_status),
                bigquery.ScalarQueryParameter("sc_explanation", "STRING", sc_explanation),
                bigquery.ArrayQueryParameter("sc_clar_qs", "STRING", sc_clar_qs),
                bigquery.ScalarQueryParameter("sc_interview_reco", "BOOL", sc_interview_reco),
                bigquery.ArrayQueryParameter("sc_suggested_qs", "STRING", sc_suggested_qs),
                bigquery.ScalarQueryParameter("match_status", "STRING", match_status),
                bigquery.ScalarQueryParameter("rule_flagged", "STRING", rule_flagged),
            ]

            job_config = bigquery.QueryJobConfig(query_parameters=params)

            # Debug: Log sanitized payload snapshot before MERGE
            try:
                debug_payload = {
                    "job_id": job_id,
                    "candidate_id": candidate_id,
                    "semantic_similarity_score": semantic_score,
                    "rule_flagged": rule_flagged,
                    "screening_decision": {
                        "status": sc_status,
                        "explanation": sc_explanation,
                        "clarification_questions": sc_clar_qs[:5],
                    },
                    "interview_recommendation": sc_interview_reco,
                    "suggested_interview_questions": sc_suggested_qs[:5],
                    "mandatory_requirements": {
                        "skills_requirement_met": mand_skills,
                        "experience_requirement_met": mand_experience,
                        "domain_requirement_met": mand_domain,
                        "education_requirement_met": mand_education,
                        "location_requirement_met": mand_location,
                    },
                    "match_scores": {
                        "overall_match_score": sc_overall,
                        "skill_match_score": sc_skill,
                        "experience_match_score": sc_experience,
                        "domain_match_score": sc_domain,
                        "education_match_score": sc_education,
                        "location_match_score": sc_location,
                    },
                    "missing_required_skills": arr_missing_required[:10],
                    "missing_preferred_skills": arr_missing_preferred[:10],
                    "key_strengths": arr_key_strengths[:10],
                    "gaps_and_concerns": arr_gaps_concerns[:10],
                    "match_status": match_status,
                }
                logger.debug("Upserting match with payload: " + json.dumps(debug_payload)[:1500])
            except Exception:
                pass

            query_job = self.bq_client.query(merge_sql, job_config=job_config)
            query_job.result()

            # Return the existing match_id if present, else the newly created one
            return existing_id or new_match_id

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

    def _touch_existing_match(self, job_id, candidate_id):
        """Update only the updated_at field for an existing match row.

        This avoids creating duplicate rows while marking the record as recently processed.
        """
        query = f"""
        UPDATE `{self.project_id}.{self.dataset_id}.{self.match_table_name}`
        SET updated_at = CURRENT_TIMESTAMP()
        WHERE job_id = @job_id AND candidate_id = @candidate_id
        """
        query_params = [
            bigquery.ScalarQueryParameter("job_id", "STRING", job_id),
            bigquery.ScalarQueryParameter("candidate_id", "STRING", candidate_id),
        ]
        job_config = bigquery.QueryJobConfig(query_parameters=query_params)
        logger.debug(f"Touching existing match updated_at for candidate_id={candidate_id}, job_id={job_id}")
        query_job = self.bq_client.query(query, job_config=job_config)
        query_job.result()

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

    def _find_job_id_by_title(self, job_title: str) -> Optional[str]:
        """Try to resolve a job_id from a job title string using a case-insensitive LIKE query.

        Returns the most recently posted matching job_id or None if no match.
        """
        try:
            if not job_title or not isinstance(job_title, str):
                return None

            # Simple sanitization and wildcard search
            clean_title = job_title.strip().lower()
            # Use LIKE with wildcards around the title terms to allow partial matches
            like_pattern = f"%{clean_title}%"

            query = f"""
            SELECT job_id
            FROM `{self.project_id}.{self.dataset_id}.{self.jd_table_name}`
            WHERE LOWER(job_title) LIKE @like_pattern
            ORDER BY posting_date DESC
            LIMIT 1
            """

            job_config = bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ScalarQueryParameter("like_pattern", "STRING", like_pattern)
                ]
            )

            results = list(self.bq_client.query(query, job_config=job_config).result())
            if not results:
                self.logger.debug(f"No JD found matching title pattern: {clean_title}")
                return None

            row = results[0]
            return row.get("job_id")
        except Exception as e:
            # Do not raise — return None to allow coordinator fallback
            try:
                self.logger.debug(f"Error finding job_id by title '{job_title}': {e}")
            except Exception:
                pass
            return None

    def workflow2_resume_to_jds(self, candidate_id, metadata_filters=None, limit=10, job_id=None):
        """
        Workflow to match a single resume (by candidate_id) against job descriptions.
        If job_id is provided, match the resume only against that single job.
        Otherwise, perform a vector search across all JDs (existing behavior).
        """
        try:
            logger.info(f"workflow2 start: candidate_id={candidate_id}, limit={limit}, job_id={job_id}")
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

            # If a specific job_id is provided, match only against that job
            if job_id:
                logger.debug(f"Running single-JD match for job_id={job_id}")
                jd_data = self._fetch_jd_data(job_id)
                if not jd_data:
                    logger.warning(f"Could not fetch data for job_id {job_id}")
                    return {
                        "candidate_id": candidate_id,
                        "candidate_name": candidate_name,
                        "total_jobs_processed": 0,
                        "matches": [],
                        "error": f"Job with job_id {job_id} not found."
                    }

                jd_embedding = jd_data.get("embeddings")
                if not jd_embedding:
                    logger.info(f"JD embedding not found for job_id={job_id}, generating a new one.")
                    jd_text = jd_data.get("job_summary") or jd_data.get("job_description") or ""
                    if not jd_text:
                        logger.warning(f"No text available to generate embedding for job_id={job_id}")
                        return {
                            "candidate_id": candidate_id,
                            "candidate_name": candidate_name,
                            "total_jobs_processed": 0,
                            "matches": [],
                            "error": "Job text empty, cannot generate embedding."
                        }
                    jd_embedding = self._generate_and_store_embedding(
                        jd_text, self.jd_table_name, "job_id", job_id
                    )
                    if not jd_embedding:
                        return {"error": "Failed to generate and store JD embedding."}

                # Calculate semantic similarity between the resume and the single JD
                semantic_score = self._calculate_semantic_match(jd_embedding, resume_embedding)
                logger.info(f"Single-JD LLM match: candidate_id={candidate_id}, job_id={job_id}, semantic_score={semantic_score:.4f}")

                # Fetch full resume data for detailed match
                full_resume_data = self._fetch_resume_data(candidate_id)
                if not full_resume_data:
                    logger.warning(f"Could not fetch full resume data for {candidate_id}")
                    return {
                        "candidate_id": candidate_id,
                        "candidate_name": candidate_name,
                        "total_jobs_processed": 0,
                        "matches": [],
                        "error": "Full resume data missing after embedding generation."
                    }

                try:
                    match_details = self._detailed_llm_match(jd_data, full_resume_data)
                    match_results = []
                    match_id = self._store_match_results(job_id, candidate_id, semantic_score, match_details)
                    if match_id:
                        match_results.append({
                            "match_id": match_id,
                            "job_id": job_id,
                            "candidate_id": candidate_id,
                            "candidate_name": candidate_name,
                            "semantic_score": semantic_score,
                            "match_details": match_details
                        })

                    logger.info(f"workflow2 single-JD complete: candidate_id={candidate_id}, job_id={job_id}, matches={len(match_results)}")
                    return {
                        "candidate_id": candidate_id,
                        "candidate_name": candidate_name,
                        "total_jobs_processed": 1,
                        "matches": match_results,
                    }

                except Exception as match_error:
                    logger.error(f"Error processing single-job match for candidate {candidate_id}: {str(match_error)}", exc_info=True)
                    return {
                        "candidate_id": candidate_id,
                        "candidate_name": candidate_name,
                        "total_jobs_processed": 1,
                        "matches": [],
                        "error": str(match_error)
                    }

            # Rest of the method remains the same (vector search across all JDs)
            logger.debug("Running JD vector search for candidate")
            vector_results = self._vector_search_jds_query(resume_embedding, limit=limit)
            if not vector_results:
                logger.info("No potential matches found via vector search.")
                return {
                    "candidate_id": candidate_id,
                    "candidate_name": candidate_name,  # Added candidate name here
                    "matches": []
                }

            # Deduplicate results by job_id
            seen_job_ids = set()
            unique_results = []
            for r in vector_results:
                jid = r.get("job_id")
                if jid in seen_job_ids:
                    continue
                seen_job_ids.add(jid)
                unique_results.append(r)

            if len(unique_results) != len(vector_results):
                logger.debug(f"Vector search returned {len(vector_results)} results; deduped to {len(unique_results)} unique job_ids")
            else:
                logger.debug(f"Vector search returned {len(vector_results)} unique results")

            match_results = []
            for match in unique_results:
                try:
                    job_id = match["job_id"]
                    logger.debug(f"Fetching JD data for job_id={job_id}")
                    jd_data = self._fetch_jd_data(job_id)
                    if not jd_data:
                        logger.warning(f"Could not fetch data for job_id {job_id}, skipping.")
                        continue

                    semantic_score = 1 - match["distance"]  # Cosine distance to similarity
                    logger.info(f"LLM match: candidate_id={candidate_id}, job_id={job_id}, semantic_score={semantic_score:.4f}")

                    logger.debug("Fetching full resume data for detailed match")
                    full_resume_data = self._fetch_resume_data(candidate_id)
                    if not full_resume_data:
                        logger.warning(f"Could not fetch full resume data for {candidate_id}")
                        continue

                    try:
                        logger.debug("Calling _detailed_llm_match")
                        match_details = self._detailed_llm_match(jd_data, full_resume_data)

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