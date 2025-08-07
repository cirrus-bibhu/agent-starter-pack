from datetime import datetime, date
import json
import os
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
                print(f"Failed to generate embeddings for {id_field_name}={id_value}")
                return None
                
            query = f"""
            UPDATE `{self.project_id}.{self.dataset_id}.{table_name}`
            SET embeddings = ARRAY{json.dumps(embeddings)}
            WHERE {id_field_name} = '{id_value}'
            """
            
            print(f"DEBUG: Executing query to update embeddings: {query[:100]}...")
            self.bq_client.query(query).result()
            print(f"Successfully updated embeddings for {id_field_name}={id_value}")
            
            return embeddings
        except Exception as e:
            print(f"Error in _generate_and_store_embedding: {str(e)}")
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
                print("Missing embedding data for semantic match calculation")
                return 0.0

            if isinstance(jd_embedding, str):
                try:
                    jd_embedding = json.loads(jd_embedding)
                except json.JSONDecodeError:
                    if jd_embedding.startswith("gs://"):
                        print(
                            f"JD embedding is a GCS URI, this should be resolved before calling this method")
                        return 0.0
                    else:
                        print(
                            f"Failed to parse JD embedding from string: {jd_embedding[:50]}...")
                        return 0.0

            if isinstance(resume_embedding, str):
                try:
                    resume_embedding = json.loads(resume_embedding)
                except json.JSONDecodeError:
                    if resume_embedding.startswith("gs://"):
                        print(
                            f"Resume embedding is a GCS URI, this should be resolved before calling this method")
                        return 0.0
                    else:
                        print(
                            f"Failed to parse resume embedding from string: {resume_embedding[:50]}...")
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
                print(
                    f"Invalid embedding format after normalization: JD type={type(jd_embedding)}, Resume type={type(resume_embedding)}")
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
                    print(
                        "Failed to convert embeddings to list format after all attempts")
                    return 0.0

            if not all(isinstance(x, (int, float)) for x in jd_embedding) or \
               not all(isinstance(x, (int, float)) for x in resume_embedding):
                print("Embeddings contain non-numeric elements")
                return 0.0

            return self._cosine_similarity(jd_embedding, resume_embedding)
        except Exception as e:
            print(f"Error calculating semantic match: {str(e)}")
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
            return results

        except Exception as e:
            print(f"Error in vector search query: {str(e)}")
            return []

    def _detailed_llm_match(self, jd_data, resume_data):
        try:
            prompt = PromptTemplate.from_template(MATCH_PROMPT)

            chain = prompt | self.llm | StrOutputParser()

            jd_json = json.dumps(jd_data, indent=2, default=default_json_serializer)
            resume_json = json.dumps(resume_data, indent=2, default=default_json_serializer)

            response = chain.invoke({
                "jd_json": jd_json,
                "resume_json": resume_json,
            })

            if response.strip().startswith("```json"):
                response = response.strip()[7:-4]

            return json.loads(response)

        except Exception as e:
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

            row_to_insert.update(match_details)

            errors = self.bq_client.insert_rows_json(f"{self.project_id}.{self.dataset_id}.{self.match_table_name}", [row_to_insert])
            if errors:
                raise Exception(f"Failed to insert match result into BigQuery: {errors}")
            
            # Removed email notification call
            return match_id

        except Exception as e:
            print(f"Error storing match results: {str(e)}")
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
            return len(results) > 0
        except Exception as e:
            print(f"Error checking for existing match: {str(e)}")
            return False

    def _get_email_template(self, template_name):
        try:
            template_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                'client_email_templates',
                template_name
            )
            with open(template_path, 'r') as f:
                return f.read()
        except Exception as e:
            print(f"Error loading email template {template_name}: {str(e)}")
            return None


    def _store_match_result(self, job_id, candidate_id, semantic_score, match_details):
        try:
            match_id = str(uuid.uuid4())
            now = datetime.now().isoformat()

            match_status = match_details.get(
                "screening_decision", {}).get("status")

            if not match_status:
                overall_score = match_details.get(
                    "match_scores", {}).get("overall_match_score", 0)
                if overall_score >= 80:
                    match_status = "Proceed Ahead"
                elif overall_score >= 60:
                    match_status = "Clarification Needed"
                else:
                    match_status = "Reject"

            row = {
                "match_id": match_id,
                "job_id": job_id,
                "candidate_id": candidate_id,
                "semantic_similarity_score": semantic_score,
                "mandatory_requirements": {
                    "skills_requirement_met": match_details.get("mandatory_requirements", {}).get("skills_requirement_met", False),
                    "experience_requirement_met": match_details.get("mandatory_requirements", {}).get("experience_requirement_met", False),
                    "domain_requirement_met": match_details.get("mandatory_requirements", {}).get("domain_requirement_met", False),
                    "education_requirement_met": match_details.get("mandatory_requirements", {}).get("education_requirement_met", False),
                    "location_requirement_met": match_details.get("mandatory_requirements", {}).get("location_requirement_met", False)
                },
                "match_scores": {
                    "overall_match_score": match_details.get("match_scores", {}).get("overall_match_score", 0),
                    "skill_match_score": match_details.get("match_scores", {}).get("skill_match_score", 0),
                    "experience_match_score": match_details.get("match_scores", {}).get("experience_match_score", 0),
                    "domain_match_score": match_details.get("match_scores", {}).get("domain_match_score", 0),
                    "education_match_score": match_details.get("match_scores", {}).get("education_match_score", 0),
                    "location_match_score": match_details.get("match_scores", {}).get("location_match_score", 0)
                },
                "missing_required_skills": match_details.get("missing_required_skills", []),
                "missing_preferred_skills": match_details.get("missing_preferred_skills", []),
                "key_strengths": match_details.get("key_strengths", []),
                "gaps_and_concerns": match_details.get("gaps_and_concerns", []),
                "screening_decision": {
                    "status": match_status,
                    "explanation": match_details.get("screening_decision", {}).get("explanation", ""),
                    "clarification_questions": match_details.get("screening_decision", {}).get("clarification_questions", []),
                    "interview_recommendation": match_details.get("screening_decision", {}).get("interview_recommendation", False),
                    "suggested_interview_questions": match_details.get("screening_decision", {}).get("suggested_interview_questions", [])
                },
                "match_status": match_status,
                "created_at": now,
                "updated_at": now
            }

            errors = self.bq_client.insert_rows_json(
                f"{self.project_id}.{self.dataset_id}.{self.match_table_name}",
                [row]
            )

            if errors:
                raise Exception(f"Error inserting row into BigQuery: {errors}")
            
            return match_id
        except Exception as e:
            raise Exception(f"Error storing match result: {str(e)}")
            
    def validate_graduation_age(self, graduation_year, birth_year):
        try:
            graduation_age = graduation_year - birth_year
            is_within_range = 21 <= graduation_age <= 24
            
            return {
                "graduation_age": graduation_age,
                "is_within_expected_range": is_within_range
            }
        except Exception as e:
            raise Exception(f"Error validating graduation age: {str(e)}")

    def workflow1_jd_to_resumes(self, job_id, metadata_filters=None, limit=10, batch_size=None):
        try:
            jd_data = self._fetch_jd_data(job_id)
            if not jd_data:
                return {
                    "job_id": job_id,
                    "error": f"JD with ID {job_id} not found",
                    "matches": [],
                    "status": "Failed",
                }

            jd_embedding = jd_data.get("embeddings")
            if not jd_embedding:
                try:
                    jd_text = jd_data.get("job_description", "")
                    jd_summary = jd_data.get("summary", "")
                    technical_requirements = ", ".join(jd_data.get("required_technical_skills", []))
                    soft_requirements = ", ".join(jd_data.get("soft_skills", []))
                    role_context = jd_data.get("role_context", "")
                    
                    combined_text = f"""
                    Job Description: {jd_text}
                    Summary: {jd_summary}
                    Technical Requirements: {technical_requirements}
                    Soft Skills: {soft_requirements}
                    Role Context: {role_context}
                    """.strip()
                    
                    jd_embedding = self._generate_and_store_embedding(
                        combined_text,
                        self.jd_table_name,
                        "job_id",
                        job_id
                    )
                    if not jd_embedding:
                        raise Exception("Failed to generate embeddings")
                        
                except Exception as e:
                    print(f"Error generating embeddings for JD {job_id}: {str(e)}")
                    return {
                        "job_id": job_id,
                        "error": "Failed to generate embeddings",
                        "matches": [],
                        "status": "Failed",
                    }

            if not metadata_filters:
                metadata_filters = {
                    "min_experience": jd_data.get("min_required_years_experience"),
                    "max_experience": jd_data.get("max_preferred_years_experience"),
                    "locations": jd_data.get("job_location_city_state", []),
                    "work_models": jd_data.get("job_work_model", []),
                    "employment_types": jd_data.get("job_employment_type", []),
                    "required_skills": jd_data.get("required_technical_skills", []),
                    "domains": jd_data.get("domains", []),
                    "certifications": jd_data.get("certifications", [])
                }

            min_exp_filter = "r.total_years_experience >= IFNULL({}, 0)"
            
            if metadata_filters and 'min_experience' in metadata_filters and metadata_filters['min_experience'] is not None:
                min_exp_filter = min_exp_filter.format(metadata_filters['min_experience'])
            else:
                min_exp_filter = min_exp_filter.format(0)
                
            location_filter = ""
            if metadata_filters and 'locations' in metadata_filters and metadata_filters['locations']:
                location_filter = " AND r.candidate_location IN UNNEST(@locations)"
                
            query = f"""
            WITH query_embedding AS (
                SELECT ARRAY{json.dumps(jd_embedding)} AS embedding
            )
            SELECT 
                r.candidate_id, r.candidate_name, 
                1 - COSINE_DISTANCE(r.embeddings, (SELECT embedding FROM query_embedding)) as similarity_score
            FROM 
                `{self.project_id}.{self.dataset_id}.{self.resume_table_name}` r
            WHERE 
                {min_exp_filter}
                AND ARRAY_LENGTH(r.embeddings) = {len(jd_embedding)} -- Ensure embedding dimensions match
                {location_filter}
            ORDER BY 
                similarity_score DESC
            LIMIT {limit * 2}
            """

            query_params = []
            if metadata_filters['locations']:
                query_params.append(bigquery.ArrayQueryParameter("locations", "STRING", metadata_filters['locations']))
            
            job_config = bigquery.QueryJobConfig(query_parameters=query_params) if query_params else None
            
            vector_results = list(self.bq_client.query(query, job_config=job_config).result())
            
            print(f"DEBUG: Found {len(vector_results)} results from vector search")
            
            semantic_matches = []
            for result in vector_results:
                resume_data = {}
                try:
                    for key, value in result.items():
                        resume_data[key] = value
                    
                    candidate_id = resume_data.get("candidate_id")
                    candidate_name = resume_data.get("candidate_name")
                    score = resume_data.get("similarity_score", 0)
                    print(f"DEBUG: Found candidate {candidate_name} (ID: {candidate_id}) with score {score:.4f}")
                    
                    semantic_matches.append({
                        "candidate_id": candidate_id,
                        "candidate_name": candidate_name,
                        "semantic_score": score,
                        "resume_data": resume_data
                    })
                except Exception as e:
                    print(f"DEBUG: Error processing result: {str(e)}")
                    continue

            if not semantic_matches:
                return {
                    "job_id": job_id,
                    "job_title": jd_data.get("job_title"),
                    "total_candidates_processed": 0,
                    "matches": [],
                    "message": "No suitable matches found",
                }

            semantic_matches.sort(key=lambda x: x["semantic_score"], reverse=True)
            top_matches = semantic_matches[:limit]

            match_results = []
            for match in top_matches:
                try:
                    graduation_year = None
                    if "education" in match["resume_data"]:
                        for edu in match["resume_data"].get("education", []):
                            if edu.get("degree_type", "").lower() in ["bachelor", "bachelors", "bachelor's", "b.s.", "b.a.", "bs", "ba"]:
                                graduation_year = edu.get("graduation_year")
                                break
                    
                    birth_year = None
                    if "personal_info" in match["resume_data"]:
                        birth_year = match["resume_data"].get("personal_info", {}).get("year_of_birth")
                    
                    graduation_age_validation = None
                    if graduation_year and birth_year:
                        try:
                            graduation_age_validation = self.validate_graduation_age(graduation_year, birth_year)
                        except Exception as e:
                            print(f"Error validating graduation age: {str(e)}")
                    
                    match_details = self._detailed_llm_match(jd_data, match["resume_data"])
                    
                    if graduation_age_validation:
                        if "education_validation" not in match_details:
                            match_details["education_validation"] = {}
                        match_details["education_validation"]["graduation_age"] = graduation_age_validation

                    match_id = self._store_match_result(
                        job_id,
                        match["candidate_id"],
                        match["semantic_score"],
                        match_details,
                    )

                    match_status = match_details.get("screening_decision", {}).get("status")
                    if not match_status:
                        overall_score = match_details.get("match_scores", {}).get("overall_match_score", 0)
                        match_status = "Proceed Ahead" if overall_score >= 80 else \
                                    "Clarification Needed" if overall_score >= 60 else \
                                    "Reject"

                    result = {
                        "match_id": match_id,
                        "candidate_id": match["candidate_id"],
                        "candidate_name": match["resume_data"].get("candidate_name"),
                        "candidate_email": match["resume_data"].get("candidate_email"),
                        "candidate_phone": match["resume_data"].get("candidate_phone"),
                        "candidate_location": match["resume_data"].get("candidate_location"),
                        "total_experience": match["resume_data"].get("total_years_experience"),
                        "semantic_score": match["semantic_score"],
                        "match_scores": match_details.get("match_scores", {}),
                        "screening_status": match_status,
                        "mandatory_requirements_met": match_details.get("mandatory_requirements", {}),
                        "technical_skills_match": list(set(jd_data.get("required_technical_skills", [])) & 
                                                    set(match["resume_data"].get("technical_skills", []))),
                        "key_strengths": match_details.get("key_strengths", []),
                        "missing_required_skills": match_details.get("missing_required_skills", []),
                        "missing_preferred_skills": match_details.get("missing_preferred_skills", []),
                        "gaps_and_concerns": match_details.get("gaps_and_concerns", []),
                        "interview_recommended": match_details.get("screening_decision", {}).get("interview_recommendation", False),
                        "explanation": match_details.get("screening_decision", {}).get("explanation", ""),
                        "clarification_questions": match_details.get("screening_decision", {}).get("clarification_questions", []),
                        "suggested_interview_questions": match_details.get("screening_decision", {}).get("suggested_interview_questions", []),
                        "education_validation": match_details.get("education_validation", {}),
                    }
                    match_results.append(result)
                    
                except Exception as match_error:
                    print(f"Error processing match for candidate {match['candidate_id']}: {str(match_error)}")

            return {
                "job_id": job_id,
                "job_title": jd_data.get("job_title"),
                "total_candidates_processed": len(vector_results),
                "candidates_matched": len(semantic_matches),
                "matches": match_results,
            }

        except Exception as e:
            print(f"An unexpected error occurred in workflow1_jd_to_resumes: {str(e)}")
            return {
                "job_id": job_id,
                "error": str(e),
                "matches": [],
                "status": "Failed",
            }

    def _fetch_all_jds(self, filters=None):
        """Fetches all job descriptions from BigQuery, optionally applying filters."""
        try:
            query = f"SELECT job_id, job_description, embeddings FROM `{self.project_id}.{self.dataset_id}.{self.jd_table_name}`"
            if filters:
                where_clauses = " AND ".join(filters)
                query += f" WHERE {where_clauses}"
            
            print(f"DEBUG: Executing query to fetch all JDs: {query[:150]}...")
            query_job = self.bq_client.query(query)
            results = [dict(row) for row in query_job.result()]
            print(f"Successfully fetched {len(results)} job descriptions.")
            return results
        except Exception as e:
            print(f"Error fetching all job descriptions: {str(e)}")
            return []

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
            print(f"DEBUG: Executing vector search query on JDs: {query[:150]}...")
            query_job = self.bq_client.query(query)
            results = [{"job_id": row["job_id"], "distance": row["distance"]} for row in query_job.result()]
            print(f"Vector search found {len(results)} potential JD matches.")
            return results
        except Exception as e:
            print(f"Error performing vector search on JDs: {str(e)}")
            return []

    def workflow2_resume_to_jds(self, candidate_id, metadata_filters=None, limit=10):
        """
        Workflow to match a single resume (by candidate_id) against all job descriptions.
        """
        try:
            print(f"Starting workflow2: Match resume {candidate_id} to all JDs.")
            resume_data = self._fetch_resume_data(candidate_id)
            if not resume_data:
                return {"error": f"Resume with candidate_id {candidate_id} not found."}

            # Get candidate name early
            candidate_name = resume_data.get("candidate_name", "Unknown Candidate")

            resume_embedding = resume_data.get("embeddings")
            if not resume_embedding:
                print("Resume embedding not found, generating a new one.")
                resume_text = resume_data.get("resume_text", "")
                if not resume_text:
                    return {"error": "Resume text is empty, cannot generate embedding."}
                resume_embedding = self._generate_and_store_embedding(
                    resume_text, self.resume_table_name, "candidate_id", candidate_id
                )
                if not resume_embedding:
                    return {"error": "Failed to generate and store resume embedding."}

            # Rest of the method remains the same
            vector_results = self._vector_search_jds_query(resume_embedding, limit=limit)
            if not vector_results:
                print("No potential matches found via vector search.")
                return {
                    "candidate_id": candidate_id,
                    "candidate_name": candidate_name,  # Added candidate name here
                    "matches": []
                }

            match_results = []
            for match in vector_results:
                try:
                    job_id = match["job_id"]
                    jd_data = self._fetch_jd_data(job_id)
                    if not jd_data:
                        print(f"Could not fetch data for job_id {job_id}, skipping.")
                        continue

                    semantic_score = 1 - match["distance"] # Cosine distance to similarity
                    
                    print(f"Performing detailed LLM match for resume {candidate_id} and job {job_id}...")
                    if self._check_for_existing_match(job_id, candidate_id):
                        print(f"DEBUG: Match already exists for candidate {candidate_id} and job {job_id}. Skipping.")
                        continue

                    full_resume_data = self._fetch_resume_data(candidate_id)
                    if not full_resume_data:
                        print(f"Could not fetch full resume data for {candidate_id}")
                        continue

                    try:
                        match_details = self._detailed_llm_match(
                            jd_data, full_resume_data)
                        
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
                        print(
                            f"Error processing match for candidate {candidate_id}: {str(match_error)}")

                except Exception as e:
                    print(f'Error processing match for job {match.get("job_id")}: {str(e)}')

            return {
                "candidate_id": candidate_id,
                "candidate_name": resume_data.get("candidate_name"),
                "total_jobs_processed": len(vector_results),
                "matches": match_results,
            }

        except Exception as e:
            print(f"An unexpected error occurred in workflow2_resume_to_jds: {str(e)}")
            return {
                "candidate_id": candidate_id,
                "error": str(e),
                "matches": [],
                "status": "Failed",
            }