import logging
from google.cloud import bigquery
from google.api_core import exceptions

logger = logging.getLogger('Agent.SchemaManager')

# Definitive schema for the job_details table
JOB_DETAILS_SCHEMA = [
    bigquery.SchemaField("job_id", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("customer_type", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("recruiter_email", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("prime_vendor_email", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("sub_vendor_email", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("prime_vendor_name", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("sub_vendor_name", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("end_client_name", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("job_title", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("job_description", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("department", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("is_relocation_allowed", "BOOLEAN", mode="NULLABLE"),
    bigquery.SchemaField("is_remote", "BOOLEAN", mode="NULLABLE"),
    bigquery.SchemaField("job_location_city", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("job_location_state", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("job_location_country", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("gcs_path", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("is_only_ex_employee_required", "BOOLEAN", mode="NULLABLE"),
    bigquery.SchemaField("required_technical_skills", "STRING", mode="REPEATED"),
    bigquery.SchemaField("preferred_technical_skills", "STRING", mode="REPEATED"),
    bigquery.SchemaField("job_domains", "RECORD", mode="REPEATED", fields=[
        bigquery.SchemaField("domain_name", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("requirement_type", "STRING", mode="NULLABLE"),
    ]),
    bigquery.SchemaField("preferred_previous_companies", "STRING", mode="REPEATED"),
    bigquery.SchemaField("min_required_years_experience", "FLOAT", mode="NULLABLE"),
    bigquery.SchemaField("max_preferred_years_experience", "FLOAT", mode="NULLABLE"),
    bigquery.SchemaField("required_skill_experience_years", "RECORD", mode="REPEATED", fields=[
        bigquery.SchemaField("skill", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("min_years", "FLOAT", mode="NULLABLE"),
    ]),
    bigquery.SchemaField("job_location_city_state", "STRING", mode="REPEATED"),
    bigquery.SchemaField("job_work_model", "STRING", mode="REPEATED"),
    bigquery.SchemaField("job_employment_type", "STRING", mode="REPEATED"),
    bigquery.SchemaField("job_geography", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("job_certifications", "STRING", mode="REPEATED"),
    bigquery.SchemaField("optional_requirements_list", "STRING", mode="REPEATED"),
    bigquery.SchemaField("questionnaire_details", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("posting_date", "DATE", mode="NULLABLE"),
    bigquery.SchemaField("closing_date", "DATE", mode="NULLABLE"),
    bigquery.SchemaField("job_summary", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("embeddings", "FLOAT", mode="REPEATED"),
    bigquery.SchemaField("job_status", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("created_timestamp", "TIMESTAMP", mode="REQUIRED"),
    bigquery.SchemaField("last_updated_timestamp", "TIMESTAMP", mode="REQUIRED"),
]

# Definitive schema for the matches table
MATCHES_SCHEMA = [
    bigquery.SchemaField("match_id", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("job_id", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("candidate_id", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("semantic_similarity_score", "FLOAT", mode="REQUIRED"),
    bigquery.SchemaField("mandatory_requirements", "RECORD", mode="NULLABLE", fields=[
        bigquery.SchemaField("skills_requirement_met", "BOOLEAN", mode="NULLABLE"),
        bigquery.SchemaField("experience_requirement_met", "BOOLEAN", mode="NULLABLE"),
        bigquery.SchemaField("domain_requirement_met", "BOOLEAN", mode="NULLABLE"),
        bigquery.SchemaField("education_requirement_met", "BOOLEAN", mode="NULLABLE"),
        bigquery.SchemaField("location_requirement_met", "BOOLEAN", mode="NULLABLE"),
    ]),
    bigquery.SchemaField("match_scores", "RECORD", mode="NULLABLE", fields=[
        bigquery.SchemaField("overall_match_score", "FLOAT", mode="NULLABLE"),
        bigquery.SchemaField("skill_match_score", "FLOAT", mode="NULLABLE"),
        bigquery.SchemaField("experience_match_score", "FLOAT", mode="NULLABLE"),
        bigquery.SchemaField("domain_match_score", "FLOAT", mode="NULLABLE"),
        bigquery.SchemaField("education_match_score", "FLOAT", mode="NULLABLE"),
        bigquery.SchemaField("location_match_score", "FLOAT", mode="NULLABLE"),
    ]),
    bigquery.SchemaField("missing_required_skills", "STRING", mode="REPEATED"),
    bigquery.SchemaField("missing_preferred_skills", "STRING", mode="REPEATED"),
    bigquery.SchemaField("key_strengths", "STRING", mode="REPEATED"),
    bigquery.SchemaField("gaps_and_concerns", "STRING", mode="REPEATED"),
    bigquery.SchemaField("screening_decision", "RECORD", mode="NULLABLE", fields=[
        bigquery.SchemaField("status", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("explanation", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("clarification_questions", "STRING", mode="REPEATED"),
    ]),
    bigquery.SchemaField("interview_recommendation", "BOOLEAN", mode="NULLABLE"),
    bigquery.SchemaField("suggested_interview_questions", "STRING", mode="REPEATED"),
    bigquery.SchemaField("rule_flagged", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("match_status", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("created_at", "TIMESTAMP", mode="NULLABLE"),
    bigquery.SchemaField("updated_at", "TIMESTAMP", mode="NULLABLE"),
]

# Definitive schema for the resumes table
RESUMES_SCHEMA = [
    bigquery.SchemaField("candidate_id", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("candidate_name", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("candidate_email", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("candidate_phone", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("candidate_location", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("linkedin_url", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("date_of_birth", "DATE", mode="NULLABLE"),
    bigquery.SchemaField("resume_summary", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("total_years_experience", "FLOAT", mode="NULLABLE"),
    bigquery.SchemaField("technical_skills", "STRING", mode="REPEATED"),
    bigquery.SchemaField("soft_skills", "STRING", mode="REPEATED"),
    bigquery.SchemaField("skill_proficiency", "RECORD", mode="REPEATED", fields=[
        bigquery.SchemaField("skill", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("level", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("years_experience", "FLOAT", mode="NULLABLE"),
    ]),
    bigquery.SchemaField("experience", "RECORD", mode="REPEATED", fields=[
        bigquery.SchemaField("company", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("role", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("start_date", "DATE", mode="NULLABLE"),
        bigquery.SchemaField("end_date", "DATE", mode="NULLABLE"),
        bigquery.SchemaField("duration_months", "INTEGER", mode="NULLABLE"),
        bigquery.SchemaField("location", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("technologies_used", "STRING", mode="REPEATED"),
    ]),
    bigquery.SchemaField("education", "RECORD", mode="REPEATED", fields=[
        bigquery.SchemaField("degree", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("institution", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("field_of_study", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("graduation_year", "INTEGER", mode="NULLABLE"),
    ]),
    bigquery.SchemaField("certifications", "STRING", mode="REPEATED"),
    bigquery.SchemaField("previous_companies", "STRING", mode="REPEATED"),
    bigquery.SchemaField("domains_worked_in", "STRING", mode="REPEATED"),
    bigquery.SchemaField("languages", "RECORD", mode="REPEATED", fields=[
        bigquery.SchemaField("language", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("proficiency", "STRING", mode="NULLABLE"),
    ]),
    bigquery.SchemaField("visa_status", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("gaps_in_experience", "RECORD", mode="REPEATED", fields=[
        bigquery.SchemaField("start_date", "DATE", mode="NULLABLE"),
        bigquery.SchemaField("end_date", "DATE", mode="NULLABLE"),
        bigquery.SchemaField("duration_months", "INTEGER", mode="NULLABLE"),
        bigquery.SchemaField("reason", "STRING", mode="NULLABLE"),
    ]),
    bigquery.SchemaField("discrepancies", "RECORD", mode="REPEATED", fields=[
        bigquery.SchemaField("type", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("description", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("severity", "STRING", mode="NULLABLE"),
    ]),
    bigquery.SchemaField("metrics", "RECORD", mode="NULLABLE", fields=[
        bigquery.SchemaField("resume_quality_score", "FLOAT", mode="NULLABLE"),
        bigquery.SchemaField("possible_ai_generated", "BOOLEAN", mode="NULLABLE"),
        bigquery.SchemaField("missing_information", "STRING", mode="REPEATED"),
        bigquery.SchemaField("inconsistencies", "STRING", mode="REPEATED"),
    ]),
    bigquery.SchemaField("work_preferences", "RECORD", mode="NULLABLE", fields=[
        bigquery.SchemaField("preferred_work_models", "STRING", mode="REPEATED"),
        bigquery.SchemaField("preferred_employment_types", "STRING", mode="REPEATED"),
        bigquery.SchemaField("preferred_locations", "STRING", mode="REPEATED"),
        bigquery.SchemaField("willing_to_relocate", "BOOLEAN", mode="NULLABLE"),
        bigquery.SchemaField("salary_expectations", "STRING", mode="NULLABLE"),
    ]),
    bigquery.SchemaField("resume_text", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("storage_uri", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("embeddings", "FLOAT", mode="REPEATED"),
    bigquery.SchemaField("created_at", "TIMESTAMP", mode="NULLABLE"),
    bigquery.SchemaField("updated_at", "TIMESTAMP", mode="NULLABLE"),
]

TABLE_SCHEMAS = {
    'job_details': JOB_DETAILS_SCHEMA,
    'resumes': RESUMES_SCHEMA,
    'matches': MATCHES_SCHEMA
}

def ensure_table_exists(client: bigquery.Client, table_id: str) -> None:
    table_name = table_id.split('.')[-1]
    schema = TABLE_SCHEMAS.get(table_name)

    if not schema:
        logger.error(f"Schema not found for table '{table_name}'. Cannot create table.")
        raise ValueError(f"No schema defined for table {table_name}")

    try:
        client.get_table(table_id)
        logger.info(f"Table {table_id} already exists.")
    except exceptions.NotFound:
        logger.warning(f"Table {table_id} not found. Creating it now.")
        try:
            table = bigquery.Table(table_id, schema=schema)

            # Set clustering fields for better query performance
            if table_name == 'job_details':
                table.clustering_fields = ['job_id']
            elif table_name == 'resumes':
                table.clustering_fields = ['candidate_id']
            elif table_name == 'matches':
                table.clustering_fields = ['job_id', 'candidate_id']

            # Create table
            table = client.create_table(table)
            logger.info(f"Successfully created table {table_id}.")
        except Exception as e:
            logger.error(f"Failed to create table {table_id}: {e}", exc_info=True)
            raise
