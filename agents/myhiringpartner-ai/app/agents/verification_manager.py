import logging
import json
import os
import re
from datetime import date, datetime
import google.generativeai as genai
from ..tools.linkedin_scrapper import ProfileExtractor as LinkedinScraper
from google.cloud import bigquery
from app.agent import BaseAgent
from app.config import config
import time
import backoff
from google.api_core import exceptions

logger = logging.getLogger('Agent.VerificationManager')

class VerificationManagerAgent(BaseAgent):
    VERIFICATION_PROMPT = """You are a world-class resume verification expert. Your task is to conduct a comprehensive analysis of a candidate's resume and LinkedIn profile to identify discrepancies, gaps, and inconsistencies. You will be provided with the candidate's resume and LinkedIn data in JSON format.

**Analysis Dimensions:**

1.  **Discrepancy Analysis:**
    *   **LinkedIn vs. Resume:** Compare job titles, companies, and dates. Note any differences.
    *   **Experience Plausibility:** Assess if skills and experience align. Flag exaggerated claims.
    *   **Education Consistency:** Check for matching degrees, institutions, and dates.

2.  **Timeline & Gap Analysis:**
    *   Construct a career timeline from all available dates.
    *   Identify and report any unexplained employment gaps of 6 months or more, providing start dates, end dates, and estimated duration.

3.  **Metrics & Summary:**
    *   List all identified inconsistencies and any significant missing information.
    *   Provide a brief overall summary of your findings.

**Output Format:**

Return your analysis as a single JSON object. Ensure all date fields are in 'YYYY-MM-DD' format.

```json
{{
  "discrepancies": [
    {{
      "type": "LinkedIn Comparison | Experience Plausibility | Education Verification",
      "description": "Description of the discrepancy.",
      "severity": "low | medium | high"
    }}
  ],
  "gaps_in_experience": [
    {{
      "start_date": "YYYY-MM-DD",
      "end_date": "YYYY-MM-DD",
      "duration_months": 6,
      "reason": "Unexplained gap between jobs."
    }}
  ],
  "metrics": {{
    "resume_quality_score": 0.85,
    "possible_ai_generated": false,
    "inconsistencies": [
      "List of identified inconsistencies."
    ],
    "missing_information": [
      "List of missing information."
    ]
  }}
}}
```

**Candidate Data:**

**Resume (JSON):**
{resume_json}

{linkedin_section}
"""

    def __init__(self):
        super().__init__("VerificationManagerAgent")
        self.config = config
        
        genai.configure(api_key=os.getenv('GEMINI_API_KEY', ''))
        self.llm = genai.GenerativeModel('gemini-1.5-pro')

        api_key = os.getenv("SCRAPIN_API_KEY")
        if not api_key:
            raise ValueError("SCRAPIN_API_KEY environment variable not set")
        self.linkedin_scraper = LinkedinScraper(api_key)
        self.bq_client = bigquery.Client()
        self.resumes_table_id = f"{self.bq_client.project}.{self.config.bq_dataset}.resumes"

    def _clean_for_json(self, value):
        if isinstance(value, (date, datetime)):
            return value.isoformat()
        elif isinstance(value, dict):
            return {k: self._clean_for_json(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [self._clean_for_json(item) for item in value]
        return value



    def _format_education(self, edu: dict) -> dict:
        return {
            "degree": edu.get('degree', '') or '',
            "institution": edu.get('institution', '') or '',
            "field_of_study": edu.get('field_of_study', '') or '',
            "graduation_year": int(edu.get('graduation_year')) if edu.get('graduation_year') is not None else None
        }

    def _generate_verification_prompt(self, resume_data: dict, linkedin_data: dict) -> str:
        """Generate prompt for verification by formatting resume and LinkedIn data."""
        cleaned_resume_data = self._clean_for_json(resume_data)
        resume_json = json.dumps(cleaned_resume_data, indent=2)

        linkedin_section = "**LinkedIn (JSON):**\nNot provided."
        if linkedin_data:
            cleaned_linkedin_data = self._clean_for_json(linkedin_data)
            linkedin_json = json.dumps(cleaned_linkedin_data, indent=2)
            linkedin_section = f"**LinkedIn (JSON):**\n{linkedin_json}"

        return self.VERIFICATION_PROMPT.format(
            resume_json=resume_json,
            linkedin_section=linkedin_section
        )

    @backoff.on_exception(
        backoff.expo,
        exceptions.BadRequest,
        max_tries=3,
        giveup=lambda e: "streaming buffer" not in str(e).lower(),
        max_time=300  # Maximum 5 minutes of retrying
    )
    def _update_bigquery_with_retry(self, update_query: str, job_config: bigquery.QueryJobConfig):
        """Execute BigQuery update with retry for streaming buffer errors"""
        return self.bq_client.query(update_query, job_config=job_config).result()

    def _parse_date(self, date_str: str):
        """Safely parse a date string into a date object."""
        if not date_str or not isinstance(date_str, str):
            return None
        try:
            return date.fromisoformat(date_str)
        except (ValueError, TypeError):
            return None

    def _update_verification_data_in_bq(self, candidate_id: str, verification_data: dict):
        """Updates the resumes table with verification data."""
        logger.info(f"Updating verification data for candidate_id: {candidate_id}")

        # Extract fields from verification data, providing empty defaults
        # Clean and validate discrepancies data
        discrepancies = verification_data.get('discrepancies', [])
        if isinstance(discrepancies, list):
            cleaned_discrepancies = []
            for item in discrepancies:
                if isinstance(item, dict):
                    # For STRUCT arrays, BQ expects a list of Row objects
                    cleaned_row = bigquery.Row(
                        (
                            str(item.get('type', '')) if item.get('type') is not None else None,
                            str(item.get('description', '')) if item.get('description') is not None else None,
                            str(item.get('severity', '')) if item.get('severity') is not None else None
                        ),
                        {"type": 0, "description": 1, "severity": 2}
                    )
                    cleaned_discrepancies.append(cleaned_row)
            discrepancies = cleaned_discrepancies
        else:
            discrepancies = [] # Default to empty list if not a list

        # Clean and validate gaps_in_experience data
        gaps_raw = verification_data.get('gaps_in_experience', [])
        gaps = []
        if isinstance(gaps_raw, list):
            for gap in gaps_raw:
                if isinstance(gap, dict):
                    duration = gap.get('duration_months')
                    start_date = self._parse_date(gap.get('start_date'))
                    end_date = self._parse_date(gap.get('end_date'))

                    # For STRUCT arrays, BQ expects a list of Row objects
                    cleaned_row = bigquery.Row(
                        (
                            start_date.strftime('%Y-%m-%d') if start_date else None,
                            end_date.strftime('%Y-%m-%d') if end_date else None,
                            int(duration) if duration is not None else None,
                            str(gap.get('reason', '')) if gap.get('reason') is not None else None
                        ),
                        {"start_date": 0, "end_date": 1, "duration_months": 2, "reason": 3}
                    )
                    gaps.append(cleaned_row)

        metrics = verification_data.get('metrics', {})

        # Extract all metrics fields, providing defaults
        resume_quality_score = metrics.get('resume_quality_score')
        possible_ai_generated = metrics.get('possible_ai_generated')
        inconsistencies = metrics.get('inconsistencies', [])
        missing_info = metrics.get('missing_information', [])

        # Build literal ARRAY<STRUCT> SQL snippets for discrepancies and gaps to avoid complex query parameters
        def _escape(value: str) -> str:
            """Escape single quotes for safe SQL literal usage."""
            return value.replace("'", "\\'") if isinstance(value, str) else value

        discrepancies_sql = "[" + ", ".join(
            f"STRUCT('{_escape(d.get('type', ''))}', '{_escape(d.get('description', ''))}', '{_escape(d.get('severity', ''))}')"
            for d in discrepancies
        ) + "]"

        gaps_sql = "[" + ", ".join(
            f"STRUCT('{gap[0]}'::DATE, '{gap[1]}'::DATE, {gap[2] if gap[2] is not None else 'NULL'}, '{_escape(gap[3])}')"
            for gap in gaps
        ) + "]"

        # Construct the MERGE statement with embedded literals
        script = f"""
        MERGE `{self.resumes_table_id}` T
        USING (SELECT @candidate_id AS candidate_id) S ON T.candidate_id = S.candidate_id
        WHEN MATCHED THEN
            UPDATE SET
                T.discrepancies = {discrepancies_sql},
                T.gaps_in_experience = {gaps_sql},
                T.metrics = STRUCT(
                    @resume_quality_score AS resume_quality_score,
                    @possible_ai_generated AS possible_ai_generated,
                    @missing_info AS missing_information,
                    @inconsistencies AS inconsistencies
                ),
                T.updated_at = CURRENT_TIMESTAMP()
        """

        # Define scalar and simple array query parameters
        query_params = [
            bigquery.ScalarQueryParameter("candidate_id", "STRING", candidate_id),
            bigquery.ScalarQueryParameter("resume_quality_score", "FLOAT64", resume_quality_score),
            bigquery.ScalarQueryParameter("possible_ai_generated", "BOOL", possible_ai_generated),
            bigquery.ArrayQueryParameter("inconsistencies", "STRING", inconsistencies),
            bigquery.ArrayQueryParameter("missing_info", "STRING", missing_info)
        ]

        job_config = bigquery.QueryJobConfig(query_parameters=query_params)

        try:
            self._update_bigquery_with_retry(script, job_config=job_config)
            logger.info(f"Successfully updated verification data for candidate_id: {candidate_id}")
        except exceptions.BadRequest as e:
            logger.error(f"BigQuery BadRequest Error: {e}", exc_info=True)
            logger.error(f"Data sent to BigQuery that caused the error:")
            logger.error(f"  - candidate_id: {candidate_id}")
            logger.error(f"  - discrepancies: {json.dumps(discrepancies, indent=2)}")
            logger.error(f"  - gaps: {json.dumps(gaps, indent=2, default=str)}")
            logger.error(f"  - metrics: {json.dumps(metrics, indent=2)}")
            raise
        except Exception as e:
            logger.error(f"An unexpected error occurred while updating verification data for {candidate_id}: {e}", exc_info=True)
            raise


    def run(self, data: dict):
        candidate_id = data.get('candidate_id')
        email_data = data.get('email_data')
        logger.info(f"Starting verification for candidate_id: {candidate_id}")
        
        try:
            # 1. Fetch resume data from BigQuery
            query = f"""
            SELECT *  
            FROM `{self.resumes_table_id}` 
            WHERE candidate_id = @candidate_id
            """
            job_config = bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ScalarQueryParameter("candidate_id", "STRING", candidate_id)
                ]
            )
            results = self.bq_client.query(query, job_config=job_config).result()
            if results.total_rows == 0:
                logger.error(f"No resume found for candidate_id: {candidate_id}")
                return {"status": "error", "message": f"No resume found for candidate_id: {candidate_id}"}
            
            resume_data = dict(next(iter(results)))
            
            # 2. Process email content to get DOB and LinkedIn URL, and update the table
            updates = {}
            dob_match = re.search(r'(?:Date of Birth|dob):\s*(\S+)', email_data.body, re.IGNORECASE)
            if dob_match:
                updates['date_of_birth'] = dob_match.group(1)
            
            linkedin_url_match = re.search(r'LinkedIn:\s*(https?://[\S]+)', email_data.body, re.IGNORECASE)
            if linkedin_url_match:
                updates['linkedin_url'] = linkedin_url_match.group(1)
            
            if updates:
                set_clauses = []
                query_params = [bigquery.ScalarQueryParameter("candidate_id", "STRING", candidate_id)]
                for field, value in updates.items():
                    param_name = f"param_{field}"
                    set_clauses.append(f"T.{field} = @{param_name}")
                    if field == 'date_of_birth':
                        query_params.append(bigquery.ScalarQueryParameter(param_name, "DATE", date.fromisoformat(value)))
                    else:
                        query_params.append(bigquery.ScalarQueryParameter(param_name, "STRING", value))

                script = f"""
                MERGE `{self.resumes_table_id}` T
                USING (SELECT @candidate_id AS candidate_id) S ON T.candidate_id = S.candidate_id
                WHEN MATCHED THEN
                    UPDATE SET {', '.join(set_clauses)}, T.updated_at = CURRENT_TIMESTAMP()
                """
                job_config = bigquery.QueryJobConfig(query_parameters=query_params)
                self._update_bigquery_with_retry(script, job_config=job_config)
                logger.info(f"Updated DOB/LinkedIn for candidate: {candidate_id}")
                # Refresh resume_data with the latest info
                resume_data.update(updates)

            # 3. Use the potentially updated LinkedIn URL for verification
            linkedin_url_to_use = resume_data.get('linkedin_url')
            verification_result = None

            linkedin_data = {}
            if linkedin_url_to_use:
                logger.info(f"Found LinkedIn URL to process: {linkedin_url_to_use}")
                try:
                    linkedin_data = self.linkedin_scraper.extract(linkedin_url_to_use)
                except Exception as e:
                    logger.warning(f"Failed to fetch LinkedIn data for {linkedin_url_to_use}: {e}. Proceeding without it.")
            else:
                logger.info("No LinkedIn URL found. Proceeding with resume data only.")
            
            prompt = self._generate_verification_prompt(resume_data, linkedin_data)
            response_text = self.llm.generate_content(prompt).text
            
            match = re.search(r"```json\n(.*?)\n```", response_text, re.DOTALL)
            if match:
                json_str = match.group(1)
            else:
                json_str = response_text

            verification_result = json.loads(json_str)
            logger.info(f"Received verification JSON from AI: {json.dumps(verification_result, indent=2)}")

            if verification_result:
                self._update_verification_data_in_bq(candidate_id, verification_result)
                logger.info(f"Successfully stored verification results for candidate {candidate_id}")
            else:
                logger.warning("Verification result was empty. Nothing to store.")

            return {
                "status": "success",
                "verification": verification_result
            }

        except Exception as e:
            logger.error(f"Error in verification process: {e}", exc_info=True)
            return {
                "status": "error",
                "message": "Verification process failed",
                "details": str(e)
            }
