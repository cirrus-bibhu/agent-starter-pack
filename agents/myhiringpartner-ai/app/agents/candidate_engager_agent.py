import json
import re
from typing import Any, Dict
from google.cloud import bigquery
from google.cloud.bigquery import StructQueryParameter, ScalarQueryParameter
from ..agent import BaseAgent, EmailData
from ..config import config
from ..tools.bq_schema_manager import ensure_table_exists

CANDIDATE_INFO_EXTRACTION_PROMPT = ''''
Your task is to extract the following information from the candidate's email reply and format it as a JSON object:
- LinkedIn Profile URL
- All education details, including degree, university/institution, major/field of study, and graduation year.
- Date of Birth (DOB)

Your response must be ONLY the JSON object, with no additional text or formatting. The JSON object must follow this structure exactly:

{
  "linkedin_url": "<LinkedIn Profile URL>",
  "education": [
    {
      "degree": "<e.g., B.S. in Computer Science>",
      "university": "<University Name>",
      "major": "<Major or Field of Study>",
      "graduation_year": "<Year>"
    }
  ],
  "dob": "<YYYY-MM-DD>"
}

If any information is not present in the email, set the corresponding value to null.
'''

class CandidateEngagerAgent(BaseAgent):
    """An agent that processes candidate replies and updates their information in BigQuery."""

    def __init__(self):
        super().__init__("CandidateEngagerAgent", model_name="gemini-1.5-pro")
        self.bq_client = self._get_bigquery_client()
        self.table_id = f"{self.bq_client.project}.{config.bq_dataset}.resumes"
        ensure_table_exists(self.bq_client, self.table_id)

    def _get_bigquery_client(self):
        try:
            return bigquery.Client()
        except Exception as e:
            self.logger.error(f"Failed to initialize BigQuery client: {e}", exc_info=True)
            return None

    def run(self, email_data: EmailData, candidate_id: str = None) -> Dict[str, Any]:
        if not candidate_id:
            return {"status": "error", "message": "Candidate ID was not provided to the agent."}

        self.logger.info(f"Processing info for candidate ID: {candidate_id}")
        info_json = {}
        education_data = None

        # Attempt to parse structured data first
        try:
            # Extract education JSON
            education_match = re.search(r'Education:\s*(\[.*?\])', email_data.body, re.DOTALL)
            if education_match:
                education_str = education_match.group(1)
                education_data = json.loads(education_str)
                self.logger.info("Successfully parsed education data using regex.")

            # Extract LinkedIn URL
            linkedin_match = re.search(r'LinkedIn:\s*(https?://[\w\./\-]+)', email_data.body)
            if linkedin_match:
                info_json['linkedin_url'] = linkedin_match.group(1)

            # Extract DOB
            dob_match = re.search(r'DOB:\s*([0-9]{4}-[0-9]{2}-[0-9]{2})', email_data.body)
            if dob_match:
                info_json['dob'] = dob_match.group(1)

        except (json.JSONDecodeError, IndexError) as e:
            self.logger.warning(f"Could not parse structured data directly, falling back to LLM. Error: {e}")
            education_data = None # Reset on failure

        # If direct parsing fails, use LLM as a fallback
        if education_data is None:
            self.logger.info("Falling back to LLM for information extraction.")
            response_text = self.model.generate_content(CANDIDATE_INFO_EXTRACTION_PROMPT + "\n\n" + email_data.body).text
            self.logger.debug(f"Raw extracted info response: {response_text}")

            if response_text.strip().startswith('```json'):
                response_text = response_text.strip()[7:-3].strip()
            
            try:
                llm_info = json.loads(response_text)
                info_json.update(llm_info) # Merge results, LLM data is fallback
                education_data = info_json.get("education", [])
            except json.JSONDecodeError:
                return {"status": "error", "message": "Failed to parse extracted information from LLM."}

        # Transform education data for BigQuery
        education_records = []
        if education_data:
            for details in education_data:
                # Ensure the record has the essential fields before adding
                if details and details.get("institution") and details.get("field_of_study"):
                    record = {
                        "degree": details.get("degree"),
                        "institution": details.get("institution"),
                        "field_of_study": details.get("field_of_study"),
                        "graduation_year": int(details.get("graduation_year")) if details.get("graduation_year") else None
                    }
                    education_records.append(record)

        # Use MERGE for a robust upsert operation
        query = f"""
        MERGE `{self.table_id}` T
        USING (SELECT @candidate_id AS candidate_id) S
        ON T.candidate_id = S.candidate_id
        WHEN MATCHED THEN
            UPDATE SET
                linkedin_url = @linkedin_url,
                education = @education,
                date_of_birth = @date_of_birth;
        """
        education_structs = []
        for rec in education_records:
            params = [
                ScalarQueryParameter("degree", "STRING", rec.get("degree")),
                ScalarQueryParameter("institution", "STRING", rec.get("institution")),
                ScalarQueryParameter("field_of_study", "STRING", rec.get("field_of_study")),
                ScalarQueryParameter("graduation_year", "INT64", rec.get("graduation_year")),
            ]
            education_structs.append(StructQueryParameter(None, *params))

        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("linkedin_url", "STRING", info_json.get("linkedin_url")),
                bigquery.ArrayQueryParameter("education", "STRUCT", education_structs),
                bigquery.ScalarQueryParameter("date_of_birth", "DATE", info_json.get("dob")),
                bigquery.ScalarQueryParameter("candidate_id", "STRING", candidate_id),
            ]
        )
        try:
            self.bq_client.query(query, job_config=job_config).result()
            return {"status": "success", "message": f"Successfully updated information for candidate {candidate_id}."}
        except Exception as e:
            self.logger.error(f"Failed to update BigQuery for candidate {candidate_id}: {e}", exc_info=True)
            return {"status": "error", "message": "Failed to update candidate information in BigQuery."}
