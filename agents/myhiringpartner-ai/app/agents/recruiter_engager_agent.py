import logging
import re
import json
import uuid
from typing import Dict, Any, Optional, Union
from datetime import datetime
from google.cloud import bigquery
from ..agent import BaseAgent, EmailData
from ..config import config
from ..tools.bq_schema_manager import ensure_table_exists
from app.agents.ex_consultant_agent import ExConsultantAgent
from app.agents.utils import get_missing_fields

ADDITIONAL_INFO_EXTRACTION_PROMPT = """You are an expert data extraction agent. Your task is to extract specific pieces of information from a recruiter's email reply and format it as a JSON object.

The user previously asked for the following information:
- is_relocation_allowed (boolean: true/false)
- is_remote (boolean: true/false)
- prime_vendor_name (string)
- prime_vendor_email (string)
- end_client_name (string)

Analyze the email body below and extract the values for these fields.

**Recruiter's Reply:**
---
{email_body}
---

**Instructions:**
1.  Read the email carefully to find the answers.
2.  Format the output as a single, clean JSON object.
3.  If a value is not found, use `null`.
4.  Ensure boolean values are `true` or `false`, not strings.
5.  Do not include any text or explanations outside of the JSON object.

**JSON Output Example:**
{{
    "is_relocation_allowed": true,
    "is_remote": false,
    "prime_vendor_name": "Tech Solutions Inc.",
    "prime_vendor_email": "contact@techsolutions.com",
    "end_client_name": "Global Innovations Corp"
}}
"""

class RecruiterEngagerAgent(BaseAgent):
    def __init__(self):
        super().__init__("RecruiterEngager", model_name="gemini-1.5-pro")
        self.logger = logging.getLogger(__name__)
        self.bq_client = None
        self.table_id = None
        self._initialize_bigquery()
    
    def _initialize_bigquery(self) -> None:
        try:
            self.bq_client = bigquery.Client()
            if self.bq_client:
                self.table_id = f"{self.bq_client.project}.{config.bq_dataset}.{config.bq_table}"
                ensure_table_exists(self.bq_client, self.table_id)
                self.logger.info("BigQuery client and table verified successfully.")
            else:
                self.logger.warning("Could not initialize BigQuery client.")
        except Exception as e:
            self.logger.error(f"Failed to initialize BigQuery: {e}", exc_info=True)
            self.bq_client = None
            self.table_id = None
    
    def _extract_job_details(self, email_data: EmailData) -> Dict[str, Any]:
        try:
            prompt = ADDITIONAL_INFO_EXTRACTION_PROMPT.format(email_body=email_data.body)
            response = self.model.generate_content(prompt)

            if not response.text:
                self.logger.error("LLM response was empty.")
                return {}

            try:
                # Clean the response to ensure it's valid JSON
                cleaned_response = response.text.strip()
                # Remove markdown code block fences if present
                if cleaned_response.startswith("```json"):
                    cleaned_response = cleaned_response[7:]
                if cleaned_response.endswith("```"):
                    cleaned_response = cleaned_response[:-3]
                
                extracted_data = json.loads(cleaned_response.strip())

                # Prepare the details for BigQuery update
                job_details = {
                    "recruiter_email": email_data.sender
                }
                job_details.update(extracted_data)

                return job_details

            except json.JSONDecodeError as e:
                self.logger.error(f"Error parsing LLM response: {e}. Response was: {response.text}")
                return {}

        except Exception as e:
            self.logger.error(f"Error extracting job details: {e}", exc_info=True)
            return {}
    
    def _update_job_in_bigquery(self, job_id: str, updates: Dict[str, Any]) -> bool:
        if not self.bq_client or not self.table_id:
            self.logger.error("BigQuery client not initialized. Cannot update job details.")
            return False
        
        if not job_id:
            self.logger.error("Job ID is required to update job details.")
            return False
        
        if not updates:
            self.logger.warning("No updates provided. Skipping BigQuery update.")
            return True
        
        try:
            update_fields = []
            query_params = []
            
            type_mapping = {
                str: "STRING",
                int: "INT64",
                float: "FLOAT64",
                bool: "BOOL",
                datetime: "TIMESTAMP",
                dict: "STRING", 
                list: "STRING",
            }
            
            updates_copy = updates.copy()
            updates_copy.pop('job_id', None)
            
            for field, value in updates_copy.items():
                if value is not None:
                    if field.startswith('_'):
                        continue
                        
                    if isinstance(value, (dict, list)):
                        param_type = "STRING"
                        param_value = json.dumps(value)
                    else:
                        param_type = type_mapping.get(type(value), "STRING")
                        param_value = value
                        
                        if isinstance(value, datetime):
                            param_value = value.isoformat()
                    
                    update_fields.append(f"`{field}` = @{field}")
                    query_params.append(
                        bigquery.ScalarQueryParameter(field, param_type, param_value)
                    )
            
            if not update_fields:
                self.logger.warning("No valid fields to update.")
                return True
            
            current_time = datetime.utcnow()
            update_fields.append("`last_updated_timestamp` = @last_updated")
            query_params.append(
                bigquery.ScalarQueryParameter("last_updated", "TIMESTAMP", current_time.isoformat())
            )
            
            # Add WHERE clause for job_id
            update_query = f"""
            UPDATE `{self.table_id}`
            SET {', '.join(update_fields)}
            WHERE job_id = @where_job_id
            """
            
            # Add job_id parameter with a unique name for WHERE clause
            query_params.append(
                bigquery.ScalarQueryParameter("where_job_id", "STRING", job_id)
            )
            
            job_config = bigquery.QueryJobConfig(query_parameters=query_params)
            
            # Log the query and parameters for debugging
            self.logger.debug(f"Update query: {update_query}")
            self.logger.debug(f"Query parameters: {query_params}")
            
            query_job = self.bq_client.query(update_query, job_config=job_config)
            query_job.result()  # Wait for the query to complete
            
            if query_job.errors:
                self.logger.error(f"BigQuery update failed: {query_job.errors}")
                return False
                
            self.logger.info(f"Successfully updated job {job_id} with {len(updates_copy)} fields.")
            return True
            
        except Exception as e:
            self.logger.error(f"Error updating job in BigQuery: {e}", exc_info=True)
            return False

    def _get_job_details_from_bigquery(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Fetches the full job details for a given job_id from BigQuery."""
        if not self.bq_client or not self.table_id:
            self.logger.error("BigQuery client not initialized. Cannot fetch job details.")
            return None

        try:
            query = f"SELECT * FROM `{self.table_id}` WHERE job_id = @job_id"
            query_params = [
                bigquery.ScalarQueryParameter("job_id", "STRING", job_id)
            ]
            job_config = bigquery.QueryJobConfig(query_parameters=query_params)
            
            query_job = self.bq_client.query(query, job_config=job_config)
            rows = list(query_job.result())

            if not rows:
                self.logger.warning(f"No job found with job_id: {job_id}")
                return None

            job_details = dict(rows[0])
            self.logger.info(f"Successfully fetched job details for job_id: {job_id}")
            return job_details

        except Exception as e:
            self.logger.error(f"Error fetching job details from BigQuery: {e}", exc_info=True)
            return None
    
    def _extract_job_id(self, email_body: str) -> Optional[str]:
        patterns = [
            r"Job[\s-]?ID[\s:]+([A-Z0-9-]+)",
            r"Job[\s-]?#[\s:]+([A-Z0-9-]+)",
            r"Reference[\s-]?(?:No\.?|Number)[\s:]+([A-Z0-9-]+)",
            r"\b(JB|JOB|REF)[\s-]?([A-Z0-9-]+)\b"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, email_body, re.IGNORECASE)
            if match:
                return match.group(1) if len(match.groups()) > 0 else match.group(0)
        
        return None
    
    def _is_complete_job_info(self, job_data: Dict[str, Any]) -> bool:
        required_fields = [
            'job_title',
            'end_client_name',
            'job_location_city',
            'job_location_state',
            'job_location_country',
            'required_technical_skills',
            'min_required_years_experience'
        ]
        
        return all(job_data.get(field) for field in required_fields)
    
    def run(self, email_data: Union[str, EmailData], **kwargs) -> Dict[str, Any]:
        try:
            if isinstance(email_data, str):
                email_data = EmailData.from_json(email_data)

            self.logger.info(f"RecruiterEngagerAgent processing email: {email_data.subject}")

            # Prioritize job_id from kwargs, otherwise extract from email
            job_id = kwargs.get('job_id') or self._extract_job_id(email_data.body)

            job_details = self._extract_job_details(email_data)
            if not job_details:
                return {
                    "status": "error",
                    "message": "Could not extract job details from the email. Please provide the information in a clear format.",
                    "job_id": job_id
                }

            # Ensure the extracted/passed job_id is in the details
            job_details['job_id'] = job_id

            result = {
                "status": "success",
                "job_id": job_id,
                "job_details": job_details,
                "next_step": ""
            }

            is_complete = self._is_complete_job_info(job_details)

            if self.bq_client and self.table_id:
                # Filter out null values so we don't overwrite existing data with nulls
                update_details = {k: v for k, v in job_details.items() if v is not None}

                if not update_details:
                    result["message"] = "No new job details were extracted from the email."
                else:
                    # First, update with the info we just received
                    update_success = self._update_job_in_bigquery(job_id, update_details)
                    if not update_success:
                        return {
                            "status": "error",
                            "message": "Failed to update job details in BigQuery",
                            "job_id": job_id
                        }
                    result["message"] = "Job details updated successfully."

                # Now, fetch the full, updated job record to check for remaining missing fields
                updated_job_data = self._get_job_details_from_bigquery(job_id)
                if not updated_job_data:
                    return {
                        "status": "error",
                        "message": f"Could not retrieve updated job details for job_id: {job_id}",
                        "job_id": job_id
                    }

                remaining_missing_fields = get_missing_fields(updated_job_data)

                if remaining_missing_fields:
                    self.logger.info(f"Still missing fields for job {job_id}: {remaining_missing_fields}. Triggering another follow-up.")
                    # This is where you would trigger another email. For now, we'll log it.
                    result["next_step"] = "follow_up_required"
                    result["missing_fields"] = remaining_missing_fields
                    result["message"] = "Job details partially updated. Follow-up required for remaining info."
                else:
                    self.logger.info(f"All required information for job {job_id} has been collected.")
                    result["next_step"] = "complete"
            else:
                result["message"] = "Job details processed but not saved (BigQuery unavailable)"
                result["next_step"] = "Database connection unavailable. Please try again later."

            return result
            
        except Exception as e:
            self.logger.error(f"Error in RecruiterEngagerAgent: {e}", exc_info=True)
            return {
                "status": "error",
                "message": f"An error occurred while processing the recruiter's reply: {str(e)}",
                "job_id": job_details.get('job_id') if 'job_details' in locals() else None
            }
