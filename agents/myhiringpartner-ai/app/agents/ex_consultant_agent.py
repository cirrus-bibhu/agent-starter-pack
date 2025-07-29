import logging
from typing import Dict, Any, List
from google.cloud import bigquery
import json
from datetime import date, datetime
from app.agent import BaseAgent
from app.config import config

class ExConsultantAgent(BaseAgent):
    """
    Agent responsible for finding and matching ex-consultants to job postings.
    """
    def __init__(self):
        super().__init__("ExConsultantAgent", model_name="gemini-1.5-pro")
        self.logger = logging.getLogger(self.__class__.__name__)

        self.bq_client = bigquery.Client()

    def _convert_dates_to_str(self, data: Any) -> Any:
        """Recursively convert date and datetime objects in a data structure to ISO format strings."""
        if isinstance(data, dict):
            return {k: self._convert_dates_to_str(v) for k, v in data.items()}
        if isinstance(data, list):
            return [self._convert_dates_to_str(i) for i in data]
        if isinstance(data, (date, datetime)):
            return data.isoformat()
        return data

    def find_ex_consultants(self, job_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        self.logger.info(f"Finding ex-consultants for job: {job_data.get('job_title')}")

        job_end_client = job_data.get('end_client_name')
        job_prime_vendor = job_data.get('prime_vendor_name')

        companies_to_match = []
        if job_end_client:
            companies_to_match.append(job_end_client)
        if job_prime_vendor:
            companies_to_match.append(job_prime_vendor)

        if not companies_to_match:
            self.logger.warning("No end client or prime vendor name provided in job data.")
            return []
        
        try:

            query_params = [
                bigquery.ArrayQueryParameter('companies', 'STRING', list(set(companies_to_match)))
            ]
            query = f"""
                SELECT DISTINCT r.*
                FROM `{self.bq_client.project}.{config.bq_dataset}.resumes` AS r,
                UNNEST(r.previous_companies) AS past_company
                WHERE past_company IN UNNEST(@companies)
            """

            job_config = bigquery.QueryJobConfig(query_parameters=query_params)
            query_job = self.bq_client.query(query, job_config=job_config)

            results = query_job.result()
            found_consultants = [dict(row) for row in results]
            self.logger.info(f"Found {len(found_consultants)} ex-consultant(s) in the database.")
            
            return self._convert_dates_to_str(found_consultants)

        except Exception as e:
            self.logger.error(f"Error querying BigQuery for ex-consultants: {e}", exc_info=True)
            return []
    
    def _generate_embeddings(self, text: str) -> List[float]:
        return []
    
    def _calculate_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        return 0.0

    def match_ex_consultants(self, job_data: Dict[str, Any], ex_consultants: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        self.logger.info(f"Matching ex-consultants for job: {job_data.get('job_title')}")

        if not ex_consultants:
            return []

        job_description = job_data.get('job_description', '')
        job_title = job_data.get('job_title', '')

        matched_results = []

        for consultant in ex_consultants:
            consultant_resume_summary = consultant.get('resume_summary', '')
            
            prompt = f"""Analyze the job and candidate summary, then provide a JSON response.

Job: {job_title}
Description: {job_description}

Candidate: {consultant.get('name')}
Summary: {consultant_resume_summary}

---
Respond with a JSON object containing 'score' (0-100) and 'reasoning' (a brief explanation).
Example: {{"score": 85, "reasoning": "Strong match based on NLP experience."}}
"""
            
            try:
                response = self.model.generate_content(prompt)
                try:
                    cleaned_response = response.text.strip()
                    if cleaned_response.startswith("```json"):
                        cleaned_response = cleaned_response[7:]
                    if cleaned_response.endswith("```"):
                        cleaned_response = cleaned_response[:-3]
                    
                    match_info = json.loads(cleaned_response)
                except (json.JSONDecodeError, AttributeError) as e:
                    self.logger.error(f"Error decoding LLM response for consultant {consultant.get('id')}: {e}. Response: '{response.text}'")
                    match_info = {"score": 0, "reasoning": "Error processing AI response."}

                consultant_copy = consultant.copy()
                consultant_copy['match_score'] = match_info.get('score', 0)
                consultant_copy['match_reasoning'] = match_info.get('reasoning', 'No reasoning provided.')
                matched_results.append(consultant_copy)
            except Exception as e:
                self.logger.error(f"Error matching consultant {consultant.get('name')}: {e}", exc_info=True)
                consultant_copy = consultant.copy()
                consultant_copy['match_score'] = 0
                consultant_copy['match_reasoning'] = 'Error during matching.'
                matched_results.append(consultant_copy)

        matched_results.sort(key=lambda x: x.get('match_score', 0), reverse=True)
        self.logger.info(f"Matched {len(matched_results)} consultants. Top 5 selected.")
        return matched_results[:5]

    def run(self, job_data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Main entry point for the ExConsultantAgent.
        """
        try:
            self.logger.info(f"ExConsultantAgent started for job: {job_data.get('job_title')}")
            
            ex_consultants = self.find_ex_consultants(job_data)
            if not ex_consultants:
                return {"status": "no_ex_consultants_found", "message": "No ex-consultants found for this job.", "job_id": job_data.get('job_id')}
            
            matched_consultants = self.match_ex_consultants(job_data, ex_consultants)
            
            return {
                "status": "success",
                "message": "Ex-consultants found and matched.",
                "job_id": job_data.get('job_id'),
                "matched_consultants": matched_consultants
            }
        except Exception as e:
            self.logger.error(f"Error in ExConsultantAgent: {e}", exc_info=True)
            return {"status": "error", "message": str(e), "job_id": job_data.get('job_id')}
