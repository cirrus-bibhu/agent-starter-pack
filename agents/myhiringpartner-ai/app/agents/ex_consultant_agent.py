import logging
from typing import Dict, Any, Union, List
from ..agent import BaseAgent, EmailData
import json

class ExConsultantAgent(BaseAgent):
    """
    Agent responsible for finding and matching ex-consultants to job postings.
    """
    def __init__(self):
        super().__init__("ExConsultantAgent", model_name="gemini-1.5-pro")
        self.logger = logging.getLogger(__name__)

    def find_ex_consultants(self, job_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        self.logger.info(f"Finding ex-consultants for job: {job_data.get('job_title')}")
        
        job_end_client = job_data.get('end_client_name', '').lower()
        job_prime_vendor = job_data.get('prime_vendor_name', '').lower()
        job_description = job_data.get('job_description', '')
        required_skills = job_data.get('required_technical_skills', [])
        
        try:
            query = {
                "$or": [
                    {"past_companies": {"$regex": job_end_client, "$options": "i"}},
                    {"past_companies": {"$regex": job_prime_vendor, "$options": "i"}}
                ]
            }
            
            company_matched_consultants = []
            
            if not company_matched_consultants:
                self.logger.warning("No consultants found by company relationship")
                return []
            
            if required_skills:
                company_matched_consultants = [
                    c for c in company_matched_consultants
                    if any(skill.lower() in [s.lower() for s in c.get('skills', [])] 
                          for skill in required_skills)
                ]
            
            job_embedding = self._generate_embeddings(job_description)
            
            for consultant in company_matched_consultants:
                consultant['similarity_score'] = self._calculate_similarity(
                    job_embedding, 
                    consultant.get('resume_embedding')
                )
            
            company_matched_consultants.sort(key=lambda x: x.get('similarity_score', 0), reverse=True)
            
            found_consultants = []
            for consultant in company_matched_consultants:
                consultant_copy = consultant.copy()
                if job_end_client and any(company.lower() == job_end_client 
                                       for company in consultant.get('past_companies', [])):
                    consultant_copy['relationship'] = 'ex_end_client'
                elif job_prime_vendor and any(company.lower() == job_prime_vendor 
                                           for company in consultant.get('past_companies', [])):
                    consultant_copy['relationship'] = 'ex_prime_vendor'
                found_consultants.append(consultant_copy)
            
            self.logger.info(f"Found {len(found_consultants)} potential ex-consultants.")
            return found_consultants
            
        except Exception as e:
            self.logger.error(f"Error finding ex-consultants: {e}", exc_info=True)
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
            
            prompt = f"""You are an expert at matching job descriptions to candidate resumes. \
            Given the following job description and candidate resume summary, \
            rate how well the candidate matches the job on a scale of 0 to 100. \
            Also, provide a brief reasoning for your score. \
            
            Job Title: {job_title}
            Job Description: {job_description}
            
            Candidate Name: {consultant.get('name')}
            Candidate Resume Summary: {consultant_resume_summary}
            
            Your response should be a JSON object with 'score' (integer) and 'reasoning' (string) fields. \
            Example: {{"score": 85, "reasoning": "Candidate has strong relevant skills."}}
            """
            
            try:
                response = self.model.generate_content(prompt)
                if response.text:
                    match_info = json.loads(response.text.strip())
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
