import os
import json
import glob
from typing import Dict, Any, List
from datetime import datetime
from ..agent import BaseAgent

class MatchingServiceAgent(BaseAgent):
    """Agent responsible for matching resumes with job postings"""
    
    def __init__(self):
        super().__init__("MatchingService", model_name="gemini-1.5-pro")
        self.job_postings_dir = "job_postings"
        self.resume_submissions_dir = "resume_submissions"
        self.matches_dir = "matches"
        os.makedirs(self.matches_dir, exist_ok=True)
    
    def load_job_posting(self, job_id: str) -> Dict[str, Any]:
        """Load a specific job posting by ID"""
        try:
            with open(os.path.join(self.job_postings_dir, job_id)) as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"Error loading job posting {job_id}: {str(e)}")
            return None

    def load_resume(self, resume_id: str) -> Dict[str, Any]:
        """Load a specific resume by ID"""
        try:
            with open(os.path.join(self.resume_submissions_dir, resume_id)) as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"Error loading resume {resume_id}: {str(e)}")
            return None

    def match_resume_to_job(self, resume_data: Dict[str, Any], job_data: Dict[str, Any]) -> Dict[str, Any]:
        """Match a resume against a job posting"""
        try:
            prompt = f"""
Analyze the match between this resume and job posting. Return a JSON object with:
- match_score (0.0 to 1.0)
- match_summary (detailed explanation)
- strengths (list of matching qualifications)
- gaps (list of missing qualifications)

Resume:
{json.dumps(resume_data.get('resume_info', {}), indent=2)}

Job:
{json.dumps(job_data.get('job_details', {}), indent=2)}

Return only the JSON object.
"""
            response = self.model.generate_content(
                prompt,
                generation_config={"temperature": 0.1}
            )
            
            match_result = json.loads(response.text)
            match_result['timestamp'] = datetime.now().isoformat()
            match_result['resume_id'] = resume_data.get('metadata', {}).get('source_email', {}).get('id')
            match_result['job_id'] = job_data.get('metadata', {}).get('source_email', {}).get('id')
            
            return match_result
            
        except Exception as e:
            self.logger.error(f"Error matching resume to job: {str(e)}")
            return {
                "match_score": 0.0,
                "match_summary": f"Error performing match: {str(e)}",
                "strengths": [],
                "gaps": [],
                "timestamp": datetime.now().isoformat()
            }

    def save_match_result(self, match_result: Dict[str, Any]) -> str:
        """Save the match result to a JSON file"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"match_{match_result['resume_id']}_{match_result['job_id']}_{timestamp}.json"
            filepath = os.path.join(self.matches_dir, filename)
            
            with open(filepath, 'w') as f:
                json.dump(match_result, f, indent=2)
            
            return filepath
        except Exception as e:
            self.logger.error(f"Error saving match result: {str(e)}")
            return None

    def find_matching_jobs(self, resume_id: str, min_score: float = 0.7) -> List[Dict[str, Any]]:
        """Find all suitable jobs for a given resume"""
        try:
            resume_data = self.load_resume(resume_id)
            if not resume_data:
                return []

            matches = []
            for job_file in glob.glob(os.path.join(self.job_postings_dir, "*.json")):
                job_data = self.load_job_posting(os.path.basename(job_file))
                if not job_data:
                    continue

                match_result = self.match_resume_to_job(resume_data, job_data)
                if match_result["match_score"] >= min_score:
                    matches.append(match_result)
                    self.save_match_result(match_result)

            return sorted(matches, key=lambda x: x["match_score"], reverse=True)

        except Exception as e:
            self.logger.error(f"Error finding matching jobs: {str(e)}")
            return []

    def find_matching_candidates(self, job_id: str, min_score: float = 0.7) -> List[Dict[str, Any]]:
        """Find all suitable candidates for a given job"""
        try:
            job_data = self.load_job_posting(job_id)
            if not job_data:
                return []

            matches = []
            for resume_file in glob.glob(os.path.join(self.resume_submissions_dir, "*.json")):
                resume_data = self.load_resume(os.path.basename(resume_file))
                if not resume_data:
                    continue

                match_result = self.match_resume_to_job(resume_data, job_data)
                if match_result["match_score"] >= min_score:
                    matches.append(match_result)
                    self.save_match_result(match_result)

            return sorted(matches, key=lambda x: x["match_score"], reverse=True)

        except Exception as e:
            self.logger.error(f"Error finding matching candidates: {str(e)}")
            return []

    def run(self, input_data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Main execution method. Accepts different operations:
        - match_resume_to_job: Match a specific resume to a specific job
        - find_matching_jobs: Find all suitable jobs for a resume
        - find_matching_candidates: Find all suitable candidates for a job
        """
        try:
            operation = input_data.get("operation")
            if not operation:
                return {"status": "error", "message": "No operation specified"}

            if operation == "match_resume_to_job":
                resume_id = input_data.get("resume_id")
                job_id = input_data.get("job_id")
                if not resume_id or not job_id:
                    return {"status": "error", "message": "Missing resume_id or job_id"}

                resume_data = self.load_resume(resume_id)
                job_data = self.load_job_posting(job_id)
                if not resume_data or not job_data:
                    return {"status": "error", "message": "Could not load resume or job data"}

                match_result = self.match_resume_to_job(resume_data, job_data)
                saved_path = self.save_match_result(match_result)
                
                return {
                    "status": "success",
                    "match_result": match_result,
                    "saved_path": saved_path
                }

            elif operation == "find_matching_jobs":
                resume_id = input_data.get("resume_id")
                min_score = input_data.get("min_score", 0.7)
                if not resume_id:
                    return {"status": "error", "message": "Missing resume_id"}

                matches = self.find_matching_jobs(resume_id, min_score)
                return {
                    "status": "success",
                    "matches": matches,
                    "total_matches": len(matches)
                }

            elif operation == "find_matching_candidates":
                job_id = input_data.get("job_id")
                min_score = input_data.get("min_score", 0.7)
                if not job_id:
                    return {"status": "error", "message": "Missing job_id"}

                matches = self.find_matching_candidates(job_id, min_score)
                return {
                    "status": "success",
                    "matches": matches,
                    "total_matches": len(matches)
                }

            else:
                return {"status": "error", "message": f"Unknown operation: {operation}"}

        except Exception as e:
            self.logger.error(f"Error in matching service: {str(e)}")
            return {"status": "error", "message": str(e)}