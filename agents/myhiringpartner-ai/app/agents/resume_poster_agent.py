import os
import json
import uuid
from typing import Dict, Any
from datetime import datetime
import re
from google.cloud import bigquery, storage
from google.cloud import aiplatform
from vertexai.language_models import TextEmbeddingModel
from ..agent import BaseAgent, EmailData
from ..config import config
from ..tools.bq_schema_manager import ensure_table_exists
from ..tools.file_processor import FileProcessor
from datetime import datetime
from jinja2 import Environment, FileSystemLoader

RESUME_ANALYSIS_PROMPT = '''You are an expert resume analyzer. Your task is to extract detailed information from the provided resume text and format it into a single, valid JSON object that strictly adheres to the schema provided below.

**CRITICAL INSTRUCTIONS:**
1.  **JSON ONLY:** Your entire response MUST be a single JSON object enclosed in a ```json markdown block. Do not include any text, comments, or formatting outside this block.
2.  **SINGLE-LINE STRINGS:** All string values within the JSON must be on a single line. Do not use multi-line strings. This is critical for parsing.
3.  **STRICT SCHEMA:** The JSON structure must exactly match the schema provided below. Do not add, remove, or rename any fields.
4.  **SYNTAX:** Pay extremely close attention to JSON syntax. Ensure all strings are in double quotes, and there are no trailing commas.
5.  **NULL VALUES:** If a value for a field cannot be found in the resume, use the JSON literal `null` (not the string "null" or an empty string).
6.  **DATES:** Format all dates as "YYYY-MM-DD". If a full date is not available, do your best to infer it or use `null`.
7.  **ESCAPING:** You MUST escape any backslashes (\\) or double quotes (") within string values by using a double backslash (\\\\) or (\"). This is especially important for the `resume_text` field.

**BigQuery Schema to Follow:**
```json
{{
  "candidate_id": "{candidate_id}",
  "candidate_name": "Full name of the candidate",
  "candidate_email": "Primary email address",
  "candidate_phone": "Primary phone number",
  "candidate_location": "City, State, Country",
  "linkedin_url": "The candidate's LinkedIn profile URL",
  "date_of_birth": "YYYY-MM-DD",
  "resume_summary": "A concise 2-3 sentence summary of the candidate's profile",
  "total_years_experience": 10.5,
  "technical_skills": ["Python", "SQL", "BigQuery", "GCP"],
  "soft_skills": ["Communication", "Problem Solving"],
  "skill_proficiency": [
    {{
      "skill": "Skill name",
      "level": "Expert/Advanced/Intermediate/Beginner",
      "years_experience": 3.5
    }}
  ],
  "experience": [
    {{
      "company": "Company Name",
      "role": "Job Title",
      "start_date": "YYYY-MM-DD",
      "end_date": "YYYY-MM-DD",
      "duration_months": 24,
      "location": "City, State",
      "technologies_used": ["Tech1", "Tech2"]
    }}
  ],
  "education": [
    {{
      "degree": "Master of Science",
      "institution": "University Name",
      "field_of_study": "Computer Science",
      "graduation_year": 2020
    }}
  ],
  "certifications": ["Google Cloud Certified - Professional Data Engineer"],
  "previous_companies": ["List of all previous companies worked at"],
  "domains_worked_in": ["Finance", "Healthcare", "Technology"],
  "languages": [
    {{
      "language": "Language name",
      "proficiency": "Native/Fluent/Intermediate/Basic"
    }}
  ],
  "visa_status": "H1B/GC/Citizen/etc",
  "work_preferences": {{
    "preferred_work_models": ["Remote", "Hybrid", "On-site"],
    "preferred_employment_types": ["Full-time", "Contract", "Part-time"],
    "willing_to_relocate": true,
    "salary_expectations": "Annual salary expectation range"
  }},
  "resume_text": "The full, plain text of the resume"
}}
```

**Resume Text to Analyze:**
```
{resume_text}
```
'''

class ResumeProcessorAgent(BaseAgent):
    """An agent that processes resumes, extracts information, and saves it to BigQuery."""

    def __init__(self):
        super().__init__("ResumeProcessor", model_name="gemini-1.5-pro")
        self.output_dir = "resume_submissions"
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        self.logger.info("Initializing ResumeProcessorAgent...")
        self.logger.info("Initializing BigQuery client...")
        self.bq_client = self._get_bigquery_client()
        
        self.storage_client = storage.Client()
        self.bucket_name = "myhiringpartner-ai_artifacts"
        self.folder_prefix = "resumes/"
        
        self.logger.info(f"Getting GCS bucket: {self.bucket_name}")
        try:
            self.bucket = self.storage_client.get_bucket(self.bucket_name)
        except Exception as e:
            self.logger.warning(f"Bucket {self.bucket_name} not found, creating...")
            self.bucket = self.storage_client.create_bucket(
                self.bucket_name,
                location="us-central1"
            )
        
        self.table_id = None
        if self.bq_client:
            self.table_id = f"{self.bq_client.project}.{config.bq_dataset}.resumes"
            try:
                ensure_table_exists(self.bq_client, self.table_id)
                self.logger.info(f"BigQuery client and table '{self.table_id}' verified successfully.")
            except Exception as e:
                self.logger.error(f"Failed to verify or create BigQuery table '{self.table_id}': {e}", exc_info=True)
                self.bq_client = None
        
        try:
            project_id = self.bq_client.project
            location = getattr(config, 'location')
            self.logger.info(f"Initializing Vertex AI Platform with project='{project_id}' and location='{location}'")
            aiplatform.init(project=project_id, location=location)
            self.embedding_model = TextEmbeddingModel.from_pretrained("text-embedding-005")
        except Exception as e:
            self.logger.error(f"Failed to initialize embedding model: {e}", exc_info=True)
            self.embedding_model = None

        # Initialize Jinja2 environment
        template_dir = os.path.join(os.path.dirname(__file__), '..', 'templates')
        self.jinja_env = Environment(loader=FileSystemLoader(template_dir))

    def send_email(self, to: str, subject: str, body: str):
        # This method should be implemented to send emails.
        # For now, it will just log the email content.
        self.logger.info(f"Sending email to: {to}")
        self.logger.info(f"Subject: {subject}")
        self.logger.info(f"Body: {body}")

    def _check_essential_information(self, resume_data: Dict[str, Any]) -> Dict[str, Any]:
        """Checks for missing essential information in the resume data."""
        missing_info = {}
        if not resume_data.get("linkedin_url"):
            missing_info["linkedin_url"] = "LinkedIn Profile URL"

        education = resume_data.get("education", [])
        if not isinstance(education, list):
            education = []

        # Check for bachelor's degree
        has_bachelors = any(
            edu.get("degree", "").lower().startswith("bachelor") 
            or edu.get("degree", "").lower().startswith("b.") 
            for edu in education
        )
        if not has_bachelors:
            missing_info["bachelors"] = "Bachelor's degree details"

        # Check for master's degree
        has_masters = any(
            edu.get("degree", "").lower().startswith("master")
            or edu.get("degree", "").lower().startswith("m.")
            for edu in education
        )
        if not has_masters:
            missing_info["masters"] = "Master's degree details"

        if not resume_data.get("date_of_birth"):
            missing_info["date_of_birth"] = "Date of Birth"

        return missing_info

    def _generate_candidate_email(self, missing_info: Dict[str, Any], resume_data: Dict[str, Any]) -> EmailData:
        """Generates an email to the candidate requesting missing information."""
        template = self.jinja_env.get_template('candidate_followup_email.html')

        # Transform education data for the template
        education_template_data = {
            'bachelors': any(
                edu.get("degree", "").lower().startswith("bachelor")
                or edu.get("degree", "").lower().startswith("b.")
                for edu in resume_data.get('education', [])
            ),
            'masters': any(
                edu.get("degree", "").lower().startswith("master")
                or edu.get("degree", "").lower().startswith("m.")
                for edu in resume_data.get('education', [])
            )
        }

        html_body = template.render(
            candidate_name=resume_data.get('candidate_name', 'there'),
            linkedin_url=resume_data.get('linkedin_url'),
            education=education_template_data,
            date_of_birth=resume_data.get('date_of_birth')
        )

        subject = f"Action Required: Additional Information for Your Application"
        return EmailData(
            subject=subject, 
            body=html_body, 
            recipient=resume_data.get('candidate_email'),
            sender="system@myhiringpartner.ai",
            timestamp=datetime.now().isoformat(),
            attachments=[]
        )

    def _get_bigquery_client(self):
        try:
            return bigquery.Client()
        except Exception as e:
            self.logger.error(f"Failed to initialize BigQuery client: {e}", exc_info=True)
            return None

    def _generate_embeddings(self, text: str) -> list:
        """Generates embeddings for the given text using Vertex AI."""
        if not self.embedding_model:
            return []
            
        try:
            response = self.embedding_model.get_embeddings([text])
            return response[0].values if response else []
        except Exception as e:
            self.logger.error(f"Failed to generate embeddings: {e}", exc_info=True)
            return []

    def _clean_json_response(self, response: str) -> Dict[str, Any]:
        """Cleans and parses the LLM's JSON response with robust error handling."""
        try:
            # 1. Aggressively find the JSON block
            match = re.search(r'```(?:json)?\n?([\s\S]*?)\n?```', response)
            if match:
                json_str = match.group(1).strip()
            else:
                start_idx = response.find('{')
                end_idx = response.rfind('}') + 1
                if start_idx != -1 and end_idx != 0:
                    json_str = response[start_idx:end_idx].strip()
                else:
                    self.logger.error("Could not extract JSON block from LLM response.")
                    self.logger.debug(f"Raw response: {response}")
                    return None

            # 2. Pre-cleaning and syntax correction
            json_str = re.sub(r'[\x00-\x1f]', '', json_str)
            json_str = re.sub(r'\}\s*\{', '}, {', json_str)
            json_str = re.sub(r'(")\s*\n\s*(")', r'\1,\n\2', json_str)
            json_str = re.sub(r'(\d)\s*\n\s*(")', r'\1,\n\2', json_str)
            json_str = re.sub(r'(\])\s*\n\s*(")', r'\1,\n\2', json_str)
            json_str = re.sub(r'(\})\s*\n\s*(")', r'\1,\n\2', json_str)

            # 3. Final parsing attempt
            try:
                cleaned_str = self._remove_trailing_commas(json_str)
                return json.loads(cleaned_str)
            except json.JSONDecodeError as e:
                self.logger.error(f"JSON parsing failed after all cleaning attempts: {e}")
                self.logger.debug(f"Problematic JSON string for debugging: {cleaned_str}")
                return None

        except Exception as e:
            self.logger.error(f"An unexpected error occurred during JSON cleaning: {e}", exc_info=True)
            return None

    def _remove_trailing_commas(self, json_str: str) -> str:

        pattern = r',(?=\\s*[}\\]])'
        return re.sub(pattern, '', json_str)

    def _upload_to_gcs(self, file_path: str, content: bytes = None) -> str:
        """Uploads content to Google Cloud Storage and returns the URI."""
        try:
            blob_name = f"{self.folder_prefix}{os.path.basename(file_path)}"
            blob = self.bucket.blob(blob_name)
            
            if content:
                blob.upload_from_string(content, content_type='application/pdf')
            else:
                blob.upload_from_filename(file_path, content_type='application/pdf')
                
            self.logger.info(f"Successfully uploaded resume to gs://{self.bucket_name}/{blob_name}")
            return f"gs://{self.bucket_name}/{blob_name}"
        except Exception as e:
            self.logger.error(f"Failed to upload to GCS: {e}", exc_info=True)
            return ""

    def analyze_resume(self, resume_text: str, candidate_id: str, email_data: EmailData = None) -> Dict[str, Any]:
        """Analyzes the resume text using the LLM to extract structured data."""
        self.logger.info("Analyzing resume text...")
        prompt = RESUME_ANALYSIS_PROMPT.format(resume_text=resume_text, candidate_id=candidate_id)
        
        try:
            response = self.model.generate_content(prompt).text
            self.logger.debug(f"Raw LLM response: {response[:500]}...")
            
            analysis = self._clean_json_response(response)
            if not analysis:
                self.logger.error("Failed to parse JSON from LLM response after cleaning.")
                return {}

            # Add candidate_id to the analysis
            analysis['candidate_id'] = candidate_id
            
            # Try to extract LinkedIn URL from email body if available
            if email_data and email_data.body:
                linkedin_match = re.search(r'(?:linkedin\.com/in/[^\s)]+|linkedin profile:?\s*(https?://[^\s)]+))', email_data.body, re.IGNORECASE)
                if linkedin_match:
                    url = linkedin_match.group(1) if linkedin_match.group(1) else linkedin_match.group(0)
                    if not url.startswith('http'):
                        url = 'https://www.' + url
                    analysis['linkedin_url'] = url
                
            self.logger.info("Successfully parsed resume analysis.")
            return analysis

        except Exception as e:
            self.logger.error(f"Failed to analyze resume: {e}", exc_info=True)
            return {}

    def save_resume_to_bq(self, resume_data: Dict[str, Any], embeddings: list, storage_uri: str) -> bool:
        """Saves the structured resume data and embeddings to BigQuery."""
        if not self.bq_client or not self.table_id:
            return False

        try:
            row = {
                **resume_data,
                "embeddings": embeddings,
                "storage_uri": storage_uri,
                "created_at": datetime.utcnow().isoformat(),
                "updated_at": datetime.utcnow().isoformat()
            }

            errors = self.bq_client.insert_rows_json(self.table_id, [row])
            if not errors:
                self.logger.info(f"Successfully inserted resume for {resume_data.get('candidate_name')} into BigQuery.")
                return True
            else:
                self.logger.error(f"BigQuery insertion errors: {errors}")
                return False
        except Exception as e:
            self.logger.error(f"Failed to save resume to BigQuery: {e}", exc_info=True)
            return False

    def run(self, email_data: EmailData) -> Dict[str, Any]:
        """Main method to process an email with a resume attachment."""
        temp_resume_path = None
        try:
            attachments = email_data.attachments
            if not attachments:
                self.logger.error("No resume attachment found.")
                return {"status": "error", "message": "No resume attachment found"}

            attachment = attachments[0]
            pdf_bytes = bytes.fromhex(attachment['content'])
            original_filename = attachment.get('filename', 'resume.pdf')
            temp_resume_path = os.path.join(self.output_dir, f"temp_{original_filename}")

            with open(temp_resume_path, 'wb') as f:
                f.write(pdf_bytes)
            self.logger.info(f"Processing temporary resume file: {temp_resume_path}")
            
            storage_uri = self._upload_to_gcs(temp_resume_path, pdf_bytes)
            if not storage_uri:
                return {"status": "error", "message": "Failed to upload resume to storage"}

            resume_text = FileProcessor.extract_text_from_file(temp_resume_path)
            if not resume_text:
                return {"status": "error", "message": "Could not extract text from PDF."}

            candidate_id = str(uuid.uuid4())
            
            analysis = self.analyze_resume(resume_text, candidate_id)
            if not analysis:
                return {"status": "error", "message": "Failed to analyze resume"}

            embeddings = self._generate_embeddings(resume_text)
            
            if self.save_resume_to_bq(analysis, embeddings, storage_uri):
                missing_info = self._check_essential_information(analysis)
                if missing_info:
                    email_to_send = self._generate_candidate_email(missing_info, analysis)
                    if email_to_send.recipient:
                        self.send_email(to=email_to_send.recipient, subject=email_to_send.subject, body=email_to_send.body)
                    return {
                        "status": "success",
                        "message": "Resume processed, but essential information is missing. An email has been sent to the recruiter.",
                        "candidate_id": candidate_id,
                        "details": {
                            "candidate_name": analysis.get("candidate_name"),
                            "storage_uri": storage_uri,
                            "missing_info": missing_info
                        }
                    }
                else:
                    return {
                        "status": "success",
                        "message": "Resume processed and saved to BigQuery.",
                        "candidate_id": candidate_id,
                        "details": {
                            "candidate_name": analysis.get("candidate_name"),
                            "storage_uri": storage_uri
                        }
                    }
            else:
                return {"status": "error", "message": "Failed to save resume data"}

        except Exception as e:
            self.logger.error(f"Error processing resume email: {e}", exc_info=True)
            return {"status": "error", "message": f"An error occurred: {str(e)}"}
        finally:
            if temp_resume_path and os.path.exists(temp_resume_path):
                os.remove(temp_resume_path)
                self.logger.info(f"Removed temporary file: {temp_resume_path}")