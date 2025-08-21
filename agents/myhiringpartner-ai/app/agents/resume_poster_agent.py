import os
import json
import uuid
import asyncio
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
from .matching_service import MatchingService

RESUME_ANALYSIS_PROMPT = '''You are an expert resume analyzer. Your task is to extract detailed information from the provided resume text and format it into a single, valid JSON object that strictly adheres to the schema provided below.

**CRITICAL INSTRUCTIONS:**
1.  **JSON ONLY:** Your entire response MUST be a single JSON object enclosed in a ```json markdown block. Do not include any text, comments, or formatting outside this block.
2.  **SINGLE-LINE STRINGS:** All string values within the JSON must be on a single line. Do not use multi-line strings. This is critical for parsing.
3.  **STRICT SCHEMA:** The JSON structure must exactly match the schema provided below. Do not add, remove, or rename any fields.
4.  **ARRAY FIELDS MUST BE JSON ARRAYS:** Any field that is defined as a repeated field in the schema (for example: technical_skills, previous_companies, domains_worked_in, certifications, and within experience -> technologies_used) MUST be a JSON array of strings. Do not return a single comma-separated string for these fields.
5.  **EXPERIENCE.technologies_used:** For each experience entry, the `technologies_used` field MUST be an array of strings (e.g. ["Python", "SQL"]). If only a comma-separated string is available, split into an array. If none, return an empty array `[]` (not null or an empty string).
6.  **SKILL_PROFICIENCY:** The skill_proficiency array must contain objects with `skill`, `level`, and `years_experience` as described in the schema.
7.  **NULL VALUES:** If a value for a field cannot be found in the resume, use the JSON literal `null` for nullable scalar fields. For repeated fields use an empty array `[]` when nothing is present.
8.  **DATES:** Format all dates as "YYYY-MM-DD". If a full date is not available, do your best to infer it or use `null`.
9.  **ESCAPING:** You MUST escape any backslashes (\\) or double quotes (\") within string values by using a double backslash (\\\\) or (\"). This is especially important for the `resume_text` field.

**BigQuery Schema to Follow:**
```json
{{
  "candidate_id": "{candidate_id}",
  "candidate_name": "Full name of the candidate",
  "candidate_email": "Primary email address",~
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
        project_id = os.getenv('GOOGLE_CLOUD_PROJECT')
        self.bucket_name = f"{project_id}-myhiringpartner-ai-artifacts"
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

        # Check for bachelor's degree (handle None values safely)
        def _norm_degree(e: Any) -> str:
            try:
                if not isinstance(e, dict):
                    return ""
                v = e.get("degree")
                return (v or "").strip().lower()
            except Exception:
                return ""

        has_bachelors = any(
            _norm_degree(edu).startswith("bachelor") or _norm_degree(edu).startswith("b.")
            for edu in education
        )
        if not has_bachelors:
            missing_info["bachelors"] = "Bachelor's degree details"

        # Check for master's degree (handle None values safely)
        has_masters = any(
            _norm_degree(edu).startswith("master") or _norm_degree(edu).startswith("m.")
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

        # Transform education data for the template (handle None values safely)
        def _norm_degree(e: Any) -> str:
            try:
                if not isinstance(e, dict):
                    return ""
                v = e.get("degree")
                return (v or "").strip().lower()
            except Exception:
                return ""

        education_list = resume_data.get('education', [])
        if not isinstance(education_list, list):
            education_list = []

        education_template_data = {
            'bachelors': any(_norm_degree(edu).startswith("bachelor") or _norm_degree(edu).startswith("b.") for edu in education_list),
            'masters': any(_norm_degree(edu).startswith("master") or _norm_degree(edu).startswith("m.") for edu in education_list)
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
            self.logger.error(f"Failed to initialize BigQuery client: {str(e)}", exc_info=True)
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
            if (match):
                json_str = match.group(1).strip()
            else:
                start_idx = response.find('{')
                end_idx = response.rfind('}') + 1
                if (start_idx != -1 and end_idx != 0):
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

    def _sanitize_resume_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize resume data to conform to BigQuery schema expectations.

        - Ensure nested repeated fields such as experience[].technologies_used are lists of strings.
        - For arrays of structs (experience, education, languages, skill_proficiency),
          replace None or empty-string date fields with None to avoid invalid BQ dates.
        - Preserve numeric fields as-is; if they are None, keep them None (nullable).
        - Ensure lists are actually lists; if not, coerce to empty list.
        """
        def sanitize_list_of_dicts(items, parent_field=None):
            if not isinstance(items, list):
                return []
            sanitized = []
            numeric_fields = ['duration_months', 'years_of_experience']  # Add all numeric fields here
            date_fields = ['start_date', 'end_date']

            for item in items:
                if not isinstance(item, dict):
                    continue

                new_item = {}
                for k, v in item.items():
                    # Ensure technologies_used (a repeated string field) is always a list
                    if k == 'technologies_used':
                        if v is None:
                            new_item[k] = []
                        elif isinstance(v, list):
                            # normalize elements to strings and drop nulls
                            new_item[k] = [str(x).strip() for x in v if x is not None and str(x).strip()]
                        elif isinstance(v, str):
                            # split on commas or semicolons and strip whitespace
                            parts = [p.strip() for p in re.split(r'[;,]', v) if p.strip()]
                            new_item[k] = parts if parts else []
                        else:
                            # fallback: coerce single value to string list
                            try:
                                new_item[k] = [str(v)]
                            except Exception:
                                new_item[k] = []
                    # Normalize date fields: convert empty strings to None, validate YYYY-MM-DD
                    elif k in date_fields:
                        if v is None or (isinstance(v, str) and not v.strip()):
                            new_item[k] = None
                        else:
                            if isinstance(v, str) and re.match(r'^\d{4}-\d{2}-\d{2}$', v.strip()):
                                new_item[k] = v.strip()
                            else:
                                if isinstance(v, str) and re.match(r'^\d{4}$', v.strip()):
                                    new_item[k] = f"{v.strip()}-01-01"
                                else:
                                    new_item[k] = None
                    # Handle numeric fields that might be empty strings or invalid
                    elif k in numeric_fields:
                        if v == '' or v is None:
                            new_item[k] = None
                        else:
                            try:
                                new_item[k] = int(v) if str(v).strip().isdigit() else v
                            except (ValueError, TypeError, AttributeError):
                                new_item[k] = None
                    # Preserve nested lists/dicts as-is where appropriate
                    elif isinstance(v, list):
                        # normalize list elements to strings where possible
                        new_item[k] = [x if not isinstance(x, dict) else x for x in v]
                    elif isinstance(v, (int, float, bool)):
                        new_item[k] = v
                    elif v is None:
                        # For string-typed fields set None -> None (leave as NULL)
                        new_item[k] = None
                    else:
                        new_item[k] = str(v)
                sanitized.append(new_item)
            return sanitized

        sanitized_data = dict(data or {})

        # Known RECORD arrays
        for field in ["experience", "education", "languages", "skill_proficiency"]:
            sanitized_data[field] = sanitize_list_of_dicts(sanitized_data.get(field, []), parent_field=field)

        # For education, ensure graduation_year is numeric or None
        if isinstance(sanitized_data.get('education'), list):
            for edu in sanitized_data['education']:
                if isinstance(edu, dict):
                    gy = edu.get('graduation_year')
                    if gy is None or (isinstance(gy, str) and not gy.strip()):
                        edu['graduation_year'] = None
                    else:
                        try:
                            if isinstance(gy, str) and re.match(r'^\d{4}$', gy.strip()):
                                edu['graduation_year'] = int(gy.strip())
                        except Exception:
                            edu['graduation_year'] = None

        # Known string arrays: technical_skills, soft_skills, certifications, previous_companies, domains_worked_in
        for field in [
            "technical_skills",
            "soft_skills",
            "certifications",
            "previous_companies",
            "domains_worked_in",
        ]:
            val = sanitized_data.get(field)
            if isinstance(val, list):
                sanitized_data[field] = [str(x) for x in val if x is not None]
            elif isinstance(val, str):
                parts = [p.strip() for p in re.split(r'[;,]', val) if p.strip()]
                sanitized_data[field] = parts
            elif val is None:
                sanitized_data[field] = []

        # Basic top-level normalizations (safe defaults)
        for sfield in [
            "candidate_name",
            "candidate_email",
            "candidate_phone",
            "candidate_location",
            "linkedin_url",
            "resume_summary",
            "visa_status",
        ]:
            if sanitized_data.get(sfield) is None:
                sanitized_data[sfield] = None  # keep as NULL at top-level (nullable)

        return sanitized_data

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
        # Ensure BigQuery client/table initialized on-demand
        if not self.bq_client:
            self.logger.warning("BigQuery client not initialized. Attempting to re-initialize...")
            self.bq_client = self._get_bigquery_client()
            if not self.bq_client:
                self.logger.error("BigQuery client could not be initialized.")
                return False
        if not self.table_id:
            self.table_id = f"{self.bq_client.project}.{config.bq_dataset}.resumes"
        try:
            ensure_table_exists(self.bq_client, self.table_id)
        except Exception as e:
            self.logger.error(f"Failed to verify or create BigQuery table '{self.table_id}': {e}", exc_info=True)
            return False

        try:
            sanitized = self._sanitize_resume_data(resume_data)
            row = {
                **sanitized,
                "embeddings": embeddings,
                "storage_uri": storage_uri,
                "created_at": datetime.utcnow().isoformat(),
                "updated_at": datetime.utcnow().isoformat()
            }

            # Debug: log the sanitized row at DEBUG level to inspect arrays/structures
            try:
                self.logger.debug(f"Sanitized row payload for BQ insert: {json.dumps(row, default=str)[:2000]}")
            except Exception:
                self.logger.debug("Sanitized row payload could not be JSON-serialized for debug logging.")

            errors = self.bq_client.insert_rows_json(self.table_id, [row])
            if not errors:
                self.logger.info(f"Successfully inserted resume for {resume_data.get('candidate_name')} into BigQuery.")
                return True
            else:
                # Log the full sanitized row when insertion fails to aid debugging
                self.logger.error(f"BigQuery insertion errors: {errors}")
                try:
                    self.logger.error(f"Full sanitized row causing error: {json.dumps(row, default=str)}")
                except Exception:
                    self.logger.error("Failed to serialize sanitized row for logging.")
                return False
        except Exception as e:
            self.logger.error(f"Failed to save resume to BigQuery: {e}", exc_info=True)
            return False

    def _extract_job_title_from_subject(self, subject: str) -> str:
        """Extract job title from subjects like:
        Returns the job title (e.g. 'Data Engineer') or None if not found.
        """
        if not subject or not isinstance(subject, str):
            return None
        try:
            cleaned = re.sub(r"[\[\]\(\)]", " ", subject)
            m = re.search(r'New application:\s*(?P<title>.*?)(?:\s+[A-Za-z]{2,}[-_ ]?\d+|\s+from\b|$)', cleaned, re.IGNORECASE)
            if m:
                return m.group('title').strip()
            m2 = re.search(r':\s*(?P<title>.*?)\s+from\b', cleaned, re.IGNORECASE)
            if m2:
                return m2.group('title').strip()
            return None
        except Exception:
            return None

    def run(self, email_data: EmailData) -> Dict[str, Any]:
        """Main method to process an email with a resume attachment."""
        temp_resume_path = None
        try:
            attachments = email_data.attachments
            if (not attachments) and getattr(email_data, 'gcs_path', None):
                attachments = [{
                    'storage_path': getattr(email_data, 'gcs_path', None),
                    'bucket_name': getattr(email_data, 'bucket_name', None),
                    'filename': 'resume.pdf'
                }]

            if not attachments:
                self.logger.error("No resume attachment found.")
                return {"status": "error", "message": "No resume attachment found"}

            attachment = attachments[0]
            pdf_bytes = None
            original_filename = attachment.get('filename', 'resume.pdf')

            # Case 1: inline hex content provided
            if 'content' in attachment and attachment.get('content'):
                try:
                    pdf_bytes = bytes.fromhex(attachment['content'])
                except Exception as e:
                    self.logger.warning(f"Failed to decode inline attachment hex content: {e}")
                    pdf_bytes = None

            # Case 2: GCS reference provided
            if pdf_bytes is None:
                storage_path = attachment.get('storage_path') or attachment.get('gcs_path')
                bucket_name = attachment.get('bucket_name') or getattr(email_data, 'bucket_name', None) or self.bucket_name
                if storage_path and bucket_name:
                    try:
                        blob = self.storage_client.bucket(bucket_name).blob(storage_path)
                        pdf_bytes = blob.download_as_bytes(timeout=60)
                        if not original_filename or original_filename == 'resume.pdf':
                            original_filename = os.path.basename(storage_path) or 'resume.pdf'
                        self.logger.info(f"Downloaded attachment from gs://{bucket_name}/{storage_path}")
                    except Exception as e:
                        self.logger.error(f"Failed to download attachment from gs://{bucket_name}/{storage_path}: {e}")
                        return {"status": "error", "message": "Failed to download attachment from storage"}

            if pdf_bytes is None:
                self.logger.error("Attachment content unavailable (no inline content and no valid GCS reference).")
                return {"status": "error", "message": "Attachment content unavailable"}

            temp_resume_path = os.path.join(self.output_dir, f"temp_{original_filename}")

            with open(temp_resume_path, 'wb') as f:
                f.write(pdf_bytes)
            self.logger.info(f"Processing temporary resume file: {temp_resume_path}")
            
            storage_uri = self._upload_to_gcs(temp_resume_path, pdf_bytes)
            if not storage_uri:
                return {"status": "error", "message": "Failed to upload resume to storage"}

            self.logger.info("Extracting text from resume file...")
            # Use asyncio to run the async method
            resume_text = asyncio.run(FileProcessor.extract_text_from_file(temp_resume_path))
            self.logger.info(f"Extracted resume text length: {len(resume_text) if resume_text else 0}")
            if not resume_text:
                return {"status": "error", "message": "Could not extract text from PDF."}

            candidate_id = str(uuid.uuid4())
            
            self.logger.info("Analyzing resume with LLM...")
            analysis = self.analyze_resume(resume_text, candidate_id)
            if not analysis:
                return {"status": "error", "message": "Failed to analyze resume"}

            self.logger.info("Generating embeddings for resume text...")
            embeddings = self._generate_embeddings(resume_text)
            
            self.logger.info("Inserting resume record into BigQuery...")
            if self.save_resume_to_bq(analysis, embeddings, storage_uri):
                match_results = None

                response = {
                    "status": "success",
                    "message": "Resume processed and saved to BigQuery.",
                    "candidate_id": candidate_id,
                    "details": {
                        "candidate_name": analysis.get('candidate_name'),
                        "storage_uri": storage_uri
                    }
                }

                if match_results is not None:
                    response['matching_results'] = match_results

                return response

        except Exception as e:
            self.logger.error(f"Error processing resume email: {e}", exc_info=True)
            return {"status": "error", "message": f"An error occurred: {str(e)}"}
        finally:
            if temp_resume_path and os.path.exists(temp_resume_path):
                os.remove(temp_resume_path)
                self.logger.info(f"Removed temporary file: {temp_resume_path}")