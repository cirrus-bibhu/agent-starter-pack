import os
import re
import json
import uuid
import base64
from typing import Dict, Any, Union

from dotenv import load_dotenv

load_dotenv()

from ..agent import BaseAgent, EmailData
from datetime import datetime, timezone, timedelta
from ..tools.linkedin_scrapper import JobExtractor
from google.cloud import bigquery
from ..config import config
from ..tools.bq_schema_manager import ensure_table_exists
from firebase_admin import firestore, initialize_app
from google.cloud.firestore_v1.base_query import FieldFilter

# Gmail and authentication imports
from google.cloud import secretmanager
import google.auth
from google.api_core import exceptions as gcloud_exceptions
from googleapiclient.discovery import build
from google.auth.transport.requests import Request
import google.auth.credentials
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import logging

EXTRACT_JOB_LINK_PROMPT = """Extract the LinkedIn job posting URL from this email. Return only the URL, nothing else.
Accept URLs that contain either '/jobs/view/' or '/comm/jobs/view/' and may include query parameters or tracking IDs.
If multiple URLs are found, return the one most likely to be the job posting.
If no URL is found, return 'NO_URL_FOUND'."""

JOB_ANALYSIS_PROMPT = '''You are an expert job description analyzer. Extract detailed information from the provided job description into a JSON format that matches the given BigQuery schema.

JOB DESCRIPTION:
{job_description}

IMPORTANT: Your entire response must be ONLY valid JSON. Do not include any explanation, markdown formatting, or text outside the JSON object. Your response must start with '{{' and end with '}}'.

Extract the following information in this exact JSON structure:
{{
  "job_id": "Unique identifier for the job",
  "customer_type": "end_client/sub_vendor/prime_vendor (if specified)",
  "recruiter_email": "Email of the recruiter who posted the job (if available in context)",
  "prime_vendor_email": "Email address of the primary vendor (if specified)",
  "sub_vendor_email": "Email address of the sub-vendor (if specified)",
  "prime_vendor_name": "Name of the primary vendor or direct agency (if specified)",
  "sub_vendor_name": "Name of the sub-vendor or secondary agency (if specified)",
  "end_client_name": "Name of the specific end-client for whom the job is intended (if specified)",
  "job_title": "The official title of the job position",
  "job_description": "The full, detailed description of the job duties and requirements.",
  "department": "The department within the end-client's organization where the job is located (if specified)",
  "is_relocation_allowed": true/false,  // Must be actual boolean value (true or false)
  "is_remote": true/false,              // Must be actual boolean value (true or false)
  "job_location_city": "Job Location City (if specified)",
  "job_location_state": "Job Location State (if specified)",
  "job_location_country": "Job Location Country (if specified)",
  "is_only_ex_employee_required": true/false,  // Must be actual boolean value (true or false)
  "required_technical_skills": ["A list of essential technical skills required for the job"],
  "preferred_technical_skills": ["A list of desirable (but not strictly required) technical skills"],
  "job_domains": [
    {{
      "domain_name": "The name of a specific job domain (e.g., 'Finance', 'Healthcare')",
      "requirement_type": "The type of requirement for this domain (e.g., 'primary', 'secondary')"
    }}
  ],
  "preferred_previous_companies": ["A list of companies from which candidates are preferred to have prior experience"],
  "min_required_years_experience": "The minimum number of years of overall professional experience required",
  "max_preferred_years_experience": "The maximum number of years of overall professional experience preferred",
  "required_skill_experience_years": [
    {{
      "skill": "The name of the specific skill",
      "min_years": "The minimum years of experience required for this particular skill"
    }}
  ],
  "job_location_city_state": ["A list of city and state combinations for the job's physical location(s)"],
  "job_work_model": ["A list of accepted work models (e.g., 'Remote', 'Hybrid', 'Onsite')"],
  "job_employment_type": ["A list of employment types (e.g., 'Full-time', 'Contract', 'Part-time')"],
  "job_geography": "A broader geographical area for the job (e.g., 'EMEA', 'North America')",
  "job_certifications": [
    {{
      "name": "Name of certification",
      "is_mandatory": true/false  // Must be actual boolean value (true or false)
    }}
  ],
  "optional_requirements_list": ["A list of additional, non-mandatory requirements or nice-to-haves"],
  "questionnaire_details": {{
    "has_questionnaire": true/false,  // Must be actual boolean value (true or false)
    "is_mandatory": true/false,       // Must be actual boolean value (true or false)
    "questionnaire_type": "skill_matrix/custom_questions/application_form/etc.",
    "questions": [
      {{
        "question_text": "The full text of the question or field to be filled",
        "field_type": "skill/experience/rating/text/date/etc."
      }}
    ]
  }},
  "posting_date": "The date the job was first made public or posted (if specified)",
  "closing_date": "The date the job is expected to close or stop accepting applications (if specified)",
  "job_summary": "A concise summary of the job listing, about 2-3 sentences."
}}

IMPORTANT GUIDELINES:
1. Only extract information that is EXPLICITLY stated in the job description.
2. If information for a field is not found, use "Not specified" for string values, empty arrays [], null for objects, or false for booleans where appropriate.
3. For boolean fields, use `true` or `false` JSON literals, not strings like `"true"` or `"false"`.
4. For numerical fields, provide numbers, not strings (e.g., 5, not "5 years").
5. Be precise and avoid making assumptions. If the JD says "5+ years", use 5 for years.
6. The entire output must be a single, valid JSON object.
7. Do not include any text, comments, or markdown formatting outside of the JSON structure.
8. Maintain the exact structure and field names as specified in the template.
'''

# Gmail-related constants
EMAIL_TOKEN_PREFIX = "email-token"
REFRESH_TOKEN_PREFIX = "refresh"
ACCESS_TOKEN_PREFIX = "access"

# Gmail configuration - these should be set in your environment or config
try:
    initialize_app()
    db = firestore.client()
except Exception as e:
    raise

# Gmail helper classes and functions
class SecretManagerError(Exception):
    """Custom exception for Secret Manager operations."""
    pass

class GmailError(Exception):
    """Custom exception for Gmail operations."""
    pass

def _get_project_id() -> str:
    """Gets the Google Cloud Project ID from the environment or default auth."""
    project_id = os.environ.get("GOOGLE_CLOUD_PROJECT")
    if not project_id:
        try:
            _, project_id = google.auth.default()
        except google.auth.exceptions.DefaultCredentialsError:
            raise EnvironmentError("GOOGLE_CLOUD_PROJECT env var not set or gcloud auth not configured.")
    return project_id

def _sanitize_id_components(*args) -> tuple:
    """Sanitizes strings to be used in Secret Manager secret IDs."""
    sanitized = []
    for component in args:
        sanitized.append(component.replace('@', '-at-').replace('.', '-dot-'))
    return tuple(sanitized)

def get_latest_secret_version(client: secretmanager.SecretManagerServiceClient, project_id: str, secret_id: str) -> str:
    """Retrieve the latest version of a secret from Secret Manager."""
    try:
        secret_path = f"projects/{project_id}/secrets/{secret_id}/versions/latest"
        response = client.access_secret_version(request={"name": secret_path})
        return response.payload.data.decode("UTF-8")
    except gcloud_exceptions.NotFound:
        raise SecretManagerError(f"Secret '{secret_id}' not found in Secret Manager")
    except Exception as e:
        raise SecretManagerError(f"Error retrieving secret '{secret_id}': {str(e)}")

def retrieve_oauth_tokens_from_secret_manager(email: str) -> tuple:
    """Retrieve OAuth tokens from Google Cloud Secret Manager."""
    try:
        secret_client = secretmanager.SecretManagerServiceClient()
        project_id = _get_project_id()
        safe_email = _sanitize_id_components(email)[0]
        
        # Correcting the secret ID to use the correct email
        refresh_token_secret_id = f"{EMAIL_TOKEN_PREFIX}_{REFRESH_TOKEN_PREFIX}_support-at-myhiringpartner-dot-ai"
        refresh_token = get_latest_secret_version(secret_client, project_id, refresh_token_secret_id)

        # Retrieve access token data
        access_token_secret_id = f"{EMAIL_TOKEN_PREFIX}_{ACCESS_TOKEN_PREFIX}_support-at-myhiringpartner-dot-ai"
        access_token_json = get_latest_secret_version(secret_client, project_id, access_token_secret_id)
        access_token_data = json.loads(access_token_json)
        
        return refresh_token, access_token_data
        
    except Exception as e:
        raise SecretManagerError(f"Failed to retrieve OAuth tokens: {str(e)}")

class SecretManagerCredentials(google.auth.credentials.Credentials):
    """Custom credentials class that uses tokens from Secret Manager."""
    
    def __init__(self, refresh_token: str, access_token_data: dict, client_id: str, client_secret: str):
        super().__init__()
        self.refresh_token = refresh_token
        self.token = access_token_data['access_token']
        self.client_id = client_id
        self.client_secret = client_secret
        self.token_uri = "https://oauth2.googleapis.com/token"
        
        # Parse expiry from stored data
        if access_token_data.get('expires_at'):
            expiry_str = access_token_data['expires_at']
            try:
                if expiry_str.endswith('Z'):
                    expiry_str = expiry_str.replace('Z', '+00:00')
                
                self.expiry = datetime.fromisoformat(expiry_str)
                
                if self.expiry.tzinfo is None:
                    self.expiry = self.expiry.replace(tzinfo=timezone.utc)
            except ValueError:
                self.expiry = None
        else:
            self.expiry = None
            
        self.scopes = access_token_data.get('scope', [])
    
    def refresh(self, request):
        """Refresh the access token using the refresh token."""
        import requests
        
        data = {
            'grant_type': 'refresh_token',
            'refresh_token': self.refresh_token,
            'client_id': self.client_id,
            'client_secret': self.client_secret,
        }
        
        response = requests.post(self.token_uri, data=data)
        response.raise_for_status()
        
        token_data = response.json()
        self.token = token_data['access_token']
        
        if 'expires_in' in token_data:
            self.expiry = datetime.now(timezone.utc) + timedelta(seconds=token_data['expires_in'])
    
    @property
    def expired(self):
        if not self.expiry:
            return False
        return datetime.now(timezone.utc) >= self.expiry

class JobPosterAgent(BaseAgent):
    def __init__(self):
        super().__init__("JobPoster", model_name="gemini-1.5-pro")

        api_key = os.getenv("SCRAPIN_API_KEY")
        if not api_key:
            raise ValueError("SCRAPIN_API_KEY is not set.")

        self.gmail_client_id = os.getenv("GMAIL_CLIENT_ID")
        self.gmail_client_secret = os.getenv("GMAIL_CLIENT_SECRET")

        if not self.gmail_client_id or not self.gmail_client_secret:
            raise ValueError("GMAIL_CLIENT_ID and GMAIL_CLIENT_SECRET must be set in the environment.")
        self.job_extractor = JobExtractor(api_key=api_key)

        self.table_id = None
        self.bq_client = self._get_bigquery_client()
        if self.bq_client:
            self.table_id = f"{self.bq_client.project}.{config.bq_dataset}.{config.bq_table}"
            try:
                ensure_table_exists(self.bq_client, self.table_id)
                self.logger.info("BigQuery client and table verified successfully.")
            except Exception as e:
                self.logger.error(f"Failed to verify or create BigQuery table '{self.table_id}': {e}", exc_info=True)
                self.bq_client = None
                self.table_id = None
        else:
            self.logger.warning("Could not initialize BigQuery client. Data will not be saved.")

        self._initialize_embedding_model()

    def _initialize_embedding_model(self):
        try:
            from google.cloud import aiplatform
            from vertexai.preview.language_models import TextEmbeddingModel
            aiplatform.init()
            self.embedding_model = TextEmbeddingModel.from_pretrained("text-embedding-005")
            self.logger.info("Vertex AI embedding model initialised.")
        except Exception as e:
            self.logger.warning(f"Failed to initialise embedding model: {e}. Embeddings will be skipped.")
            self.embedding_model = None

    def _get_bigquery_client(self):
        try:
            client = bigquery.Client()
            self.logger.info("BigQuery client created successfully")
            return client
        except Exception as e:
            self.logger.error(f"Failed to create BigQuery client: {e}")
            return None

    def _create_gmail_service(self, email: str):
        """Create Gmail service using credentials from Secret Manager."""
        self.logger.info("Creating Gmail service...")
        try:
            refresh_token, access_token_data = retrieve_oauth_tokens_from_secret_manager(email)
            
            credentials = SecretManagerCredentials(
                refresh_token=refresh_token,
                access_token_data=access_token_data,
                client_id=self.gmail_client_id,
                client_secret=self.gmail_client_secret
            )
            # Refresh token if expired
            if credentials.expired:
                self.logger.info("Access token expired, refreshing...")
                credentials.refresh(Request())
            
            # Build Gmail service
            service = build('gmail', 'v1', credentials=credentials)
            self.logger.info("Gmail service created successfully")
            return service
            
        except Exception as e:
            self.logger.error(f"Error creating Gmail service: {str(e)}")
            raise GmailError(f"Failed to create Gmail service: {str(e)}")

    def _save_email_as_draft(self, from_email: str, to_email: str, subject: str, body_html: str) -> str:
        """Save an email as draft in Gmail."""
        try:
            # Create Gmail service
            service = self._create_gmail_service(from_email)
            
            # Create the email message
            message = MIMEMultipart()
            message['To'] = to_email
            message['Subject'] = subject
            message.attach(MIMEText(body_html, 'html'))
            
            # Encode the message
            raw_message = base64.urlsafe_b64encode(message.as_bytes()).decode()
            
            # Create draft
            draft_body = {
                'message': {
                    'raw': raw_message
                }
            }
            
            draft = service.users().drafts().create(userId='me', body=draft_body).execute()
            self.logger.info(f"Draft created successfully with ID: {draft['id']}")
            
            return draft['id']
            
        except Exception as e:
            self.logger.error(f"Error creating email draft: {str(e)}")
            raise GmailError(f"Failed to create email draft: {str(e)}")

    def extract_job_link(self, email_data: EmailData) -> str:
        if email_data.metadata and "linkedin_links" in email_data.metadata:
            links = email_data.metadata["linkedin_links"]
            if links:
                return links[0]

        try:
            email_content = f"""
            Subject: {email_data.subject}
            Body: {email_data.body}
            """

            response = self.model.generate_content(EXTRACT_JOB_LINK_PROMPT + "\n\nEmail content:\n" + email_content)
            url = response.text.strip()

            # Validate LLM-returned URL first
            if url != 'NO_URL_FOUND' and (
                'linkedin.com/jobs/view/' in url or 'linkedin.com/comm/jobs/view/' in url
            ):
                return url

            # Fallback: regex parse from the raw email subject/body
            try:
                combined = f"{email_data.subject}\n{email_data.body}"
                pattern = r"https?://www\.linkedin\.com/(?:comm/)?jobs/view/[^\s>\)\]\"']+"
                match = re.search(pattern, combined)
                if match:
                    found = match.group(0)
                    return found
            except Exception:
                pass

            return None
        except Exception as e:
            self.logger.error(f"Error extracting job link: {str(e)}")
            return None

    def _extract_linkedin_job_id(self, url: str) -> str:
        if not url:
            return None
        pattern = r"/(?:comm/)?jobs/view/(?:[\w\-]*-)?(\d+)(?:[/?#]|$)"
        match = re.search(pattern, url)
        if match:
            job_id = match.group(1)
            self.logger.info(f"Extracted LinkedIn job ID: {job_id}")
            return job_id
        self.logger.warning(f"Could not extract job ID from URL: {url}")
        return None

    def _extract_internal_job_id(self, subject: str) -> str:
        """Extract an internal/vendor job ID from the email subject.

        Rules:
        - Prefer internal Job_Id like 'MHP-2529' or 'MHP2529' and normalize to 'MHP-2529'.
        - Return None if not found.
        """
        try:
            if not subject or not isinstance(subject, str):
                return None
            subj = subject.strip()
            subj = re.sub(r"[\[\]\(\)]", " ", subj)

            # Internal code, e.g., MHP-2528 or MHP2528 or MHP 2528 or MHP_2528
            m = re.search(r"\b([A-Za-z]{2,})[-_ ]?(\d{2,})\b", subj, re.IGNORECASE)
            if m:
                prefix = m.group(1)
                digits = m.group(2)
                norm = f"{prefix.upper()}-{digits}"
                self.logger.info(f"Extracted internal job ID from subject: {norm}")
                return norm

            # Labeled Job ID: "Job ID: ABC-123"
            m = re.search(r"Job[\s-]?ID[:\s]*([A-Za-z0-9\-_.]+)", subj, re.IGNORECASE)
            if m:
                return m.group(1).strip()

            return None
        except Exception:
            return None

    def scrape_job_details(self, url: str) -> Dict[str, Any]:
        try:
            self.logger.info(f"Starting job scraping for URL: {url}")
            
            scraped_data = self.job_extractor.extract(url)
            if not scraped_data:
                self.logger.error("Job extractor returned no data")
                return None
                
            self.logger.info("Job details scraped successfully")
            self.logger.debug(f"Scraped job title: {scraped_data.get('title', 'Unknown')}")
            return scraped_data
                
        except Exception as e:
            self.logger.error(f"Error scraping job details: {str(e)}")
            return None

    def _prepare_job_data(self, job_analysis: Dict[str, Any], job_url: str, email_data: EmailData) -> Dict[str, Any]:
        # Extract LinkedIn job id for the new field
        linkedin_job_id = self._extract_linkedin_job_id(job_url)
        if linkedin_job_id:
            job_analysis['linkedin_job_id'] = linkedin_job_id
        else:
            self.logger.warning("Could not parse LinkedIn job ID from URL.")

        # Derive internal job_id from subject (vendor token or labeled Job ID)
        internal_job_id = self._extract_internal_job_id(getattr(email_data, 'subject', None))
        if not internal_job_id:
            self.logger.info("No internal job ID found in subject; generating a UUID as job_id.")
            internal_job_id = str(uuid.uuid4())
        job_analysis['job_id'] = internal_job_id

        embeddings = self._generate_embeddings(structured_data=job_analysis)
        job_analysis['embeddings'] = embeddings or []

        return job_analysis

    def analyze_job_description(self, job_details: Dict[str, Any]) -> Dict[str, Any]:
        try:
            self.logger.info("Starting job description analysis with LLM")
            
            job_description = f"""
Title: {job_details.get('title', 'Not specified')}
Company: {job_details.get('companyName', 'Not specified')}
Location: {job_details.get('location', 'Not specified')}

Description:
{job_details.get('description', 'Not specified')}

Requirements:
{job_details.get('requirements', 'Not specified')}

Additional Information:
{job_details.get('additionalInfo', 'Not specified')}
"""
            analysis_prompt = JOB_ANALYSIS_PROMPT.format(job_description=job_description)

            response = self.model.generate_content(
                analysis_prompt,
                generation_config={
                    "temperature": 0.1,
                    "candidate_count": 1,
                    "top_p": 0.1,
                    "top_k": 1,
                }
            )
            
            response_text = response.text.strip()
            
            try:
                start_idx = response_text.find('{')
                end_idx = response_text.rfind('}') + 1
                
                if start_idx == -1 or end_idx == 0:
                    self.logger.error("No JSON object found in response")
                    self.logger.debug(f"Raw response: {response_text}")
                    return None
                
                json_str = response_text[start_idx:end_idx]
                
                try:
                    analysis = json.loads(json_str)
                    self.logger.info("Job description analysis completed successfully.")
                    self.logger.debug(f"Raw analysis response: {json.dumps(analysis, indent=2, default=str)}")

                    boolean_fields = [
                        "is_relocation_allowed",
                        "is_remote",
                        "is_only_ex_employee_required"
                    ]
                    for field in boolean_fields:
                        value = analysis.get(field)
                        self.logger.debug(f"Boolean field {field}: type={type(value)}, value={value}")

                    title = analysis.get("job_title", "N/A")
                    city = analysis.get("job_location_city", "N/A")
                    state = analysis.get("job_location_state", "N/A")
                    self.logger.info(f"Analyzed position: {title} at {city}, {state}")

                    skills = analysis.get("required_technical_skills", [])
                    if skills:
                        skills_str = ", ".join(skills)
                        self.logger.info(f"Required technical skills: {skills_str}")
                    
                    for field in boolean_fields:
                        if field in analysis:
                            if analysis[field] is None:
                                analysis[field] = None
                            elif isinstance(analysis[field], str):
                                analysis[field] = analysis[field].lower().strip() in ['true', 'yes', '1', 'on', 'y', 't']
                            elif isinstance(analysis[field], (int, float)):
                                analysis[field] = bool(analysis[field])
                            else:
                                analysis[field] = bool(analysis[field])

                    return analysis
                    
                except json.JSONDecodeError as je:
                    self.logger.error(f"JSON parsing error: {str(je)}")
                    self.logger.debug(f"Failed JSON string: {json_str}")
                    try:
                        fixed_json = json_str.replace('\\"', '"')
                        analysis = json.loads(fixed_json)
                        self.logger.info("Job description analysis completed successfully after fixes")
                        return analysis
                    except:
                        self.logger.error("Could not fix JSON parsing issues")
                        return None
                        
            except Exception as e:
                self.logger.error(f"Error processing response: {str(e)}")
                self.logger.debug(f"Response text: {response_text}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error in job analysis: {str(e)}")
            self.logger.debug(f"Job details: {job_details}")
            return None

    def _generate_embeddings(self, text=None, structured_data=None):
        if structured_data and not text:
            text_parts = []
            if structured_data.get("job_title") and structured_data["job_title"] != "Not Mentioned":
                text_parts.append(f"Job Title: {structured_data['job_title']}")

            location_parts = []
            if structured_data.get("job_location_city") and structured_data["job_location_city"] != "Not Mentioned":
                location_parts.append(structured_data["job_location_city"])
            if structured_data.get("job_location_state") and structured_data["job_location_state"] != "Not Mentioned":
                location_parts.append(structured_data["job_location_state"])
            if structured_data.get("job_location_country") and structured_data["job_location_country"] != "Not Mentioned":
                location_parts.append(structured_data["job_location_country"])
            if location_parts:
                text_parts.append(f"Location: {', '.join(location_parts)}")

            if structured_data.get("job_employment_type"):
                text_parts.append(f"Employment Type: {', '.join(structured_data['job_employment_type'])}")

            if structured_data.get("job_summary") and structured_data["job_summary"] != "Not Mentioned":
                text_parts.append(f"Job Summary: {structured_data['job_summary']}")

            if structured_data.get("job_description") and structured_data["job_description"] != "Not Mentioned":
                text_parts.append(f"Job Description: {structured_data['job_description']}")

            if structured_data.get("required_technical_skills"):
                text_parts.append(f"Required Skills: {', '.join(structured_data['required_technical_skills'])}")

            if structured_data.get("preferred_technical_skills"):
                text_parts.append(f"Preferred Skills: {', '.join(structured_data['preferred_technical_skills'])}")

            experience_parts = []
            if structured_data.get("min_required_years_experience"):
                experience_parts.append(f"Minimum {structured_data['min_required_years_experience']} years")
            if structured_data.get("max_preferred_years_experience"):
                experience_parts.append(f"up to {structured_data['max_preferred_years_experience']} years")
            if experience_parts:
                text_parts.append(f"Experience: {' '.join(experience_parts)}")

            if structured_data.get("job_certifications"):
                cert_items = []
                for cert in structured_data["job_certifications"]:
                    if isinstance(cert, dict):
                        cert_items.append(cert.get("name") or str(cert))
                    else:
                        cert_items.append(str(cert))
                if cert_items:
                    text_parts.append(f"Certifications: {', '.join(cert_items)}")

            text = "\n\n".join(text_parts)

        if not text:
            return None

        if not hasattr(self, 'embedding_model') or self.embedding_model is None:
            return None
        embeddings = self.embedding_model.get_embeddings([text])
        if embeddings and embeddings[0].values:
            return embeddings[0].values
        return None

    def _clean_data_for_bq(self, data: Dict[str, Any]) -> Dict[str, Any]:
        def to_bool(value):
            if value is None:
                return None
            if isinstance(value, bool):
                return value
            if isinstance(value, str):
                value = value.lower().strip()
                if value in ['null', 'none', 'n/a', 'not specified', '']:
                    return None
                return value in ['true', 'yes', '1', 'on', 'y', 't']
            if isinstance(value, (int, float)):
                return bool(value)
            return None

        boolean_fields = [
            "is_relocation_allowed",
            "is_remote",
            "is_only_ex_employee_required"
        ]

        for field in boolean_fields:
            if field in data:
                data[field] = to_bool(data[field])

        if 'questionnaire_details' in data:
            if isinstance(data['questionnaire_details'], dict):
                details = data['questionnaire_details'].copy()
                
                if 'has_questionnaire' in details:
                    details['has_questionnaire'] = to_bool(details['has_questionnaire'])
                if 'is_mandatory' in details:
                    details['is_mandatory'] = to_bool(details['is_mandatory'])
                
                try:
                    data['questionnaire_details'] = json.dumps(details)
                except (TypeError, ValueError):
                    data['questionnaire_details'] = None
            elif isinstance(data['questionnaire_details'], str):
                try:
                    parsed = json.loads(data['questionnaire_details'])
                    if isinstance(parsed, dict):
                        if 'has_questionnaire' in parsed:
                            parsed['has_questionnaire'] = to_bool(parsed['has_questionnaire'])
                        if 'is_mandatory' in parsed:
                            parsed['is_mandatory'] = to_bool(parsed['is_mandatory'])
                        data['questionnaire_details'] = json.dumps(parsed)
                    else:
                        data['questionnaire_details'] = data['questionnaire_details']
                except json.JSONDecodeError:
                    data['questionnaire_details'] = None
            else:
                data['questionnaire_details'] = None

        if 'job_certifications' in data and isinstance(data['job_certifications'], list):
            cleaned_certs = []
            for cert in data['job_certifications']:
                if isinstance(cert, dict):
                    cert_copy = cert.copy()
                    if 'is_mandatory' in cert_copy:
                        cert_copy['is_mandatory'] = to_bool(cert_copy.get('is_mandatory'))
                    cleaned_certs.append(cert_copy)
                elif isinstance(cert, str):
                    cleaned_certs.append({"name": cert, "is_mandatory": None})
                else:
                    cleaned_certs.append({"name": str(cert), "is_mandatory": None})
            
            try:
                data['job_certifications'] = json.dumps(cleaned_certs)
            except (TypeError, ValueError):
                data['job_certifications'] = None

        array_fields = [
            'required_technical_skills',
            'preferred_technical_skills',
            'job_location_city_state',
            'job_work_model',
            'job_employment_type',
            'preferred_previous_companies',
            'optional_requirements_list'
        ]
        for field in array_fields:
            if field in data:
                if not isinstance(data[field], list):
                    data[field] = []
                elif any(item is None for item in data[field]):
                    data[field] = [item for item in data[field] if item is not None]

        return data

    def save_job_details(self, job_analysis: Dict[str, Any], email_data: EmailData) -> str:
        if not self.bq_client or not self.table_id:
            self.logger.error("BigQuery client not initialized. Cannot save job details.")
            return None

        job_analysis = self._clean_data_for_bq(job_analysis)

        self.logger.debug(f"Cleaned data for BQ before saving: {json.dumps(job_analysis, indent=2, default=str)}")

        try:
            from app.tools.bq_schema_manager import JOB_DETAILS_SCHEMA
            schema_fields_map = {field.name: field for field in JOB_DETAILS_SCHEMA}

            filtered_analysis = {
                k: v for k, v in job_analysis.items()
                if k in schema_fields_map and schema_fields_map[k].field_type != 'RECORD'
            }

            current_timestamp = datetime.utcnow()
            if 'created_timestamp' not in filtered_analysis:
                filtered_analysis['created_timestamp'] = current_timestamp
            if 'last_updated_timestamp' not in filtered_analysis:
                filtered_analysis['last_updated_timestamp'] = current_timestamp
            if 'job_status' not in filtered_analysis:
                filtered_analysis['job_status'] = 'ACTIVE'
            if 'gcs_path' not in filtered_analysis:
                filtered_analysis['gcs_path'] = f"gs://job-details/{filtered_analysis.get('job_id', 'unknown')}"

            update_setters = ', '.join([f'T.`{key}` = @{key}' for key in filtered_analysis if key not in ['job_id', 'created_timestamp']])
            columns = ', '.join([f'`{k}`' for k in filtered_analysis.keys()])
            placeholders = ', '.join([f'@{key}' for key in filtered_analysis.keys()])

            merge_sql = f"""
            MERGE `{self.table_id}` T
            USING (SELECT @job_id AS job_id) S ON T.job_id = S.job_id
            WHEN MATCHED THEN
                UPDATE SET {update_setters}
            WHEN NOT MATCHED THEN
                INSERT ({columns}) VALUES ({placeholders})
            """

            query_params = []
            for key, value in filtered_analysis.items():
                field = schema_fields_map[key]
                param_type = field.field_type.upper()
                
                type_mapping = {
                    'STRING': 'STRING',
                    'FLOAT': 'FLOAT64',
                    'FLOAT64': 'FLOAT64',
                    'INTEGER': 'INT64',
                    'INT64': 'INT64',
                    'BOOL': 'BOOL',
                    'BOOLEAN': 'BOOL',
                    'DATE': 'DATE',
                    'TIMESTAMP': 'TIMESTAMP',
                    'DATETIME': 'DATETIME'
                }
                
                bq_param_type = type_mapping.get(param_type, 'STRING')
                
                if field.mode == 'REPEATED':
                    param_value = value if isinstance(value, list) else [value] if value is not None else []
                    query_params.append(bigquery.ArrayQueryParameter(key, bq_param_type, param_value))
                else:
                    if bq_param_type == 'BOOL':
                        if value is None:
                            value = None
                        elif isinstance(value, str):
                            value = value.lower().strip() in ['true', 'yes', '1', 'on', 'y', 't']
                        elif isinstance(value, (int, float)):
                            value = bool(value)
                        else:
                            value = bool(value)
                    elif bq_param_type == 'FLOAT64':
                        if value is not None:
                            try:
                                value = float(value)
                            except (ValueError, TypeError):
                                value = None
                    elif bq_param_type == 'INT64':
                        if value is not None:
                            try:
                                value = int(value)
                            except (ValueError, TypeError):
                                value = None
                    elif bq_param_type == 'TIMESTAMP':
                        if isinstance(value, str):
                            try:
                                value = datetime.fromisoformat(value.replace('Z', '+00:00'))
                            except ValueError:
                                value = current_timestamp
                        elif not isinstance(value, datetime):
                            value = current_timestamp
                    elif bq_param_type == 'DATE':
                        if isinstance(value, str):
                            try:
                                value = datetime.fromisoformat(value).date()
                            except ValueError:
                                value = None
                        elif isinstance(value, datetime):
                            value = value.date()
                    
                    query_params.append(bigquery.ScalarQueryParameter(key, bq_param_type, value))

            job_config = bigquery.QueryJobConfig(query_parameters=query_params)
            job_id = filtered_analysis.get('job_id')
            self.logger.info(f"Executing MERGE for job_id: {job_id}")
            
            self.logger.debug(f"Filtered analysis data before BigQuery insertion: {json.dumps(filtered_analysis, indent=2, default=str)}")
            
            param_info = []
            for param in query_params:
                param_dict = {'name': param.name}
                
                if isinstance(param, bigquery.ArrayQueryParameter):
                    param_dict['type'] = f"ARRAY<{param.array_type}>"
                    param_dict['value'] = str(param.values) if hasattr(param, 'values') else str(getattr(param, 'value', 'N/A'))
                elif isinstance(param, bigquery.ScalarQueryParameter):
                    param_dict['type'] = str(param.type_) if hasattr(param, 'type_') else 'UNKNOWN'
                    param_dict['value'] = str(param.value) if hasattr(param, 'value') else 'N/A'
                else:
                    param_dict['type'] = str(type(param))
                    param_dict['value'] = str(getattr(param, 'value', 'N/A'))
                
                if hasattr(param, 'value'):
                    param_dict['value_type'] = str(type(param.value))
                param_info.append(param_dict)
            
            self.logger.debug(f"Query parameters: {json.dumps(param_info, indent=2)}")
            
            try:
                query_job = self.bq_client.query(merge_sql, job_config=job_config)
                query_job.result()
            except Exception as e:
                self.logger.error(f"Error executing BigQuery query: {str(e)}")
                self.logger.debug(f"Query SQL: {merge_sql}")
                self.logger.debug(f"Query parameters: {json.dumps(param_info, indent=2)}")
                raise

            if query_job.errors:
                self.logger.error(f"BigQuery MERGE job failed: {query_job.errors}")
                return None

            self.logger.info(f"Successfully saved job details for job_id: {job_id}")
            return job_id

        except Exception as e:
            self.logger.error(f"Error in save_job_details: {e}", exc_info=True)
            return None

    def _get_job_details(self, email_data: EmailData) -> tuple[Dict[str, Any], str]:
        """Get job details from email and LinkedIn."""
        # Check if this is a recruiter reply
        if email_data.subject.lower().startswith('follow-up information request'):
            return self._process_recruiter_reply(email_data)

        job_url = self.extract_job_link(email_data)
        if not job_url:
            return None, None

        job_id = self._extract_linkedin_job_id(job_url)
        if not job_id:
            return None, None

        job_details = self.scrape_job_details(job_url)
        if not job_details:
            return None, None

        job_analysis = self.analyze_job_description(job_details)
        return job_analysis, job_url

    def _process_recruiter_reply(self, email_data: EmailData) -> tuple[Dict[str, Any], str]:
        """Process recruiter reply email containing additional job details."""
        try:
            # Extract the replied details from the email
            email_body = email_data.body
            replied_details_start = email_body.find("Recruiter's Reply:")
            if replied_details_start == -1:
                return None, None

            replied_details = json.loads(email_body[replied_details_start:])
            
            # Extract job information from replied details
            job_analysis = {
                "job_id": str(uuid.uuid4()),  # Generate a new job ID
                "recruiter_email": email_data.sender,
                "prime_vendor_name": replied_details["vendor_info"]["prime_vendor_name"],
                "prime_vendor_email": replied_details["vendor_info"]["prime_vendor_email"],
                "end_client_name": replied_details["vendor_info"]["end_client_name"],
                "is_relocation_allowed": replied_details["basic_job_info"]["is_relocation_allowed"],
                "is_remote": replied_details["basic_job_info"]["is_remote"],
                "job_location_city": "Not specified",
                "job_location_state": "Not specified",
                "job_location_country": "Not specified",
                "created_timestamp": datetime.utcnow().isoformat(),
                "last_updated_timestamp": datetime.utcnow().isoformat()
            }

            return job_analysis, None  # No LinkedIn URL since this is a reply
        except Exception as e:
            self.logger.error(f"Error processing recruiter reply: {e}", exc_info=True)
            return None, None

    def run(self, email_input: Union[str, EmailData], **kwargs) -> Dict[str, Any]:
        try:
            if isinstance(email_input, str):
                email_data = EmailData.from_json(email_input)
            elif isinstance(email_input, dict):
                email_data = EmailData(**email_input)
            else:
                email_data = email_input

            job_analysis, job_url = self._get_job_details(email_data)
            if not job_url:
                return {'status': 'error', 'message': 'No LinkedIn job posting URL found'}
            if not job_analysis:
                return {'status': 'error', 'message': f'Failed to get job details from {job_url}'}

            company_details = self._get_company_details(email_data.recipient)
            self.logger.info(f"Company details lookup result: {json.dumps(company_details, indent=2)}")
            
            if company_details:
                job_analysis.update({
                    'customer_type': company_details.get('customer_type'),
                    'prime_vendor_name': company_details.get('prime_vendor_name'),
                    'prime_vendor_email': company_details.get('prime_vendor_email'),
                    'sub_vendor_name': company_details.get('sub_vendor_name'),
                    'sub_vendor_email': company_details.get('sub_vendor_email')
                })
                self.logger.info(f"Updated job analysis with company type: {company_details.get('customer_type')}")
            else:
                self.logger.warning(f"No company details found for recipient: {email_data.recipient}")

            prepared_data = self._prepare_job_data(job_analysis, job_url, email_data)

            saved_job_id = self.save_job_details(prepared_data, email_data)
            if not saved_job_id:
                return {'status': 'error', 'message': 'Failed to save job details to BigQuery'}

            missing_info = self._check_essential_information(prepared_data)
            if not any(missing_info.values()):
                return {
                    'status': 'success',
                    'message': 'Job details processed and saved successfully. No follow-up needed.',
                    'job_analysis': prepared_data
                }

            try:
                email_generation_result = self._generate_recruiter_email(prepared_data, missing_info)
                if not email_generation_result or not isinstance(email_generation_result, tuple):
                    raise GmailError("Failed to generate email content.")

                subject, body_html, _ = email_generation_result # Original to_email is ignored

                from_email = "support@myhiringpartner.ai"
                to_email = email_data.sender

                if not to_email:
                    raise GmailError("Original sender's email is not available, cannot send follow-up.")
                draft_id = self._save_email_as_draft(from_email, to_email, subject, body_html)
                self.logger.info(f"Successfully saved draft with ID: {draft_id}")

                return {
                    'status': 'success',
                    'message': f'Job details saved. Follow-up email drafted with ID: {draft_id}',
                    'job_analysis': prepared_data,
                    'draft_id': draft_id
                }
            except (GmailError, SecretManagerError) as e:
                self.logger.error(f"Failed to create or save draft: {e}", exc_info=True)
                return {'status': 'error', 'message': f'Failed to handle draft: {e}'}

        except Exception as e:
            self.logger.error(f"Error processing job posting: {e}", exc_info=True)
            return {'status': 'error', 'message': str(e)}

    def _check_essential_information(self, job_details: Dict[str, Any]) -> Dict[str, list[str]]:
        """Check for essential missing information and categorize them."""
        missing_info = {
            'basic_info': [],
            'vendor_info': []
        }

        # Check basic job information
        basic_fields = [
            'job_location_city', 'job_location_state', 'job_location_country',
            'is_relocation_allowed', 'is_remote'
        ]
        # If a combined 'location' string is present, do not request city/state/country separately
        combined_location = (job_details.get('location') or '').strip()

        for field in basic_fields:
            value = job_details.get(field)

            # Treat explicit booleans (True or False) as present; only None or "not specified" is missing
            if field in ('is_relocation_allowed', 'is_remote'):
                if value is None or (isinstance(value, str) and value.lower().strip() == 'not specified'):
                    missing_info['basic_info'].append(field)
                continue

            # For location components, consider them satisfied if a combined 'location' exists
            if field in ('job_location_city', 'job_location_state', 'job_location_country') and combined_location:
                continue

            is_missing_none = value is None
            is_missing_str = isinstance(value, str) and value.lower().strip() == 'not specified'

            if is_missing_none or is_missing_str:
                missing_info['basic_info'].append(field)

        # Check vendor information based on customer type
        customer_type = (job_details.get('customer_type') or 'not specified').lower()
        # Helper to check missing string fields safely
        def _is_missing_str_field(val: Any) -> bool:
            return val is None or (isinstance(val, str) and val.lower().strip() == 'not specified') or (isinstance(val, str) and val.strip() == '')

        if customer_type == 'prime_vendor':
            if _is_missing_str_field(job_details.get('end_client_name')):
                missing_info['vendor_info'].append('end_client_name')
            # Also request end-client email if not present
            if _is_missing_str_field(job_details.get('end_client_email')):
                missing_info['vendor_info'].append('end_client_email')
        elif customer_type == 'sub_vendor':
            if _is_missing_str_field(job_details.get('end_client_name')):
                missing_info['vendor_info'].append('end_client_name')
            if _is_missing_str_field(job_details.get('end_client_email')):
                missing_info['vendor_info'].append('end_client_email')
            if _is_missing_str_field(job_details.get('prime_vendor_name')):
                missing_info['vendor_info'].append('prime_vendor_name')
            if _is_missing_str_field(job_details.get('prime_vendor_email')):
                missing_info['vendor_info'].append('prime_vendor_email')

        return missing_info

    def _generate_recruiter_email(self, job_details: Dict[str, Any], missing_info: Dict[str, list]) -> tuple:
        """Generate a targeted email to request specific missing information from the recruiter."""
        try:
            if not any(missing_info.values()):
                return None

            job_title = job_details.get('job_title', 'Unknown Position')
            job_id = job_details.get('job_id', 'N/A')

            # Construct the path to the template file
            template_path = os.path.join(os.path.dirname(__file__), '..', 'templates', 'follow_up_email.html')

            with open(template_path, 'r') as f:
                template_str = f.read()

            # Generate HTML rows only for truly missing fields,
            # grouping location components to keep the email concise.
            response_cell_style = 'border: 1px solid #ccc; background-color: #f9f9f9; height: 25px;'

            table_rows = []

            basic_missing = list(missing_info.get('basic_info', []) or [])
            vendor_missing = list(missing_info.get('vendor_info', []) or [])

            # Group location pieces into one concise row if any are missing
            location_keys = ['job_location_city', 'job_location_state', 'job_location_country']
            missing_location_parts = [
                ('City', 'job_location_city') if 'job_location_city' in basic_missing else None,
                ('State', 'job_location_state') if 'job_location_state' in basic_missing else None,
                ('Country', 'job_location_country') if 'job_location_country' in basic_missing else None,
            ]
            missing_location_parts = [p for p in missing_location_parts if p]

            if missing_location_parts:
                # Remove individual keys so we don't duplicate them below
                for _, key in missing_location_parts:
                    if key in basic_missing:
                        basic_missing.remove(key)
                parts_label = ", ".join([p[0] for p in missing_location_parts])
                table_rows.append(
                    f'<tr><td style="padding-right: 15px;">Location ({parts_label})</td>'
                    f'<td style="{response_cell_style}"></td></tr>'
                )

            # Map for clearer labels
            field_labels = {
                'is_remote': 'Remote (Yes/No)',
                'is_relocation_allowed': 'Relocation Allowed (Yes/No)',
                'end_client_name': 'End Client Name',
                'end_client_email': 'End Client Email',
                'prime_vendor_name': 'Prime Vendor Name',
                'prime_vendor_email': 'Prime Vendor Email',
            }

            # Add remaining basic missing fields (excluding location already handled)
            for field in basic_missing:
                label = field_labels.get(field, field.replace('_', ' ').title())
                table_rows.append(
                    f'<tr><td style="padding-right: 15px;">{label}</td>'
                    f'<td style="{response_cell_style}"></td></tr>'
                )

            # Add vendor-related missing fields
            for field in vendor_missing:
                label = field_labels.get(field, field.replace('_', ' ').title())
                table_rows.append(
                    f'<tr><td style="padding-right: 15px;">{label}</td>'
                    f'<td style="{response_cell_style}"></td></tr>'
                )

            missing_fields_table = "\n".join(table_rows)

            # Populate the template using str.replace to avoid issues with CSS braces
            email_body = template_str.replace('{job_title}', job_title)
            email_body = email_body.replace('{job_id}', job_id)
            email_body = email_body.replace('{missing_fields_table}', missing_fields_table)

            # The subject is part of the template's title, but we can return it separately if needed.
            # For now, let's create a full email string with subject.
            subject = f"Quick follow-up: {job_title}"
            
            # Mimic an email format with headers and body
            to_email = job_details.get('recruiter_email')
            if not to_email:
                self.logger.error("Recruiter email not found, cannot generate email.")
                return None

            return subject, email_body, to_email

        except Exception as e:
            self.logger.error(f"Error generating recruiter email: {e}", exc_info=True)
            return f"Error generating email: {str(e)}"

    def _get_company_details(self, email: str) -> dict:
        """Look up company details for an email address."""
        try:
            self.logger.info(f"Looking up company details for email: {email}")
            
            email = email.lower()
            
            all_companies = db.collection('companies').get()
            for doc in all_companies:
                data = doc.to_dict()

            all_recruiters = db.collection('recruiters').get()
            for doc in all_recruiters:
                data = doc.to_dict()
            
            self.logger.info(f"Executing Firestore query for email: {email}")
            recruiter_query = (db.collection('recruiters')
                              .where(filter=FieldFilter('email', '==', email))
                              .limit(1)
                              .get())
            
            recruiter_docs = list(recruiter_query)
            
            if not recruiter_docs:
                return {}
            
            recruiter_data = recruiter_docs[0].to_dict()
            self.logger.info(f"Found recruiter data: {json.dumps(recruiter_data, indent=2, default=str)}")
            
            company_id = recruiter_data.get('companyId')
            self.logger.info(f"Looking up company with ID: {company_id}")
            
            if not company_id:
                return {}
                
            company_doc = db.collection('companies').document(company_id).get()
            
            if not company_doc.exists:
                return {}
                
            company_data = company_doc.to_dict()
            
            company_type = company_data.get('companyType', '').lower()
            company_name = company_data.get('companyName')
            
            result = {
                'customer_type': company_type,
                'company_name': company_name,
                'company_id': company_id,
                'email': email
            }
            
            if company_type == 'prime_vendor':
                self.logger.info("Processing as prime vendor")
                result.update({
                    'prime_vendor_name': company_name,
                    'prime_vendor_email': email,
                    'sub_vendor_name': None,
                    'sub_vendor_email': None
                })
            elif company_type == 'sub_vendor':
                self.logger.info("Processing as sub vendor")
                result.update({
                    'sub_vendor_name': company_name,
                    'sub_vendor_email': email,
                    'prime_vendor_name': None,
                    'prime_vendor_email': None
                })
            elif company_type == 'end_client':
                self.logger.info("Processing as end client")
                result.update({
                    'end_client_name': company_name,
                    'end_client_email': email,
                    'prime_vendor_name': None,
                    'prime_vendor_email': None,
                    'sub_vendor_name': None,
                    'sub_vendor_email': None
                })
            else:
                self.logger.warning(f"Unknown company type: {company_type}")
            
            self.logger.info(f"Final company details: {json.dumps(result, indent=2, default=str)}")
            return result
            
        except Exception as e:
            self.logger.error(f"Error looking up company details for {email}: {e}", exc_info=True)
            return {}
