import os
import json
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List
from dataclasses import dataclass
from enum import Enum
from datetime import datetime
import google.generativeai as genai

genai.configure(api_key=os.getenv('GEMINI_API_KEY', ''))

EXTRACT_LINKS_PROMPT = """Extract any LinkedIn job posting URLs from this email content.
Accept URLs that contain either '/jobs/view/' or '/comm/jobs/view/'. URLs may include query parameters or tracking IDs.
Return only the URLs, one per line. If no URLs found, return 'NO_URLS_FOUND'."""


CLASSIFICATION_PROMPT = """Analyze this email and determine its category.
Look for key indicators:
- Job descriptions, requirements (job_posting)
- Personal qualifications, work history (resume_application)
- A reply providing information requested for a job (recruiter_jd_info_replied)
- A reply from a candidate providing personal info like LinkedIn, DOB, or education (candidate_info_replied)

Respond with one term: 'job_posting', 'resume_application', 'recruiter_jd_info_replied', 'candidate_info_replied'"""


@dataclass
class EmailData:
    """Structured representation of email data"""
    subject: str
    body: str
    sender: str
    recipient: str
    timestamp: str
    attachments: List[Dict[str, Any]] = None
    metadata: Dict[str, Any] = None

    def __init__(self, **kwargs):
        import dataclasses
        names = {f.name for f in dataclasses.fields(self)}
        for k, v in kwargs.items():
            if k in names:
                setattr(self, k, v)

    @classmethod
    def from_json(cls, json_str: str) -> 'EmailData':
        """Create an EmailData instance from a JSON string"""
        try:
            data = json.loads(json_str)
            return cls(
                subject=data.get('subject', ''),
                body=data.get('body', ''),
                sender=data.get('sender', ''),
                recipient=data.get('recipient', ''),
                timestamp=data.get('timestamp', ''),
                attachments=data.get('attachments'),
                metadata=data.get('metadata')
            )
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format: {str(e)}")
        except KeyError as e:
            raise ValueError(f"Missing required field: {str(e)}")
            
    def to_json(self) -> str:
        """Convert the EmailData instance to a JSON string"""
        return json.dumps({
            'subject': self.subject,
            'body': self.body,
            'sender': self.sender,
            'recipient': self.recipient,
            'timestamp': self.timestamp,
            'attachments': self.attachments,
            'metadata': self.metadata
        })

class EmailType(Enum):
    JOB_POSTING = "job_posting"
    RESUME_APPLICATION = "resume_application"
    RECRUITER_JD_INFO_REPLIED = "recruiter_jd_info_replied"
    CANDIDATE_MOVE_FORWAD_REPLY = "candidate_move_forwad_reply"
    JOB_CLOSING = "job_closing"
    UNKNOWN = "unknown"

class BaseAgent(ABC):
    """Base class for all agents in the system"""
    
    def __init__(self, name: str, model_name: str = "gemini-1.5-pro"):
        self.name = name
        self.model_name = model_name
        self.model = genai.GenerativeModel(model_name)
        self.logger = logging.getLogger(f"Agent.{name}")

    
    @abstractmethod
    def run(self, input_data: Any, **kwargs) -> Dict[str, Any]:
        """Execute the agent's primary function"""
        pass


class MainAgent(BaseAgent):
    """Main agent responsible for routing emails to appropriate specialized agents"""

    def __init__(self):
        super().__init__("MainAgent", model_name="gemini-1.5-pro")
        self.classification_prompt = CLASSIFICATION_PROMPT
    
    def classify_email(self, email_data: EmailData) -> EmailType:
        """Classify the type of email"""
        try:
            email_content = f"""
            Subject: {email_data.subject}
            Body: {email_data.body}
            """
            
            response = self.model.generate_content(
                self.classification_prompt + "\n\nEmail to classify:\n" + email_content
            )
            
            classification = response.text.strip().lower()
            
            if classification == "job_posting":
                return EmailType.JOB_POSTING
            elif classification == "resume_application":
                return EmailType.RESUME_APPLICATION
            else:
                return EmailType.UNKNOWN
                
        except Exception as e:
            self.logger.error(f"Error classifying email: {str(e)}")
            return EmailType.UNKNOWN

    def extract_links(self, email_data: EmailData) -> List[str]:
        """Extract LinkedIn job links from the email content"""
        try:
            email_content = f"""
            Subject: {email_data.subject}
            Body: {email_data.body}
            """

            response = self.model.generate_content(EXTRACT_LINKS_PROMPT + "\n\nEmail content:\n" + email_content)
            urls = response.text.strip().split('\n')
            
            return [
                url.strip()
                for url in urls
                if url.strip()
                and url.strip() != 'NO_URLS_FOUND'
                and (
                    'linkedin.com/jobs/view/' in url.strip()
                    or 'linkedin.com/comm/jobs/view/' in url.strip()
                )
            ]
            
        except Exception as e:
            self.logger.error(f"Error extracting links: {str(e)}")
            return []

    def run(self, email_data: EmailData, **kwargs) -> Dict[str, Any]:
        """Process the email and route to appropriate handler"""
        try:
            email_type = self.classify_email(email_data)
            
            linkedin_links = self.extract_links(email_data)
            
            from .agents.job_poster_agent import JobPosterAgent
            from .agents.resume_poster_agent import ResumeProcessorAgent
            
            if email_type == EmailType.JOB_POSTING:
                handler = JobPosterAgent()
                email_data.metadata = {"linkedin_links": linkedin_links}
                result = handler.run(email_data)
            elif email_type == EmailType.RESUME_APPLICATION:
                handler = ResumeProcessorAgent()
                result = handler.run(email_data)
            else:
                return {
                    'status': 'error',
                    'message': 'Unable to classify email type',
                    'email_type': 'unknown'
                }
            
            return {
                'status': 'success',
                'email_type': email_type.value,
                'linkedin_links': linkedin_links,
                'result': result
            }
            
        except Exception as e:
            self.logger.error(f"Error processing email: {str(e)}")
            return {
                'status': 'error',
                'message': str(e)
            }

root_agent = MainAgent()