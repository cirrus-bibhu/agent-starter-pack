from datetime import time
import time
import logging
import re
from typing import Dict, Any, Literal, Optional
from google.generativeai import protos
from pydantic import BaseModel, Field
import json
from app.agent import BaseAgent, EmailData, EmailType
from app.agents.candidate_engager_agent import CandidateEngagerAgent
from app.agents.job_poster_agent import JobPosterAgent
from app.agents.resume_poster_agent import ResumeProcessorAgent
from app.agents.matching_service import MatchingService
from app.agents.recruiter_engager_agent import RecruiterEngagerAgent
from app.agents.ex_consultant_agent import ExConsultantAgent
from app.agents.verification_manager import VerificationManagerAgent

ROUTING_PROMPT_TEMPLATE = """Analyze the following email and call the appropriate function to process it.

Email Content:
Subject: {subject}
From: {sender}
Body: {body}
Attachments: {attachments}

You have two possible actions:
1. Call route_to_job_poster(email_content, confidence) - For job postings, hiring announcements, or position descriptions
2. Call route_to_resume_processor(email_content, confidence) - For resumes, job applications, or candidate submissions

Key indicators for job postings:
- Job IDs or reference numbers in subject
- Links to job descriptions (especially LinkedIn)
- HR department as sender
- Words like \"position\", \"opening\", 'hire', \"job posting\"

Key indicators for resumes:
- Resume attachments (.pdf, .doc, .docx)
- Candidate introductions
- Application-related language
- Words like \"apply\", \"application\", \"resume\", \"CV\"

You must respond by calling one of these functions with:
1. email_content: The full email content provided
2. confidence: A score between 0.0-1.0 indicating your confidence

DO NOT provide any explanation, ONLY call the appropriate function."""

EMAIL_CLASSIFICATION_PROMPT = '''You are an expert email classifier for a recruitment system. Your task is to analyze incoming emails and classify them into appropriate categories.

CATEGORIES:
1. "job_posting" - Emails containing:
   - New job descriptions or position announcements
   - Job requirements and qualifications
   - Company hiring announcements
   - LinkedIn job postings or career opportunities
   - Requests to post or advertise positions

2. "resume_application" - Emails containing:
   - Job applications or resumes
   - Cover letters or introduction letters
   - Candidate qualifications and experience
   - Interest in specific job positions
   - Follow-up on submitted applications

3. "recruiter_jd_info_replied" - Emails that are replies from recruiters providing additional information about a job posting that was previously requested by the system. These emails are in response to a specific inquiry from MyHiringPartner.ai and typically contain answers to questions about the job (like relocation, location, vendor information, etc.).

4. "candidate_move_forwad_reply" - Emails from candidates providing additional information like education, date of birth, and LinkedIn profile for verification purposes.

4. "unknown" - Use only if the email clearly doesn't fit the above categories

ANALYSIS INSTRUCTIONS:
- Carefully examine both subject line and email body
- Consider the sender's intent and primary purpose
- Look for key indicators like job descriptions vs personal qualifications
- Check for presence of attachments and their types
- Analyze any links or references (especially LinkedIn job links)

EMAIL TO ANALYZE:
Subject: {{subject}}
Body: {{body}}
Sender: {{sender}}
Attachments: {{num_attachments}} files

Your response must be a JSON object matching the EmailClassification schema with:
1. category: Exactly one of ["job_posting", "resume_application", "recruiter_jd_info_replied", "candidate_move_forwad_reply", "unknown"]
2. confidence: A float between 0-1 indicating classification confidence
3. reasoning: A clear explanation of why this classification was chosen

RESPONSE FORMAT:
{
    "category": "category_name",
    "confidence": 0.95,
    "reasoning": "Clear explanation of classification..."
}'''

RECRUITER_EMAIL_PROMPT = '''You are an expert recruitment assistant. Generate a professional and friendly email to request missing information about a job posting.

JOB DETAILS:
{job_details}

MISSING INFORMATION:
{missing_fields}

Based on the available job details and missing information, generate an email that:
1. Thanks the recruiter for the job posting
2. Mentions the job title and end client name
3. Only asks for information that is actually missing (check job_details first)
4. If the recruiter is a prime vendor, ask for the end client's information.
5. If the recruiter is a sub-vendor, ask for the end client and prime vendor's information.
6. Be concise and professional

EMAIL TEMPLATE:
Subject: Regarding Job Posting: {job_title}

Hi [Recruiter Name],

Thank you for sharing the job opportunity for the {job_title} position at {end_client_name}.

To ensure we find the best candidates, could you please provide the following details:

{missing_fields_list}

[VENDOR_SECTION - Will be replaced based on customer_type and available information]

Thank you for your help!

Best regards,
MyHiringPartner.ai'''


class EmailClassification(BaseModel):
    category: Literal["job_posting", "resume_application", "recruiter_jd_info_replied", "candidate_move_forwad_reply", "unknown"] = Field(
        description="The classified category of the email"
    )
    confidence: float = Field(
        description="Confidence score for the classification (0-1)",
        ge=0.0,
        le=1.0
    )
    reasoning: str = Field(
        description="Detailed explanation of why this classification was chosen"
    )

class CoordinatorAgent(BaseAgent):
    """
    Main coordinator agent that routes incoming email to appropriate sub-agents
    based on email content analysis using Gemini.
    """
    
    def __init__(self):
        tools = [
            protos.Tool(
                function_declarations=[
                    protos.FunctionDeclaration(
                        name="route_to_job_poster",
                        description="Route email to job poster agent for processing job postings",
                        parameters=protos.Schema(
                            type=protos.Type.OBJECT,
                            properties={
                                "email_content": protos.Schema(
                                    type=protos.Type.STRING,
                                    description="The email content to be processed"
                                ),
                                "confidence": protos.Schema(
                                    type=protos.Type.NUMBER,
                                    description="Confidence level that this is a job posting (0-1)"
                                )
                            },
                            required=["email_content"]
                        )
                    ),
                    protos.FunctionDeclaration(
                        name="route_to_resume_processor",
                        description="Route email to resume processor agent for processing resume applications",
                        parameters=protos.Schema(
                            type=protos.Type.OBJECT,
                            properties={
                                "email_content": protos.Schema(
                                    type=protos.Type.STRING,
                                    description="The email content to be processed"
                                ),
                                "confidence": protos.Schema(
                                    type=protos.Type.NUMBER,
                                    description="Confidence level that this is a resume application (0-1)"
                                )
                            },
                            required=["email_content"]
                        )
                    )
                ]
            )
        ]
        
        super().__init__(name="CoordinatorAgent")
        self.tools = tools
        self.logger.info("Initializing JobPosterAgent...")
        self.job_poster_agent = JobPosterAgent()
        self.logger.info("Initializing ResumeProcessorAgent...")
        self.resume_processor_agent = ResumeProcessorAgent()
        self.logger.info("Initializing RecruiterEngagerAgent...")
        self.recruiter_engager_agent = RecruiterEngagerAgent()
        self.candidate_engager_agent = CandidateEngagerAgent()
        self.matching_service = MatchingService()
        self.ex_consultant_agent = ExConsultantAgent()
        self.verification_manager_agent = VerificationManagerAgent()
        
        self.logger = logging.getLogger("CoordinatorAgent")
        self.logger.setLevel(logging.DEBUG)

    def _get_essential_missing_fields(self, job_analysis: Dict[str, Any]) -> Dict[str, list[str]]:
        """Get essential missing fields for job posting."""
        missing_fields = {
            'basic_info': [],
            'vendor_info': []
        }

        # Check basic job information
        basic_fields = [
            'job_location_city', 'job_location_state', 'job_location_country',
            'is_relocation_allowed', 'is_remote'
        ]
        for field in basic_fields:
            value = job_analysis.get(field)
            if isinstance(value, str) and value.lower() == "not specified":
                missing_fields['basic_info'].append(field)
            elif isinstance(value, bool) and not value:
                missing_fields['basic_info'].append(field)

        # Check vendor information based on customer type
        customer_type = job_analysis.get('customer_type', 'not specified')
        if customer_type:
            customer_type = str(customer_type).lower()
            if customer_type == 'prime_vendor':
                end_client_name = job_analysis.get('end_client_name')
                if end_client_name and isinstance(end_client_name, str) and end_client_name.lower() == 'not specified':
                    missing_fields['vendor_info'].append('end_client_name')
            elif customer_type == 'sub_vendor':
                end_client_name = job_analysis.get('end_client_name')
                prime_vendor_name = job_analysis.get('prime_vendor_name')
                if end_client_name and isinstance(end_client_name, str) and end_client_name.lower() == 'not specified':
                    missing_fields['vendor_info'].append('end_client_name')
                if prime_vendor_name and isinstance(prime_vendor_name, str) and prime_vendor_name.lower() == 'not specified':
                    missing_fields['vendor_info'].append('prime_vendor_name')

        self.logger.info(f"Identified essential missing fields: {missing_fields}")
        return missing_fields
    
    def _classify_email_type(self, email_data: EmailData) -> EmailType:
        try:
            prompt = EMAIL_CLASSIFICATION_PROMPT.replace(
                "{{subject}}", email_data.subject
            ).replace(
                "{{body}}", email_data.body
            ).replace(
                "{{sender}}", email_data.sender
            ).replace(
                "{{num_attachments}}", str(len(email_data.attachments) if email_data.attachments is not None else 0)
            )
            
            response = self._generate_content(prompt)
            
            if response and hasattr(response, 'text') and response.text.strip():
                response_text = response.text.strip()
                self.logger.debug(f"Raw classification response: {response_text}")
                
                if response_text.startswith('```json'):
                    response_text = response_text[7:-3].strip()
                
                classification_data = json.loads(response_text)
                classification = EmailClassification(**classification_data)
                
                self.logger.info(f"Email classified as: {classification.category} with confidence: {classification.confidence}")
                self.logger.debug(f"Classification reasoning: {classification.reasoning}")
                
                return EmailType(classification.category)
            
            else:
                self.logger.warning("Received empty or invalid classification response")
                return EmailType.UNKNOWN
                
        except Exception as e:
            self.logger.error(f"Error classifying email: {str(e)}")
            return EmailType.UNKNOWN
    
    def _generate_content(self, prompt: str) -> Any:
        try:
            self.logger.debug(f"Generating content with prompt length: {len(prompt)}")
            response = self.model.generate_content(prompt)
            self.logger.debug("Content generation successful")
            return response
        except Exception as e:
            self.logger.error(f"Error generating content: {str(e)}")
            raise
    
    def route_to_job_poster(self, email_content: str, confidence: float = 0.8) -> Dict[str, Any]:
        try:
            self.logger.info(f"Routing to job poster agent with confidence: {confidence}")
            start_time = time.time()
            
            result = self.job_poster_agent.run(email_content)
            duration = time.time() - start_time
            self.logger.debug(f"Job poster agent completed in {duration:.2f}s")
            
            if result.get('status') == 'success':
                job_analysis = result.get('job_analysis', {})
                
                # Check for essential missing information
                missing_fields = self._get_essential_missing_fields(job_analysis)
                
                if any(missing_fields.values()):
                    self.logger.info(f"Essential missing fields identified: {missing_fields}")
                    
                    # Use the email generated by the job poster agent
                    result['recruiter_email'] = result.get('recruiter_email', "No essential information is missing.")
                
                return {
                    "status": "success",
                    "agent": "job_poster",
                    "confidence": confidence,
                    "result": result,
                    "analysis_summary": {
                        "job_title": job_analysis.get('job_title', 'N/A'),
                        "location": f"{job_analysis.get('job_location_city', 'N/A')}, {job_analysis.get('job_location_state', 'N/A')}",
                        "core_skills_count": len(job_analysis.get('required_technical_skills', [])),
                        "has_questionnaire": bool(job_analysis.get('questionnaire_details')),
                        "processing_time": f"{duration:.2f}s"
                    }
                }
            else:
                self.logger.error(f"Job posting processing failed: {result.get('message')}")
            
            return {
                "status": result.get('status', 'error'),
                "agent": "job_poster",
                "confidence": confidence,
                "result": result
            }
            
        except Exception as e:
            self.logger.error(f"Error routing to job poster: {str(e)}", exc_info=True)
            return {
                "status": "error", 
                "agent": "job_poster",
                "message": str(e)
            }

    def _format_response(self, agent_name: str, result: Dict[str, Any], confidence: float) -> Dict[str, Any]:
        if result.get("agent") == "verification_manager":
            return {
                "status": result.get("status", "error"),
                "agent": result.get("agent"),
                "confidence": confidence,
                "result": result
            }
        
        return {
            "status": result.get("status", "error"),
            "agent": agent_name,
            "confidence": confidence,
            "result": result
        }
    
    def route_to_verification_manager(self, email_data: EmailData) -> Dict[str, Any]:
        try:
            # Extract candidate ID from subject
            candidate_id_match = re.search(r'\(ID: ([a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12})\)', email_data.subject)
            if not candidate_id_match:
                self.logger.error("Could not extract candidate_id from email subject.")
                return {"status": "error", "message": "Missing candidate_id in subject"}
            
            candidate_id = candidate_id_match.group(1)
            
            # Pass the entire email data to verification manager
            result = self.verification_manager_agent.run({
                "candidate_id": candidate_id,
                "email_data": email_data
            })
            
            return {"status": "success", "agent": "verification_manager", "result": result}

        except Exception as e:
            self.logger.error(f"Error routing to verification manager: {str(e)}", exc_info=True)
            return {
                "status": "error", 
                "agent": "verification_manager",
                "message": str(e)
            }

    def _extract_job_id_from_subject(self, subject: str) -> Optional[str]:
        """Extract job ID from email subject."""
        # Look for patterns like "Position 12345" or "JobID: 12345" etc.
        import re
        patterns = [
            r'Position\s+(\d+)',  # Matches "Position 12345"
            r'Job[\s-]?ID[\s:]+(\d+)',  # Matches "Job ID: 12345" or "Job-ID 12345"
            r'\b(\d{4,})\b'  # Matches any 4+ digit number
        ]
        
        for pattern in patterns:
            match = re.search(pattern, subject)
            if match:
                return match.group(1)
        return None

    def route_to_resume_processor(self, email_content: str, confidence: float = 0.8) -> Dict[str, Any]:
        try:
            self.logger.info(f"Routing to resume processor agent with confidence: {confidence}")
            start_time = time.time()
            
            email_data = EmailData.from_json(email_content)
            result = self.resume_processor_agent.run(email_data)
            duration = time.time() - start_time
            self.logger.debug(f"Resume processor agent completed in {duration:.2f}s")
            
            if result.get('status') == 'success':
                analysis = result.get('analysis', {})
                self.logger.info(f"Successfully processed resume for: {analysis.get('candidate_name', 'N/A')}")
                
                # Prepare the response
                response = {
                    "status": "success",
                    "agent": "resume_processor",
                    "confidence": confidence,
                    "result": result,
                    "analysis_summary": {
                        "candidate_name": analysis.get('candidate_name', 'N/A'),
                        "total_experience": analysis.get('total_experience_years', 0),
                        "top_skills": analysis.get('top_skills', [])[:3],
                        "processing_time": f"{duration:.2f}s"
                    }
                }
                
                # Extract candidate_id and job_id
                candidate_id = result.get('candidate_id')
                if not candidate_id and isinstance(result.get('result'), dict):
                    candidate_id = result['result'].get('candidate_id')
                
                job_id = None
                if hasattr(email_data, 'subject'):
                    job_id = self._extract_job_id_from_subject(email_data.subject)
                
                if candidate_id and job_id:
                    self.logger.info(f"Calling matching service for candidate_id={candidate_id} and job_id={job_id}")
                    try:
                        # First, ensure the job posting exists in the database
                        try:
                            job_data = self.matching_service._fetch_jd_data(job_id)
                            if not job_data:
                                self.logger.warning(f"Job ID {job_id} not found in the database")
                                response['matching_result'] = {
                                    'status': 'warning',
                                    'message': f'Job ID {job_id} not found in the database',
                                    'candidate_id': candidate_id,
                                    'job_id': job_id
                                }
                                return response
                                
                            # If job exists, find matching candidates
                            match_result = self.matching_service.workflow1_jd_to_resumes(job_id)
                            response['matching_result'] = match_result
                            self.logger.info(f"Matching service completed with status: {match_result.get('status', 'unknown')}")
                            
                        except Exception as e:
                            self.logger.error(f"Error in matching service: {str(e)}", exc_info=True)
                            response['matching_result'] = {
                                'status': 'error',
                                'message': f'Error in matching service: {str(e)}',
                                'candidate_id': candidate_id,
                                'job_id': job_id
                            }
                            
                    except Exception as e:
                        self.logger.error(f"Error calling matching service: {str(e)}", exc_info=True)
                        response['matching_result'] = {
                            'status': 'error',
                            'message': f'Failed to call matching service: {str(e)}',
                            'candidate_id': candidate_id,
                            'job_id': job_id
                        }
                else:
                    missing = []
                    if not candidate_id:
                        missing.append('candidate_id')
                    if not job_id:
                        missing.append('job_id (from email subject)')
                    warning_msg = f"Cannot run matching service - missing: {', '.join(missing)}"
                    self.logger.warning(warning_msg)
                    response['matching_result'] = {
                        'status': 'warning',
                        'message': warning_msg,
                        'candidate_id': candidate_id,
                        'job_id': job_id
                    }
                
                return response
            else:
                self.logger.error(f"Resume processing failed: {result.get('message')}")
            
            return {
                "status": result.get('status', 'error'),
                "agent": "resume_processor",
                "confidence": confidence,
                "result": result
            }
            
        except Exception as e:
            self.logger.error(f"Error routing to resume processor: {str(e)}", exc_info=True)
            return {
                "status": "error", 
                "agent": "resume_processor",
                "message": str(e)
            }

    def _format_recruiter_email(self, job_analysis: Dict[str, Any], missing_fields: list[str]) -> str:
        """Format the recruiter email with appropriate vendor section based on customer type."""
        missing_fields_list = "\n".join([f"- {field.replace('_', ' ').title()}" for field in missing_fields])
        
        email_template = f"""Subject: Regarding Job Posting: {job_analysis.get('job_title')}

Hi Bibhu,

Thank you for sharing the job opportunity for the {job_analysis.get('job_title')} position at {job_analysis.get('end_client_name')}.

"""
        
        if missing_fields:
            email_template += f"""To ensure we find the best candidates, could you please provide the following details:

{missing_fields_list}

"""

        customer_type = job_analysis.get('customer_type')
        if customer_type == 'prime_vendor':
            email_template += """Could you also please share any specific information about the end client that would be helpful in identifying suitable candidates? This might include company culture, team structure, or project details.

"""
        elif customer_type == 'sub_vendor':
            prime_vendor_info_missing = 'prime_vendor_name' in missing_fields or 'prime_vendor_email' in missing_fields
            if prime_vendor_info_missing:
                email_template += """As you are a sub-vendor for this role, could you also please provide:
- Prime Vendor Name
- Prime Vendor Contact Information (email address)
- End Client Contact Information (if different from the hiring manager contact)

"""

        email_template += """Thank you for your help!

Best regards,
MyHiringPartner.ai"""

        return email_template

    def run(self, email_json: str, **kwargs) -> Dict[str, Any]:
        try:
            self.logger.info("Coordinator agent started")
            start_time = time.time()
            
            if isinstance(email_json, str):
                email_json = json.loads(email_json)

            if 'receiver' in email_json and 'recipient' not in email_json:
                email_json['recipient'] = email_json.pop('receiver')

            email_data = EmailData(**email_json)
            self.logger.debug(f"Processing email: {email_data.subject}")
            
            classification = self._classify_email_type(email_data)
            classification_time = time.time() - start_time
            self.logger.debug(f"Classification completed in {classification_time:.2f}s")
            
            if classification == EmailType.JOB_POSTING:
                result = self.route_to_job_poster(email_json, confidence=0.9)
                if result.get("status") == "success" and result.get("details", {}).get("job_id"):
                    job_id = result["details"]["job_id"]
                    self.logger.info(f"Job posting successful. Triggering matching service for job_id: {job_id}")
                    try:
                        match_results = self.matching_service.workflow1_jd_to_resumes(job_id=job_id)
                        self.logger.info(f"Matching service completed for job_id: {job_id}. Results: {match_results}")
                        result['matching_results'] = match_results
                    except Exception as e:
                        self.logger.error(f"Matching service failed for job_id {job_id}: {e}", exc_info=True)
                        result['matching_results'] = {"status": "error", "message": str(e)}
                return result

            elif classification == EmailType.RESUME_APPLICATION:
                result = self.route_to_resume_processor(email_json, confidence=0.9)
                if result.get("status") == "success" and result.get("details", {}).get("candidate_id"):
                    candidate_id = result["details"]["candidate_id"]
                    self.logger.info(f"Resume processing successful. Triggering matching service for candidate_id: {candidate_id}")
                    try:
                        match_results = self.matching_service.workflow2_resume_to_jds(candidate_id=candidate_id)
                        self.logger.info(f"Matching service completed for candidate_id: {candidate_id}. Results: {match_results}")
                        result['matching_results'] = match_results
                    except Exception as e:
                        self.logger.error(f"Matching service failed for candidate_id {candidate_id}: {e}", exc_info=True)
                        result['matching_results'] = {"status": "error", "message": str(e)}
                return result
            elif classification == EmailType.RECRUITER_JD_INFO_REPLIED:
                self.logger.info("Routing to recruiter engager agent...")
                recruiter_engager_result = self.recruiter_engager_agent.run(email_data)
                
                if recruiter_engager_result.get("status") == "success":
                    self.logger.info("Recruiter engager agent successful. Triggering ex-consultant agent...")
                    job_data = recruiter_engager_result.get("job_details", {})
                    if job_data:
                        ex_consultant_result = self.ex_consultant_agent.run(job_data=job_data)
                        # Persist ex-consultant matches to BigQuerychecl 
                        matched_consultants = ex_consultant_result.get("matched_consultants", [])
                        job_id = job_data.get("job_id")
                        for consultant in matched_consultants:
                            candidate_id = consultant.get("candidate_id") or consultant.get("id")
                            if not candidate_id or not job_id:
                                self.logger.warning(f"Skipping match insert: missing candidate_id or job_id (job_id={job_id}, candidate_id={candidate_id})")
                                continue
                            try:
                                if not self.matching_service._check_for_existing_match(job_id, candidate_id):
                                    # Prepare a minimal match_details dict for storage
                                    match_details = {
                                        "match_scores": {"overall_match_score": consultant.get("match_score", 0)},
                                        "screening_decision": {"status": "Proceed Ahead", "explanation": consultant.get("match_reasoning", "")},
                                        "mandatory_requirements": {},
                                        "rule_flagged": "",
                                        "missing_required_skills": [],
                                        "missing_preferred_skills": [],
                                        "key_strengths": [],
                                        "gaps_and_concerns": []
                                    }
                                    match_id = self.matching_service._store_match_result(job_id, candidate_id, consultant.get("match_score", 0), match_details)
                                    self.logger.info(f"Inserted ex-consultant match for job_id={job_id}, candidate_id={candidate_id}, match_id={match_id}")
                                else:
                                    self.logger.info(f"Match already exists for job_id={job_id}, candidate_id={candidate_id}")
                            except Exception as e:
                                self.logger.error(f"Failed to insert ex-consultant match for job_id={job_id}, candidate_id={candidate_id}: {e}")
                        return self._format_response("ex_consultant_search_completed", ex_consultant_result, 0.9)
                    else:
                        self.logger.warning("No job data returned from recruiter engager agent.")
                        return self._format_response("recruiter_jd_info_replied", recruiter_engager_result, 0.9)
                else:
                    self.logger.error("Recruiter engager agent failed.")
                    return self._format_response("recruiter_jd_info_replied", recruiter_engager_result, 0.9)
            elif classification == EmailType.CANDIDATE_MOVE_FORWAD_REPLY:
                self.logger.info("Routing to verification manager agent...")
                result = self.route_to_verification_manager(email_data)
                if result["status"] == "success":
                    return self._format_response("candidate_move_forwad_reply", result, 0.9)
                else:
                    return result
            else:
                self.logger.warning(f"Could not classify email: {email_data.subject}")
                return {
                    "status": "unclassified",
                    "message": "Could not determine email type",
                    "email_subject": email_data.subject
                }
                
        except Exception as e:
            self.logger.error(f"Error in coordinator agent: {str(e)}", exc_info=True)
            return {
                "status": "error",
                "message": str(e)
            }