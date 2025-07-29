import logging
import json
import os
import re
from datetime import date, datetime
import google.generativeai as genai
from ..tools.linkedin_scrapper import ProfileExtractor as LinkedinScraper
from google.cloud import bigquery
from app.agent import BaseAgent
from app.config import config
import time
import backoff
from google.api_core import exceptions

logger = logging.getLogger('Agent.VerificationManager')

class VerificationManagerAgent(BaseAgent):
    AI_FABRICATED_RESUME_VERIFICATION_PROMPT = """You are a world-class resume fraud detection expert. Your task is to conduct a forensic analysis of a candidate's resume to identify signs of fabrication, AI generation, or fraudulent claims.

**Analysis Dimensions:**

1.  **Authenticity and Fabrication Analysis:**
    *   **Content Scrutiny:** Scrutinize the resume for common red flags of fabrication:
        *   **Academic Experience as Work:** Identifying academic projects, university coursework, or degrees being misrepresented as professional work experience.
        *   **Vague & Generic Content:** Job descriptions are overly generic, full of buzzwords, and lack specific, measurable achievements.
        *   **Unrealistic Progression:** A perfectly linear career path with no gaps, overlaps, or logical setbacks.
        *   **Fully Fabricated History:** The entire work history appears implausible or invented.
    *   **Timeline Plausibility:** Check for major logical flaws, such as a late graduation age that is inconsistent with the claimed experience, without any alternate explanation.

2.  **Final Verdict Guidance (Strict):**
    *   **Reject the candidate for clear signs of fabrication:**
        *   The resume misrepresents academic achievements (e.g., a Master's degree) as professional work experience.
        *   The work history is clearly fabricated, containing generic descriptions and lacking verifiable details.
        *   The entire profile is logically inconsistent and appears to be fake.

**Output Format:**

Return your analysis as a single JSON object. Focus the 'discrepancies' on authenticity and fabrication concerns.

```json
{{
  "discrepancies": [
    {{
      "type": "Authenticity Concern | Fabricated Experience",
      "description": "Red Flag: Candidate's Master's Degree period is counted as professional work experience.",
      "severity": "high"
    }}
  ],
  "gaps_in_experience": [],
  "final_verdict": "Proceed | Reject",
  "verdict_reason": "Resume is likely fabricated. It misrepresents academic qualifications as professional experience."
}}
```

**Candidate Data:**

**Resume (JSON):**
{{resume_json}}

{{linkedin_section}}
"""

    EDUCATION_TIMELINE_VERIFICATION_PROMPT = """You are a world-class resume verification expert. Your task is to conduct a focused analysis of a candidate's education and career timeline for plausibility and coherence.

**Analysis Dimensions:**

1.  **Education & Timeline Plausibility Analysis:**
    *   **Education Verification:**
        *   Compare the education details (degrees, institutions, dates) on the resume with LinkedIn.
        *   If a Master's degree is listed, verify that a Bachelor's degree is also present and that the timeline between them is logical. Flag if the Bachelor's degree is missing.
    *   **Timeline Coherence:** Analyze the relationship between education and work timelines. Flag inconsistencies such as:
        *   Full-time work experience overlapping with full-time education periods (e.g., starting a job in 2012 but claiming a Master's degree completed in 2014).
        *   Professional work experience starting significantly before the completion of a foundational degree (e.g., Bachelor's in 2017, but experience starting in 2015).
    *   **Age and Experience Plausibility:** Assess if the claimed graduation age and career progression are realistic. Flag anomalies like an unusually late graduation age without a clear explanation.

2.  **Final Verdict Guidance (Strict):**
    *   **Reject the candidate for critical timeline inconsistencies:**
        *   The timeline is illogical (e.g., holding a full-time senior role before completing a Bachelor's degree) without any reasonable explanation.
        *   A claimed degree's timeline directly and inexplicably conflicts with the work history provided.

**Output Format:**

Return your analysis as a single JSON object. Focus the 'discrepancies' on timeline and education issues.

```json
{{
  "discrepancies": [
    {{
      "type": "Timeline Plausibility | Education Verification",
      "description": "Candidate claims a Bachelor's degree completed in 2017, but work experience starts in 2015, two years prior to graduation.",
      "severity": "high"
    }}
  ],
  "gaps_in_experience": [],
  "final_verdict": "Proceed | Reject",
  "verdict_reason": "Reject due to critical timeline inconsistency; professional work experience overlaps with full-time education."
}}
```

**Candidate Data:**

**Resume (JSON):**
{{resume_json}}

{{linkedin_section}}
"""

    STANDARD_VERIFICATION_PROMPT = """You are a world-class resume verification expert. Your task is to conduct a strict analysis of a candidate's resume and LinkedIn profile to identify discrepancies.

**Analysis Dimensions:**

1.  **Resume vs. LinkedIn Discrepancy Analysis:**
    *   **Experience Verification:** For each job on the resume, find the corresponding entry on LinkedIn. Compare job titles, companies, and employment dates. Note any discrepancies where the resume claims a longer duration or a more senior title.
    *   **Skills Verification:** Check if the key skills and certifications listed on the resume are supported by the LinkedIn profile.
    *   **LinkedIn Availability:** If the LinkedIn profile is not provided or is inaccessible, note this as a medium-severity discrepancy.

2.  **Timeline & Gap Analysis:**
    *   Construct a career timeline based on the resume.
    *   Identify and report any unexplained employment gaps of 6 months or more.

3.  **Final Verdict Guidance (Strict):**
    *   **Reject the candidate if the RESUME OVERSTATES experience:**
        *   The total work experience on the resume is exaggerated by more than **12 months** compared to what is verifiable on LinkedIn.
        *   A specific job's duration on the resume is inflated by more than **6 months** compared to LinkedIn.
        *   Key technical skills are listed on the resume but are completely absent from the LinkedIn profile.
    *   **Do NOT reject if LinkedIn shows MORE experience.** Note it as a 'low' severity discrepancy, but this is not grounds for rejection.

**Output Format:**

Return your analysis as a single JSON object with the specified structure.

```json
{{
  "discrepancies": [
    {{
      "type": "LinkedIn Comparison",
      "description": "Description of the discrepancy.",
      "severity": "low | medium | high"
    }}
  ],
  "gaps_in_experience": [
    {{
      "start_date": "YYYY-MM-DD",
      "end_date": "YYYY-MM-DD",
      "duration_months": 6
    }}
  ],
  "final_verdict": "Proceed | Reject",
  "verdict_reason": "A brief justification for the verdict based on the guidelines."
}}
```

**Candidate Data:**

**Resume (JSON):**
{{resume_json}}

{{linkedin_section}}
"""

    VERIFICATION_PROMPT = AI_FABRICATED_RESUME_VERIFICATION_PROMPT


    def __init__(self):
        super().__init__("VerificationManagerAgent")
        self.config = config
        
        genai.configure(api_key=os.getenv('GEMINI_API_KEY', ''))
        self.llm = genai.GenerativeModel('gemini-1.5-pro')

        api_key = os.getenv("SCRAPIN_API_KEY")
        if not api_key:
            raise ValueError("SCRAPIN_API_KEY environment variable not set")
        self.linkedin_scraper = LinkedinScraper(api_key)
        self.bq_client = bigquery.Client()
        self.resumes_table_id = f"{self.bq_client.project}.{self.config.bq_dataset}.resumes"

    def _clean_for_json(self, value):
        if isinstance(value, (date, datetime)):
            return value.isoformat()
        elif isinstance(value, dict):
            return {k: self._clean_for_json(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [self._clean_for_json(item) for item in value]
        return value



    def _format_education(self, edu: dict) -> dict:
        return {
            "degree": edu.get('degree', '') or '',
            "institution": edu.get('institution', '') or '',
            "field_of_study": edu.get('field_of_study', '') or '',
            "graduation_year": int(edu.get('graduation_year')) if edu.get('graduation_year') is not None else None
        }

    def _generate_verification_prompt(self, resume_data: dict, linkedin_data: dict) -> str:
        """Generate prompt for verification by formatting resume and LinkedIn data."""
        cleaned_resume_data = self._clean_for_json(resume_data)
        resume_json = json.dumps(cleaned_resume_data, indent=2)

        linkedin_section = "**LinkedIn (JSON):**\nNot provided."
        if linkedin_data:
            cleaned_linkedin_data = self._clean_for_json(linkedin_data)
            linkedin_json = json.dumps(cleaned_linkedin_data, indent=2)
            linkedin_section = f"**LinkedIn (JSON):**\n{linkedin_json}"

        return self.VERIFICATION_PROMPT.format(
            resume_json=resume_json,
            linkedin_section=linkedin_section
        )

    @backoff.on_exception(
        backoff.expo,
        exceptions.BadRequest,
        max_tries=3,
        giveup=lambda e: "streaming buffer" not in str(e).lower(),
        max_time=300  # Maximum 5 minutes of retrying
    )
    def _update_bigquery_with_retry(self, update_query: str, job_config: bigquery.QueryJobConfig):
        """Execute BigQuery update with retry for streaming buffer errors"""
        return self.bq_client.query(update_query, job_config=job_config).result()

    def _parse_date(self, date_str: str):
        """Safely parse a date string into a date object."""
        if not date_str or not isinstance(date_str, str):
            return None
        try:
            return date.fromisoformat(date_str)
        except (ValueError, TypeError):
            return None

    def _update_verification_data_in_bq(self, candidate_id: str, verification_data: dict):
        """Updates the resumes table with verification data."""
        logger.info(f"Updating verification data for candidate_id: {candidate_id}")

        # Extract fields from verification data, providing empty defaults
        # Clean and validate discrepancies data
        discrepancies = verification_data.get('discrepancies', [])
        if isinstance(discrepancies, list):
            cleaned_discrepancies = []
            for item in discrepancies:
                if isinstance(item, dict):
                    # For STRUCT arrays, BQ expects a list of Row objects
                    cleaned_row = bigquery.Row(
                        (
                            str(item.get('type', '')) if item.get('type') is not None else None,
                            str(item.get('description', '')) if item.get('description') is not None else None,
                            str(item.get('severity', '')) if item.get('severity') is not None else None
                        ),
                        {"type": 0, "description": 1, "severity": 2}
                    )
                    cleaned_discrepancies.append(cleaned_row)
            discrepancies = cleaned_discrepancies
        else:
            discrepancies = [] # Default to empty list if not a list

        # Clean and validate gaps_in_experience data
        gaps_raw = verification_data.get('gaps_in_experience', [])
        gaps = []
        if isinstance(gaps_raw, list):
            for gap in gaps_raw:
                if isinstance(gap, dict):
                    duration = gap.get('duration_months')
                    start_date = self._parse_date(gap.get('start_date'))
                    end_date = self._parse_date(gap.get('end_date'))

                    # For STRUCT arrays, BQ expects a list of Row objects
                    cleaned_row = bigquery.Row(
                        (
                            start_date.strftime('%Y-%m-%d') if start_date else None,
                            end_date.strftime('%Y-%m-%d') if end_date else None,
                            int(duration) if duration is not None else None,
                            str(gap.get('reason', '')) if gap.get('reason') is not None else None
                        ),
                        {"start_date": 0, "end_date": 1, "duration_months": 2, "reason": 3}
                    )
                    gaps.append(cleaned_row)

        metrics = verification_data.get('metrics', {})
        final_verdict = verification_data.get('final_verdict')
        verdict_reason = verification_data.get('verdict_reason')

        # Extract all metrics fields, providing defaults
        resume_quality_score = metrics.get('resume_quality_score')
        inconsistencies = metrics.get('inconsistencies', [])
        missing_info = metrics.get('missing_information', [])

        # Use a default value for resume_quality_score if it's None
        if resume_quality_score is None:
            resume_quality_score = 0.0  # Or any other suitable default
            logger.warning("resume_quality_score was None, defaulting to 0.0")

        # Use a default for final_verdict if it's None
        if final_verdict is None:
            final_verdict = "needs_review"
            logger.warning("final_verdict was None, defaulting to 'needs_review'")

        # Use a default for verdict_reason if it's None
        if verdict_reason is None:
            verdict_reason = "AI verdict was not provided."
            logger.warning("verdict_reason was None, defaulting to 'AI verdict was not provided.'")

        # Construct the MERGE statement with query parameters
        script = f"""
        MERGE `{self.resumes_table_id}` T
        USING (SELECT @candidate_id AS candidate_id) S ON T.candidate_id = S.candidate_id
        WHEN MATCHED THEN
            UPDATE SET
                T.discrepancies = @discrepancies,
                T.gaps_in_experience = @gaps_in_experience,
                T.metrics = STRUCT(
                    @resume_quality_score AS resume_quality_score,
                    @missing_info AS missing_information,
                    @inconsistencies AS inconsistencies
                ),
                T.final_verdict = @final_verdict,
                T.verdict_reason = @verdict_reason,
                T.updated_at = CURRENT_TIMESTAMP()
        """

        # Define scalar and simple array query parameters
        query_params = [
            bigquery.ScalarQueryParameter("candidate_id", "STRING", candidate_id),
            bigquery.ArrayQueryParameter(
                "discrepancies",
                "STRUCT<type STRING, description STRING, severity STRING>",
                discrepancies
            ),
            bigquery.ArrayQueryParameter(
                "gaps_in_experience",
                "STRUCT<start_date STRING, end_date STRING, duration_months INT64, reason STRING>",
                gaps
            ),
            bigquery.ScalarQueryParameter("resume_quality_score", "FLOAT64", resume_quality_score),
            bigquery.ArrayQueryParameter("inconsistencies", "STRING", inconsistencies),
            bigquery.ArrayQueryParameter("missing_info", "STRING", missing_info),
            bigquery.ScalarQueryParameter("final_verdict", "STRING", final_verdict),
            bigquery.ScalarQueryParameter("verdict_reason", "STRING", verdict_reason)
        ]

        job_config = bigquery.QueryJobConfig(query_parameters=query_params)

        try:
            self._update_bigquery_with_retry(script, job_config=job_config)
            logger.info(f"Successfully updated verification data for candidate_id: {candidate_id}")
        except exceptions.BadRequest as e:
            logger.error(f"BigQuery BadRequest Error: {e}", exc_info=True)
            logger.error(f"Data sent to BigQuery that caused the error:")
            logger.error(f"  - candidate_id: {candidate_id}")
            logger.error(f"  - discrepancies: {json.dumps(discrepancies, indent=2, default=str)}")
            logger.error(f"  - gaps: {json.dumps(gaps, indent=2, default=str)}")
            logger.error(f"  - metrics: {json.dumps(metrics, indent=2)}")
            logger.error(f"  - final_verdict: {final_verdict}")
            logger.error(f"  - verdict_reason: {verdict_reason}")
            raise
        except Exception as e:
            logger.error(f"An unexpected error occurred while updating verification data for {candidate_id}: {e}", exc_info=True)
            raise


    def run(self, data: dict):
        candidate_id = data.get('candidate_id')
        email_data = data.get('email_data')
        logger.info(f"Starting verification for candidate_id: {candidate_id}")
        
        try:
            # 1. Fetch resume data from BigQuery
            query = f"""
            SELECT *  
            FROM `{self.resumes_table_id}` 
            WHERE candidate_id = @candidate_id
            """
            job_config = bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ScalarQueryParameter("candidate_id", "STRING", candidate_id)
                ]
            )
            results = self.bq_client.query(query, job_config=job_config).result()
            if results.total_rows == 0:
                logger.error(f"No resume found for candidate_id: {candidate_id}")
                return {"status": "error", "message": f"No resume found for candidate_id: {candidate_id}"}
            
            resume_data = dict(next(iter(results)))
            
            # 2. If email data is present, extract LinkedIn URL
            linkedin_url_from_email = None
            if email_data and email_data.body:
                linkedin_url_match = re.search(r'(https?://(?:www\.)?linkedin\.com/in/[\S]+)', email_data.body, re.IGNORECASE)
                if linkedin_url_match:
                    linkedin_url_from_email = linkedin_url_match.group(1)
                    # Update the resume_data in memory for this run
                    resume_data['linkedin_url'] = linkedin_url_from_email
                    logger.info(f"Found LinkedIn URL in email body: {linkedin_url_from_email}")

            # 3. Use the potentially updated LinkedIn URL for verification
            linkedin_url_to_use = resume_data.get('linkedin_url')
            verification_result = None

            linkedin_data = {}
            if linkedin_url_to_use:
                logger.info(f"Found LinkedIn URL to process: {linkedin_url_to_use}")
                try:
                    linkedin_data = self.linkedin_scraper.extract(linkedin_url_to_use)
                except Exception as e:
                    logger.warning(f"Failed to fetch LinkedIn data for {linkedin_url_to_use}: {e}. Proceeding without it.")
            else:
                logger.info("No LinkedIn URL found. Proceeding with resume data only.")
            
            prompt = self._generate_verification_prompt(resume_data, linkedin_data)
            response_text = self.llm.generate_content(prompt).text
            
            match = re.search(r"```json\n(.*?)\n```", response_text, re.DOTALL)
            if match:
                json_str = match.group(1)
            else:
                json_str = response_text

            verification_result = json.loads(json_str)
            logger.info(f"Received verification JSON from AI: {json.dumps(verification_result, indent=2)}")

            if verification_result:
                update_success = self._update_verification_data_in_bq(candidate_id, verification_result)
                if not update_success:
                    logger.error(f"Failed to update BigQuery with verification data for candidate {candidate_id}")

                sanitized_name = re.sub(r'[^a-zA-Z0-9_]', '', candidate_name.replace(' ', '_')).lower()
                output_filename = f"{sanitized_name}_verification.json"
                
                output_dir = "verification_results"
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                
                full_output_path = os.path.join(output_dir, output_filename)

                with open(full_output_path, 'w') as f:
                    json.dump(verification_result, f, indent=2)
                
                logger.info(f"Verification result for '{candidate_name}' saved to {full_output_path}")
                print("\n--- Verification Result ---")
                print(json.dumps(verification_result, indent=2))
                print("--- End Verification Result ---\n")

            else:
                logger.warning("Verification result was empty. Nothing to store.")

            return {
                "status": "success",
                "verification": verification_result
            }

        except Exception as e:
            logger.error(f"Error in verification process: {e}", exc_info=True)
            return {
                "status": "error",
                "message": "Verification process failed",
                "details": str(e)
            }
