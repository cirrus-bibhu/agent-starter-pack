import json
from dotenv import load_dotenv

load_dotenv()

from .coordinator import CoordinatorAgent

def test_email_processing():
    print("--- Testing Email Processing ---")
    
    job_posting_email = {
        "id": "email_job_post_001",
        "subject": "Your job is posted for the role of Python Developer",
        "body": """
Dear Hiring Team,

Please review the details for our new job opening.
This is an urgent hire for our division.
The full job description can be found here: https://www.linkedin.com/jobs/view/4269275339


Best regards,
HR Department
""",
        "sender": "brmohapatra404@gmail.com",
        "recipient": "bibhu@myhiringpartner.ai",    
        "timestamp": "2025-07-03T16:45:00Z",
        "attachments": []
    }

    coordinator = CoordinatorAgent()
    
    print("\nProcessing Job Posting Email:")
    job_result = coordinator.run(json.dumps(job_posting_email))
    print("Job Posting Result:")

    if job_result.get("status") == "success" and job_result.get("result", {}).get("recruiter_email"):
        print("\n--- Generated Recruiter Email ---")
        print(job_result["result"]["recruiter_email"])
        print("---------------------------------")

if __name__ == "__main__":
    test_email_processing()
