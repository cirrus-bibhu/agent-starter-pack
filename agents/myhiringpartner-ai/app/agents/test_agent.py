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
The full job description can be found here: https://www.linkedin.com/jobs/view/4270079854


Best regards,
HR Department
""",
        "sender": "test.recruiter@krestasoftech.com",
        "recipient": "bibhu@mavlra.com",    
        "timestamp": "2025-07-03T16:45:00Z",
        "attachments": []
    }

    coordinator = CoordinatorAgent()
    
    print("\nProcessing Job Posting Email:")
    job_result = coordinator.run(json.dumps(job_posting_email))
    print("Job Posting Result:")
    print(json.dumps(job_result, indent=2))

    if job_result.get("status") == "success" and job_result.get("result", {}).get("recruiter_email"):
        print("\n--- Generated Recruiter Email ---")
        print(job_result["result"]["recruiter_email"])
        print("---------------------------------")

if __name__ == "__main__":
    test_email_processing()
