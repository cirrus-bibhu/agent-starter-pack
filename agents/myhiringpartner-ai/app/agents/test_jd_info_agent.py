import json
from dotenv import load_dotenv

load_dotenv()

from app.agents.coordinator import CoordinatorAgent

def test_recruiter_reply_processing():
    print("--- Testing Recruiter JD Info Reply Processing ---")
    
    job_id = "4256698922"
    
    recruiter_reply_email = {
        "id": "email_recruiter_reply_001",
        "subject": f"Re: Follow-up Information Request for Python Developer (Job ID: {job_id})",
        "body": f"""
Hi Team,

Please find the requested details for Job ID: {job_id} below:

Basic Job Information:
- Is Relocation Allowed: Yes
- Is Remote: True

Vendor Information:
- Prime Vendor Name: Kresta Softech Private Limited
- Prime Vendor Email: prime@krestasofttech.com
- End Client Name: XYZ Corporation
- End Client Email: contact@xyzcorp.com

Let me know if you need anything else.

Thanks,
Recruiter
""",
        "sender": "test.recruiter@krestasoftech.com",
        "recipient": "bibhu@myhiringpartner.ai",
        "timestamp": "2025-07-04T10:00:00Z",
        "attachments": []
    }

    coordinator = CoordinatorAgent()
    
    print("\nProcessing Recruiter's Reply Email:")
    result = coordinator.run(email_json=json.dumps(recruiter_reply_email), job_id=job_id)
    
    print("Processing Result:")
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    test_recruiter_reply_processing()
