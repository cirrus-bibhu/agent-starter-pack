import os
import sys
import json
from dotenv import load_dotenv

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

load_dotenv()

from app.agents.coordinator import CoordinatorAgent

def test_candidate_reply():
    print("--- Testing Candidate Info Reply Processing ---")
    
    candidate_id = "73a63660-6243-4465-b8d9-58e559e536fe"
    
    candidate_reply_email = {
        "id": "email_candidate_reply_001",
        "subject": f"Re: Action Required: Additional Information for Your Application (ID: {candidate_id})",
        "body": """Hi,

Thank you for following up. Here is my date of birth: 1982-05-07, linkedin profile: https://www.linkedin.com/in/gowthami-k-a04599267/

Best regards,
Harikrishna Katari""",
        "sender": "krishna.apps415@gmail.com",
        "recipient": "system@myhiringpartner.ai",
        "timestamp": "2025-07-11T17:12:00Z",
        "attachments": []
    }

    coordinator = CoordinatorAgent()

    print("\nProcessing Candidate Reply Email:")
    result = coordinator.run(json.dumps(candidate_reply_email))
    print("\nEmail Processing Result:")
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    test_candidate_reply()