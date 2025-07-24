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
    
    candidate_id = "ddbb2c4e-bf7a-4f09-9b2e-f638f0236ce1"
    
    candidate_reply_email = {
        "id": "email_candidate_reply_001",
        "subject": f"Re: Action Required: Additional Information for Your Application (ID: {candidate_id})",
        "body": """Hi,

        """,
        "sender": "akhil.ramadugu@example.com",
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