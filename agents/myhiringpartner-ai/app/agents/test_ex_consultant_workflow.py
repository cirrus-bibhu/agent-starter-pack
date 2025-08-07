import json
from dotenv import load_dotenv
import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

load_dotenv()

from unittest.mock import patch
from app.agents.coordinator import CoordinatorAgent

def test_ex_consultant_workflow_with_found_candidates():
    """
    Tests the end-to-end workflow, simulating a scenario where ex-consultants ARE found.
    """
    print("--- Testing Ex-Consultant Workflow---")

    job_id = "4256698922"
    recruiter_reply_email = {
        "id": "email_ex_consultant_test_002",
        "subject": f"Re: Job Info for AI Engineer (Job ID: {job_id})",
        "body": f"""
        Hi,
        Here are the details for Job ID {job_id}:
        - End Client Name: Global Tech Innovations
        - Prime Vendor Name: Strategic Staffing Solutions
        - Job Title: Senior AI Engineer
        - Job Description: Seeking an experienced AI engineer with a background in machine learning and natural language processing.
        """,
        "sender": "recruiter@strategicstaffing.com",
        "recipient": "bibhu@myhiringpartner.ai",
        "timestamp": "2025-07-25T15:00:00Z",
        "attachments": []
    }

    coordinator = CoordinatorAgent()
    print("\nProcessing Recruiter's Reply to trigger Ex-Consultant Workflow with REAL data...")
    result = coordinator.run(email_json=json.dumps(recruiter_reply_email), job_id=job_id)

    print("\n--- Final Processing Result ---")
    print(json.dumps(result, indent=2))
    
    # Basic assertion to check for success
    assert result['status'] == 'success'
    assert result['agent'] == 'ex_consultant_search_completed'
    assert len(result['result']['matched_consultants']) > 0
    print("\nSuccessfully found ex-consultants.")

if __name__ == "__main__":
    test_ex_consultant_workflow_with_found_candidates()
