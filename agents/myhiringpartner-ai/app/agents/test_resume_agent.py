import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

import json
from dotenv import load_dotenv

load_dotenv()

from .coordinator import CoordinatorAgent

def test_resume_email_processing():
    print("--- Testing Resume Application Email Processing ---")

    # The PDF is located one directory above 'app/agents'
    pdf_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Gowthami -Java-LinkedInMisMatch.docx'))
    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()

    resume_email = {
        "id": "email_resume_app_001",
        "subject": "Resume Application for  Position MAV-2419",
        "body": """
Dear Hiring Team,

Please find attached my resume for your consideration for the Java Developer role.

Best regards,
Gowthami
""",
        "sender": "gowthami.apps415@gmail.com",
        "recipient": "bibhu@mavlra.com",
        "timestamp": "2025-07-11T17:12:00Z",
        "attachments": [
            {
                "filename": "Gowthami -Java-LinkedInMisMatch.docx",
                "content_type": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                "content": pdf_bytes.hex()
            }
        ]
    }

    coordinator = CoordinatorAgent()

    print("\nProcessing Resume Application Email:")
    resume_result = coordinator.run(json.dumps(resume_email))
    print("Resume Processing Result:")
    print(json.dumps(resume_result, indent=2))

    if resume_result.get("status") == "success" and resume_result.get("result"):
        print("\n--- Extracted Candidate Details ---")
        print(json.dumps(resume_result["result"], indent=2))
        print("-----------------------------------")

if __name__ == "__main__":
    test_resume_email_processing()
