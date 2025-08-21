import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

import json
from dotenv import load_dotenv
load_dotenv()

from .coordinator import CoordinatorAgent

# Try to create a sample DOC resume if it doesn't exist
SAMPLE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'sample_resumes'))
SAMPLE_DOC = os.path.join(SAMPLE_DIR, 'Data_Engineer_Sample.doc')

try:
    os.makedirs(SAMPLE_DIR, exist_ok=True)
except Exception:
    pass

if not os.path.exists(SAMPLE_DOC):
    # Create a simple plain-text .doc file to simulate legacy Word documents
    try:
        SAMPLE_TEXT = (
            'Mayur Rawte\n'
            'mayur.rawte@example.com | +1-555-123-4567 | Bangalore, India\n\n'
            'Professional Summary\n'
            'Data Engineer with 5+ years experience building ETL pipelines, data warehouses, and scalable data platforms. '
            'Skilled in Python, SQL, BigQuery, Airflow and GCP.\n\n'
            'Experience\n'
            'Senior Data Engineer, Acme Corp — Jan 2021 to Present\n'
            '• Designed and operated ETL pipelines using Apache Airflow and Python.\n'
            '• Implemented BigQuery schemas and optimized queries for low latency analytics.\n\n'
            'Education\n'
            'B.Tech, Computer Science — 2018, National Institute of Technology\n\n'
            'Skills\n'
            'Python, SQL, BigQuery, GCP, Apache Airflow, Data Modeling, ETL\n'
        )
        with open(SAMPLE_DOC, 'w', encoding='utf-8') as f:
            f.write(SAMPLE_TEXT)
        print(f"Created sample legacy .doc resume: {SAMPLE_DOC}")
    except Exception as e:
        print(f"Warning: failed to create .doc sample ({e}). Falling back to creating a plain text file.")
        try:
            SAMPLE_TXT = SAMPLE_DOC.replace('.doc', '.txt')
            with open(SAMPLE_TXT, 'w', encoding='utf-8') as f:
                f.write('Mayur Rawte\nmayur.rawte@example.com\nData Engineer\nSkills: Python, SQL, BigQuery, GCP, Airflow')
            SAMPLE_DOC = SAMPLE_TXT
        except Exception:
            pass


def test_resume_email_processing():
    print('--- Testing Resume Application Email Processing ---')

    # Use the .doc sample created above
    with open(SAMPLE_DOC, 'rb') as f:
        resume_bytes = f.read()

    # 1) First, simulate a job posting email so the job exists in job_details
    posting_email = {
        "id": "email_job_post_001",
        "subject": "Your job is posted: [Application Lead-MHP 2529]",
        "body": """
Dear Hiring Team,

Please review the details for our new job opening.
This is an urgent hire for our division.
The full job description can be found here: https://www.linkedin.com/jobs/view/4259823054

Thanks,
Recruiter
""",
        "sender": "recruiter@example.com",
        "recipient": "support@myhiringpartner.ai",
        "timestamp": "2025-08-19T10:00:00Z"
    }

    # 2) Then, simulate an application email that references the same internal job id in the subject
    resume_email = {
        "id": "email_resume_app_001",
        "subject": "New application: Application Lead-MHP2529 from John Doe",
        "body": """
Dear Hiring Team,

Please find attached my resume for your consideration for the Production Agent role.

Best regards,
Mayur Rawte
""",
        "sender": "mayur.rawte@example.com",
        "recipient": "support@myhiringpartner.ai",
        "timestamp": "2025-08-19T12:00:00Z",
        "attachments": [
            {
                "filename": os.path.basename(SAMPLE_DOC),
                "content_type": "application/msword",
                "content": resume_bytes.hex()
            }
        ]
    }

    coordinator = CoordinatorAgent()

    print('\nProcessing Resume Application Email:')
    resume_result = coordinator.run(json.dumps(resume_email))
    print('Resume Processing Result:')
    print(json.dumps(resume_result, indent=2))

    if resume_result.get("status") == "success" and resume_result.get("result"):
        print('\n--- Extracted Candidate Details ---')
        print(json.dumps(resume_result["result"], indent=2))
        print('-----------------------------------')


if __name__ == "__main__":
    test_resume_email_processing()
