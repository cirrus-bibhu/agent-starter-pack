import os
import uuid
from datetime import datetime
from google.cloud import bigquery
from dotenv import load_dotenv

# --- Configuration ---
# Load environment variables from .env file
load_dotenv()

# Get BigQuery project and dataset from environment variables
PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT")
DATASET_ID = os.getenv("BQ_DATASET", "myhiringpartner_ai")
TABLE_ID = "resumes"

# --- Sample Candidate Data ---
# This candidate has worked at 'Global Tech Innovations', making them an 'ex-consultant' for our test case.
CANDIDATE_ID = f"candidate_{uuid.uuid4()}"

CANDIDATE_DATA = {
    "candidate_id": CANDIDATE_ID,
    "candidate_name": "John Doe (Test Candidate)",
    "candidate_email": "john.doe.test@example.com",
    "experience": [
        {
            "company": "Global Tech Innovations", # This matches the 'end_client_name' in the test job
            "role": "Senior AI Engineer",
            "start_date": "2018-01-01",
            "end_date": "2023-12-31"
        }
    ],
    "technical_skills": ["Python", "Machine Learning", "TensorFlow"],
    "resume_summary": "An experienced AI Engineer with a strong background in developing and deploying machine learning models. Proven expertise in Python, TensorFlow, and cloud platforms. Previously worked at Global Tech Innovations.",
    "previous_companies": ["Global Tech Innovations", "Another Tech Co"],
    "created_at": datetime.utcnow().isoformat(),
    "updated_at": datetime.utcnow().isoformat(),
    # Add other required fields with default values if necessary
    "storage_uri": f"gs://myhiringpartner-ai_artifacts/resumes/{CANDIDATE_ID}.pdf"
}

def insert_data_to_bigquery():
    """Connects to BigQuery and inserts the sample candidate data."""
    try:
        client = bigquery.Client(project=PROJECT_ID)
        table_ref = client.dataset(DATASET_ID).table(TABLE_ID)
        
        print(f"Attempting to insert data into {PROJECT_ID}.{DATASET_ID}.{TABLE_ID}")
        
        errors = client.insert_rows_json(table_ref, [CANDIDATE_DATA])
        
        if not errors:
            print("--- Success! ---")
            print(f"Successfully inserted candidate '{CANDIDATE_DATA['candidate_name']}' with ID: {CANDIDATE_ID}")
        else:
            print("--- Error ---")
            print("Encountered errors while inserting rows:")
            for error in errors:
                print(f"Row: {error['index']} Errors: {error['errors']}")

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        print("Please ensure you have authenticated with Google Cloud (`gcloud auth application-default login`) and that the project/dataset is correct.")

if __name__ == "__main__":
    if not PROJECT_ID:
        print("Error: GOOGLE_CLOUD_PROJECT environment variable not set.")
    else:
        insert_data_to_bigquery()
