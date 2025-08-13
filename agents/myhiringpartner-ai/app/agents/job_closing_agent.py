import logging
import os
from google.cloud import bigquery
from ..config import config

class JobClosingAgent:
    def __init__(self):
        self.logger = logging.getLogger("JobClosingAgent")
        self.logger.setLevel(logging.DEBUG)
        self.bq_client = bigquery.Client()
        # Resolve project and dataset from environment if provided, otherwise fall back
        project = os.getenv("GOOGLE_CLOUD_PROJECT") or self.bq_client.project
        dataset = os.getenv("BQ_DATASET") or config.bq_dataset
        table = config.bq_table  # defaults to "job_details"
        self.table_id = f"{project}.{dataset}.{table}"

    def run(self, job_id: str):
        self.logger.info(f"Attempting to close job_id: {job_id}")
        try:
            query = f"""
                UPDATE `{self.table_id}`
                SET job_status = 'closed'
                WHERE job_id = @job_id
            """

            job_config = bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ScalarQueryParameter("job_id", "STRING", job_id),
                ]
            )

            query_job = self.bq_client.query(query, job_config=job_config)
            query_job.result()  # Wait for the job to complete

            if query_job.num_dml_affected_rows > 0:
                self.logger.info(f"Successfully closed job_id: {job_id}. Rows affected: {query_job.num_dml_affected_rows}")
                return {"status": "success", "message": f"Job {job_id} has been closed."}
            else:
                self.logger.warning(f"Job_id: {job_id} not found or already closed.")
                return {"status": "not_found", "message": f"Job {job_id} not found or status was not updated."}

        except Exception as e:
            self.logger.error(f"Error closing job {job_id}: {e}", exc_info=True)
            return {"status": "error", "message": str(e)}
