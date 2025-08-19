from google.cloud import bigquery

PROJECT_ID = "saas-agent-qa"
SOURCE_DATASET_ID = "myhiringpartner_ai"
BACKUP_DATASET_ID = "mhp_bkp"
TABLES_TO_PROCESS = ["job_details", "resumes", "matches"]

def backup_and_clear_tables():
    client = bigquery.Client(project=PROJECT_ID)

    backup_dataset_ref = client.dataset(BACKUP_DATASET_ID)
    try:
        client.get_dataset(backup_dataset_ref)
        print(f"Dataset '{BACKUP_DATASET_ID}' already exists.")
    except Exception:
        print(f"Creating dataset '{BACKUP_DATASET_ID}'...")
        client.create_dataset(bigquery.Dataset(backup_dataset_ref))
        print(f"Dataset '{BACKUP_DATASET_ID}' created.")

    for table_id in TABLES_TO_PROCESS:
        source_table_ref = client.dataset(SOURCE_DATASET_ID).table(table_id)
        backup_table_ref = client.dataset(BACKUP_DATASET_ID).table(table_id)

        print(f"Copying table '{table_id}' from '{SOURCE_DATASET_ID}' to '{BACKUP_DATASET_ID}'...")
        
        job_config = bigquery.CopyJobConfig()
        job_config.write_disposition = bigquery.WriteDisposition.WRITE_TRUNCATE
        
        try:
            copy_job = client.copy_table(
                source_table_ref,
                backup_table_ref,
                job_config=job_config,
            )
            copy_job.result()
            print(f"Table '{table_id}' copied successfully.")
        except Exception as e:
            print(f"An error occurred while copying table {table_id}: {e}")
            return

    print("\nBackup complete. Now clearing original tables.")

    for table_id in TABLES_TO_PROCESS:
        full_table_id = f"{PROJECT_ID}.{SOURCE_DATASET_ID}.{table_id}"
        print(f"Clearing table '{full_table_id}'...")
        
        query = f"TRUNCATE TABLE `{full_table_id}`"
        
        try:
            query_job = client.query(query)
            query_job.result()
            print(f"Table '{full_table_id}' cleared successfully.")
        except Exception as e:
            print(f"An error occurred while clearing table {full_table_id}: {e}")

    print("\nAll operations completed successfully.")

if __name__ == "__main__":
    backup_and_clear_tables()
