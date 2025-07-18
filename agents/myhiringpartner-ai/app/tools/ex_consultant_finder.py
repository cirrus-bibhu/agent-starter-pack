import logging
from typing import List, Dict, Any
from google.cloud import bigquery
import json

class ExConsultantFinder:
    def __init__(self, project_id: str, dataset_id: str, resume_table: str):
        self.logger = logging.getLogger(__name__)
        self.project_id = project_id
        self.dataset_id = dataset_id
        self.resume_table = resume_table
        self.bq_client = bigquery.Client(project=project_id)
    
    def _vector_search_query(self, query_embedding: List[float], limit: int = 20, end_client_name: str = None, prime_vendor_name: str = None) -> List[Dict[str, Any]]:
        try:
            filter_clauses = []
            if end_client_name:
                filter_clauses.append(f"LOWER('{end_client_name}') IN (SELECT LOWER(company) FROM UNNEST(past_companies) AS company)")
            if prime_vendor_name:
                filter_clauses.append(f"LOWER('{prime_vendor_name}') IN (SELECT LOWER(company) FROM UNNEST(past_companies) AS company)")

            base_filter_sql = ""
            if filter_clauses:
                base_filter_sql = f"base_table_filter => '({' OR '.join(filter_clauses)})'"

            query = f"""
            WITH query_embedding AS (
                SELECT {json.dumps(query_embedding)} AS embedding
            )
            SELECT 
                r.*,
                1 - distance AS similarity_score
            FROM 
                VECTOR_SEARCH(
                    TABLE `{self.project_id}.{self.dataset_id}.{self.resume_table}`,
                    'embeddings',
                    (SELECT embedding FROM query_embedding),
                    distance_type => 'COSINE',
                    top_k => {limit},
                    {base_filter_sql}
                )
            ORDER BY similarity_score DESC
            """

            self.logger.info(f"Executing BigQuery vector search with filter: {base_filter_sql}")
            query_job = self.bq_client.query(query)
            results = list(query_job.result())
            return [dict(row) for row in results]

        except Exception as e:
            self.logger.error(f"Error in vector search query: {e}", exc_info=True)
            return []
    
    def find_ex_consultants(
        self,
        job_description: str,
        end_client_name: str,
        prime_vendor_name: str,
        max_results: int = 20
    ) -> Dict[str, List[Dict[str, Any]]]:
        try:
            job_embedding = [0.1] * 768

            filtered_candidates = self._vector_search_query(
                job_embedding, 
                max_results, 
                end_client_name,
                prime_vendor_name
            )

            result = {
                "all_matches": filtered_candidates,
                "ex_end_client": [],
                "ex_prime_vendor": []
            }

            if filtered_candidates:
                lower_end_client = end_client_name.lower() if end_client_name else ""
                lower_prime_vendor = prime_vendor_name.lower() if prime_vendor_name else ""

                for candidate in filtered_candidates:
                    past_companies = [c.lower() for c in candidate.get('past_companies', []) if c]
                    
                    is_ex_client = lower_end_client and lower_end_client in past_companies
                    is_ex_vendor = lower_prime_vendor and lower_prime_vendor in past_companies

                    if is_ex_client:
                        result['ex_end_client'].append(candidate)
                    if is_ex_vendor:
                        result['ex_prime_vendor'].append(candidate)

            self.logger.info(f"Found {len(filtered_candidates)} total matches. "
                           f"{len(result['ex_end_client'])} ex-client matches, "
                           f"{len(result['ex_prime_vendor'])} ex-vendor matches")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error finding ex-consultants: {e}", exc_info=True)
            return {"all_matches": [], "ex_end_client": [], "ex_prime_vendor": []}
