from google.cloud import bigquery
from google.oauth2 import service_account
import pandas as pd
import os

class BigQueryClient:
    def __init__(self, credentials_path=None, project_id=None):
        """Initialize BigQuery client with optional credentials."""
        self.credentials_path = credentials_path or os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
        self.project_id = project_id or os.getenv('GOOGLE_CLOUD_PROJECT')
        self.client = self._authenticate()

    def _authenticate(self):
        """Authenticate with BigQuery using service account credentials."""
        if self.credentials_path:
            credentials = service_account.Credentials.from_service_account_file(
                self.credentials_path,
                scopes=["https://www.googleapis.com/auth/cloud-platform"]
            )
            return bigquery.Client(credentials=credentials, project=self.project_id)
        return bigquery.Client(project=self.project_id)

    def query_to_dataframe(self, query):
        """Execute a BigQuery SQL query and return results as pandas DataFrame."""
        query_job = self.client.query(query)
        return query_job.to_dataframe()

    def get_github_repos(self, language=None, min_stars=100):
        """Get GitHub repository data from public dataset."""
        base_query = """
            SELECT 
                repo_name, 
                watch_count as stars,
                fork_count as forks,
                language
            FROM `bigquery-public-data.github_repos.repos`
            {where_clause}
            LIMIT 1000
        """
        
        where_clause = ""
        if language:
            where_clause = f"WHERE language = '{language}' AND watch_count >= {min_stars}"
        elif min_stars:
            where_clause = f"WHERE watch_count >= {min_stars}"
            
        return self.query_to_dataframe(base_query.format(where_clause=where_clause))
