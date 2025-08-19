import os
import json
import logging
from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv

from google.cloud import secretmanager
import google.auth
from google.api_core import exceptions as gcloud_exceptions
from googleapiclient.discovery import build
from google.auth.transport.requests import Request
import google.auth.credentials

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

EMAIL_TOKEN_PREFIX = "email-token"
REFRESH_TOKEN_PREFIX = "refresh"
ACCESS_TOKEN_PREFIX = "access"

class SecretManagerError(Exception):
    pass

class GmailError(Exception):
    pass

def _get_project_id() -> str:
    project_id = os.environ.get("GOOGLE_CLOUD_PROJECT")
    if not project_id:
        try:
            _, project_id = google.auth.default()
        except google.auth.exceptions.DefaultCredentialsError:
            raise EnvironmentError("GOOGLE_CLOUD_PROJECT env var not set or gcloud auth not configured.")
    return project_id

def _sanitize_id_components(*args) -> tuple:
    sanitized = []
    for component in args:
        sanitized.append(component.replace('@', '-at-').replace('.', '-dot-'))
    return tuple(sanitized)

def get_latest_secret_version(client: secretmanager.SecretManagerServiceClient, project_id: str, secret_id: str) -> str:
    try:
        secret_path = f"projects/{project_id}/secrets/{secret_id}/versions/latest"
        response = client.access_secret_version(request={"name": secret_path})
        return response.payload.data.decode("UTF-8")
    except gcloud_exceptions.NotFound:
        raise SecretManagerError(f"Secret '{secret_id}' not found in Secret Manager")
    except Exception as e:
        raise SecretManagerError(f"Error retrieving secret '{secret_id}': {str(e)}")

def retrieve_oauth_tokens_from_secret_manager(email: str) -> tuple:
    try:
        secret_client = secretmanager.SecretManagerServiceClient()
        project_id = _get_project_id()
        safe_email = _sanitize_id_components(email)[0]
        
        refresh_token_secret_id = f"{EMAIL_TOKEN_PREFIX}_{REFRESH_TOKEN_PREFIX}_{safe_email}"
        refresh_token = get_latest_secret_version(secret_client, project_id, refresh_token_secret_id)
        
        access_token_secret_id = f"{EMAIL_TOKEN_PREFIX}_{ACCESS_TOKEN_PREFIX}_{safe_email}"
        access_token_json = get_latest_secret_version(secret_client, project_id, access_token_secret_id)
        access_token_data = json.loads(access_token_json)
        
        return refresh_token, access_token_data
    except Exception as e:
        raise SecretManagerError(f"Failed to retrieve OAuth tokens: {str(e)}")

class SecretManagerCredentials(google.auth.credentials.Credentials):
    def __init__(self, refresh_token: str, access_token_data: dict, client_id: str, client_secret: str):
        super().__init__()
        self.refresh_token = refresh_token
        self.token = access_token_data['access_token']
        self.client_id = client_id
        self.client_secret = client_secret
        self.token_uri = "https://oauth2.googleapis.com/token"
        
        if access_token_data.get('expires_at'):
            expiry_str = access_token_data['expires_at']
            try:
                if expiry_str.endswith('Z'):
                    expiry_str = expiry_str.replace('Z', '+00:00')
                self.expiry = datetime.fromisoformat(expiry_str)
                if self.expiry.tzinfo is None:
                    self.expiry = self.expiry.replace(tzinfo=timezone.utc)
            except ValueError:
                self.expiry = None
        else:
            self.expiry = None
        self.scopes = access_token_data.get('scope', [])
    
    def refresh(self, request):
        import requests
        data = {
            'grant_type': 'refresh_token',
            'refresh_token': self.refresh_token,
            'client_id': self.client_id,
            'client_secret': self.client_secret,
        }
        response = requests.post(self.token_uri, data=data)
        response.raise_for_status()
        token_data = response.json()
        self.token = token_data['access_token']
        if 'expires_in' in token_data:
            self.expiry = datetime.now(timezone.utc) + timedelta(seconds=token_data['expires_in'])
    
    @property
    def expired(self):
        if not self.expiry:
            return False
        return datetime.now(timezone.utc) >= self.expiry

def create_gmail_service(email: str, client_id: str, client_secret: str):
    try:
        refresh_token, access_token_data = retrieve_oauth_tokens_from_secret_manager(email)
        credentials = SecretManagerCredentials(
            refresh_token=refresh_token,
            access_token_data=access_token_data,
            client_id=client_id,
            client_secret=client_secret
        )
        if credentials.expired:
            credentials.refresh(Request())
        service = build('gmail', 'v1', credentials=credentials)
        return service
    except Exception as e:
        raise GmailError(f"Failed to create Gmail service: {str(e)}")


def main():
    load_dotenv()
    
    CLIENT_ID = os.getenv("GMAIL_CLIENT_ID")
    CLIENT_SECRET = os.getenv("GMAIL_CLIENT_SECRET")
    
    if not CLIENT_ID or not CLIENT_SECRET:
        logger.error("Missing environment variables: GMAIL_CLIENT_ID and GMAIL_CLIENT_SECRET must be set.")
        return

    try:
        email_address = 'support@myhiringpartner.ai'
        
        service = create_gmail_service(email_address, CLIENT_ID, CLIENT_SECRET)
        
        results = service.users().messages().list(userId='me', labelIds=['INBOX'], maxResults=15).execute()
        messages = results.get('messages', [])

        if not messages:
            print("No messages found in your inbox.")
        else:
            print("Fetching recent emails from your inbox...\n")
            for message in messages:
                msg = service.users().messages().get(userId='me', id=message['id'], format='metadata', metadataHeaders=['Subject']).execute()
                headers = msg['payload']['headers']
                subject = next((header['value'] for header in headers if header['name'] == 'Subject'), 'No Subject')
                
                print(f"Subject: {subject}")
                print(f"Message ID: {msg['id']}")
                print("---")

    except (SecretManagerError, GmailError, Exception) as e:
        print(f"\n❌ An error occurred: {e}")

if __name__ == "__main__":
    main()
