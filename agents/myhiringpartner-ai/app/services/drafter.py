import os
import json
import logging
import base64
from datetime import datetime, timezone, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dotenv import load_dotenv

# Google Cloud and Gmail libraries
from google.cloud import secretmanager
import google.auth
from google.api_core import exceptions as gcloud_exceptions
from googleapiclient.discovery import build
from google.auth.transport.requests import Request
import google.auth.credentials


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Prefixes for Secret Manager secret IDs (matching your storage script)
EMAIL_TOKEN_PREFIX = "email-token"
REFRESH_TOKEN_PREFIX = "refresh"
ACCESS_TOKEN_PREFIX = "access"


# Custom Exceptions
class SecretManagerError(Exception):
    """Custom exception for Secret Manager operations."""
    pass


class GmailError(Exception):
    """Custom exception for Gmail operations."""
    pass


# --- Helper Functions ---


def _get_project_id() -> str:
    """Gets the Google Cloud Project ID from the environment or default auth."""
    project_id = os.environ.get("GOOGLE_CLOUD_PROJECT")
    if not project_id:
        try:
            _, project_id = google.auth.default()
        except google.auth.exceptions.DefaultCredentialsError:
            raise EnvironmentError("GOOGLE_CLOUD_PROJECT env var not set or gcloud auth not configured.")
    return project_id


def _sanitize_id_components(*args) -> tuple:
    """Sanitizes strings to be used in Secret Manager secret IDs."""
    sanitized = []
    for component in args:
        # Replace characters not allowed in secret IDs with a hyphen
        sanitized.append(component.replace('@', '-at-').replace('.', '-dot-'))
    return tuple(sanitized)


# --- Token Retrieval Functions ---


def get_latest_secret_version(client: secretmanager.SecretManagerServiceClient, project_id: str, secret_id: str) -> str:
    """Retrieve the latest version of a secret from Secret Manager."""
    try:
        secret_path = f"projects/{project_id}/secrets/{secret_id}/versions/latest"
        response = client.access_secret_version(request={"name": secret_path})
        return response.payload.data.decode("UTF-8")
    except gcloud_exceptions.NotFound:
        raise SecretManagerError(f"Secret '{secret_id}' not found in Secret Manager")
    except Exception as e:
        raise SecretManagerError(f"Error retrieving secret '{secret_id}': {str(e)}")


def retrieve_oauth_tokens_from_secret_manager(email: str) -> tuple:
    """
    Retrieve OAuth tokens from Google Cloud Secret Manager.
    
    Args:
        email: User's email address.
        
    Returns:
        tuple: (refresh_token, access_token_data_dict)
    """
    try:
        secret_client = secretmanager.SecretManagerServiceClient()
        project_id = _get_project_id()
        safe_email = _sanitize_id_components(email)[0]
        
        # Retrieve refresh token
        refresh_token_secret_id = f"{EMAIL_TOKEN_PREFIX}_{REFRESH_TOKEN_PREFIX}_{safe_email}"
        refresh_token = get_latest_secret_version(secret_client, project_id, refresh_token_secret_id)
        
        # Retrieve access token data
        access_token_secret_id = f"{EMAIL_TOKEN_PREFIX}_{ACCESS_TOKEN_PREFIX}_{safe_email}"
        access_token_json = get_latest_secret_version(secret_client, project_id, access_token_secret_id)
        access_token_data = json.loads(access_token_json)
        
        logger.info(f"Successfully retrieved OAuth tokens for {email}")
        return refresh_token, access_token_data
        
    except Exception as e:
        logger.error(f"Error retrieving tokens from Secret Manager: {str(e)}")
        raise SecretManagerError(f"Failed to retrieve OAuth tokens: {str(e)}")


# --- Custom Credentials Class ---


class SecretManagerCredentials(google.auth.credentials.Credentials):
    """Custom credentials class that uses tokens from Secret Manager."""
    
    def __init__(self, refresh_token: str, access_token_data: dict, client_id: str, client_secret: str):
        super().__init__()
        self.refresh_token = refresh_token
        self.token = access_token_data['access_token']
        self.client_id = client_id
        self.client_secret = client_secret
        self.token_uri = "https://oauth2.googleapis.com/token"
        
        # Parse expiry from stored data
        if access_token_data.get('expires_at'):
            expiry_str = access_token_data['expires_at']
            # Handle different datetime formats and ensure timezone awareness
            try:
                if expiry_str.endswith('Z'):
                    expiry_str = expiry_str.replace('Z', '+00:00')
                
                self.expiry = datetime.fromisoformat(expiry_str)
                
                # Ensure timezone awareness
                if self.expiry.tzinfo is None:
                    self.expiry = self.expiry.replace(tzinfo=timezone.utc)
            except ValueError:
                # If parsing fails, set to None to force refresh
                logger.warning(f"Could not parse expiry time: {expiry_str}")
                self.expiry = None
        else:
            self.expiry = None
            
        self.scopes = access_token_data.get('scope', [])
    
    def refresh(self, request):
        """Refresh the access token using the refresh token."""
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
        
        # Update expiry if provided
        if 'expires_in' in token_data:
            self.expiry = datetime.now(timezone.utc) + timedelta(seconds=token_data['expires_in'])
    
    @property
    def expired(self):
        """Check if the token is expired."""
        if not self.expiry:
            return False
        return datetime.now(timezone.utc) >= self.expiry


# --- Gmail Functions ---


def create_gmail_service(email: str, client_id: str, client_secret: str):
    """Create Gmail service using credentials from Secret Manager."""
    try:
        # Retrieve tokens from Secret Manager
        refresh_token, access_token_data = retrieve_oauth_tokens_from_secret_manager(email)
        
        # Create custom credentials
        credentials = SecretManagerCredentials(
            refresh_token=refresh_token,
            access_token_data=access_token_data,
            client_id=client_id,
            client_secret=client_secret
        )
        
        # Refresh token if expired
        if credentials.expired:
            logger.info("Access token expired, refreshing...")
            credentials.refresh(Request())
        
        # Build Gmail service
        service = build('gmail', 'v1', credentials=credentials)
        logger.info("Gmail service created successfully")
        return service
        
    except Exception as e:
        logger.error(f"Error creating Gmail service: {str(e)}")
        raise GmailError(f"Failed to create Gmail service: {str(e)}")


def create_sample_email_draft(service, to_email: str = None, subject: str = None, body: str = None):
    """Create a sample email draft in Gmail."""
    try:
        # Default sample content
        if not to_email:
            to_email = "example@example.com"
        if not subject:
            subject = "Sample Email Draft - Test Message"
        if not body:
            body = """Hello,

This is a sample email draft created automatically using the Gmail API.

This message demonstrates:
- Successful OAuth authentication
- Token retrieval from Google Secret Manager
- Gmail API integration
- Draft creation functionality

Best regards,
Automated Email System

---
Generated on: {}
""".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

        # Create the email message
        message = MIMEMultipart()
        message['To'] = to_email
        message['Subject'] = subject
        message.attach(MIMEText(body, 'plain'))
        
        # Encode the message
        raw_message = base64.urlsafe_b64encode(message.as_bytes()).decode()
        
        # Create draft
        draft_body = {
            'message': {
                'raw': raw_message
            }
        }
        
        draft = service.users().drafts().create(userId='me', body=draft_body).execute()
        logger.info(f"Draft created successfully with ID: {draft['id']}")
        
        return draft
        
    except Exception as e:
        logger.error(f"Error creating email draft: {str(e)}")
        raise GmailError(f"Failed to create email draft: {str(e)}")


# --- Main Function ---


def main():
    """Main function to create a Gmail draft using credentials from Secret Manager."""

    load_dotenv()
    
    # Configuration - Load from environment variables
    CLIENT_ID = os.getenv("GMAIL_CLIENT_ID")
    CLIENT_SECRET = os.getenv("GMAIL_CLIENT_SECRET")

    # Validate that environment variables are set
    if not CLIENT_ID or not CLIENT_SECRET:
        raise ValueError("Missing environment variables: GMAIL_CLIENT_ID and GMAIL_CLIENT_SECRET must be set.")
    
    try:
        # Get email address
        email_address = 'bibhu@myhiringpartner.ai'
        if not email_address:
            logger.error("Email address cannot be empty.")
            return
        
        # Optional: Get custom draft content
        print("\n--- Optional: Customize your draft (press Enter to use defaults) ---")
        to_email = input("To email (default: example@example.com): ").strip()
        subject = input("Subject (default: Sample Email Draft): ").strip()
        body = input("Body (default: auto-generated): ").strip()
        
        # Use defaults if empty
        to_email = to_email if to_email else None
        subject = subject if subject else None
        body = body if body else None
        
        # Create Gmail service
        logger.info("Creating Gmail service...")
        service = create_gmail_service(email_address, CLIENT_ID, CLIENT_SECRET)
        
        # Create sample draft
        logger.info("Creating email draft...")
        draft = create_sample_email_draft(service, to_email, subject, body)
        
        print(f"\nSUCCESS! Email draft created successfully!")
        print(f"Draft ID: {draft['id']}")
        print(f"To: {to_email or 'example@example.com'}")
        print(f"Subject: {subject or 'Sample Email Draft - Test Message'}")
        print("\nYou can find this draft in your Gmail Drafts folder.")
        
    except (SecretManagerError, GmailError, ValueError) as e:
        logger.error(f"Application error: {e}")
        print(f"\n❌ Error: {e}")
    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user.")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        print(f"\n❌ Unexpected error: {e}")


if __name__ == "__main__":
    main()