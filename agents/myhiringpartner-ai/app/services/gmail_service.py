import os
import json
import logging
from datetime import datetime, timezone

# Google Cloud and OAuth libraries
from dotenv import load_dotenv
from google.cloud import secretmanager
import google.auth
from google.api_core import exceptions as gcloud_exceptions
from google_auth_oauthlib.flow import InstalledAppFlow

# --- Configuration ---
# Your credentials from the previous step
load_dotenv()
    
CLIENT_ID = os.getenv("GMAIL_CLIENT_ID")
CLIENT_SECRET = os.getenv("GMAIL_CLIENT_SECRET")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Scopes define the level of access you are requesting from the user.
SCOPES = [
    'https://www.googleapis.com/auth/gmail.readonly',
    'https://www.googleapis.com/auth/gmail.compose',  # Required for creating drafts
    'https://www.googleapis.com/auth/gmail.modify'    # Required for modifying Gmail data
]

# Prefixes for Secret Manager secret IDs
EMAIL_TOKEN_PREFIX = "email-token"
REFRESH_TOKEN_PREFIX = "refresh"
ACCESS_TOKEN_PREFIX = "access"

# Custom Exception for clarity
class SecretManagerError(Exception):
    """Custom exception for Secret Manager operations."""
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

def _create_secret_if_not_exists(client: secretmanager.SecretManagerServiceClient, project_id: str, secret_id: str):
    """Creates a secret in Secret Manager if it doesn't already exist."""
    parent = f"projects/{project_id}"
    secret_path = f"{parent}/secrets/{secret_id}"
    try:
        client.get_secret(request={"name": secret_path})
    except gcloud_exceptions.NotFound:
        logger.info(f"Secret '{secret_id}' not found. Creating it.")
        client.create_secret(request={"parent": parent, "secret_id": secret_id, "secret": {"replication": {"automatic": {}}}})


def store_oauth_tokens_in_secret_manager(
    email: str,
    tokens: dict,
    service: str
) -> str:
    """Store OAuth tokens in Google Cloud Secret Manager based only on email.
    
    Args:
        email: User's email address.
        tokens: OAuth tokens including refresh_token and access_token.
        service: Service identifier (e.g., 'gmail').
        
    Returns:
        str: Name of the created refresh token version, or a message indicating none was created.
    """
    if not all(key in tokens for key in ['access_token', 'token_type', 'scope', 'expires_at']):
        raise ValueError("Missing required access token data")

    try:
        secret_client = secretmanager.SecretManagerServiceClient()
        project_id = _get_project_id()
        safe_email = _sanitize_id_components(email)[0]
        
        refresh_token_version_name = "N/A (No new refresh token was issued)"

        # --- Store Refresh Token (only if it was provided) ---
        if tokens.get('refresh_token'):
            refresh_token_secret_id = f"{EMAIL_TOKEN_PREFIX}_{REFRESH_TOKEN_PREFIX}_{safe_email}"
            refresh_token_path = f"projects/{project_id}/secrets/{refresh_token_secret_id}"
            
            _create_secret_if_not_exists(secret_client, project_id, refresh_token_secret_id)
            
            refresh_token_version = secret_client.add_secret_version(
                request={"parent": refresh_token_path, "payload": {"data": tokens['refresh_token'].encode("UTF-8")}}
            )
            logger.info(f"Stored new refresh token in: {refresh_token_secret_id}")
            refresh_token_version_name = refresh_token_version.name
        else:
            logger.warning(f"No refresh token provided for {email}. Skipping refresh token storage. This is expected if consent was previously granted.")

        # --- Store Access Token ---
        access_token_secret_id = f"{EMAIL_TOKEN_PREFIX}_{ACCESS_TOKEN_PREFIX}_{safe_email}"
        access_token_path = f"projects/{project_id}/secrets/{access_token_secret_id}"
        
        access_token_data = {
            'access_token': tokens['access_token'],
            'token_type': tokens['token_type'],
            'scope': tokens['scope'],
            'expires_at': tokens['expires_at'].isoformat(),
            'service': service
        }
        
        _create_secret_if_not_exists(secret_client, project_id, access_token_secret_id)
        
        secret_client.add_secret_version(
            request={"parent": access_token_path, "payload": {"data": json.dumps(access_token_data).encode("UTF-8")}}
        )
        logger.info(f"Stored access token metadata in: {access_token_secret_id}")
        logger.info(f"Successfully stored OAuth tokens for {email} in Secret Manager.")
        return refresh_token_version_name

    except Exception as e:
        logger.error(f"Error storing tokens in Secret Manager: {str(e)}")
        raise SecretManagerError(f"Failed to store OAuth tokens: {str(e)}")


# --- SIMPLIFIED Main Execution Logic ---

def main():
    """
    Main function to run the OAuth flow and store the credentials.
    """
    client_config = {
        "installed": {
            "client_id": CLIENT_ID,
            "client_secret": CLIENT_SECRET,
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "redirect_uris": ["http://localhost:8080"]
        }
    }
    
    # In your main() function
    try:
        flow = InstalledAppFlow.from_client_config(client_config, SCOPES)
        # Adding prompt='consent' forces the user to re-approve access,
        # which results in a new refresh token being issued.
        credentials = flow.run_local_server(port=8080, prompt='consent')
    except Exception as e:
        logger.error(f"An error occurred during the OAuth flow: {e}")
        return

    token_data = {
        "refresh_token": credentials.refresh_token,
        "access_token": credentials.token,
        "token_type": "Bearer",
        "scope": list(credentials.scopes),
        "expires_at": credentials.expiry
    }

    # SIMPLIFIED: Only ask for the email address
    email_address = input("Enter the email address you just authenticated with: ")
    if not email_address:
        logger.error("Email address cannot be empty.")
        return
        
    try:
        # SIMPLIFIED: No longer passing company_id
        secret_version_name = store_oauth_tokens_in_secret_manager(
            email=email_address,
            tokens=token_data,
            service='gmail'
        )
        logger.info(f"SUCCESS! Tokens stored. Refresh token version name: {secret_version_name}")
    except (ValueError, SecretManagerError) as e:
        logger.error(f"Failed to store tokens: {e}")

if __name__ == "__main__":
    main()
