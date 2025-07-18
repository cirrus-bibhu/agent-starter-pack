import requests
import re
import json
import logging
from urllib.parse import urlparse, parse_qs, unquote

# Configure logging to only show INFO and above
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class JobExtractor:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://api.scrapin.io/enrichment/jobs/details"
    
    def format_url(self, url):
        if 'linkedin.com/e/v2' in url:
            parsed = urlparse(url)
            params = parse_qs(parsed.query)
            if 'url' in params:
                url = unquote(params['url'][0]).replace('%2E', '.')
        
        job_id = re.search(r'(\d{10})', url)
        if job_id:
            return f"https://www.linkedin.com/jobs/view/{job_id.group(1)}"
        return url
    
    def extract(self, job_url):
        try:
            formatted_url = self.format_url(job_url)
            headers = {"X-API-KEY": self.api_key}
            params = {"url": formatted_url}
            
            response = requests.get(self.base_url, headers=headers, params=params)
            response.raise_for_status()
            
            json_response = response.json()
            
            # Check for successful response and return the job data
            if json_response.get('success') and 'job' in json_response:
                return json_response['job']
            else:
                raise Exception(f"API returned unsuccessful response: {json_response}")
                
        except Exception as e:
            logger.error(f"Error extracting job details: {str(e)}")
            raise Exception(f"Error extracting job details: {str(e)}")

class ProfileExtractor:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://api.scrapin.io/enrichment/profile"
    
    def format_url(self, url):
        if 'linkedin.com/e/v2' in url:
            parsed = urlparse(url)
            params = parse_qs(parsed.query)
            if 'url' in params:
                url = unquote(params['url'][0]).replace('%2E', '.')
        
        profile_match = re.search(r'linkedin\.com/in/([^/?]+)', url)
        if profile_match:
            username = profile_match.group(1)
            return f"https://www.linkedin.com/in/{username}"
        return url
    
    def extract(self, profile_url):
        formatted_url = self.format_url(profile_url)
        headers = {"X-API-KEY": self.api_key}
        params = {"linkedInUrl": formatted_url}
        response = requests.get(self.base_url, headers=headers, params=params)
        return response.json()
