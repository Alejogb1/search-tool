import sys
import os
import yaml
from google_auth_oauthlib.flow import InstalledAppFlow

def main():
    # Load credentials from google-ads.yaml
    with open('./google-ads.yaml', 'r') as f:
        config = yaml.safe_load(f)

    client_id = config['client_id']
    client_secret = config['client_secret']
    scopes = ['https://www.googleapis.com/auth/adwords']

    # Create a client secrets dictionary
    client_secrets = {
        'installed': {
            'client_id': client_id,
            'client_secret': client_secret,
            'auth_uri': 'https://accounts.google.com/o/oauth2/auth',
            'token_uri': 'https://oauth2.googleapis.com/token',
        }
    }

    flow = InstalledAppFlow.from_client_config(
        client_config=client_secrets,
        scopes=scopes
    )
    flow.redirect_uri = 'http://localhost:8080/'
    
    credentials = flow.run_local_server(
        port=8080,
        authorization_prompt_message="Please visit this URL to authorize this application: {url}",
        access_type='offline',
        prompt='consent'
    )
    
    # Print the refresh token
    print("Copy the following refresh token to your google-ads.yaml file:")
    print(credentials.refresh_token)

if __name__ == '__main__':
    main()
