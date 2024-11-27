import os
import base64
import tempfile
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
import email
from apscheduler.schedulers.background import BackgroundScheduler
from app import InvoiceProcessor, Config
import pickle

class GmailConfig:
    # File paths and directories
    BASE_TEMP_DIR = tempfile.gettempdir()
    APP_DIR = os.path.join(BASE_TEMP_DIR, 'cashflow-app')
    
    # Gmail credentials handling
    GMAIL_CREDENTIALS_JSON = os.getenv('GMAIL_CREDENTIALS_FILE')
    GMAIL_CREDS_PATH = os.path.join(APP_DIR, 'gmail_credentials.json')
    TOKEN_PATH = os.path.join(APP_DIR, 'gmail_token.pickle')
    
    # Set up credentials file
    try:
        if GMAIL_CREDENTIALS_JSON:
            print("Found Gmail credentials in environment")
            credentials_content = base64.b64decode(GMAIL_CREDENTIALS_JSON).decode('utf-8')
            
            # Ensure APP_DIR exists
            os.makedirs(APP_DIR, exist_ok=True)
            
            # Write credentials to file
            with open(GMAIL_CREDS_PATH, 'w') as f:
                f.write(credentials_content)
            print(f"Successfully wrote Gmail credentials to {GMAIL_CREDS_PATH}")
        else:
            print("Warning: No Gmail credentials found in environment")
            GMAIL_CREDS_PATH = None
    except Exception as e:
        print(f"Error setting up Gmail credentials: {e}")
        GMAIL_CREDS_PATH = None

# email_processor.py
class EmailProcessor:
    def __init__(self):
        self.SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']
        self.invoice_processor = InvoiceProcessor()
        self.scheduler = None
        
        # Only set up Gmail service if credentials exist
        if GmailConfig.GMAIL_CREDS_PATH:
            try:
                self.service = self.setup_gmail_service()
                self.scheduler = BackgroundScheduler()
                self.setup_scheduler()
                print("Email processor initialized successfully")
            except Exception as e:
                print(f"Failed to initialize email processor: {e}")
        else:
            print("Email processor not initialized - missing Gmail credentials")


    def setup_gmail_service(self):
        creds = None
        
        # Load existing token if available
        if os.path.exists(GmailConfig.TOKEN_PATH):
            with open(GmailConfig.TOKEN_PATH, 'rb') as token:
                creds = pickle.load(token)

        # If no valid credentials available, let's get them
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                if not os.path.exists(GmailConfig.GMAIL_CREDS_PATH):
                    raise Exception("Gmail credentials file not found")
                    
                flow = InstalledAppFlow.from_client_secrets_file(
                    GmailConfig.GMAIL_CREDS_PATH, self.SCOPES)
                creds = flow.run_local_server(port=0)
                
            # Save the credentials for future use
            with open(GmailConfig.TOKEN_PATH, 'wb') as token:
                pickle.dump(creds, token)

        return build('gmail', 'v1', credentials=creds)


    def setup_scheduler(self):
        self.scheduler.add_job(self.check_new_emails, 'interval', minutes=15)
        self.scheduler.start()

    def check_new_emails(self):
        try:
            # Customize this query based on your needs
            query = "in:inbox is:unread has:attachment filename:pdf"
            print(f"Checking for new emails with query: {query}")
            
            results = self.service.users().messages().list(userId='me', q=query).execute()
            messages = results.get('messages', [])

            if messages:
                print(f"Found {len(messages)} new messages")
                for message in messages:
                    self.process_message(message)
            else:
                print("No new messages found")
                
        except Exception as e:
            print(f"Error processing emails: {e}")

    def process_message(self, message):
        try:
            msg = self.service.users().messages().get(userId='me', id=message['id']).execute()
            attachments = self.get_attachments(msg)
            
            for attachment in attachments:
                if attachment['filename'].lower().endswith('.pdf'):
                    print(f"Processing PDF: {attachment['filename']}")
                    with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_file:
                        temp_file.write(base64.urlsafe_b64decode(attachment['data']))
                        success = self.invoice_processor.process_pdf(temp_file.name)
                        
                        if success:
                            self.mark_as_read(message['id'])
                            print(f"Successfully processed invoice from email: {attachment['filename']}")
                        else:
                            print(f"Failed to process invoice: {attachment['filename']}")
        except Exception as e:
            print(f"Error processing message: {e}")

    def get_attachments(self, message):
        attachments = []
        if 'parts' in message['payload']:
            for part in message['payload']['parts']:
                if part['filename'] and 'data' in part['body']:
                    attachment = {
                        'filename': part['filename'],
                        'data': part['body']['data']
                    }
                    attachments.append(attachment)
        return attachments

    def mark_as_read(self, message_id):
        self.service.users().messages().modify(
            userId='me',
            id=message_id,
            body={'removeLabelIds': ['UNREAD']}
        ).execute()

if __name__ == "__main__":
    # This allows you to run the email processor independently
    processor = EmailProcessor()
    try:
        print("Email processor started. Press Ctrl+C to stop.")
        while True:
            pass  # Keep the script running
    except KeyboardInterrupt:
        print("Shutting down email processor...")
        processor.scheduler.shutdown()