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
from config import Config, InvoiceProcessor, GmailConfig



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