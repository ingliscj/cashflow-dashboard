import os
import tempfile
import base64
from dotenv import load_dotenv
from oauth2client.service_account import ServiceAccountCredentials
import gspread
from anthropic import Anthropic
import json
import re
from decimal import Decimal
from typing import Optional, Dict, List
from datetime import datetime
import pandas as pd
import numpy as np
import traceback
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from apscheduler.schedulers.background import BackgroundScheduler
import pickle

# For PDF processing
from PyPDF2 import PdfReader

# Only load .env file in development
if not os.getenv('RENDER'):
    print("Loading .env file for local development")
    load_dotenv()
else:
    print("Running on Render, using environment variables")



class Config:
# Only load .env file in development
    if not os.getenv('RENDER'):
        print("Loading .env file for local development")
        load_dotenv()
    else:
        print("Running on Render, using environment variables")
    # API Keys and Tokens
    ANTHROPIC_API_KEY = os.getenv('API_KEY')  
    SLACK_BOT_TOKEN = os.getenv('SLACK_BOT_TOKEN')
    
    # Google Sheet Configuration
    GOOGLE_SHEET_NAME = os.getenv('GOOGLE_SHEET_NAME')
    GOOGLE_SHEET_KEY = os.getenv('GOOGLE_SHEET_KEY')

    
    # File paths and directories
    BASE_TEMP_DIR = tempfile.gettempdir()
    APP_DIR = os.path.join(BASE_TEMP_DIR, 'cashflow-app')
    LOCAL_FOLDER = os.getenv('LOCAL_FOLDER', APP_DIR)
    PROCESSED_FILES_LOG = os.getenv('PROCESSED_FILES_LOG', os.path.join(APP_DIR, 'processed.json'))
    
    # Handle Google credentials
    GOOGLE_CREDENTIALS_JSON = os.getenv('GOOGLE_CREDENTIALS_FILE')  # Changed to match CodeSpaces secret name
    GOOGLE_CREDENTIALS_FILE = os.path.join(APP_DIR, 'google_credentials.json')
    
    # Set up Google credentials file
    try:
        if GOOGLE_CREDENTIALS_JSON:
            print("Found Google credentials in environment")
            credentials_content = base64.b64decode(GOOGLE_CREDENTIALS_JSON).decode('utf-8')
            
            # Ensure APP_DIR exists
            os.makedirs(APP_DIR, exist_ok=True)
            
            # Write credentials to file
            with open(GOOGLE_CREDENTIALS_FILE, 'w') as f:
                f.write(credentials_content)
            print(f"Successfully wrote credentials to {GOOGLE_CREDENTIALS_FILE}")
        else:
            print("Warning: No Google credentials found in environment")
            GOOGLE_CREDENTIALS_FILE = None
    except Exception as e:
        print(f"Error setting up Google credentials: {e}")
        GOOGLE_CREDENTIALS_FILE = None

    # Initialize Anthropic client configuration
    if not ANTHROPIC_API_KEY:
        print("Warning: No Anthropic API key found in environment")

    # Validate required configurations
    @classmethod
    def validate_config(cls):
        missing_vars = []
        required_vars = {
            'API_KEY': cls.ANTHROPIC_API_KEY,
            'SLACK_BOT_TOKEN': cls.SLACK_BOT_TOKEN,
            'GOOGLE_CREDENTIALS_FILE': cls.GOOGLE_CREDENTIALS_JSON,
            'GOOGLE_SHEET_NAME': cls.GOOGLE_SHEET_NAME,
            'GOOGLE_SHEET_KEY': cls.GOOGLE_SHEET_KEY
        }
        
        for var_name, var_value in required_vars.items():
            if not var_value:
                missing_vars.append(var_name)
        
        if missing_vars:
            print(f"Missing required environment variables: {', '.join(missing_vars)}")
            return False
        return True
  

    
    SUPPORTED_CURRENCIES = ['AED', 'USD', 'GBP', 'EUR']
    DEFAULT_CURRENCY = 'AED'


    # Add Ziina entity validation
    ZIINA_ENTITIES = {
        "ZIINA FZ LLC": ["ZIINA FZ LLC", "ZIINA FZ-LLC", "ZIINA FZ", "ZIINA FZ L.L.C"],
        "ZIINA LTD": ["ZIINA LTD", "ZIINA LIMITED", "ZIINA LTD."],
        "ZIINA INC": ["ZIINA INC", "ZIINA INC.", "ZIINA INCORPORATED"],
        "ZIINA PAYMENT LLC": ["ZIINA PAYMENT LLC", "ZIINA PAYMENT", "ZIINA PAYMENTS LLC", "ZIINA PAYMENTS"]
    }
 
        # Currency settings
    SUPPORTED_CURRENCIES = ['AED', 'USD', 'GBP', 'EUR']
    DEFAULT_CURRENCY = 'AED'
    CURRENCIES = {
        'AED': {
            'symbols': ['AED', 'Dirham', 'د.إ'],
            'decimal_places': 2
        },
        'USD': {
            'symbols': ['USD', '$', 'Dollar'],
            'decimal_places': 2
        },
        'GBP': {
            'symbols': ['GBP', '£', 'Pound'],
            'decimal_places': 2
        },
        'EUR': {
            'symbols': ['EUR', '€', 'Euro'],
            'decimal_places': 2
        }
    }
    

     # Define all sheet columns in order
    SHEET_HEADERS = [
        "Invoice Number",
        "Invoice Date",
        "Due Date",
        "Payment Terms",
        "Account Name",
        "Account TRN",
        "Account Address",
        "Account City",
        "Account Country",
        "Billing Company Name",
        "Billing Address",
        "Billing City",
        "Billing Country",
        "Billing TRN",
        "Subtotal Amount",
        "Subtotal Currency",
        "Tax Amount",
        "Tax Currency",
        "Tax Rate",
        "Total Amount",
        "Total Currency",
        "Currency Conversion Original Amount",
        "Currency Conversion Original Currency",
        "Currency Conversion Rate",
        "Bank Name",
        "Bank Branch",
        "Bank Address",
        "Bank Swift Code",
        "Bank Account Name",
        "AED IBAN",
        "USD IBAN",
        "GBP IBAN",
        "EUR IBAN",
        "Reference Number",
        "Line Items Detail",
        "Raw Line Items"  # JSON string of all line items
    ]


    # Exchange rate patterns
    EXCHANGE_PATTERNS = [
        r'(\d+(?:,\d{3})*(?:\.\d{2})?)\s*([A-Z]{3})\s*[@at]*\s*(?:rate)?\s*(\d+(?:\.\d+)?)',
        r'converted.*?(\d+(?:,\d{3})*(?:\.\d{2})?)\s*([A-Z]{3})',
        r'(\d+(?:,\d{3})*(?:\.\d{2})?)\s*([A-Z]{3})\s*\*\s*(\d+(?:\.\d+)?)'
    ]

    OPTIMIZATION_PERIODS = {
        "weekly": 7,
        "monthly": 30,
        "quarterly": 90
    }

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


class InvoiceProcessor:
    def __init__(self):
        try:
            print("Initializing InvoiceProcessor...")
            print(f"Config.GOOGLE_CREDENTIALS_FILE = {Config.GOOGLE_CREDENTIALS_FILE}")
            print(f"Config.GOOGLE_CREDENTIALS_JSON exists: {bool(Config.GOOGLE_CREDENTIALS_JSON)}")
            print(f"Config.APP_DIR = {Config.APP_DIR}")
            
            scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
            
            if not Config.GOOGLE_CREDENTIALS_FILE:
                raise ValueError("Google credentials file path is None")
                
            if not os.path.exists(Config.GOOGLE_CREDENTIALS_FILE):
                raise FileNotFoundError(f"Credentials file not found at: {Config.GOOGLE_CREDENTIALS_FILE}")
                
            creds = ServiceAccountCredentials.from_json_keyfile_name(Config.GOOGLE_CREDENTIALS_FILE, scope)
            self.google_client = gspread.authorize(creds)
            self.sheet = self.google_client.open(Config.GOOGLE_SHEET_NAME).sheet1
            self.processed_invoices = self.load_processed_invoices()
            self.config = Config()
            self.claude_client = Anthropic(api_key=Config.ANTHROPIC_API_KEY)
        
        except Exception as e:
            print(f"Error initializing InvoiceProcessor: {str(e)}")
            print(f"Current working directory: {os.getcwd()}")
            raise


    def detect_currency(self, text: str, amount: str) -> tuple:
        """
        Detect currency from text and amount
        Returns (amount, currency, exchange_rate, original_currency)
        """
        # Check for explicit exchange rates
        for pattern in Config.EXCHANGE_PATTERNS:
            match = re.search(pattern, text)
            if match:
                original_amount = match.group(1)
                original_currency = match.group(2)
                rate = match.group(3) if len(match.groups()) > 2 else None
                return original_amount, original_currency, rate, None

        # Check for currency symbols/codes
        amount_str = str(amount).upper()
        for currency, info in Config.CURRENCIES.items():
            for symbol in info['symbols']:
                if symbol.upper() in amount_str:
                    cleaned_amount = re.sub(r'[^\d.]', '', amount_str)
                    return cleaned_amount, currency, None, None

        # Default to AED if no currency found
        return re.sub(r'[^\d.]', '', str(amount)), 'AED', None, None

    def format_amount(self, amount_str: str, currency: str = None, text_context: str = "") -> str:
        """Format amount with proper currency and handling of conversions"""
        if not amount_str:
            return f"0.00 {currency or 'AED'}"

        try:
            # Detect currency and any conversions
            amount, detected_currency, exchange_rate, original_currency = self.detect_currency(text_context, amount_str)
            
            # Format the decimal amount
            decimal_amount = Decimal(str(amount))
            currency_info = Config.CURRENCIES.get(detected_currency, Config.CURRENCIES['AED'])
            formatted_amount = f"{decimal_amount:,.{currency_info['decimal_places']}f}"
            
            # If there's a conversion, include it
            if exchange_rate and original_currency:
                return f"{formatted_amount} {detected_currency} (converted from {original_currency} at rate {exchange_rate})"
            
            return f"{formatted_amount} {detected_currency}"
            
        except Exception as e:
            print(f"Error formatting amount {amount_str}: {e}")
            return f"0.00 {currency or 'AED'}"

    def load_processed_invoices(self) -> List[str]:
        """Load list of processed invoice numbers"""
        if os.path.exists(Config.PROCESSED_FILES_LOG):
            try:
                with open(Config.PROCESSED_FILES_LOG, "r") as f:
                    return json.load(f)
            except json.JSONDecodeError:
                print(f"Error reading {Config.PROCESSED_FILES_LOG}. Starting fresh.")
                return []
        return []
        
    def save_processed_invoice(self, invoice_number: str):
        """Save invoice number to processed list"""
        if invoice_number not in self.processed_invoices:
            self.processed_invoices.append(invoice_number)
            with open(Config.PROCESSED_FILES_LOG, "w") as f:
                json.dump(self.processed_invoices, f)

    def is_duplicate(self, invoice_number: str, values: list) -> bool:
        """Check if invoice is already processed or exists in sheet"""
        if invoice_number in self.processed_invoices:
            print(f"Invoice {invoice_number} found in processed files log. Skipping.")
            return True

        existing_rows = self.sheet.get_all_values()
        for row in existing_rows:
            if row and (row[0] == invoice_number or row[:len(values)] == values):
                print(f"Invoice {invoice_number} already exists in sheet. Skipping.")
                return True
        return False        

    def normalize_ziina_entity(self, name: str) -> str:
        """Normalize Ziina entity names to standard format"""
        if not name:
            return ""
            
        # Convert to uppercase for comparison
        name_upper = name.upper().strip()
        
        # Check against all variations
        for standard_name, variations in Config.ZIINA_ENTITIES.items():
            if any(var.upper() in name_upper for var in variations):
                return standard_name
                
        # If no match found, try partial matches
        for standard_name, variations in Config.ZIINA_ENTITIES.items():
            if "ZIINA" in name_upper:
                # Log potential new variation
                print(f"Warning: Found new Ziina entity variation: {name}")
                # Return closest match based on 'FZ', 'LTD', 'INC', or 'PAYMENT' keywords
                if "FZ" in name_upper or "FREE ZONE" in name_upper:
                    return "ZIINA FZ LLC"
                elif "LTD" in name_upper or "LIMITED" in name_upper:
                    return "ZIINA LTD"
                elif "INC" in name_upper:
                    return "ZIINA INC"
                elif "PAYMENT" in name_upper:
                    return "ZIINA PAYMENT LLC"
                
        # If still no match, log warning and return original
        print(f"Warning: Unrecognized entity name: {name}")
        return name

    def validate_invoice_data(self, json_data: Dict) -> Dict:
        """Validate and normalize invoice data"""
        # Deep copy to avoid modifying original
        data = json_data.copy()
        
        # Normalize Account Name if it's a Ziina entity
        if "Client" in data:
            client_name = data["Client"].get("Name", "")
            normalized_name = self.normalize_ziina_entity(client_name)
            if normalized_name != client_name:
                print(f"Normalized client name from '{client_name}' to '{normalized_name}'")
                data["Client"]["Name"] = normalized_name
                
        # Also check Account Name field if present
        if "Account" in data:
            account_name = data["Account"].get("Name", "")
            normalized_name = self.normalize_ziina_entity(account_name)
            if normalized_name != account_name:
                print(f"Normalized account name from '{account_name}' to '{normalized_name}'")
                data["Account"]["Name"] = normalized_name

        return data

    def send_to_claude(self, text: str) -> Optional[Dict]:
        """Send text to Claude with multi-currency support"""
        schema = {
            "Invoice Number": "",
            "Invoice Date": "",
            "Due Date": "",
            "Payment Terms": "Due On Receipt",
            "Account Name": "",
            "TRN": "",
            "Billing Contact": {
                "name": "",
                "address": "",
                "city": "",
                "country": "",
                "trn": ""
            },
            "Line Items": [
                {
                    "description": "",
                    "quantity": 0.0,
                    "unit_price": {
                        "amount": "",
                        "currency": "",
                        "conversion": {
                            "original_amount": "",
                            "original_currency": "",
                            "rate": ""
                        }
                    },
                    "tax_rate": "",
                    "amount": {
                        "value": "",
                        "currency": ""
                    }
                }
            ],
            "Amounts": {
                "subtotal": {
                    "value": "",
                    "currency": ""
                },
                "tax": {
                    "value": "",
                    "currency": ""
                },
                "total": {
                    "value": "",
                    "currency": ""
                },
                "conversion": {
                    "original_amount": "",
                    "original_currency": "",
                    "rate": "",
                    "target_currency": ""
                }
            },
            "Bank Details": {
                "account_name": "",
                "bank_name": "",
                "branch_address": "",
                "swift_code": "",
                "accounts": [
                    {
                        "currency": "",
                        "iban": ""
                    }
                ]
            }
        }

        system_prompt = """You are a specialized invoice parsing assistant. Follow these rules:
        1. The account must be one of these Ziina entities:
           - Ziina FZ LLC
           - Ziina Ltd
           - Ziina Inc
           - Ziina Payment LLC
        2. Detect and preserve all currency information
        3. Capture any currency conversion rates or calculations
        4. Extract ALL identification numbers (Invoice, TRN, Reference)
        5. Preserve exact formatting of numbers and dates
        6. Include all address components and tax numbers
        """

        try:
            response = self.claude_client.messages.create(
                model="claude-3-opus-20240229",
                max_tokens=4096,
                system=system_prompt,
                messages=[
                    {
                        "role": "user",
                        "content": (
                            "Extract data from this invoice using the exact schema below. "
                            "Preserve original currencies and amounts. "
                            "Ensure Ziina entity names are normalized to standard format. "
                            f"Schema:\n{json.dumps(schema, indent=2)}\n\n"
                            f"Invoice text:\n{text}"
                        )
                    }
                ]
            )
            
            json_data = json.loads(response.content[0].text.strip())
            
            # Normalize the Ziina entity name
            if "Account Name" in json_data:
                json_data["Account Name"] = self.normalize_ziina_entity(json_data["Account Name"])
            
            return json_data
            
        except Exception as e:
            print(f"Error from Claude: {e}")
            return None

    def format_for_sheets(self, json_data: Dict) -> List[str]:
        """Format JSON data for sheets with all available fields"""
        try:
            # Extract billing contact details
            billing = json_data.get("Billing Contact", {})
            
            # Get amounts with currency information
            amounts = json_data.get("Amounts", {})
            
            # Get currency conversion details
            conversion = amounts.get("conversion", {})
            
            # Get bank details
            bank = json_data.get("Bank Details", {})
            bank_accounts = bank.get("accounts", [])
            
            # Get IBANs by currency
            def get_iban_for_currency(currency):
                account = next(
                    (acc for acc in bank_accounts if acc.get("currency") == currency),
                    {}
                )
                return account.get("iban", "")

            # Format line items with full detail
            line_items = json_data.get("Line Items", [])
            line_item_details = []
            for item in line_items:
                unit_price = item.get("unit_price", {})
                price_str = self.format_amount(
                    unit_price.get("amount", ""),
                    unit_price.get("currency", ""),
                    json_data.get("raw_text", "")
                )
                
                line_detail = (
                    f"Description: {item.get('description', '')}\n"
                    f"Quantity: {item.get('quantity', '')}\n"
                    f"Unit Price: {price_str}\n"
                    f"Tax Rate: {item.get('tax_rate', '')}\n"
                    f"Amount: {self.format_amount(item.get('amount', {}).get('value', ''), item.get('amount', {}).get('currency', ''))}"
                )
                line_item_details.append(line_detail)

            # Create row with all available data
            return [
                json_data.get("Invoice Number", ""),
                json_data.get("Invoice Date", ""),
                json_data.get("Due Date", ""),
                json_data.get("Payment Terms", ""),
                json_data.get("Account Name", ""),
                json_data.get("TRN", ""),
                json_data.get("Account", {}).get("Address", {}).get("Lines", [""])[0],
                json_data.get("Account", {}).get("Address", {}).get("City", ""),
                json_data.get("Account", {}).get("Address", {}).get("Country", ""),
                billing.get("name", ""),
                billing.get("address", ""),
                billing.get("city", ""),
                billing.get("country", ""),
                billing.get("trn", ""),
                amounts.get("subtotal", {}).get("value", ""),
                amounts.get("subtotal", {}).get("currency", ""),
                amounts.get("tax", {}).get("value", ""),
                amounts.get("tax", {}).get("currency", ""),
                json_data.get("tax_rate", ""),
                amounts.get("total", {}).get("value", ""),
                amounts.get("total", {}).get("currency", ""),
                conversion.get("original_amount", ""),
                conversion.get("original_currency", ""),
                conversion.get("rate", ""),
                bank.get("bank_name", ""),
                bank.get("branch", ""),
                bank.get("branch_address", ""),
                bank.get("swift_code", ""),
                bank.get("account_name", ""),
                get_iban_for_currency("AED"),
                get_iban_for_currency("USD"),
                get_iban_for_currency("GBP"),
                get_iban_for_currency("EUR"),
                json_data.get("Reference", ""),
                "\n\n".join(line_item_details),
                json.dumps(line_items, indent=2)  # Store raw JSON for future reference
            ]

        except Exception as e:
            print(f"Error formatting sheet data: {e}")
            print("JSON data:", json.dumps(json_data, indent=2))
            raise

    def validate_headers(self):
        """Validate sheet headers match expected format"""
        try:
            current_headers = self.sheet.row_values(1)
            if current_headers != Config.SHEET_HEADERS:
                print("\nWarning: Sheet headers don't match expected format")
                print("\nExpected headers:")
                for h in Config.SHEET_HEADERS:
                    print(f"- {h}")
                print("\nCurrent headers:")
                for h in current_headers:
                    print(f"- {h}")
                
                # Option to update headers
                if not current_headers:  # Empty sheet
                    print("\nUpdating sheet headers...")
                    self.sheet.insert_row(Config.SHEET_HEADERS, 1)
                    print("Headers updated successfully!")
                else:
                    print("\nPlease manually update the sheet headers or create a new sheet.")
            else:
                print("Sheet headers validated successfully!")
        except Exception as e:
            print(f"Error validating headers: {e}")

    def create_new_sheet(self):
        """Create a new sheet with correct headers"""
        try:
            # Create new worksheet
            worksheet = self.google_client.create(f"Invoice Tracker {datetime.now().strftime('%Y-%m-%d')}")
            
            # Add headers
            worksheet.sheet1.insert_row(Config.SHEET_HEADERS, 1)
            
            # Format headers
            worksheet.sheet1.format('A1:AI1', {
                "backgroundColor": {"red": 0.9, "green": 0.9, "blue": 0.9},
                "textFormat": {"bold": True},
                "horizontalAlignment": "CENTER"
            })
            
            print(f"Created new sheet: {worksheet.title}")
            return worksheet.sheet1
            
        except Exception as e:
            print(f"Error creating new sheet: {e}")
            raise

    def process_pdf(self, file_path: str) -> bool:
        """Process PDF file and extract text"""
        try:
            reader = PdfReader(file_path)
            text = ""
            for page in reader.pages:
                text += page.extract_text()
            
            print(f"Extracted text length: {len(text)}")
            return self.process_invoice_text(text)
        except Exception as e:
            print(f"Error processing PDF {file_path}: {e}")
            return False

    def process_invoice_text(self, text: str) -> bool:
        """Process extracted text through Claude and add to sheet"""
        json_data = self.send_to_claude(text)
        if not json_data:
            return False

        sheet_data = self.format_for_sheets(json_data)
        return self.append_to_sheet(sheet_data)

    def append_to_sheet(self, values: List[str]) -> bool:
        """Append row to sheet with duplicate checking"""
        invoice_number = values[0]
        
        if self.is_duplicate(invoice_number, values):
            return False
            
        self.sheet.append_row(values)
        self.save_processed_invoice(invoice_number)
        print(f"Invoice {invoice_number} added to sheet successfully!")
        return True

    def test_entity_normalization():
        processor = InvoiceProcessor()
        test_cases = [
        "ZIINA FZ-LLC",
        "Ziina FZ LLC",
        "ZIINA FZ",
        "Ziina Payment LLC",
        "ZIINA PAYMENTS",
        "Ziina Ltd.",
        "ZIINA LIMITED",
        "Ziina Inc.",
        "Unknown Entity"
        ]
    
        print("\nTesting Ziina entity normalization:")
        for test in test_cases:
            normalized = processor.normalize_ziina_entity(test)
            print(f"Original: {test}")
            print(f"Normalized: {normalized}")
            print()

    def get_optimization_data(self) -> Dict:
        """Get data needed for optimization"""
        return {
            "invoice_data": self.format_for_optimization(),
            "entity_mapping": Config.ZIINA_ENTITIES
        }
    
    def format_for_optimization(self) -> List[Dict]:
        """Format invoice data for optimization"""
        # Get all processed invoices
        data = self.sheet.get_all_records()
        
        # Format for optimization
        formatted_data = []
        for row in data:
            formatted_data.append({
                "invoice_number": row["Invoice Number"],
                "due_date": row["Due Date"],
                "amount": self.extract_amount(row["Total Amount"]),
                "currency": self.extract_currency(row["Total Currency"]),
                "entity": row["Account Name"],
                "vendor": row["Billing Company Name"]
            })
            
        return formatted_data

    def extract_amount(self, amount_str: str) -> float:
        """Extract numeric amount from string"""
        try:
            return float(re.sub(r'[^\d.]', '', amount_str))
        except:
            return 0.0

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
