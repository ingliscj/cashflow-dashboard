
import base64
import tempfile
from pathlib import Path
import json
import os
import requests
from flask import Flask, request, jsonify, send_file, send_from_directory
from PyPDF2 import PdfReader
from oauth2client.service_account import ServiceAccountCredentials
import gspread
from anthropic import Anthropic
import re
from decimal import Decimal
from typing import Optional, Dict, List
from datetime import datetime
import pandas as pd 

class Config:
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

class JSONEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle NumPy and Pandas types"""
    def default(self, obj):
        try:
            if isinstance(obj, (np.integer, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif pd.isna(obj):
                return None
            elif isinstance(obj, pd.Timestamp):
                return obj.strftime('%Y-%m-%d')
            return super(JSONEncoder, self).default(obj)
        except:
            return str(obj)


class CashflowOptimizer:
    def __init__(self):
        """Initialize with proper Google Sheets connection"""
        try:
            print("Initializing CashflowOptimizer...")

            # Set up Google Sheets authentication
            scope = [
                "https://spreadsheets.google.com/feeds",
                "https://www.googleapis.com/auth/drive",
                "https://www.googleapis.com/auth/spreadsheets"
            ]

            # Debug print credential file path
            print(f"Looking for credentials file at: {Config.GOOGLE_CREDENTIALS_FILE}")

            # Check if credentials file exists
            if not os.path.exists(Config.GOOGLE_CREDENTIALS_FILE):
                raise FileNotFoundError(f"Credentials file not found at: {Config.GOOGLE_CREDENTIALS_FILE}")

            # Initialize credentials
            credentials = ServiceAccountCredentials.from_json_keyfile_name(
                Config.GOOGLE_CREDENTIALS_FILE,
                scope
            )

            print("Credentials loaded successfully")

            # Initialize Google Sheets client
            self.google_client = gspread.authorize(credentials)
            print("Google Sheets client authorized")

            # Open the spreadsheet
            try:
                self.workbook = self.google_client.open(Config.GOOGLE_SHEET_NAME)
                print(f"Opened spreadsheet: {Config.GOOGLE_SHEET_NAME}")

                # Get the first sheet
                self.sheet = self.workbook.sheet1
                print("Accessed first sheet successfully")

                # Verify we can read data
                test_data = self.sheet.get_all_records()
                print(f"Successfully connected to sheet. Found {len(test_data)} rows")

            except gspread.exceptions.SpreadsheetNotFound:
                raise Exception(f"Spreadsheet '{Config.GOOGLE_SHEET_NAME}' not found")
            except gspread.exceptions.WorksheetNotFound:
                raise Exception("First worksheet not found")
            except Exception as e:
                raise Exception(f"Error accessing spreadsheet: {str(e)}")

        except Exception as e:
            print(f"Error initializing CashflowOptimizer: {str(e)}")
            raise

    def clean_amount(self, amount_str: str) -> float:
        """Clean amount string and convert to float"""
        try:
            if pd.isna(amount_str) or amount_str == '':
                return 0.0
            
            # Convert to string if not already
            amount_str = str(amount_str)
            
            # Remove currency symbols and text
            # This will handle "1,234.56 AED" or "AED 1,234.56" or "1.234,56 EUR"
            cleaned = re.sub(r'[^\d.,\-]', '', amount_str)
            
            if not cleaned:
                return 0.0
            
            # Handle different number formats
            if ',' in cleaned and '.' in cleaned:
                if cleaned.rindex(',') > cleaned.rindex('.'):
                    # European format (1.234,56)
                    cleaned = cleaned.replace('.', '').replace(',', '.')
                else:
                    # US format (1,234.56)
                    cleaned = cleaned.replace(',', '')
            elif ',' in cleaned:
                # If comma is used as decimal separator (1234,56)
                if len(cleaned.split(',')[1]) == 2:
                    cleaned = cleaned.replace(',', '.')
                else:
                    # If comma is used as thousand separator (1,234)
                    cleaned = cleaned.replace(',', '')
            
            return float(cleaned)
            
        except Exception as e:
            print(f"Error cleaning amount '{amount_str}': {e}")
            return 0.0

    def parse_date(self, date_str) -> Optional[pd.Timestamp]:
        """Parse dates with custom formats"""
        if pd.isna(date_str) or not str(date_str).strip():
            return None
            
        try:
            # If already a datetime
            if isinstance(date_str, (datetime, pd.Timestamp)):
                return pd.Timestamp(date_str)
                
            date_str = str(date_str).strip()
            
            # Handle "26th November" format
            if 'th' in date_str.lower() or 'st' in date_str.lower() or 'nd' in date_str.lower() or 'rd' in date_str.lower():
                date_str = re.sub(r'(?<=\d)(st|nd|rd|th)', '', date_str, flags=re.IGNORECASE)
                if len(date_str.split()) == 2:
                    date_str = f"{date_str} {datetime.now().year}"
            
            # Try specific formats
            formats = [
                '%d %B %Y',     # 26 November 2024
                '%d %b %Y',     # 26 Nov 2024
                '%d.%m.%Y',     # 15.11.2024
                '%d/%m/%Y',     # 15/11/2024
                '%d-%m-%Y',     # 15-11-2024
                '%Y-%m-%d',     # 2024-11-15
                '%d %B',        # 26 November
                '%d %b',        # 26 Nov
                '%B %d %Y',     # November 26 2024
                '%b %d %Y'      # Nov 26 2024
            ]
            
            for fmt in formats:
                try:
                    parsed_date = pd.to_datetime(date_str, format=fmt)
                    # If no year specified, use current year
                    if parsed_date.year == 1900:
                        parsed_date = parsed_date.replace(year=datetime.now().year)
                    return parsed_date
                except:
                    continue
            
            # Try pandas default parser as last resort
            return pd.to_datetime(date_str)
            
        except Exception as e:
            print(f"Error parsing date '{date_str}': {e}")
            return None

    def get_invoice_data(self) -> pd.DataFrame:
        """Get invoice data from Google Sheets"""
        try:
            data = self.sheet.get_all_records()
            df = pd.DataFrame(data)
            
            if df.empty:
                return df
            
            # Process date columns
            date_columns = ['Invoice Date', 'Due Date']
            for col in date_columns:
                if col in df.columns:
                    print(f"\nProcessing {col}")
                    print("Sample before:", df[col].head())
                    df[col] = df[col].apply(self.parse_date)
                    print("Sample after:", df[col].head())

            # Process amount columns
            amount_columns = ['Total Amount', 'Subtotal Amount', 'Tax Amount']
            for col in amount_columns:
                if col in df.columns:
                    print(f"\nProcessing {col}")
                    print("Sample before:", df[col].head())
                    df[col] = df[col].apply(self.clean_amount)
                    print("Sample after:", df[col].head())

            return df
            
        except Exception as e:
            print(f"Error getting invoice data: {e}")
            return pd.DataFrame()

    def generate_optimization_report(self, days_ahead: int = 30) -> Dict:
        """Generate report with proper JSON serialization"""
        try:
            df = self.get_invoice_data()
            
            if df.empty:
                return {
                    "message": "No data found in sheet",
                    "period": f"Next {days_ahead} days",
                    "total_upcoming_payments": 0,
                    "payments_by_entity": {},
                    "daily_recommendations": []
                }

            # Filter for upcoming payments
            today = pd.Timestamp.now()
            end_date = today + pd.Timedelta(days=days_ahead)
            
            print("\nDate range in data:")
            print("Due Date range:", df['Due Date'].min(), "to", df['Due Date'].max())
            
            upcoming_payments = df[
                (df['Due Date'].notna()) & 
                (df['Due Date'] >= today) & 
                (df['Due Date'] <= end_date)
            ].copy()
            
            print(f"\nFound {len(upcoming_payments)} upcoming payments")

            if upcoming_payments.empty:
                return {
                    "message": "No upcoming payments found",
                    "period": f"Next {days_ahead} days",
                    "total_upcoming_payments": 0,
                    "payments_by_entity": {},
                    "daily_recommendations": []
                }

            # Calculate totals by entity
            total_amount = float(upcoming_payments['Total Amount'].sum())
            
            # Create entity totals
            payments_by_entity = {}
            for entity, group in upcoming_payments.groupby('Account Name'):
                payments_by_entity[str(entity)] = float(group['Total Amount'].sum())

            # Generate recommendations
            recommendations = []
            for _, row in upcoming_payments.iterrows():
                try:
                    recommendations.append({
                        'date': row['Due Date'].strftime('%Y-%m-%d'),
                        'entity': str(row['Account Name']),
                        'amount': float(row['Total Amount']),
                        'invoice_number': str(row['Invoice Number']),
                        'billing_company': str(row.get('Billing Company Name', '')),
                        'recommendation': self.get_payment_recommendation(row)
                    })
                except Exception as e:
                    print(f"Error processing recommendation: {e}")
                    continue

            # Create the response
            response = {
                'period': f"Next {days_ahead} days",
                'total_upcoming_payments': total_amount,
                'payments_by_entity': payments_by_entity,
                'daily_recommendations': sorted(recommendations, key=lambda x: x['date']),
                'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'summary': {
                    'total_invoices': len(upcoming_payments),
                    'date_range': {
                        'start': today.strftime('%Y-%m-%d'),
                        'end': end_date.strftime('%Y-%m-%d')
                    }
                }
            }

            print("\nGenerated response:", response)  # Debug print
            return response

        except Exception as e:
            print(f"Error generating optimization report: {e}")
            print("Traceback:", traceback.format_exc())  # Add this for more detailed error info
            return {
                "error": str(e),
                "message": "Error generating report",
                "period": f"Next {days_ahead} days",
                "total_upcoming_payments": 0,
                "payments_by_entity": {},
                "daily_recommendations": []
            }

    def generate_payment_schedule(self, days_ahead: int = 30) -> pd.DataFrame:
        """Generate payment schedule for the next X days"""
        try:
            df = self.get_invoice_data()
            
            # Filter for unpaid invoices due within specified days
            today = pd.Timestamp.now()
            end_date = today + pd.Timedelta(days=days_ahead)
            
            upcoming_payments = df[
                (df['Due Date'].notna()) &
                (df['Due Date'] >= today) & 
                (df['Due Date'] <= end_date)
            ].copy()
            
            # Sort by due date
            upcoming_payments = upcoming_payments.sort_values('Due Date')
            
            # Group by entity and due date
            grouped_payments = upcoming_payments.groupby([
                'Account Name', 
                'Due Date'
            ]).agg({
                'Total Amount': 'sum',
                'Invoice Number': lambda x: ', '.join(x),
                'Billing Company Name': lambda x: ', '.join(set(x))
            }).reset_index()
            
            return grouped_payments
            
        except Exception as e:
            print(f"Error generating payment schedule: {e}")
            return pd.DataFrame()

    def get_entity_summary(self) -> pd.DataFrame:
        """Get summary of payments by entity"""
        try:
            df = self.get_invoice_data()

            summary = df.groupby('Account Name').agg({
                'Total Amount': 'sum',
                'Invoice Number': 'count',
                'Due Date': ['min', 'max']
            }).reset_index()

            # Flatten column names
            summary.columns = ['Entity', 'Total Outstanding', 'Invoice Count', 'Earliest Due', 'Latest Due']

            # Add average invoice amount
            summary['Average Invoice Amount'] = summary['Total Outstanding'] / summary['Invoice Count']

            # Format dates
            summary['Earliest Due'] = summary['Earliest Due'].dt.strftime('%Y-%m-%d')
            summary['Latest Due'] = summary['Latest Due'].dt.strftime('%Y-%m-%d')

            return summary

        except Exception as e:
            print(f"Error getting entity summary: {e}")
            return pd.DataFrame()

    def get_daily_payments(self) -> pd.DataFrame:
        """Get daily payment amounts"""
        try:
            df = self.get_invoice_data()

            daily_payments = df.groupby(['Due Date', 'Account Name']).agg({
                'Total Amount': 'sum',
                'Invoice Number': lambda x: ', '.join(x)
            }).reset_index()

            # Sort by date
            daily_payments = daily_payments.sort_values('Due Date')

            return daily_payments

        except Exception as e:
            print(f"Error getting daily payments: {e}")
            return pd.DataFrame()

    def get_payment_recommendation(self, row: pd.Series) -> str:
        """Generate payment recommendation with improved error handling"""
        try:
            due_date = pd.to_datetime(row['Due Date'])
            days_to_due = (due_date - pd.Timestamp.now()).days
            amount = float(row['Total Amount'])
            currency = row['Total Currency']


            if days_to_due <= 0:
                return f"OVERDUE: Immediate payment of {amount:,.2f} {currency} required"
            elif days_to_due < 3:
                return f"URGENT: Payment of {amount:,.2f} {currency} due in {days_to_due} days"
            elif days_to_due < 7:
                return f"Schedule payment of {amount:,.2f} {currency} this week"
            else:
                return f"Plan payment of {amount:,.2f} {currency} due in {days_to_due} days"
    
        except Exception as e:
            print(f"Error generating recommendation: {e}")
            return "Unable to generate recommendation"

    def get_entity_analysis(self, entity_name: str) -> Dict:
        """Get detailed analysis for specific entity"""
        try:
            df = self.get_invoice_data()
            entity_data = df[df['Account Name'] == entity_name]

            if entity_data.empty:
                return {"error": f"No data found for entity: {entity_name}"}

            analysis = {
                "entity_name": entity_name,
                "total_outstanding": float(entity_data['Total Amount'].sum()),
                "invoice_count": len(entity_data),
                "average_amount": float(entity_data['Total Amount'].mean()),
                "largest_invoice": float(entity_data['Total Amount'].max()),
                "smallest_invoice": float(entity_data['Total Amount'].min()),
                "upcoming_payments": entity_data[
                    entity_data['Due Date'] >= datetime.now()
                ].sort_values('Due Date').apply(
                    lambda x: {
                        "date": x['Due Date'].strftime('%Y-%m-%d'),
                        "amount": float(x['Total Amount']),
                        "invoice": x['Invoice Number']
                    }, axis=1
                ).tolist()
            }

            return analysis

        except Exception as e:
            print(f"Error getting entity analysis: {e}")
            return {"error": str(e)}       


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



# Initialize Flask app
app = Flask(__name__)
processor = InvoiceProcessor()
optimizer = CashflowOptimizer()
app.json_encoder = JSONEncoder
from email_processor import EmailProcessor
email_processor = EmailProcessor()        


@app.route('/')
def home():
    return jsonify({
        "endpoints": {
            "invoice_processing": "/slack/events",
            "cashflow_dashboard": "/cashflow",
            "entity_analysis": "/cashflow/entity/<entity_name>",
            "daily_payments": "/cashflow/daily",
            "entity_summary": "/cashflow/summary",
            "custom_report": "/cashflow/report"
        }
    })

@app.route('/cashflow/report', methods=['POST'])
def generate_report():
    """Generate custom cashflow report"""
    try:
        data = request.json
        days = data.get('days', 30)
        optimizer = CashflowOptimizer()
        report = optimizer.generate_optimization_report(days)
        return jsonify(report)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/cashflow/entity/<entity_name>', methods=['GET'])
def entity_analysis(entity_name):
    """Get analysis for specific entity"""
    try:
        optimizer = CashflowOptimizer()
        df = optimizer.get_invoice_data()
        
        # Filter for entity
        entity_data = df[df['Account Name'] == entity_name]
        
        analysis = {
            "total_outstanding": float(entity_data['Total Amount'].sum()),
            "invoice_count": len(entity_data),
            "upcoming_payments": entity_data[
                entity_data['Due Date'] >= datetime.now()
            ].to_dict(orient='records'),
            "average_invoice_amount": float(entity_data['Total Amount'].mean()),
            "largest_invoice": float(entity_data['Total Amount'].max()),
            "earliest_due_date": entity_data['Due Date'].min().strftime('%Y-%m-%d'),
            "latest_due_date": entity_data['Due Date'].max().strftime('%Y-%m-%d')
        }
        
        return jsonify(analysis)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/cashflow/daily', methods=['GET'])
def daily_payments():
    """Get daily payment schedule"""
    try:
        optimizer = CashflowOptimizer()
        daily = optimizer.get_daily_payments()
        return jsonify(daily.to_dict(orient='records'))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/dashboard')
def serve_dashboard():
    return send_file('templates/index.html')

@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

@app.route('/cashflow/summary', methods=['GET'])
def entity_summary():
    """Get summary for all entities"""
    try:
        optimizer = CashflowOptimizer()
        summary = optimizer.get_entity_summary()
        return jsonify(summary.to_dict(orient='records'))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/slack/events', methods=['POST'])
def slack_events():
    """Handle Slack events"""
    event_data = request.json
    
    # Handle Slack verification
    if "challenge" in event_data:
        return jsonify({'challenge': event_data['challenge']})
    
    # Process file share events
    event_type = event_data.get("event", {}).get("type")
    if event_type == "file_shared":
        file_id = event_data["event"]["file_id"]
        process_slack_file(file_id)
    
    return jsonify({"status": "ok"})

def process_slack_file(file_id: str):
    """Download and process file from Slack with cloud storage handling"""
    try:
        file_info_url = f"https://slack.com/api/files.info"
        headers = {"Authorization": f"Bearer {Config.SLACK_BOT_TOKEN}"}
        params = {"file": file_id}
        
        response = requests.get(file_info_url, headers=headers, params=params)
        file_info = response.json()
        
        if not file_info.get("ok"):
            print(f"Error fetching file info: {file_info.get('error')}")
            return
        
        file_url = file_info["file"]["url_private"]
        file_name = file_info["file"]["name"]
        
        # Use temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            # Download file
            file_response = requests.get(file_url, headers=headers, stream=True)
            if file_response.ok:
                tmp_file.write(file_response.content)
                tmp_file.flush()
                
                print(f"Processing file: {tmp_file.name}")
                success = processor.process_pdf(tmp_file.name)
                
                if success:
                    print(f"Successfully processed {file_name}")
                else:
                    print(f"Failed to process {file_name}")
            else:
                print(f"Error downloading file: {file_response.status_code}")
                
            # Clean up
            os.unlink(tmp_file.name)
            
    except Exception as e:
        print(f"Error processing Slack file: {e}")

    # Add error handlers
@app.errorhandler(404)
def not_found_error(error):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500

@app.route('/debug/sheet', methods=['GET'])
def debug_sheet():
    try:
        optimizer = CashflowOptimizer()
        data = optimizer.sheet.get_all_records()
        
        # Convert data to be JSON serializable
        sample_data = []
        if data:
            for row in data[:2]:  # Get first two rows
                sample_row = {}
                for k, v in row.items():
                    if isinstance(v, (np.integer, np.int64)):
                        sample_row[k] = int(v)
                    elif isinstance(v, (np.floating, np.float64)):
                        sample_row[k] = float(v)
                    else:
                        sample_row[k] = str(v)
                sample_data.append(sample_row)
        
        return jsonify({
            "total_rows": len(data),
            "sample": sample_data,
            "columns": list(data[0].keys()) if data else []
        })
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/debug/processed', methods=['GET'])
def debug_processed():
    """Debug endpoint to check processed data"""
    try:
        optimizer = CashflowOptimizer()
        df = optimizer.get_invoice_data()
        
        return jsonify({
            "dataframe_info": {
                "shape": df.shape,
                "columns": list(df.columns),
                "non_null_counts": df.count().to_dict(),
                "sample_dates": df['Due Date'].head().tolist() if 'Due Date' in df else [],
                "sample_amounts": df['Total Amount'].head().tolist() if 'Total Amount' in df else []
            }
        })
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/test/connection', methods=['GET'])
def test_connection():
    """Test Google Sheets connection"""
    try:
        print("Testing connection...")
        optimizer = CashflowOptimizer()
        
        # Try to read data
        data = optimizer.sheet.get_all_records()
        
        return jsonify({
            "status": "success",
            "message": "Connected successfully",
            "sheet_name": Config.GOOGLE_SHEET_NAME,
            "rows_found": len(data),
            "columns": list(data[0].keys()) if data else []
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e),
            "sheet_name": Config.GOOGLE_SHEET_NAME
        }), 500


@app.route('/debug/data', methods=['GET'])
def debug_data():
    try:
        optimizer = CashflowOptimizer()
        
        # Get raw data
        raw_data = optimizer.sheet.get_all_records()
        
        # Get processed DataFrame
        df = optimizer.get_invoice_data()
        
        # Check date processing
        today = pd.Timestamp.now()
        if not df.empty:
            date_info = {
                "due_dates": df['Due Date'].dt.strftime('%Y-%m-%d').tolist()[:5],
                "today": today.strftime('%Y-%m-%d'),
                "future_dates": df[df['Due Date'] > today].shape[0],
                "valid_dates": df['Due Date'].notna().sum()
            }
        else:
            date_info = "No data in DataFrame"

        return jsonify({
            "raw_data_sample": raw_data[:2] if raw_data else "No raw data",
            "total_rows": len(raw_data),
            "dataframe_info": {
                "shape": df.shape,
                "columns": list(df.columns),
                "non_null_counts": df.count().to_dict() if not df.empty else {},
                "sample_amounts": df['Total Amount'].head().tolist() if not df.empty else [],
                "date_processing": date_info
            }
        })
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/debug/dates', methods=['GET'])
def debug_dates():
    """Debug date processing in detail"""
    try:
        optimizer = CashflowOptimizer()
        raw_data = optimizer.sheet.get_all_records()
        
        # Show raw date values
        date_samples = {
            'raw_dates': {
                'Invoice Date': [row.get('Invoice Date') for row in raw_data[:5]],
                'Due Date': [row.get('Due Date') for row in raw_data[:5]]
            }
        }
        
        # Process each date manually for debugging
        processed_dates = {
            'Invoice Date': [],
            'Due Date': []
        }
        
        for col in ['Invoice Date', 'Due Date']:
            for date_str in date_samples['raw_dates'][col]:
                try:
                    parsed = optimizer.parse_date(date_str)
                    processed_dates[col].append({
                        'original': date_str,
                        'parsed': parsed.strftime('%Y-%m-%d') if parsed else None,
                        'success': parsed is not None
                    })
                except Exception as e:
                    processed_dates[col].append({
                        'original': date_str,
                        'error': str(e),
                        'success': False
                    })
        
        date_samples['processed_dates'] = processed_dates
        
        # Get full DataFrame processing results
        df = optimizer.get_invoice_data()
        if not df.empty:
            date_samples['summary'] = {
                'total_rows': len(df),
                'valid_dates': {
                    'Invoice Date': df['Invoice Date'].notna().sum(),
                    'Due Date': df['Due Date'].notna().sum()
                },
                'future_dates': len(df[df['Due Date'] >= pd.Timestamp.now()]),
                'date_ranges': {
                    'Invoice Date': {
                        'min': df['Invoice Date'].min().strftime('%Y-%m-%d') if df['Invoice Date'].notna().any() else None,
                        'max': df['Invoice Date'].max().strftime('%Y-%m-%d') if df['Invoice Date'].notna().any() else None
                    },
                    'Due Date': {
                        'min': df['Due Date'].min().strftime('%Y-%m-%d') if df['Due Date'].notna().any() else None,
                        'max': df['Due Date'].max().strftime('%Y-%m-%d') if df['Due Date'].notna().any() else None
                    }
                }
            }
        
        return jsonify(date_samples)
        
    except Exception as e:
        return jsonify({
            "error": str(e),
            "location": "debug_dates"
        })     

@app.route('/cashflow', methods=['GET'])
def cashflow_dashboard():
    """Get cashflow dashboard with error handling"""
    try:
        optimizer = CashflowOptimizer()
        report = optimizer.generate_optimization_report(30)
        
        if not report:
            return jsonify({
                "message": "No data available",
                "period": "Next 30 days",
                "total_upcoming_payments": 0,
                "payments_by_entity": {},
                "daily_recommendations": []
            })
            
        return jsonify(report)
        
    except Exception as e:
        print(f"Error in cashflow endpoint: {e}")
        return jsonify({
            "error": str(e),
            "message": "Error processing request",
            "period": "Next 30 days",
            "total_upcoming_payments": 0,
            "payments_by_entity": {},
            "daily_recommendations": []
        }), 500
@app.route('/debug/cashflow', methods=['GET'])
def debug_cashflow():
    """Debug endpoint for cashflow data"""
    try:
        optimizer = CashflowOptimizer()
        df = optimizer.get_invoice_data()
        
        # Get upcoming payments
        today = pd.Timestamp.now()
        end_date = today + pd.Timedelta(days=30)
        
        upcoming = df[
            (df['Due Date'].notna()) & 
            (df['Due Date'] >= today) & 
            (df['Due Date'] <= end_date)
        ]
        
        return jsonify({
            "data_info": {
                "total_rows": len(df),
                "upcoming_rows": len(upcoming),
                "date_range": {
                    "min": df['Due Date'].min().strftime('%Y-%m-%d') if not df.empty else None,
                    "max": df['Due Date'].max().strftime('%Y-%m-%d') if not df.empty else None
                }
            },
            "sample_data": upcoming.head(2).to_dict('records') if not upcoming.empty else [],
            "amount_summary": {
                "total": float(upcoming['Total Amount'].sum()) if not upcoming.empty else 0,
                "by_entity": upcoming.groupby('Account Name')['Total Amount'].sum().to_dict() if not upcoming.empty else {}
            }
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    if not Config.validate_config():
        print("Missing required configuration. Please check your environment variables.")
        sys.exit(1)

    
    port = int(os.getenv("PORT", 3001))
    app.run(host='0.0.0.0', port=port)
