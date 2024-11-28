
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
from dotenv import load_dotenv
from config import Config, InvoiceProcessor, EmailProcessor


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
            
            print("Sample data from sheet:", df[['Invoice Number', 'Total Amount', 'Total Currency']].head())  # Debug print
            
            # Process date columns
            date_columns = ['Invoice Date', 'Due Date']
            for col in date_columns:
                if col in df.columns:
                    df[col] = df[col].apply(self.parse_date)

            # Process amount columns but keep currency information
            amount_columns = ['Total Amount', 'Subtotal Amount', 'Tax Amount']
            for col in amount_columns:
                if col in df.columns:
                    df[col] = df[col].apply(self.clean_amount)

            return df
            
        except Exception as e:
            print(f"Error getting invoice data: {e}")
            return pd.DataFrame()
            
        except Exception as e:
            print(f"Error getting invoice data: {e}")
            return pd.DataFrame()

    def generate_optimization_report(self, days_ahead: int = 30) -> Dict:
        try:
            print("Starting report generation...")  # Debug log
            df = self.get_invoice_data()
            print(f"Retrieved {len(df)} invoices")  # Debug log
            
            today = pd.Timestamp.now()
            
            # Process each invoice
            invoice_list = []
            for _, row in df.iterrows():
                due_date = row['Due Date'] if pd.notnull(row['Due Date']) else None
                status = 'past_due' if (due_date and due_date < today) else 'upcoming'
                
                invoice_data = {
                    'invoice_number': str(row['Invoice Number']),
                    'entity': str(row['Account Name']),
                    'due_date': due_date.strftime('%Y-%m-%d') if due_date else None,
                    'amount': float(row['Total Amount']),
                    'currency': str(row.get('Total Currency', 'AED')),
                    'description': str(row.get('Line Items Detail', '')),
                    'status': status
                }
                print(f"Processing invoice: {invoice_data}")  # Debug log
                invoice_list.append(invoice_data)

            # Calculate totals
            past_due_total = sum(inv['amount'] for inv in invoice_list if inv['status'] == 'past_due')
            upcoming_total = sum(inv['amount'] for inv in invoice_list if inv['status'] == 'upcoming')

            response = {
                'period': f"Next {days_ahead} days",
                'invoice_list': invoice_list,
                'total_invoices': len(invoice_list),
                'past_due_total': past_due_total,
                'upcoming_total': upcoming_total,
                'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            print(f"Generated response: {response}")  # Debug log
            return response

        except Exception as e:
            print(f"Error in generate_optimization_report: {e}")
            traceback.print_exc()
            return {
                "error": str(e),
                "message": "Error generating report",
                "period": f"Next {days_ahead} days",
                "invoice_list": [],
                "total_invoices": 0
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



# Initialize Flask app
app = Flask(__name__)
processor = InvoiceProcessor()
optimizer = CashflowOptimizer()
app.json_encoder = JSONEncoder


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

@app.route('/cashflow')
def cashflow_dashboard():
    try:
        print("Processing cashflow request...")
        optimizer = CashflowOptimizer()
        df = optimizer.get_invoice_data()
        
        today = pd.Timestamp.now()
        
        # Process each invoice
        invoice_list = []
        for _, row in df.iterrows():
            due_date = row['Due Date'] if pd.notnull(row['Due Date']) else None
            status = 'past_due' if (due_date and due_date < today) else 'upcoming'
            
            invoice_list.append({
                'invoice_number': str(row['Invoice Number']),
                'entity': str(row['Account Name']),
                'due_date': due_date.strftime('%Y-%m-%d') if due_date else None,
                'amount': float(row['Total Amount']),
                'currency': str(row.get('Total Currency', 'AED')),
                'description': str(row.get('Line Items Detail', '')),
                'status': status
            })

        # Calculate totals by currency and status
        totals = {
            'past_due': {},
            'upcoming': {}
        }
        
        for invoice in invoice_list:
            status = invoice['status']
            currency = invoice['currency']
            amount = invoice['amount']
            
            if currency not in totals[status]:
                totals[status][currency] = 0
            totals[status][currency] += amount

        return jsonify({
            'period': f"Next 30 days",
            'invoice_list': invoice_list,
            'total_invoices': len(invoice_list),
            'totals': totals,
            'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })

    except Exception as e:
        print(f"Error in cashflow endpoint: {e}")
        return jsonify({
            "error": str(e),
            "message": "Error processing request",
            "period": "Next 30 days",
            "invoice_list": [],
            "total_invoices": 0,
            "totals": {'past_due': {}, 'upcoming': {}}
        }), 500

@app.route('/debug/email')
def debug_email():
    try:
        return jsonify({
            "email_processor_running": hasattr(email_processor, 'scheduler') and email_processor.scheduler.running,
            "gmail_credentials": {
                "exists": os.path.exists(GmailConfig.GMAIL_CREDS_PATH),
                "path": GmailConfig.GMAIL_CREDS_PATH
            },
            "token": {
                "exists": os.path.exists(GmailConfig.TOKEN_PATH),
                "path": GmailConfig.TOKEN_PATH
            }
        })
    except Exception as e:
        return jsonify({"error": str(e)})


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

@app.route('/test_data')
def test_data():
    try:
        optimizer = CashflowOptimizer()
        df = optimizer.get_invoice_data()
        raw_data = df.to_dict('records')
        return jsonify({
            'raw_data': raw_data,
            'row_count': len(df),
            'columns': list(df.columns)
        })
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == "__main__":
    if not Config.validate_config():
        print("Missing required configuration. Please check your environment variables.")
        sys.exit(1)

    
    port = int(os.getenv("PORT", 3001))
    app.run(host='0.0.0.0', port=port)
