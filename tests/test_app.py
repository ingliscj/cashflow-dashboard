import pytest
from unittest.mock import Mock, patch
import os
import base64
from app import Config, CashflowOptimizer, InvoiceProcessor

@pytest.fixture
def mock_env_vars():
    """Setup environment variables for testing"""
    env_vars = {
        'API_KEY': 'test_api_key',
        'SLACK_BOT_TOKEN': 'test_slack_token',
        'GOOGLE_SHEET_NAME': 'test_sheet',
        'GOOGLE_SHEET_KEY': 'test_key',
        'GOOGLE_CREDENTIALS_FILE': base64.b64encode(b'{"test": "credentials"}').decode('utf-8')
    }
    with patch.dict(os.environ, env_vars):
        yield env_vars

def test_config_initialization(mock_env_vars):
    """Test Config class initialization and environment variable loading"""
    config = Config()
    assert config.ANTHROPIC_API_KEY == 'test_api_key'
    assert config.SLACK_BOT_TOKEN == 'test_slack_token'
    assert config.GOOGLE_SHEET_NAME == 'test_sheet'
    assert os.path.exists(config.APP_DIR)
    assert config.validate_config() == True

@patch('app.ServiceAccountCredentials')
@patch('app.gspread')
def test_invoice_processor_initialization(mock_gspread, mock_credentials, mock_env_vars):
    """Test InvoiceProcessor initialization and API connections"""
    # Setup mocks
    mock_client = Mock()
    mock_worksheet = Mock()
    mock_gspread.authorize.return_value = mock_client
    mock_client.open.return_value.sheet1 = mock_worksheet

    # Initialize processor
    processor = InvoiceProcessor()
    
    # Verify API clients were initialized
    assert processor.google_client is not None
    assert processor.claude_client is not None
    assert processor.sheet is not None

@patch('app.gspread')
def test_cashflow_optimizer_sheet_connection(mock_gspread, mock_env_vars):
    """Test CashflowOptimizer Google Sheets connection"""
    # Setup mock
    mock_client = Mock()
    mock_worksheet = Mock()
    mock_gspread.authorize.return_value = mock_client
    mock_client.open.return_value.sheet1 = mock_worksheet
    mock_worksheet.get_all_records.return_value = [
        {
            'Invoice Number': '001',
            'Due Date': '2024-12-01',
            'Total Amount': '1000',
            'Account Name': 'Test Company'
        }
    ]

    # Initialize optimizer
    optimizer = CashflowOptimizer()
    
    # Test data retrieval
    df = optimizer.get_invoice_data()
    assert not df.empty
    assert 'Invoice Number' in df.columns
    assert 'Due Date' in df.columns
    assert 'Total Amount' in df.columns

def test_date_parsing():
    """Test date parsing functionality"""
    optimizer = CashflowOptimizer()
    test_dates = [
        '26 November 2024',
        '26 Nov 2024',
        '15.11.2024',
        '15/11/2024',
        '2024-11-15'
    ]
    for date_str in test_dates:
        parsed = optimizer.parse_date(date_str)
        assert parsed is not None, f"Failed to parse date: {date_str}"

def test_amount_cleaning():
    """Test amount cleaning functionality"""
    optimizer = CashflowOptimizer()
    test_amounts = [
        ('1,234.56', 1234.56),
        ('AED 1,234.56', 1234.56),
        ('1.234,56', 1234.56),
        ('$1,234.56', 1234.56)
    ]
    for amount_str, expected in test_amounts:
        cleaned = optimizer.clean_amount(amount_str)
        assert abs(cleaned - expected) < 0.01, f"Failed to clean amount: {amount_str}"