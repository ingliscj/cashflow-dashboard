import pytest
from unittest.mock import Mock, patch
import os
import base64
from app import Config, CashflowOptimizer, InvoiceProcessor

@pytest.fixture
def mock_env_vars(monkeypatch):
    """Setup environment variables for testing"""
    # Clear any existing environment variables first
    for key in ['API_KEY', 'SLACK_BOT_TOKEN', 'GOOGLE_SHEET_NAME', 'GOOGLE_SHEET_KEY', 'GOOGLE_CREDENTIALS_FILE']:
        monkeypatch.delenv(key, raising=False)
    
    # Set test values
    test_vars = {
        'API_KEY': 'test_api_key',
        'SLACK_BOT_TOKEN': 'test_slack_token',
        'GOOGLE_SHEET_NAME': 'test_sheet',
        'GOOGLE_SHEET_KEY': 'test_key',
        'GOOGLE_CREDENTIALS_FILE': base64.b64encode(b'{"test": "credentials"}').decode('utf-8')
    }
    
    for key, value in test_vars.items():
        monkeypatch.setenv(key, value)
    
    return test_vars

def test_config_initialization(mock_env_vars):
    """Test Config class initialization and environment variable loading"""
    with patch.dict('os.environ', mock_env_vars, clear=True):
        config = Config()  # Now creates a new instance with our mocked environment
        assert config.ANTHROPIC_API_KEY == mock_env_vars['API_KEY']
        assert config.SLACK_BOT_TOKEN == mock_env_vars['SLACK_BOT_TOKEN']
        assert config.GOOGLE_SHEET_NAME == mock_env_vars['GOOGLE_SHEET_NAME']
        assert os.path.exists(config.APP_DIR)
        assert config.validate_config() == True