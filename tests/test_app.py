import pytest
from unittest.mock import Mock, patch
import os
import base64
from app import Config, CashflowOptimizer, InvoiceProcessor

def test_config():
    """Basic test to verify config can validate"""
    result = Config.validate_config()
    assert result is not None  # Just verify the method runs