"""
PII Masking and Data Privacy Utilities
Handles masking of personally identifiable information in logs and displays
"""

import hashlib
import re
from typing import Any, Dict, Optional


def mask_name(name: str) -> str:
    """Mask a person's name for privacy"""
    if not name or len(name) < 2:
        return "***"
    
    # Show first letter and last letter, mask middle
    if len(name) <= 4:
        return name[0] + "*" * (len(name) - 2) + name[-1] if len(name) > 2 else "***"
    
    return name[:2] + "*" * (len(name) - 4) + name[-2:]


def mask_account_number(account_number: str) -> str:
    """Mask account number, showing only last 4 digits"""
    if not account_number:
        return "****"
    
    account_str = str(account_number)
    if len(account_str) <= 4:
        return "*" * len(account_str)
    
    return "*" * (len(account_str) - 4) + account_str[-4:]


def hash_identifier(identifier: str, salt: str = "aml360") -> str:
    """Create a consistent hash of an identifier for tracking without exposing PII"""
    if not identifier:
        return ""
    
    return hashlib.sha256(f"{salt}_{identifier}".encode()).hexdigest()[:16]


def mask_email(email: str) -> str:
    """Mask email address"""
    if not email or '@' not in email:
        return "***@***.***"
    
    parts = email.split('@')
    username = parts[0]
    domain = parts[1]
    
    masked_username = username[0] + "*" * (len(username) - 1) if len(username) > 1 else "*"
    masked_domain = domain[:2] + "*" * max(0, len(domain) - 2)
    
    return f"{masked_username}@{masked_domain}"


def mask_transaction_for_display(transaction: Dict[str, Any]) -> Dict[str, Any]:
    """Mask PII fields in a transaction for safe display"""
    masked = transaction.copy()
    
    # Mask names
    if 'originator_name' in masked:
        masked['originator_name'] = mask_name(masked['originator_name'])
    
    if 'beneficiary_name' in masked:
        masked['beneficiary_name'] = mask_name(masked['beneficiary_name'])
    
    # Mask account numbers
    if 'originator_account_number' in masked:
        masked['originator_account_number'] = mask_account_number(masked['originator_account_number'])
    
    if 'beneficiary_account_number' in masked:
        masked['beneficiary_account_number'] = mask_account_number(masked['beneficiary_account_number'])
    
    if 'account_id' in masked:
        # For display, we can show a hashed version
        masked['account_id_display'] = hash_identifier(masked['account_id'])
    
    return masked


def mask_transaction_for_logging(transaction: Dict[str, Any]) -> Dict[str, Any]:
    """Mask PII fields in a transaction for safe logging"""
    logged = transaction.copy()
    
    # Hash identifiers for consistent tracking without exposing PII
    if 'transaction_id' in logged:
        logged['transaction_id_hash'] = hash_identifier(logged['transaction_id'])
    
    if 'account_id' in logged:
        logged['account_id_hash'] = hash_identifier(logged['account_id'])
    
    # Remove PII fields entirely from logs
    pii_fields = [
        'originator_name', 'beneficiary_name',
        'originator_address', 'beneficiary_address',
        'originator_account_number', 'beneficiary_account_number'
    ]
    
    for field in pii_fields:
        if field in logged:
            del logged[field]
    
    return logged


def sanitize_payment_instruction(instruction: str) -> str:
    """Sanitize payment instruction by removing potential PII patterns"""
    if not instruction:
        return ""
    
    # Remove email addresses
    instruction = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]', instruction)
    
    # Remove phone numbers (various formats)
    instruction = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '[PHONE]', instruction)
    instruction = re.sub(r'\b\+?\d{1,3}[-.\s]?\d{1,4}[-.\s]?\d{1,4}[-.\s]?\d{1,9}\b', '[PHONE]', instruction)
    
    # Remove credit card numbers (simple pattern)
    instruction = re.sub(r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b', '[CARD]', instruction)
    
    return instruction
