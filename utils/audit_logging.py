"""
Audit Logging Module for AML 360
Provides comprehensive audit trail for all scoring and investigation activities
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import hashlib

from utils.pii_masking import mask_transaction_for_logging, hash_identifier


# Configure audit logger
audit_logger = logging.getLogger('aml_audit')
audit_logger.setLevel(logging.INFO)

# Create audit directory if it doesn't exist
audit_dir = Path('audit')
audit_dir.mkdir(exist_ok=True)

# File handler for audit logs with rotation
audit_log_file = audit_dir / f'audit_{datetime.now().strftime("%Y%m%d")}.log'
file_handler = logging.FileHandler(audit_log_file)
file_handler.setLevel(logging.INFO)

# Format: timestamp, transaction_id_hash, event_type, details
formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
file_handler.setFormatter(formatter)
audit_logger.addHandler(file_handler)


def log_transaction_scored(
    transaction_id: str,
    account_id: str,
    total_score: int,
    suspicious: bool,
    rules_version: str = "1.0",
    model_version: Optional[str] = None,
    input_hash: Optional[str] = None,
    transaction_data: Optional[Dict[str, Any]] = None
) -> None:
    """Log when a transaction is scored"""
    
    # Create input hash if not provided
    if input_hash is None:
        input_hash = hashlib.md5(f"{transaction_id}_{account_id}".encode()).hexdigest()[:16]
    
    log_entry = {
        'event': 'TRANSACTION_SCORED',
        'transaction_id_hash': hash_identifier(transaction_id),
        'account_id_hash': hash_identifier(account_id),
        'input_hash': input_hash,
        'rules_version': rules_version,
        'model_version': model_version or 'N/A',
        'total_score': total_score,
        'suspicious_flag': suspicious,
        'timestamp': datetime.now().isoformat()
    }
    
    # Add masked transaction data if provided
    if transaction_data:
        masked_data = mask_transaction_for_logging(transaction_data)
        log_entry['transaction_metadata'] = masked_data
    
    audit_logger.info(json.dumps(log_entry))


def log_manual_transaction_entry(
    transaction_id: str,
    user_action: str,
    total_score: int,
    suspicious: bool
) -> None:
    """Log manual transaction entry and scoring"""
    
    log_entry = {
        'event': 'MANUAL_ENTRY',
        'transaction_id_hash': hash_identifier(transaction_id),
        'user_action': user_action,
        'total_score': total_score,
        'suspicious_flag': suspicious,
        'timestamp': datetime.now().isoformat()
    }
    
    audit_logger.info(json.dumps(log_entry))


def log_investigation_action(
    transaction_id: str,
    action: str,
    details: Optional[Dict[str, Any]] = None
) -> None:
    """Log investigation actions (export, case creation, review)"""
    
    log_entry = {
        'event': 'INVESTIGATION_ACTION',
        'transaction_id_hash': hash_identifier(transaction_id),
        'action': action,
        'details': details or {},
        'timestamp': datetime.now().isoformat()
    }
    
    audit_logger.info(json.dumps(log_entry))


def log_batch_processing(
    batch_id: str,
    total_transactions: int,
    flagged_count: int,
    processing_time_seconds: float
) -> None:
    """Log batch processing statistics"""
    
    log_entry = {
        'event': 'BATCH_PROCESSED',
        'batch_id': batch_id,
        'total_transactions': total_transactions,
        'flagged_count': flagged_count,
        'processing_time_seconds': processing_time_seconds,
        'timestamp': datetime.now().isoformat()
    }
    
    audit_logger.info(json.dumps(log_entry))


def log_model_inference(
    transaction_id: str,
    model_version: str,
    prediction: float,
    decision_threshold: float
) -> None:
    """Log ML model inference"""
    
    log_entry = {
        'event': 'MODEL_INFERENCE',
        'transaction_id_hash': hash_identifier(transaction_id),
        'model_version': model_version,
        'prediction_score': prediction,
        'decision_threshold': decision_threshold,
        'timestamp': datetime.now().isoformat()
    }
    
    audit_logger.info(json.dumps(log_entry))


def log_rag_query(
    query_text: str,
    transaction_id: Optional[str] = None,
    evidence_count: int = 0
) -> None:
    """Log RAG chatbot queries"""
    
    # Hash the query for tracking without exposing content
    query_hash = hashlib.md5(query_text.encode()).hexdigest()[:16]
    
    log_entry = {
        'event': 'RAG_QUERY',
        'query_hash': query_hash,
        'transaction_id_hash': hash_identifier(transaction_id) if transaction_id else None,
        'evidence_retrieved_count': evidence_count,
        'timestamp': datetime.now().isoformat()
    }
    
    audit_logger.info(json.dumps(log_entry))


def log_error(
    error_type: str,
    error_message: str,
    context: Optional[Dict[str, Any]] = None
) -> None:
    """Log errors and exceptions"""
    
    log_entry = {
        'event': 'ERROR',
        'error_type': error_type,
        'error_message': error_message,
        'context': mask_transaction_for_logging(context) if context else {},
        'timestamp': datetime.now().isoformat()
    }
    
    audit_logger.error(json.dumps(log_entry))


def create_case_file(
    case_id: str,
    transactions: list,
    investigator_notes: str = ""
) -> str:
    """Create a case file for investigation"""
    
    case_dir = Path('cases')
    case_dir.mkdir(exist_ok=True)
    
    case_file = case_dir / f'case_{case_id}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    
    case_data = {
        'case_id': case_id,
        'created_at': datetime.now().isoformat(),
        'investigator_notes': investigator_notes,
        'transactions': [mask_transaction_for_logging(t) for t in transactions],
        'transaction_count': len(transactions)
    }
    
    with open(case_file, 'w') as f:
        json.dump(case_data, f, indent=2)
    
    log_investigation_action(
        transaction_id=case_id,
        action='CASE_CREATED',
        details={'transaction_count': len(transactions), 'case_file': str(case_file)}
    )
    
    return str(case_file)
