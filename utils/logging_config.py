"""
AML 360 Logging Configuration
Centralized logging setup for the entire application
"""

import logging
import logging.handlers
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

def setup_logging(
    name: Optional[str] = None,
    level: str = "INFO",
    log_dir: str = "logs",
    max_bytes: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5,
    console_output: bool = True
) -> logging.Logger:
    """
    Set up centralized logging configuration
    
    Args:
        name: Logger name (defaults to calling module)
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: Directory to store log files
        max_bytes: Maximum size of each log file before rotation
        backup_count: Number of backup log files to keep
        console_output: Whether to output logs to console
        
    Returns:
        Configured logger instance
    """
    
    # Create logs directory if it doesn't exist
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    # Get logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # Avoid duplicate handlers
    if logger.handlers:
        return logger
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        fmt='%(asctime)s | %(name)s | %(levelname)s | %(filename)s:%(lineno)d | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    simple_formatter = logging.Formatter(
        fmt='%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # File handler with rotation
    file_handler = logging.handlers.RotatingFileHandler(
        filename=os.path.join(log_dir, 'aml360.log'),
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding='utf-8'
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(detailed_formatter)
    logger.addHandler(file_handler)
    
    # Error file handler (separate file for errors)
    error_handler = logging.handlers.RotatingFileHandler(
        filename=os.path.join(log_dir, 'aml360_errors.log'),
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding='utf-8'
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(detailed_formatter)
    logger.addHandler(error_handler)
    
    # Console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, level.upper()))
        console_handler.setFormatter(simple_formatter)
        logger.addHandler(console_handler)
    
    return logger

def setup_audit_logging(log_dir: str = "logs/audit") -> logging.Logger:
    """
    Set up audit logging for compliance and security monitoring
    
    Args:
        log_dir: Directory for audit logs
        
    Returns:
        Audit logger instance
    """
    
    # Create audit logs directory
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    # Get audit logger
    audit_logger = logging.getLogger('aml360.audit')
    audit_logger.setLevel(logging.INFO)
    
    # Avoid duplicate handlers
    if audit_logger.handlers:
        return audit_logger
    
    # Audit formatter (structured for parsing)
    audit_formatter = logging.Formatter(
        fmt='%(asctime)s | AUDIT | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Daily rotating file handler for audit logs
    audit_handler = logging.handlers.TimedRotatingFileHandler(
        filename=os.path.join(log_dir, 'audit.log'),
        when='midnight',
        interval=1,
        backupCount=365,  # Keep 1 year of audit logs
        encoding='utf-8'
    )
    audit_handler.setLevel(logging.INFO)
    audit_handler.setFormatter(audit_formatter)
    audit_logger.addHandler(audit_handler)
    
    # Don't propagate audit logs to parent loggers
    audit_logger.propagate = False
    
    return audit_logger

def log_transaction_scoring(
    transaction_id: str,
    total_score: int,
    suspicious: bool,
    model_version: Optional[str] = None,
    rules_version: Optional[str] = None,
    user_id: Optional[str] = None,
    processing_time_ms: Optional[float] = None
):
    """
    Log transaction scoring event for audit trail
    
    Args:
        transaction_id: Unique transaction identifier
        total_score: Final risk score
        suspicious: Whether transaction was flagged
        model_version: ML model version used
        rules_version: Rules engine version
        user_id: User who initiated scoring
        processing_time_ms: Processing time in milliseconds
    """
    
    audit_logger = setup_audit_logging()
    
    audit_data = {
        'event_type': 'TRANSACTION_SCORING',
        'transaction_id': transaction_id,
        'total_score': total_score,
        'suspicious': suspicious,
        'model_version': model_version or 'unknown',
        'rules_version': rules_version or 'v1.0',
        'user_id': user_id or 'system',
        'processing_time_ms': processing_time_ms
    }
    
    # Format as structured log entry
    audit_message = ' | '.join([f"{k}={v}" for k, v in audit_data.items() if v is not None])
    audit_logger.info(audit_message)

def log_database_operation(
    operation: str,
    table_name: str,
    record_count: int,
    user_id: Optional[str] = None,
    success: bool = True,
    error_message: Optional[str] = None
):
    """
    Log database operations for audit trail
    
    Args:
        operation: Type of operation (INSERT, UPDATE, DELETE, SELECT)
        table_name: Database table affected
        record_count: Number of records affected
        user_id: User who performed operation
        success: Whether operation was successful
        error_message: Error message if operation failed
    """
    
    audit_logger = setup_audit_logging()
    
    audit_data = {
        'event_type': 'DATABASE_OPERATION',
        'operation': operation,
        'table_name': table_name,
        'record_count': record_count,
        'user_id': user_id or 'system',
        'success': success,
        'error_message': error_message
    }
    
    audit_message = ' | '.join([f"{k}={v}" for k, v in audit_data.items() if v is not None])
    
    if success:
        audit_logger.info(audit_message)
    else:
        audit_logger.error(audit_message)

def log_api_access(
    endpoint: str,
    method: str,
    user_id: Optional[str] = None,
    ip_address: Optional[str] = None,
    response_status: Optional[int] = None,
    processing_time_ms: Optional[float] = None
):
    """
    Log API access for security monitoring
    
    Args:
        endpoint: API endpoint accessed
        method: HTTP method (GET, POST, etc.)
        user_id: User accessing the API
        ip_address: Client IP address
        response_status: HTTP response status code
        processing_time_ms: API processing time
    """
    
    audit_logger = setup_audit_logging()
    
    audit_data = {
        'event_type': 'API_ACCESS',
        'endpoint': endpoint,
        'method': method,
        'user_id': user_id or 'anonymous',
        'ip_address': ip_address,
        'response_status': response_status,
        'processing_time_ms': processing_time_ms
    }
    
    audit_message = ' | '.join([f"{k}={v}" for k, v in audit_data.items() if v is not None])
    audit_logger.info(audit_message)

def log_model_training(
    model_type: str,
    training_data_size: int,
    model_version: str,
    metrics: dict,
    user_id: Optional[str] = None,
    training_time_minutes: Optional[float] = None
):
    """
    Log ML model training events
    
    Args:
        model_type: Type of model trained
        training_data_size: Number of training samples
        model_version: Version of the trained model
        metrics: Training metrics (accuracy, precision, etc.)
        user_id: User who initiated training
        training_time_minutes: Training duration in minutes
    """
    
    audit_logger = setup_audit_logging()
    
    # Flatten metrics for logging
    metrics_str = ' | '.join([f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" 
                             for k, v in metrics.items()])
    
    audit_data = {
        'event_type': 'MODEL_TRAINING',
        'model_type': model_type,
        'training_data_size': training_data_size,
        'model_version': model_version,
        'user_id': user_id or 'system',
        'training_time_minutes': training_time_minutes,
        'metrics': metrics_str
    }
    
    audit_message = ' | '.join([f"{k}={v}" for k, v in audit_data.items() if v is not None])
    audit_logger.info(audit_message)

class AMLLogger:
    """
    Convenience class for AML-specific logging operations
    """
    
    def __init__(self, name: str):
        self.logger = setup_logging(name)
        self.audit_logger = setup_audit_logging()
    
    def info(self, message: str, **kwargs):
        """Log info message with optional audit data"""
        self.logger.info(message)
        if kwargs:
            self._log_audit('INFO', message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message with optional audit data"""
        self.logger.warning(message)
        if kwargs:
            self._log_audit('WARNING', message, **kwargs)
    
    def error(self, message: str, **kwargs):
        """Log error message with optional audit data"""
        self.logger.error(message)
        if kwargs:
            self._log_audit('ERROR', message, **kwargs)
    
    def debug(self, message: str, **kwargs):
        """Log debug message"""
        self.logger.debug(message)
    
    def _log_audit(self, level: str, message: str, **kwargs):
        """Internal method to log audit data"""
        audit_data = {
            'level': level,
            'message': message,
            **kwargs
        }
        audit_message = ' | '.join([f"{k}={v}" for k, v in audit_data.items()])
        self.audit_logger.info(audit_message)

# Configure root logger to reduce noise from external libraries
def configure_external_loggers():
    """Configure logging levels for external libraries to reduce noise"""
    
    # Reduce log levels for common noisy libraries
    noisy_loggers = [
        'urllib3.connectionpool',
        'requests.packages.urllib3.connectionpool',
        'chromadb',
        'sentence_transformers',
        'transformers',
        'sklearn',
        'matplotlib',
        'PIL'
    ]
    
    for logger_name in noisy_loggers:
        logging.getLogger(logger_name).setLevel(logging.WARNING)

# Initialize external logger configuration
configure_external_loggers()

# Export main functions
__all__ = [
    'setup_logging',
    'setup_audit_logging',
    'log_transaction_scoring',
    'log_database_operation',
    'log_api_access',
    'log_model_training',
    'AMLLogger'
]
