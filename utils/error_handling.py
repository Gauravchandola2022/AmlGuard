"""
Error Handling and Data Validation Utilities
Handles edge cases and validates transaction data
"""

from datetime import datetime
from typing import Dict, Any, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def validate_and_clean_transaction(txn: Dict[str, Any]) -> Tuple[Dict[str, Any], list]:
    """
    Validate and clean transaction data, handling edge cases
    Returns: (cleaned_transaction, list_of_warnings)
    """
    cleaned = txn.copy()
    warnings = []
    
    # Handle missing currency - default to USD
    if not cleaned.get('currency') or cleaned.get('currency_code'):
        if cleaned.get('currency_code'):
            cleaned['currency'] = cleaned['currency_code']
        elif not cleaned.get('currency'):
            cleaned['currency'] = 'USD'
            warnings.append("Missing currency, defaulted to USD")
            logger.warning(f"Transaction {cleaned.get('transaction_id', 'UNKNOWN')}: Missing currency, defaulted to USD")
    
    # Handle missing or invalid amount
    amount_field = 'amount' if 'amount' in cleaned else 'transaction_amount'
    if amount_field in cleaned:
        try:
            amount = float(cleaned[amount_field])
            if amount < 0:
                amount = 0
                warnings.append("Negative amount converted to 0")
            cleaned['amount'] = amount
            cleaned['transaction_amount'] = amount
        except (ValueError, TypeError):
            cleaned['amount'] = 0
            cleaned['transaction_amount'] = 0
            warnings.append("Invalid amount, set to 0")
            logger.error(f"Transaction {cleaned.get('transaction_id', 'UNKNOWN')}: Invalid amount")
    else:
        cleaned['amount'] = 0
        cleaned['transaction_amount'] = 0
        warnings.append("Missing amount, set to 0")
    
    # Handle unknown country codes - treat as Level_1 (lowest risk)
    for country_field in ['originator_country', 'beneficiary_country']:
        if country_field in cleaned:
            country_code = str(cleaned[country_field]).upper()
            if len(country_code) > 3:
                cleaned[country_field] = country_code[:2]
                warnings.append(f"Country code truncated: {country_field}")
            elif not country_code or country_code == 'UNKNOWN':
                cleaned[country_field] = 'XX'  # Unknown country code
                warnings.append(f"Unknown country code in {country_field}, set to XX (treated as Level_1)")
    
    # Handle missing payment_instruction
    if not cleaned.get('payment_instruction'):
        cleaned['payment_instruction'] = ""
        warnings.append("Missing payment_instruction, set to empty string")
    
    # Handle malformed dates
    if 'transaction_date' in cleaned:
        try:
            # Try to parse the date
            date_value = cleaned['transaction_date']
            if isinstance(date_value, str):
                # Try ISO format first
                try:
                    parsed_date = datetime.fromisoformat(date_value.replace('Z', '+00:00'))
                    cleaned['transaction_date'] = parsed_date.isoformat()
                except:
                    # Try other common formats
                    for fmt in ['%Y-%m-%d %H:%M:%S', '%Y-%m-%d', '%m/%d/%Y', '%d/%m/%Y']:
                        try:
                            parsed_date = datetime.strptime(date_value, fmt)
                            cleaned['transaction_date'] = parsed_date.isoformat()
                            break
                        except:
                            continue
                    else:
                        # If all parsing fails, use current time
                        cleaned['transaction_date'] = datetime.now().isoformat()
                        warnings.append("Malformed date, using current timestamp")
                        logger.error(f"Transaction {cleaned.get('transaction_id', 'UNKNOWN')}: Malformed date '{date_value}'")
        except Exception as e:
            cleaned['transaction_date'] = datetime.now().isoformat()
            warnings.append(f"Date parsing error: {str(e)}")
    else:
        cleaned['transaction_date'] = datetime.now().isoformat()
        warnings.append("Missing transaction_date, using current timestamp")
    
    # Ensure required fields exist
    required_fields = {
        'transaction_id': f'TXN_UNKNOWN_{datetime.now().timestamp()}',
        'account_id': 'ACCOUNT_UNKNOWN',
        'payment_type': 'UNKNOWN'
    }
    
    for field, default_value in required_fields.items():
        if not cleaned.get(field):
            cleaned[field] = default_value
            warnings.append(f"Missing {field}, set to {default_value}")
    
    # Add account_key for compatibility
    if 'account_id' in cleaned and 'account_key' not in cleaned:
        cleaned['account_key'] = cleaned['account_id']
    
    return cleaned, warnings


def handle_referential_api_failure(fallback_data: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Handle referential API unavailability with fallback data
    """
    logger.warning("Referential API unavailable, using fallback data")
    
    if fallback_data:
        return fallback_data
    
    # Default fallback
    return {
        'exchange_rates': {
            'USD': 1.0, 'EUR': 0.91, 'GBP': 0.78, 'INR': 83.2,
            'CNY': 7.10, 'JPY': 142.5, 'AED': 3.67, 'BRL': 5.00
        },
        'high_risk_countries': {
            'Level_1': ['DE', 'US', 'FR', 'GB', 'CA'],
            'Level_2': ['AE', 'BR', 'IN', 'ZA', 'MX'],
            'Level_3': ['IR', 'KP', 'SY', 'RU', 'CU'],
            'scores': {'Level_1': 2, 'Level_2': 4, 'Level_3': 10}
        },
        'suspicious_keywords': [
            'gift', 'donation', 'offshore', 'cash', 'urgent',
            'invoice 999', 'crypto', 'Hawala', 'Shell', 'bearer', 'sensitive'
        ]
    }


def log_validation_errors(transaction_id: str, warnings: list) -> None:
    """Log all validation warnings for a transaction"""
    if warnings:
        error_log_file = 'logs/aml360_errors.log'
        with open(error_log_file, 'a') as f:
            timestamp = datetime.now().isoformat()
            f.write(f"{timestamp} | {transaction_id} | Validation warnings: {'; '.join(warnings)}\n")


def safe_currency_conversion(amount: float, from_currency: str, exchange_rates: Dict[str, float]) -> float:
    """Safely convert currency with fallbacks"""
    try:
        if from_currency == 'USD':
            return amount
        
        rate = exchange_rates.get(from_currency)
        if rate is None or rate == 0:
            logger.warning(f"Unknown or invalid exchange rate for {from_currency}, treating as USD")
            return amount
        
        # Convert to USD
        return amount / rate
    except Exception as e:
        logger.error(f"Currency conversion error: {e}, treating as USD")
        return amount
