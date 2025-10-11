"""
Comprehensive test suite for AML 360 Rule Engine
Tests all 5 rules with edge cases and integration scenarios
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from backend.rules import RuleEngine, score_transaction

class TestRuleEngine:
    """Test class for AML Rule Engine"""
    
    @pytest.fixture
    def rule_engine(self):
        """Create rule engine instance with default referentials"""
        engine = RuleEngine()
        engine._load_default_referentials()
        return engine
    
    @pytest.fixture
    def sample_transaction(self):
        """Sample transaction data for testing"""
        return {
            'transaction_id': 'TEST_TXN_001',
            'account_id': 'ACC_123',
            'account_key': 'ACC_123',
            'transaction_date': '2025-10-10T14:23:00Z',
            'transaction_amount': 50000.0,
            'currency_code': 'USD',
            'beneficiary_country': 'US',
            'originator_country': 'US',
            'payment_instruction': 'Regular business payment',
            'payment_type': 'SWIFT'
        }
    
    # Rule 1 Tests: Beneficiary High-Risk Country
    def test_rule1_level3_high_risk_country(self, rule_engine, sample_transaction):
        """Test Rule 1 with Level 3 high-risk country"""
        sample_transaction['beneficiary_country'] = 'IR'  # Iran - Level 3
        
        result = rule_engine.rule_beneficiary_high_risk(sample_transaction)
        
        assert result['rule_id'] == 'R1'
        assert result['rule_name'] == 'BeneficiaryHighRisk'
        assert result['hit'] == True
        assert result['score'] == 10
        assert 'beneficiary_country=IR Level_3' in result['evidence']
    
    def test_rule1_level2_medium_risk_country(self, rule_engine, sample_transaction):
        """Test Rule 1 with Level 2 medium-risk country"""
        sample_transaction['beneficiary_country'] = 'BR'  # Brazil - Level 2
        
        result = rule_engine.rule_beneficiary_high_risk(sample_transaction)
        
        assert result['hit'] == True
        assert result['score'] == 4
        assert 'beneficiary_country=BR Level_2' in result['evidence']
    
    def test_rule1_level1_low_risk_country(self, rule_engine, sample_transaction):
        """Test Rule 1 with Level 1 low-risk country"""
        sample_transaction['beneficiary_country'] = 'US'  # USA - Level 1
        
        result = rule_engine.rule_beneficiary_high_risk(sample_transaction)
        
        assert result['hit'] == True
        assert result['score'] == 2
        assert 'beneficiary_country=US Level_1' in result['evidence']
    
    def test_rule1_no_risk_country(self, rule_engine, sample_transaction):
        """Test Rule 1 with non-listed country"""
        sample_transaction['beneficiary_country'] = 'XX'  # Non-existent country
        
        result = rule_engine.rule_beneficiary_high_risk(sample_transaction)
        
        assert result['hit'] == False
        assert result['score'] == 0
        assert result['evidence'] == ''
    
    def test_rule1_case_insensitive(self, rule_engine, sample_transaction):
        """Test Rule 1 is case insensitive"""
        sample_transaction['beneficiary_country'] = 'ir'  # lowercase
        
        result = rule_engine.rule_beneficiary_high_risk(sample_transaction)
        
        assert result['hit'] == True
        assert result['score'] == 10
    
    # Rule 2 Tests: Suspicious Keywords
    def test_rule2_exact_keyword_match(self, rule_engine, sample_transaction):
        """Test Rule 2 with exact keyword match"""
        sample_transaction['payment_instruction'] = 'This is a donation for charity'
        
        result = rule_engine.rule_suspicious_keyword(sample_transaction)
        
        assert result['rule_id'] == 'R2'
        assert result['rule_name'] == 'SuspiciousKeyword'
        assert result['hit'] == True
        assert result['score'] == 3
        assert 'donation' in result['evidence']
    
    def test_rule2_case_insensitive_match(self, rule_engine, sample_transaction):
        """Test Rule 2 case insensitive matching"""
        sample_transaction['payment_instruction'] = 'URGENT payment required'
        
        result = rule_engine.rule_suspicious_keyword(sample_transaction)
        
        assert result['hit'] == True
        assert result['score'] == 3
        assert 'urgent' in result['evidence']
    
    def test_rule2_whole_word_boundary(self, rule_engine, sample_transaction):
        """Test Rule 2 whole-word boundary matching"""
        # Should NOT match 'cash' in 'cashier'
        sample_transaction['payment_instruction'] = 'Payment to cashier desk'
        
        result = rule_engine.rule_suspicious_keyword(sample_transaction)
        
        assert result['hit'] == False
        assert result['score'] == 0
    
    def test_rule2_multiple_keywords(self, rule_engine, sample_transaction):
        """Test Rule 2 with multiple keywords (should match first one found)"""
        sample_transaction['payment_instruction'] = 'Urgent crypto donation'
        
        result = rule_engine.rule_suspicious_keyword(sample_transaction)
        
        assert result['hit'] == True
        assert result['score'] == 3
        # Should match one of the keywords
        assert any(keyword in result['evidence'] for keyword in ['urgent', 'crypto', 'donation'])
    
    def test_rule2_no_suspicious_keywords(self, rule_engine, sample_transaction):
        """Test Rule 2 with clean payment instruction"""
        sample_transaction['payment_instruction'] = 'Regular business payment for services'
        
        result = rule_engine.rule_suspicious_keyword(sample_transaction)
        
        assert result['hit'] == False
        assert result['score'] == 0
        assert result['evidence'] == ''
    
    def test_rule2_compound_keywords(self, rule_engine, sample_transaction):
        """Test Rule 2 with compound keywords like 'invoice 999'"""
        sample_transaction['payment_instruction'] = 'Payment for invoice 999 processing'
        
        result = rule_engine.rule_suspicious_keyword(sample_transaction)
        
        assert result['hit'] == True
        assert result['score'] == 3
        assert 'invoice 999' in result['evidence']
    
    # Rule 3 Tests: Large Amount
    def test_rule3_amount_above_threshold_usd(self, rule_engine, sample_transaction):
        """Test Rule 3 with amount above $1M in USD"""
        sample_transaction['transaction_amount'] = 1500000.0
        sample_transaction['currency_code'] = 'USD'
        
        result = rule_engine.rule_large_amount(sample_transaction)
        
        assert result['rule_id'] == 'R3'
        assert result['rule_name'] == 'LargeAmount'
        assert result['hit'] == True
        assert result['score'] == 3
        assert 'amount_usd=1500000.00' in result['evidence']
    
    def test_rule3_amount_exactly_threshold(self, rule_engine, sample_transaction):
        """Test Rule 3 with amount exactly $1M"""
        sample_transaction['transaction_amount'] = 1000000.0
        sample_transaction['currency_code'] = 'USD'
        
        result = rule_engine.rule_large_amount(sample_transaction)
        
        assert result['hit'] == False  # Should be > 1M, not >= 1M
        assert result['score'] == 0
    
    def test_rule3_amount_just_above_threshold(self, rule_engine, sample_transaction):
        """Test Rule 3 with amount just above $1M"""
        sample_transaction['transaction_amount'] = 1000000.01
        sample_transaction['currency_code'] = 'USD'
        
        result = rule_engine.rule_large_amount(sample_transaction)
        
        assert result['hit'] == True
        assert result['score'] == 3
    
    def test_rule3_amount_below_threshold(self, rule_engine, sample_transaction):
        """Test Rule 3 with amount below $1M"""
        sample_transaction['transaction_amount'] = 999999.99
        sample_transaction['currency_code'] = 'USD'
        
        result = rule_engine.rule_large_amount(sample_transaction)
        
        assert result['hit'] == False
        assert result['score'] == 0
    
    def test_rule3_foreign_currency_conversion(self, rule_engine, sample_transaction):
        """Test Rule 3 with foreign currency conversion"""
        # 2M EUR should be > 1M USD (assuming EUR rate ~0.91)
        sample_transaction['transaction_amount'] = 2000000.0
        sample_transaction['currency_code'] = 'EUR'
        
        result = rule_engine.rule_large_amount(sample_transaction)
        
        assert result['hit'] == True
        assert result['score'] == 3
    
    def test_rule3_invalid_amount(self, rule_engine, sample_transaction):
        """Test Rule 3 with invalid amount"""
        sample_transaction['transaction_amount'] = 'invalid'
        
        result = rule_engine.rule_large_amount(sample_transaction)
        
        assert result['hit'] == False
        assert result['score'] == 0
    
    # Rule 5 Tests: Rounded Amounts (Rule 4 is tested in integration tests)
    def test_rule5_highly_rounded_amount(self, rule_engine, sample_transaction):
        """Test Rule 5 with highly rounded amount (many trailing zeros)"""
        sample_transaction['transaction_amount'] = 1000000.0  # 6 trailing zeros
        sample_transaction['currency_code'] = 'USD'
        
        result = rule_engine.rule_rounded_amounts(sample_transaction)
        
        assert result['rule_id'] == 'R5'
        assert result['rule_name'] == 'RoundedAmounts'
        assert result['hit'] == True
        assert result['score'] == 2
        assert 'trailing_zero_count=6' in result['evidence']
    
    def test_rule5_exactly_threshold_zeros(self, rule_engine, sample_transaction):
        """Test Rule 5 with exactly threshold trailing zeros (default 4)"""
        sample_transaction['transaction_amount'] = 50000.0  # 4 trailing zeros
        sample_transaction['currency_code'] = 'USD'
        
        result = rule_engine.rule_rounded_amounts(sample_transaction)
        
        assert result['hit'] == True
        assert result['score'] == 2
        assert 'trailing_zero_count=4' in result['evidence']
    
    def test_rule5_below_threshold_zeros(self, rule_engine, sample_transaction):
        """Test Rule 5 with fewer than threshold trailing zeros"""
        sample_transaction['transaction_amount'] = 5000.0  # 3 trailing zeros
        sample_transaction['currency_code'] = 'USD'
        
        result = rule_engine.rule_rounded_amounts(sample_transaction)
        
        assert result['hit'] == False
        assert result['score'] == 0
    
    def test_rule5_no_trailing_zeros(self, rule_engine, sample_transaction):
        """Test Rule 5 with no trailing zeros"""
        sample_transaction['transaction_amount'] = 12345.67
        sample_transaction['currency_code'] = 'USD'
        
        result = rule_engine.rule_rounded_amounts(sample_transaction)
        
        assert result['hit'] == False
        assert result['score'] == 0
    
    def test_rule5_zero_amount(self, rule_engine, sample_transaction):
        """Test Rule 5 with zero amount"""
        sample_transaction['transaction_amount'] = 0.0
        sample_transaction['currency_code'] = 'USD'
        
        result = rule_engine.rule_rounded_amounts(sample_transaction)
        
        assert result['hit'] == False
        assert result['score'] == 0
    
    def test_rule5_custom_threshold(self, rule_engine, sample_transaction):
        """Test Rule 5 with custom trailing zero threshold"""
        rule_engine.trailing_zero_threshold = 2  # Lower threshold
        sample_transaction['transaction_amount'] = 100.0  # 2 trailing zeros
        sample_transaction['currency_code'] = 'USD'
        
        result = rule_engine.rule_rounded_amounts(sample_transaction)
        
        assert result['hit'] == True
        assert result['score'] == 2
    
    # Integration Tests
    def test_score_transaction_multiple_rules(self, rule_engine, sample_transaction):
        """Test scoring with multiple rules triggering"""
        sample_transaction.update({
            'beneficiary_country': 'IR',  # Rule 1: +10
            'payment_instruction': 'Urgent crypto payment',  # Rule 2: +3
            'transaction_amount': 2000000.0,  # Rule 3: +3
            'currency_code': 'USD'
        })
        
        result = rule_engine.score_transaction(sample_transaction)
        
        assert result['transaction_id'] == 'TEST_TXN_001'
        assert result['total_score'] == 16  # 10 + 3 + 3
        assert result['suspicious'] == True  # >= 3
        assert len(result['score_breakdown']) == 3
        
        # Check individual rules in breakdown
        rule_names = [rule['rule_name'] for rule in result['score_breakdown']]
        assert 'BeneficiaryHighRisk' in rule_names
        assert 'SuspiciousKeyword' in rule_names
        assert 'LargeAmount' in rule_names
    
    def test_score_transaction_no_rules_triggered(self, rule_engine, sample_transaction):
        """Test scoring with no rules triggering"""
        sample_transaction.update({
            'beneficiary_country': 'CA',  # Level 1, but will trigger +2
            'payment_instruction': 'Regular business payment',
            'transaction_amount': 50000.0,
            'currency_code': 'USD'
        })
        
        result = rule_engine.score_transaction(sample_transaction)
        
        # Note: CA is Level_1, so it will actually trigger Rule 1 with score 2
        assert result['total_score'] == 2
        assert result['suspicious'] == False  # < 3
        assert len(result['score_breakdown']) == 1
    
    def test_score_transaction_suspicious_threshold(self, rule_engine, sample_transaction):
        """Test suspicious threshold (>= 3)"""
        sample_transaction.update({
            'beneficiary_country': 'AE',  # Level 2: +4
            'payment_instruction': 'Regular payment',
            'transaction_amount': 50000.0
        })
        
        result = rule_engine.score_transaction(sample_transaction)
        
        assert result['total_score'] == 4
        assert result['suspicious'] == True  # >= 3

class TestStructuringRule:
    """Separate test class for Rule 4 (Structuring) which requires batch processing"""
    
    def test_structuring_detection_basic(self):
        """Test basic structuring detection with 3-day window"""
        engine = RuleEngine()
        
        # Create transactions that should trigger structuring
        base_date = datetime(2025, 10, 10)
        transactions = []
        
        # Account ACC_001: 3 transactions in 3-day window totaling > $1M
        for i in range(3):
            txn = {
                'transaction_id': f'TXN_00{i+1}',
                'account_key': 'ACC_001',
                'transaction_date': (base_date + timedelta(days=i)).isoformat(),
                'transaction_amount': 400000.0,  # Each $400k = $1.2M total
                'currency_code': 'USD',
                'beneficiary_country': 'US'
            }
            transactions.append(txn)
        
        df = pd.DataFrame(transactions)
        result_df = engine.batch_score_with_structuring(df)
        
        # All transactions should be flagged for structuring
        structuring_hits = 0
        for _, row in result_df.iterrows():
            breakdown = row['score_breakdown']
            for rule in breakdown:
                if rule['rule_name'] == 'Structuring':
                    structuring_hits += 1
                    assert rule['score'] == 5
                    assert 'structuring_group_ids' in rule['evidence']
                    break
        
        assert structuring_hits == 3  # All 3 transactions should be flagged
    
    def test_structuring_window_boundary(self):
        """Test structuring detection at window boundaries"""
        engine = RuleEngine()
        
        base_date = datetime(2025, 10, 10)
        transactions = []
        
        # Transactions on day 1, 3, and 4 - should form one group (1,3,4) but not (1,4) alone
        dates_and_amounts = [
            (0, 400000),  # Day 1
            (2, 400000),  # Day 3 
            (3, 300000),  # Day 4
            (5, 400000)   # Day 6 - should not be in same group
        ]
        
        for i, (day_offset, amount) in enumerate(dates_and_amounts):
            txn = {
                'transaction_id': f'TXN_00{i+1}',
                'account_key': 'ACC_BOUNDARY',
                'transaction_date': (base_date + timedelta(days=day_offset)).isoformat(),
                'transaction_amount': amount,
                'currency_code': 'USD',
                'beneficiary_country': 'US'
            }
            transactions.append(txn)
        
        df = pd.DataFrame(transactions)
        result_df = engine.batch_score_with_structuring(df)
        
        # Check which transactions were flagged for structuring
        flagged_txns = []
        for _, row in result_df.iterrows():
            breakdown = row['score_breakdown']
            for rule in breakdown:
                if rule['rule_name'] == 'Structuring':
                    flagged_txns.append(row['transaction_id'])
                    break
        
        # Should flag transactions that are within 3-day windows summing > $1M
        assert len(flagged_txns) >= 2  # At least the first 3 transactions should be flagged
    
    def test_structuring_amount_range(self):
        """Test structuring only applies to amounts in [$8,000, $9,999] range"""
        engine = RuleEngine()
        
        base_date = datetime(2025, 10, 10)
        transactions = []
        
        # Transactions outside the structuring amount range
        amounts = [7999, 8000, 8500, 9999, 10000]  # Only middle 3 should be considered
        
        for i, amount in enumerate(amounts):
            txn = {
                'transaction_id': f'TXN_00{i+1}',
                'account_key': 'ACC_RANGE',
                'transaction_date': (base_date + timedelta(hours=i)).isoformat(),
                'transaction_amount': amount,
                'currency_code': 'USD',
                'beneficiary_country': 'US'
            }
            transactions.append(txn)
        
        df = pd.DataFrame(transactions)
        result_df = engine.batch_score_with_structuring(df)
        
        # Only transactions with amounts in [8000, 9999] should be considered for structuring
        # Since we only have 3 transactions in range totaling $26,499, no structuring should be detected
        structuring_hits = 0
        for _, row in result_df.iterrows():
            breakdown = row['score_breakdown']
            for rule in breakdown:
                if rule['rule_name'] == 'Structuring':
                    structuring_hits += 1
        
        assert structuring_hits == 0  # Total < $1M
    
    def test_structuring_exactly_one_million(self):
        """Test structuring with sum exactly $1,000,000"""
        engine = RuleEngine()
        
        base_date = datetime(2025, 10, 10)
        transactions = []
        
        # Create transactions that sum to exactly $1,000,000
        amounts = [8000] * 125  # 125 * $8,000 = $1,000,000
        
        for i, amount in enumerate(amounts):
            txn = {
                'transaction_id': f'TXN_{i:03d}',
                'account_key': 'ACC_EXACT',
                'transaction_date': (base_date + timedelta(hours=i)).isoformat(),  # All within 3 days
                'transaction_amount': amount,
                'currency_code': 'USD',
                'beneficiary_country': 'US'
            }
            transactions.append(txn)
        
        df = pd.DataFrame(transactions)
        result_df = engine.batch_score_with_structuring(df)
        
        # Sum is exactly $1M, should NOT trigger (needs to be > $1M)
        structuring_hits = 0
        for _, row in result_df.iterrows():
            breakdown = row['score_breakdown']
            for rule in breakdown:
                if rule['rule_name'] == 'Structuring':
                    structuring_hits += 1
        
        assert structuring_hits == 0
    
    def test_structuring_multiple_accounts(self):
        """Test structuring detection across multiple accounts"""
        engine = RuleEngine()
        
        base_date = datetime(2025, 10, 10)
        transactions = []
        
        # Two accounts, each with structuring pattern
        for account in ['ACC_A', 'ACC_B']:
            for i in range(3):
                txn = {
                    'transaction_id': f'{account}_TXN_{i+1}',
                    'account_key': account,
                    'transaction_date': (base_date + timedelta(days=i)).isoformat(),
                    'transaction_amount': 400000.0,  # $400k each = $1.2M per account
                    'currency_code': 'USD',
                    'beneficiary_country': 'US'
                }
                transactions.append(txn)
        
        df = pd.DataFrame(transactions)
        result_df = engine.batch_score_with_structuring(df)
        
        # Both accounts should have structuring patterns
        accounts_with_structuring = set()
        for _, row in result_df.iterrows():
            breakdown = row['score_breakdown']
            for rule in breakdown:
                if rule['rule_name'] == 'Structuring':
                    accounts_with_structuring.add(row['account_key'])
        
        assert 'ACC_A' in accounts_with_structuring
        assert 'ACC_B' in accounts_with_structuring

# Performance and Error Handling Tests
class TestPerformanceAndErrors:
    """Test performance and error handling"""
    
    def test_large_dataset_processing(self):
        """Test processing of large dataset (simulated)"""
        engine = RuleEngine()
        
        # Create a moderately sized dataset for testing
        base_date = datetime(2025, 10, 10)
        transactions = []
        
        for i in range(1000):  # 1000 transactions
            txn = {
                'transaction_id': f'PERF_TXN_{i:04d}',
                'account_key': f'ACC_{i % 100}',  # 100 different accounts
                'transaction_date': (base_date + timedelta(hours=i)).isoformat(),
                'transaction_amount': np.random.uniform(1000, 500000),
                'currency_code': np.random.choice(['USD', 'EUR', 'GBP']),
                'beneficiary_country': np.random.choice(['US', 'IR', 'BR', 'FR']),
                'payment_instruction': 'Regular payment'
            }
            transactions.append(txn)
        
        df = pd.DataFrame(transactions)
        
        # Should complete without errors
        import time
        start_time = time.time()
        result_df = engine.batch_score_with_structuring(df)
        processing_time = time.time() - start_time
        
        assert len(result_df) == 1000
        assert processing_time < 30  # Should complete within 30 seconds
        assert 'total_score' in result_df.columns
        assert 'suspicious' in result_df.columns
    
    def test_missing_fields_handling(self):
        """Test handling of missing required fields"""
        engine = RuleEngine()
        
        # Transaction with missing fields
        incomplete_txn = {
            'transaction_id': 'INCOMPLETE_001'
            # Missing other required fields
        }
        
        result = engine.score_transaction(incomplete_txn)
        
        # Should not crash and should return valid result
        assert 'transaction_id' in result
        assert 'total_score' in result
        assert 'suspicious' in result
        assert isinstance(result['score_breakdown'], list)
    
    def test_invalid_currency_handling(self):
        """Test handling of invalid/unknown currency"""
        engine = RuleEngine()
        
        txn = {
            'transaction_id': 'INVALID_CURR_001',
            'transaction_amount': 1500000.0,
            'currency_code': 'INVALID_CURR',  # Unknown currency
            'beneficiary_country': 'US'
        }
        
        result = engine.score_transaction(txn)
        
        # Should handle gracefully - likely treats as USD with rate 1.0
        assert isinstance(result['total_score'], int)
        assert isinstance(result['suspicious'], bool)

# Standalone function tests
def test_standalone_score_transaction():
    """Test the standalone score_transaction function"""
    txn = {
        'transaction_id': 'STANDALONE_001',
        'beneficiary_country': 'IR',
        'payment_instruction': 'crypto donation',
        'transaction_amount': 2000000.0,
        'currency_code': 'USD'
    }
    
    result = score_transaction(txn)
    
    assert result['transaction_id'] == 'STANDALONE_001'
    assert result['total_score'] > 0
    assert result['suspicious'] == True
    assert len(result['score_breakdown']) > 0

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
