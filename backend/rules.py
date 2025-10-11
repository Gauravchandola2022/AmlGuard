"""
AML 360 Rule Engine
Implements 5 deterministic rules for transaction monitoring
"""

import pandas as pd
import numpy as np
import re
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import requests
from collections import defaultdict
import math
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RuleEngine:
    def __init__(self, referentials_url: str = "http://localhost:8001/api"):
        self.referentials_url = referentials_url
        self.exchange_rates = {}
        self.high_risk_countries = {}
        self.suspicious_keywords = []
        self.fuzzy_matching = False  # Configurable
        self.trailing_zero_threshold = 4  # Configurable
        
        # Load referentials
        self._load_referentials()
    
    def _load_referentials(self):
        """Load referential data from API or fallback to defaults"""
        try:
            # Exchange rates
            response = requests.get(f"{self.referentials_url}/exchange-rates", timeout=5)
            if response.status_code == 200:
                self.exchange_rates = response.json()["rates"]
            
            # High-risk countries
            response = requests.get(f"{self.referentials_url}/high-risk-countries", timeout=5)
            if response.status_code == 200:
                data = response.json()
                self.high_risk_countries = data
            
            # Suspicious keywords
            response = requests.get(f"{self.referentials_url}/suspicious-keywords", timeout=5)
            if response.status_code == 200:
                self.suspicious_keywords = response.json()["keywords"]
                
        except Exception as e:
            logger.warning(f"Could not load referentials from API: {e}. Using defaults.")
            self._load_default_referentials()
    
    def _load_default_referentials(self):
        """Load default referential data"""
        self.exchange_rates = {
            "USD": 1.0, "EUR": 0.91, "GBP": 0.78, "INR": 83.2,
            "CNY": 7.10, "JPY": 142.5, "AED": 3.67, "BRL": 5.00
        }
        self.high_risk_countries = {
            "Level_1": ["DE", "US", "FR", "GB", "CA"],
            "Level_2": ["AE", "BR", "IN", "ZA", "MX"],
            "Level_3": ["IR", "KP", "SY", "RU", "CU"],
            "scores": {"Level_1": 2, "Level_2": 4, "Level_3": 10}
        }
        self.suspicious_keywords = [
            "gift", "donation", "offshore", "cash", "urgent",
            "invoice 999", "crypto", "Hawala", "Shell", "bearer", "sensitive"
        ]
    
    def convert_to_usd(self, amount: float, currency: str) -> float:
        """Convert amount to USD using exchange rates"""
        if currency == "USD":
            return amount
        
        rate = self.exchange_rates.get(currency, 1.0)
        if rate == 0:
            rate = 1.0
        
        return amount / rate
    
    def rule_beneficiary_high_risk(self, txn: Dict[str, Any]) -> Dict[str, Any]:
        """Rule #1: Beneficiary country in high-risk list"""
        country = txn.get("beneficiary_country", "").upper()
        
        # Find country level
        level = None
        for level_name, countries in self.high_risk_countries.items():
            if level_name.startswith("Level_") and country in countries:
                level = level_name
                break
        
        if level:
            score = self.high_risk_countries["scores"][level]
            return {
                "rule_id": "R1",
                "rule_name": "BeneficiaryHighRisk",
                "hit": True,
                "score": score,
                "evidence": f"beneficiary_country={country} {level}"
            }
        
        return {
            "rule_id": "R1",
            "rule_name": "BeneficiaryHighRisk",
            "hit": False,
            "score": 0,
            "evidence": ""
        }
    
    def rule_suspicious_keyword(self, txn: Dict[str, Any]) -> Dict[str, Any]:
        """Rule #2: Suspicious keyword in payment_instruction"""
        payment_instruction = str(txn.get("payment_instruction", "")).lower()
        
        for keyword in self.suspicious_keywords:
            keyword_lower = keyword.lower()
            
            # Whole-word boundary matching
            pattern = r'\b' + re.escape(keyword_lower) + r'\b'
            if re.search(pattern, payment_instruction):
                return {
                    "rule_id": "R2",
                    "rule_name": "SuspiciousKeyword",
                    "hit": True,
                    "score": 3,
                    "evidence": f"payment_instruction contains '{keyword}'"
                }
            
            # Optional fuzzy matching (if enabled)
            if self.fuzzy_matching:
                if self._fuzzy_match(keyword_lower, payment_instruction):
                    return {
                        "rule_id": "R2",
                        "rule_name": "SuspiciousKeyword",
                        "hit": True,
                        "score": 3,
                        "evidence": f"payment_instruction fuzzy match '{keyword}'"
                    }
        
        return {
            "rule_id": "R2",
            "rule_name": "SuspiciousKeyword",
            "hit": False,
            "score": 0,
            "evidence": ""
        }
    
    def _fuzzy_match(self, keyword: str, text: str) -> bool:
        """Simple fuzzy matching with edit distance <= 1"""
        words = re.findall(r'\b\w+\b', text)
        for word in words:
            if self._edit_distance(keyword, word) <= 1:
                return True
        return False
    
    def _edit_distance(self, s1: str, s2: str) -> int:
        """Calculate edit distance between two strings"""
        if len(s1) > len(s2):
            s1, s2 = s2, s1
        
        distances = range(len(s1) + 1)
        for i2, c2 in enumerate(s2):
            distances_ = [i2 + 1]
            for i1, c1 in enumerate(s1):
                if c1 == c2:
                    distances_.append(distances[i1])
                else:
                    distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
            distances = distances_
        return distances[-1]
    
    def rule_large_amount(self, txn: Dict[str, Any]) -> Dict[str, Any]:
        """Rule #3: Amount > $1,000,000 USD equivalent"""
        try:
            amount = float(txn.get("transaction_amount", 0))
            currency = txn.get("currency_code", "USD")
            
            amount_usd = self.convert_to_usd(amount, currency)
            
            if amount_usd > 1_000_000:
                return {
                    "rule_id": "R3",
                    "rule_name": "LargeAmount",
                    "hit": True,
                    "score": 3,
                    "evidence": f"amount_usd={amount_usd:.2f}"
                }
        except (ValueError, TypeError):
            pass
        
        return {
            "rule_id": "R3",
            "rule_name": "LargeAmount",
            "hit": False,
            "score": 0,
            "evidence": ""
        }
    
    def rule_rounded_amounts(self, txn: Dict[str, Any]) -> Dict[str, Any]:
        """Rule #5: Rounded amounts (detect trailing zeros)"""
        try:
            amount = float(txn.get("transaction_amount", 0))
            currency = txn.get("currency_code", "USD")
            
            amount_usd = self.convert_to_usd(amount, currency)
            
            # Count trailing zeros in integer part
            integer_part = int(amount_usd)
            if integer_part == 0:
                trailing_zeros = 0
            else:
                trailing_zeros = 0
                while integer_part % 10 == 0:
                    trailing_zeros += 1
                    integer_part //= 10
            
            if trailing_zeros >= self.trailing_zero_threshold:
                return {
                    "rule_id": "R5",
                    "rule_name": "RoundedAmounts",
                    "hit": True,
                    "score": 2,
                    "evidence": f"trailing_zero_count={trailing_zeros}"
                }
        except (ValueError, TypeError):
            pass
        
        return {
            "rule_id": "R5",
            "rule_name": "RoundedAmounts",
            "hit": False,
            "score": 0,
            "evidence": ""
        }
    
    def score_transaction(self, txn: Dict[str, Any], referentials: Optional[Dict] = None, state: Optional[Dict] = None) -> Dict[str, Any]:
        """Score a single transaction against all rules"""
        transaction_id = txn.get("transaction_id", "")
        
        # Apply individual rules
        rule_results = []
        
        # Rule 1: Beneficiary high-risk
        rule_results.append(self.rule_beneficiary_high_risk(txn))
        
        # Rule 2: Suspicious keywords
        rule_results.append(self.rule_suspicious_keyword(txn))
        
        # Rule 3: Large amount
        rule_results.append(self.rule_large_amount(txn))
        
        # Rule 5: Rounded amounts
        rule_results.append(self.rule_rounded_amounts(txn))
        
        # Calculate total score
        total_score = sum(rule["score"] for rule in rule_results if rule["hit"])
        
        # Determine if suspicious
        suspicious = total_score >= 3
        
        return {
            "transaction_id": transaction_id,
            "score_breakdown": [rule for rule in rule_results if rule["hit"]],
            "total_score": total_score,
            "suspicious": suspicious
        }
    
    def batch_score_with_structuring(self, df: pd.DataFrame) -> pd.DataFrame:
        """Score all transactions in batch, including structuring rule"""
        results = []
        
        # Prepare data
        df_work = df.copy()
        df_work['transaction_date'] = pd.to_datetime(df_work['transaction_date'])
        df_work['amount_usd'] = df_work.apply(
            lambda row: self.convert_to_usd(
                float(row.get('transaction_amount', 0)), 
                row.get('currency_code', 'USD')
            ), axis=1
        )
        
        # Rule 4: Structuring detection
        structuring_hits = self._detect_structuring(df_work)
        
        # Score each transaction
        for idx, row in df_work.iterrows():
            txn = row.to_dict()
            
            # Get base rules score
            score_result = self.score_transaction(txn)
            
            # Add structuring rule if hit
            if txn['transaction_id'] in structuring_hits:
                structuring_evidence = structuring_hits[txn['transaction_id']]
                structuring_rule = {
                    "rule_id": "R4",
                    "rule_name": "Structuring",
                    "hit": True,
                    "score": 5,
                    "evidence": structuring_evidence
                }
                score_result["score_breakdown"].append(structuring_rule)
                score_result["total_score"] += 5
                score_result["suspicious"] = score_result["total_score"] >= 3
            
            # Store results
            result_row = row.to_dict()
            result_row.update({
                'total_score': score_result["total_score"],
                'suspicious': score_result["suspicious"],
                'score_breakdown': score_result["score_breakdown"]
            })
            results.append(result_row)
        
        return pd.DataFrame(results)
    
    def _detect_structuring(self, df: pd.DataFrame) -> Dict[str, str]:
        """Detect structuring patterns - 3-day sum of amounts in [$8,000, $9,999] range per account > $1,000,000"""
        structuring_hits = {}
        
        # Filter transactions in structuring range
        structuring_df = df[
            (df['amount_usd'] >= 8000) & 
            (df['amount_usd'] <= 9999)
        ].copy()
        
        if structuring_df.empty:
            return structuring_hits
        
        # Sort by account and date
        structuring_df = structuring_df.sort_values(['account_key', 'transaction_date'])
        
        # Group by account
        for account_id, group in structuring_df.groupby('account_key'):
            group = group.reset_index(drop=True)
            
            # Check each 3-day window
            for i in range(len(group)):
                current_date = group.iloc[i]['transaction_date']
                window_end = current_date + timedelta(days=2)  # 3-day window inclusive
                
                # Get transactions in 3-day window
                window_txns = group[
                    (group['transaction_date'] >= current_date) &
                    (group['transaction_date'] <= window_end)
                ]
                
                # Check if sum exceeds threshold
                window_sum = window_txns['amount_usd'].sum()
                if window_sum > 1_000_000:
                    # Mark all transactions in this window
                    txn_ids = window_txns['transaction_id'].tolist()
                    evidence = f"structuring_group_ids={txn_ids}, sum_usd={window_sum:.2f}"
                    
                    for txn_id in txn_ids:
                        structuring_hits[txn_id] = evidence
        
        return structuring_hits

def score_transaction(txn: Dict[str, Any], referentials: Optional[Dict] = None, state: Optional[Dict] = None) -> Dict[str, Any]:
    """Standalone function to score a single transaction"""
    engine = RuleEngine()
    return engine.score_transaction(txn, referentials, state)
