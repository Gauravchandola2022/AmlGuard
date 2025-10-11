"""
AML 360 Database Module
Handles database operations for flagged transactions
"""

import sqlite3
import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging
from contextlib import contextmanager
from datetime import datetime
import os

from utils.logging_config import setup_logging

logger = setup_logging(__name__)

class AMLDatabase:
    def __init__(self, db_path: str = "data/aml360.db"):
        self.db_path = db_path
        
        # Create database directory if it doesn't exist
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        self._init_database()
    
    def _init_database(self):
        """Initialize database with required tables"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Create flagged_transactions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS flagged_transactions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    transaction_id TEXT UNIQUE NOT NULL,
                    account_id TEXT,
                    transaction_date TIMESTAMP,
                    amount_usd REAL,
                    total_score INTEGER,
                    suspicious BOOLEAN,
                    score_breakdown TEXT,
                    shap_summary TEXT,
                    rag_retrieval TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create cases table for investigations
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS investigation_cases (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    case_id TEXT UNIQUE NOT NULL,
                    title TEXT,
                    status TEXT DEFAULT 'open',
                    priority TEXT DEFAULT 'medium',
                    transaction_ids TEXT,
                    notes TEXT,
                    assigned_to TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create audit_log table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS audit_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    transaction_id TEXT,
                    input_hash TEXT,
                    model_version TEXT,
                    rules_version TEXT,
                    total_score INTEGER,
                    suspicious_flag BOOLEAN,
                    operation TEXT,
                    user_id TEXT,
                    details TEXT
                )
            """)
            
            conn.commit()
            logger.info("Database initialized successfully")
    
    @contextmanager
    def get_connection(self):
        """Get database connection with context manager"""
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row  # Enable dict-like access
            yield conn
        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"Database error: {e}")
            raise
        finally:
            if conn:
                conn.close()
    
    def insert_flagged_transaction(self, transaction_data: Dict[str, Any]) -> bool:
        """Insert or update flagged transaction"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # Prepare data
                data = {
                    'transaction_id': transaction_data.get('transaction_id'),
                    'account_id': transaction_data.get('account_id') or transaction_data.get('account_key'),
                    'transaction_date': transaction_data.get('transaction_date'),
                    'amount_usd': transaction_data.get('amount_usd'),
                    'total_score': transaction_data.get('total_score'),
                    'suspicious': transaction_data.get('suspicious'),
                    'score_breakdown': json.dumps(transaction_data.get('score_breakdown', [])),
                    'shap_summary': json.dumps(transaction_data.get('shap_summary', {})),
                    'rag_retrieval': json.dumps(transaction_data.get('rag_retrieval', {}))
                }
                
                # Insert or replace
                cursor.execute("""
                    INSERT OR REPLACE INTO flagged_transactions 
                    (transaction_id, account_id, transaction_date, amount_usd, 
                     total_score, suspicious, score_breakdown, shap_summary, rag_retrieval)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    data['transaction_id'], data['account_id'], data['transaction_date'],
                    data['amount_usd'], data['total_score'], data['suspicious'],
                    data['score_breakdown'], data['shap_summary'], data['rag_retrieval']
                ))
                
                conn.commit()
                return True
                
        except Exception as e:
            logger.error(f"Error inserting flagged transaction: {e}")
            return False
    
    def batch_insert_flagged_transactions(self, transactions: List[Dict[str, Any]]) -> int:
        """Batch insert flagged transactions"""
        inserted_count = 0
        
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                batch_data = []
                for txn in transactions:
                    if txn.get('suspicious', False):  # Only insert flagged transactions
                        data = (
                            txn.get('transaction_id'),
                            txn.get('account_id') or txn.get('account_key'),
                            txn.get('transaction_date'),
                            txn.get('amount_usd'),
                            txn.get('total_score'),
                            txn.get('suspicious'),
                            json.dumps(txn.get('score_breakdown', [])),
                            json.dumps(txn.get('shap_summary', {})),
                            json.dumps(txn.get('rag_retrieval', {}))
                        )
                        batch_data.append(data)
                
                if batch_data:
                    cursor.executemany("""
                        INSERT OR REPLACE INTO flagged_transactions 
                        (transaction_id, account_id, transaction_date, amount_usd, 
                         total_score, suspicious, score_breakdown, shap_summary, rag_retrieval)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, batch_data)
                    
                    conn.commit()
                    inserted_count = len(batch_data)
                    
                logger.info(f"Batch inserted {inserted_count} flagged transactions")
                
        except Exception as e:
            logger.error(f"Error in batch insert: {e}")
        
        return inserted_count
    
    def get_flagged_transactions(self, 
                               limit: int = 1000, 
                               offset: int = 0,
                               filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Get flagged transactions with optional filtering"""
        try:
            with self.get_connection() as conn:
                query = "SELECT * FROM flagged_transactions WHERE 1=1"
                params = []
                
                # Apply filters
                if filters:
                    if filters.get('date_from'):
                        query += " AND transaction_date >= ?"
                        params.append(filters['date_from'])
                    
                    if filters.get('date_to'):
                        query += " AND transaction_date <= ?"
                        params.append(filters['date_to'])
                    
                    if filters.get('min_score'):
                        query += " AND total_score >= ?"
                        params.append(filters['min_score'])
                    
                    if filters.get('max_score'):
                        query += " AND total_score <= ?"
                        params.append(filters['max_score'])
                    
                    if filters.get('country'):
                        query += " AND (score_breakdown LIKE ? OR score_breakdown LIKE ?)"
                        params.extend([f'%{filters["country"]}%', f'%{filters["country"]}%'])
                    
                    if filters.get('account_id'):
                        query += " AND account_id = ?"
                        params.append(filters['account_id'])
                
                query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
                params.extend([limit, offset])
                
                cursor = conn.cursor()
                cursor.execute(query, params)
                
                rows = cursor.fetchall()
                
                # Convert to dict and parse JSON fields
                transactions = []
                for row in rows:
                    txn = dict(row)
                    
                    # Parse JSON fields
                    try:
                        txn['score_breakdown'] = json.loads(txn['score_breakdown'] or '[]')
                    except:
                        txn['score_breakdown'] = []
                    
                    try:
                        txn['shap_summary'] = json.loads(txn['shap_summary'] or '{}')
                    except:
                        txn['shap_summary'] = {}
                    
                    try:
                        txn['rag_retrieval'] = json.loads(txn['rag_retrieval'] or '{}')
                    except:
                        txn['rag_retrieval'] = {}
                    
                    transactions.append(txn)
                
                return transactions
                
        except Exception as e:
            logger.error(f"Error retrieving flagged transactions: {e}")
            return []
    
    def get_transaction_by_id(self, transaction_id: str) -> Optional[Dict[str, Any]]:
        """Get specific transaction by ID"""
        transactions = self.get_flagged_transactions(limit=1, filters={'transaction_id': transaction_id})
        return transactions[0] if transactions else None
    
    def get_dashboard_stats(self) -> Dict[str, Any]:
        """Get dashboard statistics"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                stats = {}
                
                # Total flagged transactions
                cursor.execute("SELECT COUNT(*) as total FROM flagged_transactions")
                stats['total_flagged'] = cursor.fetchone()['total']
                
                # Average score
                cursor.execute("SELECT AVG(total_score) as avg_score FROM flagged_transactions")
                result = cursor.fetchone()
                stats['avg_score'] = round(result['avg_score'] or 0, 2)
                
                # Score distribution
                cursor.execute("""
                    SELECT 
                        COUNT(CASE WHEN total_score BETWEEN 3 AND 5 THEN 1 END) as low_risk,
                        COUNT(CASE WHEN total_score BETWEEN 6 AND 10 THEN 1 END) as medium_risk,
                        COUNT(CASE WHEN total_score > 10 THEN 1 END) as high_risk
                    FROM flagged_transactions
                """)
                score_dist = cursor.fetchone()
                stats['score_distribution'] = dict(score_dist)
                
                # Recent activity (last 7 days)
                cursor.execute("""
                    SELECT COUNT(*) as recent_count 
                    FROM flagged_transactions 
                    WHERE created_at >= datetime('now', '-7 days')
                """)
                stats['recent_flagged'] = cursor.fetchone()['recent_count']
                
                # Top countries
                cursor.execute("""
                    SELECT account_id, COUNT(*) as count
                    FROM flagged_transactions 
                    GROUP BY account_id 
                    ORDER BY count DESC 
                    LIMIT 10
                """)
                top_accounts = cursor.fetchall()
                stats['top_accounts'] = [dict(row) for row in top_accounts]
                
                return stats
                
        except Exception as e:
            logger.error(f"Error getting dashboard stats: {e}")
            return {}
    
    def create_investigation_case(self, case_data: Dict[str, Any]) -> bool:
        """Create new investigation case"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT INTO investigation_cases 
                    (case_id, title, status, priority, transaction_ids, notes, assigned_to)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    case_data.get('case_id'),
                    case_data.get('title'),
                    case_data.get('status', 'open'),
                    case_data.get('priority', 'medium'),
                    json.dumps(case_data.get('transaction_ids', [])),
                    case_data.get('notes'),
                    case_data.get('assigned_to')
                ))
                
                conn.commit()
                return True
                
        except Exception as e:
            logger.error(f"Error creating investigation case: {e}")
            return False
    
    def log_audit_event(self, event_data: Dict[str, Any]):
        """Log audit event"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT INTO audit_log 
                    (transaction_id, input_hash, model_version, rules_version, 
                     total_score, suspicious_flag, operation, user_id, details)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    event_data.get('transaction_id'),
                    event_data.get('input_hash'),
                    event_data.get('model_version'),
                    event_data.get('rules_version'),
                    event_data.get('total_score'),
                    event_data.get('suspicious_flag'),
                    event_data.get('operation'),
                    event_data.get('user_id'),
                    json.dumps(event_data.get('details', {}))
                ))
                
                conn.commit()
                
        except Exception as e:
            logger.error(f"Error logging audit event: {e}")

# Global database instance
db = AMLDatabase()

def get_database() -> AMLDatabase:
    """Get database instance"""
    return db
