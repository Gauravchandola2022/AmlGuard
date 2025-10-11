"""
AML 360 Batch Scoring Script
Processes CSV transactions and applies rule engine scoring
"""

import pandas as pd
import argparse
import json
import time
from pathlib import Path
import logging
from typing import Optional
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from backend.rules import RuleEngine

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

class BatchScorer:
    def __init__(self, referential_url: str = "http://localhost:8001/api"):
        self.rule_engine = RuleEngine(referential_url)
        
    def load_transactions(self, file_path: str) -> pd.DataFrame:
        """Load transactions from CSV file"""
        try:
            df = pd.read_csv(file_path)
            logger.info(f"Loaded {len(df)} transactions from {file_path}")
            
            # Validate required columns
            required_cols = [
                'transaction_id', 'transaction_date', 'transaction_amount', 
                'currency_code', 'beneficiary_country', 'payment_instruction'
            ]
            
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                logger.warning(f"Missing columns: {missing_cols}")
                # Try to map alternative column names
                if 'account_id' not in df.columns and 'account_key' in df.columns:
                    df['account_id'] = df['account_key']
                if 'amount' not in df.columns and 'transaction_amount' in df.columns:
                    df['amount'] = df['transaction_amount']
                if 'currency' not in df.columns and 'currency_code' in df.columns:
                    df['currency'] = df['currency_code']
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading transactions: {e}")
            raise
    
    def process_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process batch of transactions with rule engine"""
        logger.info("Starting batch processing...")
        start_time = time.time()
        
        # Apply rule engine
        scored_df = self.rule_engine.batch_score_with_structuring(df)
        
        # Add metadata
        scored_df['processed_at'] = pd.Timestamp.now()
        
        processing_time = time.time() - start_time
        flagged_count = scored_df['suspicious'].sum()
        
        logger.info(f"Batch processing complete:")
        logger.info(f"  - Total transactions: {len(scored_df)}")
        logger.info(f"  - Flagged transactions: {flagged_count}")
        logger.info(f"  - Flagged rate: {flagged_count/len(scored_df)*100:.2f}%")
        logger.info(f"  - Processing time: {processing_time:.2f} seconds")
        
        return scored_df
    
    def save_results(self, df: pd.DataFrame, output_path: str):
        """Save scored results to CSV with PII masking for export"""
        try:
            # Import PII masking
            sys.path.append(str(Path(__file__).parent.parent))
            from utils.pii_masking import mask_name, mask_account_number, sanitize_payment_instruction
            
            # Create a copy for export
            export_df = df.copy()
            
            # Mask PII fields for export
            if 'originator_name' in export_df.columns:
                export_df['originator_name'] = export_df['originator_name'].apply(lambda x: mask_name(str(x)) if pd.notna(x) else x)
            
            if 'beneficiary_name' in export_df.columns:
                export_df['beneficiary_name'] = export_df['beneficiary_name'].apply(lambda x: mask_name(str(x)) if pd.notna(x) else x)
            
            # Remove or mask address fields
            address_fields = ['originator_address1', 'originator_address2', 'beneficiary_address1', 'beneficiary_address2']
            for field in address_fields:
                if field in export_df.columns:
                    export_df[field] = '***MASKED***'
            
            # Mask account numbers - show only last 4 digits
            if 'originator_account_number' in export_df.columns:
                export_df['originator_account_number'] = export_df['originator_account_number'].apply(
                    lambda x: mask_account_number(str(x)) if pd.notna(x) else x
                )
            
            if 'beneficiary_account_number' in export_df.columns:
                export_df['beneficiary_account_number'] = export_df['beneficiary_account_number'].apply(
                    lambda x: mask_account_number(str(x)) if pd.notna(x) else x
                )
            
            # Sanitize payment instructions
            if 'payment_instruction' in export_df.columns:
                export_df['payment_instruction'] = export_df['payment_instruction'].apply(
                    lambda x: sanitize_payment_instruction(str(x)) if pd.notna(x) else x
                )
            
            # Convert score_breakdown to JSON string for CSV export
            if 'score_breakdown' in export_df.columns:
                export_df['score_breakdown_json'] = export_df['score_breakdown'].apply(json.dumps)
                # Remove the list column
                export_df = export_df.drop('score_breakdown', axis=1)
            
            # Create output directory if it doesn't exist
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            export_df.to_csv(output_path, index=False)
            logger.info(f"Results saved to {output_path} (PII masked)")
            
        except Exception as e:
            logger.error(f"Error saving results: {e}")
            raise
    
    def generate_summary_report(self, df: pd.DataFrame) -> dict:
        """Generate summary report of scoring results"""
        try:
            total_transactions = len(df)
            flagged_transactions = df['suspicious'].sum()
            flagged_rate = (flagged_transactions / total_transactions * 100) if total_transactions > 0 else 0
            
            # Rule breakdown
            rule_hits = {}
            for _, row in df.iterrows():
                if isinstance(row.get('score_breakdown'), list):
                    for rule in row['score_breakdown']:
                        rule_name = rule.get('rule_name', 'Unknown')
                        rule_hits[rule_name] = rule_hits.get(rule_name, 0) + 1
            
            # Amount statistics
            amount_stats = {}
            if 'amount_usd' in df.columns:
                amount_stats = {
                    'min': float(df['amount_usd'].min()),
                    'max': float(df['amount_usd'].max()),
                    'mean': float(df['amount_usd'].mean()),
                    'median': float(df['amount_usd'].median())
                }
            
            # Top beneficiary countries by flags
            top_countries = {}
            if flagged_transactions > 0:
                flagged_df = df[df['suspicious'] == True]
                country_counts = flagged_df['beneficiary_country'].value_counts().head(10)
                top_countries = country_counts.to_dict()
            
            report = {
                'summary': {
                    'total_transactions': int(total_transactions),
                    'flagged_transactions': int(flagged_transactions),
                    'flagged_rate_percent': round(flagged_rate, 2),
                    'processing_timestamp': pd.Timestamp.now().isoformat()
                },
                'rule_hits': rule_hits,
                'amount_statistics_usd': amount_stats,
                'top_flagged_countries': top_countries
            }
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating summary report: {e}")
            return {}

def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(description="AML 360 Batch Transaction Scorer")
    parser.add_argument("--input", required=True, help="Input CSV file path")
    parser.add_argument("--output", required=True, help="Output CSV file path")
    parser.add_argument("--referential", default="http://localhost:8001/api", 
                       help="Referential service URL")
    parser.add_argument("--report", help="Optional summary report JSON file path")
    parser.add_argument("--chunk-size", type=int, default=10000, 
                       help="Chunk size for processing large files")
    
    args = parser.parse_args()
    
    try:
        # Initialize scorer
        scorer = BatchScorer(args.referential)
        
        # Load and process
        df = scorer.load_transactions(args.input)
        
        # Process in chunks for large files
        if len(df) > args.chunk_size:
            logger.info(f"Processing in chunks of {args.chunk_size}")
            results = []
            
            for i in range(0, len(df), args.chunk_size):
                chunk = df.iloc[i:i+args.chunk_size]
                logger.info(f"Processing chunk {i//args.chunk_size + 1}/{(len(df)-1)//args.chunk_size + 1}")
                scored_chunk = scorer.process_batch(chunk)
                results.append(scored_chunk)
            
            scored_df = pd.concat(results, ignore_index=True)
        else:
            scored_df = scorer.process_batch(df)
        
        # Save results
        scorer.save_results(scored_df, args.output)
        
        # Generate report if requested
        if args.report:
            report = scorer.generate_summary_report(scored_df)
            with open(args.report, 'w') as f:
                json.dump(report, f, indent=2)
            logger.info(f"Summary report saved to {args.report}")
        
        logger.info("Batch scoring completed successfully!")
        
    except Exception as e:
        logger.error(f"Batch scoring failed: {e}")
        exit(1)

if __name__ == "__main__":
    main()
