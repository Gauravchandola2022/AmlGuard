"""
ML Model Training Script for AML 360
Trains a RandomForest classifier on scored transaction data
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from backend.ml_model import AMLMLModel
from backend.rules import RuleEngine
from backend.database import get_database


def generate_synthetic_training_data(n_samples: int = 1000) -> pd.DataFrame:
    """Generate synthetic training data for model training"""
    
    np.random.seed(42)
    
    # Generate transaction data
    transactions = []
    
    countries = ['US', 'GB', 'FR', 'DE', 'CA', 'AE', 'BR', 'IN', 'ZA', 'MX', 'IR', 'KP', 'SY', 'RU', 'CU']
    payment_types = ['SWIFT', 'ACH', 'WIRE', 'SEPA', 'IMPS', 'NEFT']
    currencies = ['USD', 'EUR', 'GBP', 'INR', 'CNY', 'JPY', 'AED', 'BRL']
    
    for i in range(n_samples):
        # Generate features with some correlation to suspicious activity
        is_suspicious = np.random.random() < 0.15  # 15% suspicious rate
        
        # High-risk countries more likely for suspicious transactions
        if is_suspicious:
            beneficiary_country = np.random.choice(['IR', 'KP', 'SY', 'RU', 'CU'], p=[0.3, 0.2, 0.2, 0.2, 0.1])
            amount = np.random.choice([
                np.random.uniform(8000, 9999),  # Structuring range
                np.random.uniform(1000000, 5000000),  # Large amount
                np.random.uniform(100000, 200000) * 10  # Rounded amount
            ])
        else:
            beneficiary_country = np.random.choice(['US', 'GB', 'FR', 'DE', 'CA', 'AE', 'BR', 'IN'])
            amount = np.random.uniform(100, 50000)
        
        transaction = {
            'transaction_id': f'TXN_{i:06d}',
            'account_id': f'ACC_{np.random.randint(1, 100):03d}',
            'transaction_date': f'2025-{np.random.randint(1, 13):02d}-{np.random.randint(1, 29):02d}',
            'originator_country': np.random.choice(countries),
            'beneficiary_country': beneficiary_country,
            'amount': amount,
            'currency': np.random.choice(currencies),
            'payment_type': np.random.choice(payment_types),
            'payment_instruction': 'test payment',
            'is_suspicious': is_suspicious
        }
        
        transactions.append(transaction)
    
    return pd.DataFrame(transactions)


def main():
    """Train and save the ML model"""
    
    print("AML 360 Model Training Script")
    print("=" * 50)
    
    # Check if we have real data in database
    db = get_database()
    real_data = db.get_flagged_transactions(limit=10000)
    
    if real_data and len(real_data) > 100:
        print(f"Using {len(real_data)} flagged transactions from database")
        df = pd.DataFrame(real_data)
        # Add some normal transactions for balance
        synthetic_normal = generate_synthetic_training_data(n_samples=len(real_data) * 5)
        df = pd.concat([df, synthetic_normal], ignore_index=True)
    else:
        print("Generating synthetic training data...")
        df = generate_synthetic_training_data(n_samples=2000)
    
    # Score transactions using rule engine
    print("Scoring transactions with rule engine...")
    rule_engine = RuleEngine()
    
    scored_transactions = []
    for _, row in df.iterrows():
        txn_dict = row.to_dict()
        score_result = rule_engine.score_transaction(txn_dict, {})
        txn_dict.update(score_result)
        scored_transactions.append(txn_dict)
    
    df_scored = pd.DataFrame(scored_transactions)
    
    # Create target variable
    # In real scenario, this would be actual labels from investigations
    # For training, we'll use suspicious flag with some noise
    if 'is_suspicious' in df_scored.columns:
        df_scored['label'] = df_scored['is_suspicious'].astype(int)
    else:
        # Use suspicious flag from rules as proxy
        df_scored['label'] = df_scored['suspicious'].astype(int)
    
    # Ensure we have both classes
    if df_scored['label'].sum() == 0:
        print("Warning: No positive samples, adding synthetic positive samples")
        # Add some positive samples
        positive_indices = df_scored.nlargest(int(len(df_scored) * 0.1), 'total_score').index
        df_scored.loc[positive_indices, 'label'] = 1
    
    print(f"Training data: {len(df_scored)} transactions")
    print(f"Positive samples: {df_scored['label'].sum()} ({df_scored['label'].mean()*100:.1f}%)")
    
    # Train model
    print("\nTraining model...")
    model = AMLMLModel()
    
    # Split data by date
    df_scored['transaction_date'] = pd.to_datetime(df_scored['transaction_date'])
    df_scored = df_scored.sort_values('transaction_date')
    
    split_idx = int(len(df_scored) * 0.8)
    train_df = df_scored.iloc[:split_idx]
    test_df = df_scored.iloc[split_idx:]
    
    # Train - the model expects a dataframe with a 'label' column or 'is_suspicious' column
    metrics = model.train(train_df, target_col='label')
    
    print("\nTraining Metrics:")
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            print(f"  {key}: {value:.4f}")
    
    # Note: Test set evaluation can be done separately after model is saved
    print(f"\nTest set size: {len(test_df)} transactions")
    
    # Save model
    import joblib
    model_path = Path('models') / 'rf_model.joblib'
    model_path.parent.mkdir(exist_ok=True)
    
    model.save_model(str(model_path))
    print(f"\nModel saved to: {model_path}")
    
    # Test SHAP explanation
    if len(test_df) > 0:
        print("\nTesting SHAP explainability...")
        try:
            explanation = model.explain_prediction(test_df.head(1), transaction_idx=0)
            print("SHAP explanation generated successfully:")
            if 'top_features' in explanation:
                print("  Top contributing features:")
                for feat in explanation['top_features'][:3]:
                    print(f"    - {feat['feature']}: {feat.get('shap_value', feat.get('importance', 0)):.4f}")
        except Exception as e:
            print(f"  Warning: SHAP explanation failed: {e}")
    
    print("\nTraining complete!")


if __name__ == "__main__":
    main()
