"""
AML 360 ML Model
Training and inference for transaction scoring with explainability
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
import joblib
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from utils.logging_config import setup_logging

logger = setup_logging(__name__)

class AMLMLModel:
    def __init__(self, model_path: Optional[str] = None):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = []
        self.model_version = None
        self.explainer = None
        
        if model_path and Path(model_path).exists():
            self.load_model(model_path)
    
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for ML model"""
        features_df = pd.DataFrame()
        
        # Rule-based features
        if 'score_breakdown' in df.columns:
            # Extract rule hits as binary features
            rule_features = self._extract_rule_features(df)
            features_df = pd.concat([features_df, rule_features], axis=1)
        
        # Total rule score
        if 'total_score' in df.columns:
            features_df['total_rule_score'] = df['total_score']
        
        # Amount features
        if 'amount_usd' in df.columns:
            features_df['amount_usd'] = df['amount_usd']
            features_df['amount_usd_log'] = np.log1p(df['amount_usd'])
        elif 'transaction_amount' in df.columns:
            # Convert to USD if not already done
            features_df['amount_usd'] = df['transaction_amount']  # Assume USD for simplicity
            features_df['amount_usd_log'] = np.log1p(df['transaction_amount'])
        
        # Payment type (one-hot encoding)
        if 'payment_type' in df.columns:
            payment_dummies = pd.get_dummies(df['payment_type'], prefix='payment_type')
            features_df = pd.concat([features_df, payment_dummies], axis=1)
        
        # Country risk scores
        if 'originator_country' in df.columns:
            features_df['originator_risk_score'] = df['originator_country'].apply(self._get_country_risk_score)
        if 'beneficiary_country' in df.columns:
            features_df['beneficiary_risk_score'] = df['beneficiary_country'].apply(self._get_country_risk_score)
        
        # Account aggregates (if available)
        if 'account_id' in df.columns or 'account_key' in df.columns:
            account_col = 'account_id' if 'account_id' in df.columns else 'account_key'
            agg_features = self._create_account_aggregates(df, account_col)
            features_df = pd.concat([features_df, agg_features], axis=1)
        
        # Fill missing values
        features_df = features_df.fillna(0)
        
        return features_df
    
    def _extract_rule_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract rule-based features from score breakdown"""
        rule_features = pd.DataFrame()
        
        # Initialize rule columns
        rule_names = ['BeneficiaryHighRisk', 'SuspiciousKeyword', 'LargeAmount', 'Structuring', 'RoundedAmounts']
        
        for rule_name in rule_names:
            rule_features[f'rule_{rule_name}_hit'] = 0
            rule_features[f'rule_{rule_name}_score'] = 0
        
        # Extract features from score breakdown
        for idx, row in df.iterrows():
            score_breakdown = row.get('score_breakdown', [])
            if isinstance(score_breakdown, str):
                try:
                    score_breakdown = json.loads(score_breakdown)
                except:
                    score_breakdown = []
            
            if isinstance(score_breakdown, list):
                for rule in score_breakdown:
                    if isinstance(rule, dict):
                        rule_name = rule.get('rule_name', '')
                        if rule_name in rule_names:
                            rule_features.loc[idx, f'rule_{rule_name}_hit'] = 1
                            rule_features.loc[idx, f'rule_{rule_name}_score'] = rule.get('score', 0)
        
        return rule_features
    
    def _get_country_risk_score(self, country_code: str) -> float:
        """Get risk score for a country"""
        # Default risk mapping (simplified)
        high_risk_mapping = {
            'Level_1': 1.0,  # Low risk
            'Level_2': 2.0,  # Medium risk  
            'Level_3': 3.0   # High risk
        }
        
        # High-risk countries mapping (simplified)
        level_1 = ["DE", "US", "FR", "GB", "CA"]
        level_2 = ["AE", "BR", "IN", "ZA", "MX"]
        level_3 = ["IR", "KP", "SY", "RU", "CU"]
        
        if country_code in level_1:
            return 1.0
        elif country_code in level_2:
            return 2.0
        elif country_code in level_3:
            return 3.0
        else:
            return 1.0  # Default to low risk
    
    def _create_account_aggregates(self, df: pd.DataFrame, account_col: str) -> pd.DataFrame:
        """Create account-level aggregate features"""
        agg_features = pd.DataFrame(index=df.index)
        
        # Sort by date for rolling calculations
        if 'transaction_date' in df.columns:
            df_sorted = df.sort_values(['transaction_date']).copy()
            df_sorted['transaction_date'] = pd.to_datetime(df_sorted['transaction_date'])
            
            # Simple aggregates (last 30 days)
            agg_features['count_tx_30d'] = df_sorted.groupby(account_col).cumcount() + 1
            
            if 'amount_usd' in df.columns:
                agg_features['sum_tx_30d'] = df_sorted.groupby(account_col)['amount_usd'].cumsum()
                agg_features['avg_tx_30d'] = df_sorted.groupby(account_col)['amount_usd'].expanding().mean().values
            
            if 'suspicious' in df.columns:
                agg_features['count_suspicious_30d'] = df_sorted.groupby(account_col)['suspicious'].cumsum()
        else:
            # Fallback to simple counts
            account_counts = df[account_col].value_counts()
            agg_features['count_tx_30d'] = df[account_col].map(account_counts)
            agg_features['sum_tx_30d'] = 0
            agg_features['avg_tx_30d'] = 0
            agg_features['count_suspicious_30d'] = 0
        
        return agg_features
    
    def train(self, df: pd.DataFrame, target_col: str = 'suspicious') -> Dict[str, Any]:
        """Train the ML model"""
        logger.info("Starting ML model training...")
        
        # Prepare features and target
        X = self.prepare_features(df)
        y = df[target_col].astype(int)
        
        # Store feature names
        self.feature_names = X.columns.tolist()
        
        # Time-based split (train on earlier dates, test on later)
        if 'transaction_date' in df.columns:
            df_sorted = df.sort_values('transaction_date').copy()
            split_idx = int(len(df_sorted) * 0.8)  # 80% for training
            
            train_indices = df_sorted.index[:split_idx]
            test_indices = df_sorted.index[split_idx:]
            
            X_train = X.loc[train_indices]
            X_test = X.loc[test_indices]
            y_train = y.loc[train_indices]
            y_test = y.loc[test_indices]
        else:
            # Fallback to random split
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
        
        # Handle class imbalance
        class_weights = compute_class_weight(
            'balanced', 
            classes=np.unique(y_train), 
            y=y_train
        )
        class_weight_dict = dict(zip(np.unique(y_train), class_weights))
        
        # Train model
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight=class_weight_dict,
            random_state=42,
            n_jobs=-1
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Fit model
        self.model.fit(X_train_scaled, y_train)
        
        # Predictions
        y_pred = self.model.predict(X_test_scaled)
        y_prob = self.model.predict_proba(X_test_scaled)[:, 1]
        
        # Calculate metrics
        metrics = {
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'auc_roc': roc_auc_score(y_test, y_prob),
            'train_size': len(X_train),
            'test_size': len(X_test),
            'positive_class_ratio': y_train.sum() / len(y_train)
        }
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Initialize SHAP explainer
        if SHAP_AVAILABLE:
            try:
                self.explainer = shap.TreeExplainer(self.model)
            except Exception as e:
                logger.warning(f"Could not initialize SHAP explainer: {e}")
        else:
            logger.info("SHAP not available, using fallback explanations")
        
        self.model_version = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        logger.info("Model training completed:")
        logger.info(f"  - Precision: {metrics['precision']:.3f}")
        logger.info(f"  - Recall: {metrics['recall']:.3f}")
        logger.info(f"  - F1: {metrics['f1']:.3f}")
        logger.info(f"  - AUC-ROC: {metrics['auc_roc']:.3f}")
        
        return {
            'metrics': metrics,
            'feature_importance': feature_importance.to_dict('records'),
            'classification_report': classification_report(y_test, y_pred, output_dict=True)
        }
    
    def predict(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Make predictions on new data"""
        if self.model is None:
            raise ValueError("Model not trained or loaded")
        
        # Prepare features
        X = self.prepare_features(df)
        
        # Ensure feature alignment
        missing_features = set(self.feature_names) - set(X.columns)
        for feature in missing_features:
            X[feature] = 0
        
        X = X[self.feature_names]  # Reorder columns
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Predictions
        predictions = self.model.predict(X_scaled)
        probabilities = self.model.predict_proba(X_scaled)[:, 1]
        
        return {
            'predictions': predictions.tolist(),
            'probabilities': probabilities.tolist(),
            'model_version': self.model_version
        }
    
    def explain_prediction(self, df: pd.DataFrame, transaction_idx: int = 0) -> Dict[str, Any]:
        """Generate SHAP explanation for a specific transaction"""
        if self.explainer is None:
            return self._fallback_explanation(df, transaction_idx)
        
        try:
            # Prepare features
            X = self.prepare_features(df)
            
            # Ensure feature alignment
            missing_features = set(self.feature_names) - set(X.columns)
            for feature in missing_features:
                X[feature] = 0
            
            X = X[self.feature_names]
            X_scaled = self.scaler.transform(X)
            
            # Get SHAP values for specific transaction
            shap_values = self.explainer.shap_values(X_scaled[transaction_idx:transaction_idx+1])
            
            # For binary classification, take positive class SHAP values
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
            
            shap_values = shap_values[0]  # Get first (and only) row
            
            # Create feature contribution pairs
            feature_contributions = []
            for feature, shap_val in zip(self.feature_names, shap_values):
                feature_contributions.append({
                    'feature': feature,
                    'value': float(X.iloc[transaction_idx][feature]),
                    'shap_value': float(shap_val),
                    'contribution': 'positive' if shap_val > 0 else 'negative'
                })
            
            # Sort by absolute SHAP value and get top 3
            feature_contributions.sort(key=lambda x: abs(x['shap_value']), reverse=True)
            top_features = feature_contributions[:3]
            
            return {
                'top_features': top_features,
                'base_value': float(self.explainer.expected_value[1] if isinstance(self.explainer.expected_value, np.ndarray) else self.explainer.expected_value),
                'prediction_value': float(sum(shap_values)) + float(self.explainer.expected_value[1] if isinstance(self.explainer.expected_value, np.ndarray) else self.explainer.expected_value)
            }
            
        except Exception as e:
            logger.warning(f"SHAP explanation failed: {e}")
            return self._fallback_explanation(df, transaction_idx)
    
    def _fallback_explanation(self, df: pd.DataFrame, transaction_idx: int) -> Dict[str, Any]:
        """Fallback explanation based on feature importance"""
        try:
            X = self.prepare_features(df)
            
            # Get feature importances
            if hasattr(self.model, 'feature_importances_'):
                importances = self.model.feature_importances_
                
                # Create simple explanations based on feature values and importance
                explanations = []
                for feature, importance in zip(self.feature_names, importances):
                    if feature in X.columns:
                        value = X.iloc[transaction_idx][feature]
                        explanations.append({
                            'feature': feature,
                            'value': float(value),
                            'importance': float(importance),
                            'contribution': 'positive' if value > 0 else 'neutral'
                        })
                
                # Sort by importance and get top 3
                explanations.sort(key=lambda x: x['importance'], reverse=True)
                top_features = explanations[:3]
                
                return {
                    'top_features': top_features,
                    'explanation_type': 'feature_importance_fallback'
                }
            
        except Exception as e:
            logger.error(f"Fallback explanation failed: {e}")
        
        return {
            'top_features': [],
            'explanation_type': 'unavailable',
            'error': 'Could not generate explanation'
        }
    
    def save_model(self, path: str):
        """Save trained model to disk"""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'model_version': self.model_version,
            'explainer': self.explainer
        }
        
        joblib.dump(model_data, path)
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load trained model from disk"""
        model_data = joblib.load(path)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        self.model_version = model_data.get('model_version', 'unknown')
        self.explainer = model_data.get('explainer')
        
        logger.info(f"Model loaded from {path}, version: {self.model_version}")

def train_model_from_csv(csv_path: str, model_save_path: str, target_col: str = 'suspicious') -> Dict[str, Any]:
    """Train model from CSV file"""
    df = pd.read_csv(csv_path)
    
    model = AMLMLModel()
    training_results = model.train(df, target_col)
    model.save_model(model_save_path)
    
    return training_results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="AML ML Model Training")
    parser.add_argument("--input", required=True, help="Input CSV file")
    parser.add_argument("--output", required=True, help="Output model file path")
    parser.add_argument("--target", default="suspicious", help="Target column name")
    
    args = parser.parse_args()
    
    results = train_model_from_csv(args.input, args.output, args.target)
    print("Training completed!")
    print(f"Results: {json.dumps(results['metrics'], indent=2)}")
