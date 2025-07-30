import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import xgboost as xgb
import joblib
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.data_preprocessing import DataPreprocessor

class EnsembleClassifier:
    """
    Ensemble classifier for student placement prediction
    """
    
    def __init__(self, model_type: str = "stacked"):
        self.model_type = model_type
        self.model = None
        self.feature_names = None
        self.cv_results = None
        
    def build_model(self, X: pd.DataFrame, y: pd.Series):
        """Build either RandomForest or Stacked ensemble"""
        self.feature_names = X.columns.tolist()
        
        if self.model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                max_features='sqrt',
                random_state=config.RANDOM_STATE,
                n_jobs=-1
            )
            print("🏗️ Built RandomForest Ensemble")
            
        elif self.model_type == 'stacked':
            # Base models
            rf = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=config.RANDOM_STATE,
                n_jobs=-1
            )
            
            lr = LogisticRegression(
                random_state=config.RANDOM_STATE,
                max_iter=1000
            )
            
            xgb_model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=8,
                learning_rate=0.1,
                random_state=config.RANDOM_STATE,
                eval_metric='logloss'
            )
            
            # Meta model
            meta_model = LogisticRegression(
                random_state=config.RANDOM_STATE,
                max_iter=1000
            )
            
            # Stacking classifier
            self.model = StackingClassifier(
                estimators=[
                    ('rf', rf),
                    ('lr', lr),
                    ('xgb', xgb_model)
                ],
                final_estimator=meta_model,
                cv=3,
                n_jobs=-1
            )
            print("🏗️ Built Stacked Ensemble (RF + LR + XGBoost -> LR)")
        
        return self.model
    
    def evaluate_with_cv(self, X: pd.DataFrame, y: pd.Series, cv_folds: int = 5) -> dict:
        """Evaluate model using Stratified K-Fold Cross-Validation"""
        print(f"\n🔄 Performing {cv_folds}-Fold Stratified Cross-Validation...")
        
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=config.RANDOM_STATE)
        
        scoring = ['accuracy', 'precision', 'recall', 'f1']
        cv_results = cross_validate(
            self.model, X, y, 
            cv=skf, 
            scoring=scoring,
            n_jobs=-1,
            verbose=1
        )
        
        results = {
            'accuracy': {
                'mean': np.mean(cv_results['test_accuracy']),
                'std': np.std(cv_results['test_accuracy']),
                'scores': cv_results['test_accuracy']
            },
            'precision': {
                'mean': np.mean(cv_results['test_precision']),
                'std': np.std(cv_results['test_precision']),
                'scores': cv_results['test_precision']
            },
            'recall': {
                'mean': np.mean(cv_results['test_recall']),
                'std': np.std(cv_results['test_recall']),
                'scores': cv_results['test_recall']
            },
            'f1_score': {
                'mean': np.mean(cv_results['test_f1']),
                'std': np.std(cv_results['test_f1']),
                'scores': cv_results['test_f1']
            }
        }
        
        self.cv_results = results
        return results
    
    def train(self, X: pd.DataFrame, y: pd.Series):
        """Train the final model on all data"""
        print("🎯 Training final model on complete dataset...")
        return self.model.fit(X, y)
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions"""
        return self.model.predict(X)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Get prediction probabilities"""
        return self.model.predict_proba(X)
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance (if available)"""
        if hasattr(self.model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            return importance_df
        else:
            print("⚠️ Feature importance not available for this model type")
            return None
    
    def save_model(self, file_path: str = None):
        """Save model to file"""
        if file_path is None:
            file_path = config.MODEL_PATH
        joblib.dump(self, file_path)
        print(f"✅ Model saved to {file_path}")
    
    @staticmethod
    def load_model(file_path: str = None):
        """Load model from file"""
        if file_path is None:
            file_path = config.MODEL_PATH
        return joblib.load(file_path)

def train_placement_model(data_path: str = None, model_type: str = None) -> tuple:
    """
    Complete training pipeline for student placement prediction
    """
    print("🚀 Starting Student Placement Prediction Model Training...")
    print("=" * 60)
    
    if model_type is None:
        model_type = config.MODEL_TYPE
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor()
    
    # Load and prepare data
    df = preprocessor.load_data(data_path)
    df_processed = preprocessor.prepare_data(df)
    
    print(f"\n📊 Dataset Info:")
    print(f"Shape: {df_processed.shape}")
    print(f"Features: {preprocessor.get_feature_names()}")
    
    # Separate features and target
    X = df_processed[preprocessor.get_feature_names()]
    y = df_processed[config.TARGET_COLUMN]
    
    print(f"\n📈 Class Distribution:")
    class_counts = np.bincount(y)
    total = len(y)
    print(f"Not Placed (0): {class_counts[0]} ({class_counts[0]/total*100:.1f}%)")
    print(f"Placed (1): {class_counts[1]} ({class_counts[1]/total*100:.1f}%)")
    
    # Build and evaluate model
    ensemble = EnsembleClassifier(model_type=model_type)
    model = ensemble.build_model(X, y)
    
    # Cross-validation evaluation
    cv_results = ensemble.evaluate_with_cv(X, y, cv_folds=config.CV_FOLDS)
    
    # Print results
    print("\n📈 Cross-Validation Results:")
    print("=" * 40)
    for metric, values in cv_results.items():
        print(f"{metric.capitalize():12}: {values['mean']:.4f} (±{values['std']:.4f})")
    
    # Train final model
    ensemble.train(X, y)
    
    # Feature importance
    importance_df = ensemble.get_feature_importance()
    if importance_df is not None:
        print("\n🎯 Top 5 Important Features:")
        print(importance_df.head().to_string(index=False))
    
    # Save model and preprocessor
    ensemble.save_model()
    preprocessor.save_preprocessor()
    
    print("\n✅ Training completed successfully!")
    print("=" * 60)
    
    return ensemble, preprocessor, cv_results