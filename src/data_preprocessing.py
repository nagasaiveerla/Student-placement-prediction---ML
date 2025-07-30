import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from typing import Dict, Any
import joblib
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

class DataPreprocessor:
    """
    Data preprocessing pipeline for student placement prediction
    """
    
    def __init__(self):
        self.label_encoders = {}
        self.feature_columns = config.FEATURE_COLUMNS
        self.target_column = config.TARGET_COLUMN
        self.id_column = config.ID_COLUMN
        
    def load_data(self, file_path: str = None) -> pd.DataFrame:
        """Load data from Excel file"""
        if file_path is None:
            file_path = config.DATA_PATH
            
        try:
            df = pd.read_csv(file_path)
            print(f"✅ Data loaded successfully. Shape: {df.shape}")
            return df
        except FileNotFoundError:
            print("❌ csv file not found. Creating sample data for demonstration.")
            return self._create_sample_data()
    
    def _create_sample_data(self) -> pd.DataFrame:
        """Create sample data if Excel file not found"""
        np.random.seed(config.RANDOM_STATE)
        n_samples = 300
        
        df = pd.DataFrame({
            self.id_column: [f'CLG{str(i).zfill(4)}' for i in range(1, n_samples+1)],
            'IQ': np.random.normal(100, 15, n_samples).astype(int),
            'Prev_Sem_Result': np.random.uniform(5.0, 10.0, n_samples).round(2),
            'CGPA': np.random.uniform(5.0, 10.0, n_samples).round(2),
            'Academic_Perform': np.random.randint(1, 11, n_samples),
            'Internship_Experience': np.random.choice(['Yes', 'No'], n_samples),
            'Extra_Curricular': np.random.randint(0, 11, n_samples),
            'Communication_Skill': np.random.randint(1, 11, n_samples),
            'Projects_Completed': np.random.randint(0, 11, n_samples),
            self.target_column: np.random.choice(['Yes', 'No'], n_samples)
        })
        
        return df
    
    def prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare data for training:
        - Drop College_ID
        - Encode categorical columns
        - Handle missing values
        """
        df_processed = df.copy()
        
        # Drop ID column if present
        if self.id_column in df_processed.columns:
            df_processed = df_processed.drop(self.id_column, axis=1)
            print(f"✅ Dropped {self.id_column} column")
        
        # Handle missing values
        numeric_columns = df_processed.select_dtypes(include=[np.number]).columns
        categorical_columns = df_processed.select_dtypes(include=['object']).columns
        
        # Fill numeric missing values with mean
        for col in numeric_columns:
            if df_processed[col].isnull().sum() > 0:
                df_processed[col].fillna(df_processed[col].mean(), inplace=True)
                print(f"✅ Filled {df_processed[col].isnull().sum()} missing values in {col}")
        
        # Fill categorical missing values with mode
        for col in categorical_columns:
            if df_processed[col].isnull().sum() > 0:
                df_processed[col].fillna(df_processed[col].mode()[0], inplace=True)
                print(f"✅ Filled {df_processed[col].isnull().sum()} missing values in {col}")
        
        # Encode categorical columns
        categorical_cols = ['Internship_Experience', self.target_column]
        
        for col in categorical_cols:
            if col in df_processed.columns:
                le = LabelEncoder()
                df_processed[col] = le.fit_transform(df_processed[col])
                self.label_encoders[col] = le
                print(f"✅ Encoded {col}: {dict(zip(le.classes_, le.transform(le.classes_)))}")
        
        return df_processed
    
    def transform_features(self, features_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Transform new features for prediction"""
        features_copy = features_dict.copy()
        
        # Encode Internship_Experience if present
        if 'Internship_Experience' in features_copy:
            if 'Internship_Experience' in self.label_encoders:
                value = features_copy['Internship_Experience']
                if isinstance(value, str):
                    features_copy['Internship_Experience'] = self.label_encoders['Internship_Experience'].transform([value])[0]
        
        return features_copy
    
    def get_feature_names(self) -> list:
        """Get list of feature column names"""
        return self.feature_columns
    
    def save_preprocessor(self, file_path: str = None):
        """Save preprocessor to file"""
        if file_path is None:
            file_path = config.PREPROCESSOR_PATH
        joblib.dump(self, file_path)
        print(f"✅ Preprocessor saved to {file_path}")
    
    @staticmethod
    def load_preprocessor(file_path: str = None):
        """Load preprocessor from file"""
        if file_path is None:
            file_path = config.PREPROCESSOR_PATH
        return joblib.load(file_path)