"""
Configuration file for Student Placement Prediction System
Contains all the settings, paths, and parameters for the project.
"""

import os
from pathlib import Path

# =============================================================================
# PROJECT PATHS
# =============================================================================

BASE_DIR = Path(__file__).parent.absolute()

DATA_DIR = BASE_DIR / "data"
DATA_PATH = DATA_DIR / "college_student_placement_dataset.csv"
SAMPLE_DATA_PATH = DATA_DIR / "sample_data.csv"

MODELS_DIR = BASE_DIR / "models"
MODEL_PATH = MODELS_DIR / "placement_model.pkl"
PREPROCESSOR_PATH = MODELS_DIR / "preprocessor.pkl"
SCALER_PATH = MODELS_DIR / "scaler.pkl"

OUTPUTS_DIR = BASE_DIR / "outputs"
REPORTS_DIR = OUTPUTS_DIR / "reports"
PLOTS_DIR = OUTPUTS_DIR / "plots"

LOGS_DIR = BASE_DIR / "logs"
LOG_FILE = LOGS_DIR / "training.log"

# =============================================================================
# MODEL CONFIGURATION
# =============================================================================

MODEL_TYPE = "stacked"  # Options: "random_forest", "stacked"

RANDOM_FOREST_PARAMS = {
    'n_estimators': 100,
    'max_depth': 10,
    'min_samples_split': 5,
    'min_samples_leaf': 2,
    'random_state': 42,
    'n_jobs': -1
}

XGBOOST_PARAMS = {
    'n_estimators': 100,
    'max_depth': 6,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'n_jobs': -1
}

LOGISTIC_REGRESSION_PARAMS = {
    'random_state': 42,
    'max_iter': 1000
}

CV_FOLDS = 5
CV_SCORING = ['accuracy', 'precision', 'recall', 'f1']
RANDOM_STATE = 42

ENSEMBLE_METHODS = ['voting', 'stacking']
ENSEMBLE_METHOD = 'stacking'

ENABLE_HYPERPARAMETER_TUNING = False
PARAM_GRID = {
    'random_forest': {
        'n_estimators': [50, 100, 200],
        'max_depth': [5, 10, 15, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    },
    'xgboost': {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 6, 9],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.8, 0.9, 1.0]
    }
}

# =============================================================================
# DATA PREPROCESSING
# =============================================================================
ID_COLUMN = 'College_ID'
TARGET_COLUMN = 'Placement'

CATEGORICAL_COLUMNS = ['Internship_Experience']

NUMERICAL_COLUMNS = [
    'IQ', 'Prev_Sem_Result', 'CGPA', 'Academic_Performance',
    'Extra_Curricular_Score', 'Communication_Skills', 'Projects_Completed'
]

FEATURE_COLUMNS = CATEGORICAL_COLUMNS + NUMERICAL_COLUMNS

COLUMNS_TO_DROP = ['College_ID']

FILL_NUMERICAL_WITH = 'median'
FILL_CATEGORICAL_WITH = 'mode'

SCALING_METHOD = 'standard'

EXPECTED_COLUMNS = [
    'College_ID', 'IQ', 'Prev_Sem_Result', 'CGPA', 'Academic_Performance',
    'Internship_Experience', 'Extra_Curricular_Score',
    'Communication_Skills', 'Projects_Completed', 'Placement'
]

COLUMN_MAPPINGS = {
    # Add mappings if column names need to be standardized
}

# =============================================================================
# TRAINING CONFIGURATION
# =============================================================================

TEST_SIZE = 0.2
VALIDATION_SIZE = 0.2

HANDLE_IMBALANCE = True
BALANCING_METHOD = 'smote'

FEATURE_SELECTION = True
FEATURE_SELECTION_METHOD = 'mutual_info'
MAX_FEATURES = 20

# =============================================================================
# MODEL EVALUATION
# =============================================================================

EVALUATION_METRICS = [
    'accuracy', 'precision', 'recall', 'f1_score', 'roc_auc'
]

THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7]

# =============================================================================
# STREAMLIT APP CONFIGURATION
# =============================================================================

APP_TITLE = "🎓 Student Placement Prediction System"
APP_DESCRIPTION = """
This application predicts whether a student will be placed in a company based on various academic and demographic factors.
"""

SIDEBAR_TITLE = "Student Information"

INPUT_FIELDS = {
    'Internship_Experience': {
        'label': 'Internship Experience',
        'type': 'selectbox',
        'options': ['Yes', 'No'],
        'help': 'Whether the student has internship experience'
    },
    'IQ': {
        'label': 'IQ Score',
        'type': 'number_input',
        'min_value': 60,
        'max_value': 160,
        'value': 100,
        'help': 'IQ level of the student'
    },
    'Prev_Sem_Result': {
        'label': 'Previous Semester Result',
        'type': 'number_input',
        'min_value': 0.0,
        'max_value': 10.0,
        'value': 7.0,
        'help': 'GPA of previous semester'
    },
    'CGPA': {
        'label': 'Current CGPA',
        'type': 'number_input',
        'min_value': 0.0,
        'max_value': 10.0,
        'value': 7.5,
        'help': 'Cumulative Grade Point Average'
    },
    'Academic_Performance': {
        'label': 'Academic Performance',
        'type': 'number_input',
        'min_value': 0,
        'max_value': 10,
        'value': 7,
        'help': 'Rating of academic performance (out of 10)'
    },
    'Extra_Curricular_Score': {
        'label': 'Extra Curricular Score',
        'type': 'number_input',
        'min_value': 0,
        'max_value': 10,
        'value': 5,
        'help': 'Extra-curricular activity involvement score'
    },
    'Communication_Skills': {
        'label': 'Communication Skills',
        'type': 'number_input',
        'min_value': 0,
        'max_value': 10,
        'value': 6,
        'help': 'Rating of communication ability (out of 10)'
    },
    'Projects_Completed': {
        'label': 'Projects Completed',
        'type': 'number_input',
        'min_value': 0,
        'max_value': 10,
        'value': 2,
        'help': 'Number of academic/technical projects completed'
    }
}

# =============================================================================
# FASTAPI CONFIGURATION
# =============================================================================

API_TITLE = "Student Placement Prediction API"
API_DESCRIPTION = """
REST API for predicting student placement outcomes based on academic and demographic data.
"""
API_VERSION = "1.0.0"

API_HOST = "0.0.0.0"
API_PORT = 8000

CORS_ORIGINS = ["*"]
CORS_METHODS = ["GET", "POST"]
CORS_HEADERS = ["*"]

# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================

LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
        },
        'detailed': {
            'format': '%(asctime)s [%(levelname)s] %(name)s:%(lineno)d: %(message)s'
        }
    },
    'handlers': {
        'default': {
            'level': 'INFO',
            'formatter': 'standard',
            'class': 'logging.StreamHandler',
        },
        'file': {
            'level': 'INFO',
            'formatter': 'detailed',
            'class': 'logging.FileHandler',
            'filename': str(LOG_FILE),
            'mode': 'a',
        },
    },
    'loggers': {
        '': {
            'handlers': ['default', 'file'],
            'level': 'INFO',
            'propagate': False
        }
    }
}

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def create_directories():
    """Create necessary directories if they don't exist."""
    directories = [DATA_DIR, MODELS_DIR, OUTPUTS_DIR, REPORTS_DIR, PLOTS_DIR, LOGS_DIR]
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

def get_model_config(model_type=None):
    """Get model configuration based on model type."""
    if model_type is None:
        model_type = MODEL_TYPE
    
    if model_type == "random_forest":
        return RANDOM_FOREST_PARAMS
    elif model_type == "stacked":
        return {
            'random_forest': RANDOM_FOREST_PARAMS,
            'xgboost': XGBOOST_PARAMS,
            'meta_learner': LOGISTIC_REGRESSION_PARAMS
        }
    else:
        raise ValueError(f"Unknown model type: {model_type}")

# =============================================================================
# VALIDATION
# =============================================================================

def validate_config():
    """Validate configuration settings."""
    errors = []
    if MODEL_TYPE not in ['random_forest', 'stacked']:
        errors.append(f"Invalid MODEL_TYPE: {MODEL_TYPE}")
    if not 0 < TEST_SIZE < 1:
        errors.append(f"TEST_SIZE must be between 0 and 1, got: {TEST_SIZE}")
    if not 0 < VALIDATION_SIZE < 1:
        errors.append(f"VALIDATION_SIZE must be between 0 and 1, got: {VALIDATION_SIZE}")
    if errors:
        raise ValueError("Configuration validation failed:\n" + "\n".join(errors))

# =============================================================================
# INITIALIZATION
# =============================================================================

create_directories()
validate_config()

if __name__ == "__main__":
    print("Configuration loaded successfully!")
    print(f"Base directory: {BASE_DIR}")
    print(f"Model type: {MODEL_TYPE}")
    print(f"Data path: {DATA_PATH}")
    print(f"Model path: {MODEL_PATH}")
