#!/usr/bin/env python3
"""
Main training script for Student Placement Prediction System
Run this script to train the model and save it for later use.

Usage:
    python train_model.py
    python train_model.py --model_type random_forest
    python train_model.py --data_path "custom_data.xlsx"
"""

import argparse
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.model_training import train_placement_model

def main():
    parser = argparse.ArgumentParser(description='Train Student Placement Prediction Model')
    parser.add_argument('--data_path', type=str, default=None,
                       help='Path to the Excel dataset (default: uses config.DATA_PATH)')
    parser.add_argument('--model_type', type=str, choices=['random_forest', 'stacked'],
                       default=None, help='Type of model to train (default: uses config.MODEL_TYPE)')
    
    args = parser.parse_args()
    
    try:
        # Train the model
        ensemble, preprocessor, cv_results = train_placement_model(
            data_path=args.data_path,
            model_type=args.model_type
        )
        
        print("\n🎉 Training completed successfully!")
        print("\n📋 Next steps:")
        print("1. Run Streamlit app: python run_streamlit.py")
        print("2. Run FastAPI server: python run_fastapi.py")
        print("3. Or use: streamlit run src/streamlit_app.py")
        print("4. Or use: uvicorn src.fastapi_app:app --reload")
        
        return True
        
    except Exception as e:
        print(f"❌ Training failed: {str(e)}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)