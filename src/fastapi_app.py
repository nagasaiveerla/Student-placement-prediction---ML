from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, List
import pandas as pd
import numpy as np
import os
import sys

# Include parent path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from src.model_training import EnsembleClassifier
from src.data_preprocessing import DataPreprocessor

# -----------------------------
# Pydantic Model
# -----------------------------
class StudentFeatures(BaseModel):
    IQ: int = Field(..., ge=70, le=150, description="Student's IQ score")
    Prev_Sem_Result: float = Field(..., ge=0.0, le=10.0, description="Previous semester result")
    CGPA: float = Field(..., ge=0.0, le=10.0, description="Cumulative GPA")
    Academic_Performance: int = Field(..., ge=1, le=10, description="Academic performance rating")
    Internship_Experience: str = Field(..., pattern="^(Yes|No)$", description="Internship experience")
    Extra_Curricular_Score: int = Field(..., ge=0, le=10, description="Extra-curricular score")
    Communication_Skills: int = Field(..., ge=1, le=10, description="Communication skills rating")
    Projects_Completed: int = Field(..., ge=0, le=10, description="Number of projects completed")

    class Config:
        schema_extra = {
            "example": {
                "IQ": 110,
                "Prev_Sem_Result": 8.2,
                "CGPA": 8.5,
                "Academic_Performance": 8,
                "Internship_Experience": "Yes",
                "Extra_Curricular_Score": 7,
                "Communication_Skills": 8,
                "Projects_Completed": 5
            }
        }

class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    probabilities: Dict[str, float]
    feature_importance: Dict[str, float] = None

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_type: str = None
    version: str = "1.0.0"

# -----------------------------
# FastAPI App Initialization
# -----------------------------
app = FastAPI(
    title="Student Placement Prediction API",
    description="API for predicting student placement using machine learning",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

model = None
preprocessor = None

@app.on_event("startup")
async def load_models():
    global model, preprocessor
    try:
        model = EnsembleClassifier.load_model()
        preprocessor = DataPreprocessor.load_preprocessor()
        print("✅ Models loaded successfully at startup")
    except FileNotFoundError:
        print("❌ Model files not found. Please run training first.")
        model = None
        preprocessor = None

@app.get("/", response_model=Dict[str, str])
async def root():
    return {
        "message": "Student Placement Prediction API",
        "version": "1.0.0",
        "docs": "/docs"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="healthy" if model else "unhealthy",
        model_loaded=model is not None,
        model_type=model.model_type if model else None
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict_placement(features: StudentFeatures):
    if model is None or preprocessor is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run training and restart server.")
    
    try:
        features_dict = features.dict()
        processed_features = preprocessor.transform_features(features_dict)

        # ✅ Ensure feature order matches training
        ordered_features = {col: processed_features[col] for col in config.FEATURE_COLUMNS}
        feature_df = pd.DataFrame([ordered_features])

        prediction = model.predict(feature_df)[0]
        prediction_proba = model.predict_proba(feature_df)[0]

        result = "Placed" if prediction == 1 else "Not Placed"
        confidence = float(prediction_proba[1] if prediction == 1 else prediction_proba[0])

        response_data = {
            "prediction": result,
            "confidence": confidence,
            "probabilities": {
                "Not Placed": float(prediction_proba[0]),
                "Placed": float(prediction_proba[1])
            }
        }

        if hasattr(model.model, 'feature_importances_'):
            response_data["feature_importance"] = dict(zip(
                model.feature_names,
                model.model.feature_importances_.tolist()
            ))

        return PredictionResponse(**response_data)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict/batch")
async def predict_batch(features_list: List[StudentFeatures]):
    if model is None or preprocessor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        results = []
        for features in features_list:
            features_dict = features.dict()
            processed_features = preprocessor.transform_features(features_dict)

            # ✅ Ensure consistent feature order
            ordered_features = {col: processed_features[col] for col in config.FEATURE_COLUMNS}
            feature_df = pd.DataFrame([ordered_features])

            prediction = model.predict(feature_df)[0]
            prediction_proba = model.predict_proba(feature_df)[0]

            result = "Placed" if prediction == 1 else "Not Placed"
            confidence = float(prediction_proba[1] if prediction == 1 else prediction_proba[0])

            results.append({
                "prediction": result,
                "confidence": confidence,
                "probabilities": {
                    "Not Placed": float(prediction_proba[0]),
                    "Placed": float(prediction_proba[1])
                }
            })

        return {"predictions": results}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
