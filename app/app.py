from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent   #app/ folder
ROOT_DIR = BASE_DIR.parent                   #project root folder

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from typing import Literal
from pydantic import BaseModel, Field
import pandas as pd
import joblib
import shap

app = FastAPI()

class CustomerInput(BaseModel):
    CreditScore: int
    Geography: Literal["France", "Spain", "Germany"]
    Gender: Literal["Male", "Female"]
    Age: int
    Tenure: int
    Balance: float
    NumOfProducts: int
    HasCrCard: Literal[0, 1]
    IsActiveMember: Literal[0, 1]
    EstimatedSalary: float
    SatisfactionScore: Literal[1, 2, 3, 4, 5]
    CardType: Literal["DIAMOND", "GOLD", "SILVER", "PLATINUM"]
    PointEarned: int
    
model = joblib.load(ROOT_DIR / "model" / "best_model_f1.pkl")
classifier = model.named_steps['classifier']
preprocessor = model.named_steps['preprocessor']

explainer = shap.TreeExplainer(model.named_steps['classifier'])

#Features names After Preprocessing
feature_names = [n.split("__", 1)[1] for n in preprocessor.get_feature_names_out()]

@app.post("/predict")
def predict_churn(input_data: CustomerInput):
    # Convert input data to DataFrame
    input_df = pd.DataFrame([input_data.dict()])
    
    # Make prediction
    prediction = model.predict(input_df)[0]
    prediction_proba = model.predict_proba(input_df)[0]
    
    # Preprocess the input data
    preprocessed_data = preprocessor.transform(input_df)
    
    # Generate SHAP values
    sv = explainer.shap_values(preprocessed_data)         
    shap_vals = sv[0, :, 1]
    
    # Create a summary of SHAP values for the input instance
    explanation = dict(zip(feature_names, [float(v) for v in shap_vals]))
    
    return {
        "prediction": "Likely to churn" if prediction == 1 else "Likely to stay",
        "prediction_probability": float(prediction_proba[1]),
        "confidence": round(float(max(prediction_proba)) * 100, 1),
        "explanation": explanation
    }
    
@app.get("/", response_class=HTMLResponse)
def index():
    with open(BASE_DIR / "index.html", encoding="utf-8") as f:
        return f.read()