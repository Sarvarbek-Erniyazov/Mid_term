import pandas as pd
import numpy as np
import joblib
import os
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List



app = FastAPI(
    title="Avtomobil Kilometr Bashorati API",
    description="RandomForestRegressor modeliga asoslangan avtomobilning yurgan masofasini (Kilometr) bashorat qilish xizmati."
)


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(PROJECT_ROOT, 'models', 'best_model.joblib')


try:
    pipeline = joblib.load(MODEL_PATH)
    MODEL_LOADED = True
    print(f" Model muvaffaqiyatli yuklandi: {MODEL_PATH}")
except Exception as e:
    MODEL_LOADED = False
    print(f" XATO: Model yuklanmadi. Iltimos, 'best_model.joblib' mavjudligini tekshiring. Sabab: {e}")



class CarFeatures(BaseModel):
    
    price: float
    powerPS: float
    FE_CarAge: float
    FE_AvgLogPricePerBrand: float
    FE_AvgLogPowerPerModel: float
    FE_ModelPopularity: float
    FE_LogKilometerByAge: float
    
    
    abtest: str
    vehicleType: str
    gearbox: str
    model: str
    fuelType: str
    brand: str
    notRepairedDamage: str
    FE_IsManual: int
    FE_HasDamage: int
    FE_RegionPrefix: int

    
    class Config:
        json_schema_extra = {
            "example": {
                "price": 9.81,
                "powerPS": 5.25,
                "FE_CarAge": 1.79,
                "FE_AvgLogPricePerBrand": 8.61,
                "FE_AvgLogPowerPerModel": 4.75,
                "FE_ModelPopularity": 0.036,
                "FE_LogKilometerByAge": 69763.8,
                "abtest": "test",
                "vehicleType": "coupe",
                "gearbox": "manuell",
                "model": "a5",
                "fuelType": "diesel",
                "brand": "audi",
                "notRepairedDamage": "ja",
                "FE_IsManual": 1,
                "FE_HasDamage": 1,
                "FE_RegionPrefix": 66
            }
        }



@app.get("/")
def home():
    
    status = "Tayyor" if MODEL_LOADED else "Xato: Model yuklanmagan"
    return {"status": status, "service": "Avtomobil Kilometr Bashorati (ML-Deployment)"}

@app.post("/predict/")
def predict_kilometer(features: CarFeatures):
   
    if not MODEL_LOADED:
        return {"error": "Model yuklanmagan, bashorat qilib bo'lmaydi."}

    try:
        
        input_data = features.model_dump()
        input_df = pd.DataFrame([input_data])
        
        
        prediction_log = pipeline.predict(input_df)[0]
        
        
        predicted_kilometer = np.expm1(prediction_log) 
        
        
        RMSE_ERROR = 4028 
        
        return {
            "kiritilgan_xususiyatlar": input_data,
            "bashorat_kilometr": f"{predicted_kilometer:.0f} km",
            "aniqlik_diapazoni": f"± {RMSE_ERROR} km (95% ishonchlilik bilan taxminan ± {2 * RMSE_ERROR} km)",
            "log_bashorati": float(prediction_log)
        }

    except Exception as e:
        return {"error": "Bashorat jarayonida kutilmagan xato yuz berdi", "detail": str(e)}

