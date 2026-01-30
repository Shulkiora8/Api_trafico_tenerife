from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI(title="API Predicción Tráfico Canarias")

# Cargar modelo
model = joblib.load("best_model_rf.joblib")

class TrafficInput(BaseModel):
    carretera_codigo: str
    tramo_nombre: str
    estacion_id: int
    estacion_nombre: str
    velocidad_media: float
    imd_ascendentes: float
    imd_descendentes: float
    imd_pesados: int
    año: int

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(input_data: TrafficInput):
    try:
        X = np.array([[input_data.velocidad_media,
                       input_data.imd_ascendentes,
                       input_data.imd_descendentes,
                       input_data.imd_pesados,
                       input_data.año]])
        prediction = model.predict(X)
        return {"imd_total_pred": float(prediction[0])}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
