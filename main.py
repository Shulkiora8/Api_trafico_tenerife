from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import mlflow.sklearn
import os


mlflow.set_tracking_uri("databricks")

URL_MODELO = "models:/workspace.default.trafico_tenerife3/1" 

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        app.state.modelo = mlflow.sklearn.load_model(URL_MODELO)
        print("Modelo cargado desde MLflow")
    except Exception as e:
        print("Error cargando modelo desde MLflow:", e)
        app.state.modelo = None

    yield

    print("Aplicación detenida")

app = FastAPI(
    title="API Predicción Tráfico Canarias",
    lifespan=lifespan
)

origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TrafficInput(BaseModel):
    carretera_codigo: str
    año: int

@app.get("/health")
def health():
    return {"status": "ok" if app.state.modelo else "ko"}


@app.post("/predict")
def predict(input_data: TrafficInput):
    if not app.state.modelo:
        raise HTTPException(status_code=500, detail="Modelo no cargado")

    X = pd.DataFrame([{
        "carretera_codigo": input_data.carretera_codigo,
        "año": input_data.año
    }])

    try:
        pred = app.state.modelo.predict(X)
        return {"imd_total_pred": float(pred[0])}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

