from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
import joblib
import numpy as np
import pandas as pd
from fastapi.responses import HTMLResponse, RedirectResponse,JSONResponse
import json
from typing import Optional, List

# Variabili globali
loaded_model = None

# CORS Support
origins = [
    f'http://0.0.0.0:8008',
    f'http://localhost:8008',
    #f'http://localhost:8080',
]

app = FastAPI()

app.add_middleware(CORSMiddleware,
                    allow_origins=origins,
                    allow_credentials=True,
                    allow_methods=["*"],
                    allow_headers=["*"],
                    )

# Template setup
templates = Jinja2Templates(directory="templates")

#@app.get("/")
@app.get("/test", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request" : request})

@app.on_event("startup")
async def startup_event():
    print("Start")
    # caricamento modelli
    # load all stuff
    global loaded_model
    loaded_model = joblib.load('LRClassifier.pkl')

async def shutdon_envent():
    print("Shutdown")

class IrisSpecies(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float
    startDatetime:  Optional[float] =  None

@app.post('/predict',response_class=JSONResponse )
def predict_species(iris: IrisSpecies):
    data = iris.dict()
    #print(data)
    data_in = [[data['sepal_length'], data['sepal_width'], data['petal_length'], data['petal_width']]]
    prediction = loaded_model.predict(data_in)
    probability = loaded_model.predict_proba(data_in).max()
    return json.dumps({
                        'prediction': prediction[0],
                        'probability': probability
                        })