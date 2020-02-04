# pylint: disable=no-name-in-module
from enum import Enum

import joblib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from starlette.requests import Request
from starlette.staticfiles import StaticFiles
from starlette.templating import Jinja2Templates

from dataManipulation import retrieve_data

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory='templates')


class ModelName(str, Enum):
    LRModel = 'LogisticRegression'
    KNModel = 'KNN'
    NNModel = 'MultiLayerPerceptron'
    SVModel = 'SupportVectorClassifier'


class Settings(BaseModel):
    sepalLen: float
    sepalWid: float
    petalLen: float
    petalWid: float


@app.get('/')
async def home(request: Request):
    return templates.TemplateResponse('index.html', {'request': request})


@app.get('/data/')
async def data():
    return retrieve_data()


@app.post('/predict/')
async def predict(settings: Settings, modelName: ModelName = 'LogisticRegression'):
    data = settings.dict()
    newFlower = [data['sepalLen'],
                 data['sepalWid'],
                 data['petalLen'],
                 data['petalWid']]
    newFlower = np.asarray(newFlower).reshape(1, -1)

    if modelName == ModelName.LRModel:
        model = joblib.load('models/LRModel.joblib')
    elif modelName == ModelName.KNModel:
        model = joblib.load('models/KNNModel.joblib')
    elif modelName == ModelName.NNModel:
        model = joblib.load('models/MLPModel.joblib')
    else:
        model = joblib.load('models/SVModel.joblib')

    pred = model.predict(newFlower).tolist()[0]

    return {**data, 'modelName': modelName, 'prediction': pred}
