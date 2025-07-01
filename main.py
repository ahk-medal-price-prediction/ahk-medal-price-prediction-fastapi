from fastapi import FastAPI
from schemas import Data
from medal_prediction import medal_predict_data
from mould_prediction import mould_predict_data

app = FastAPI()


@app.get("/")
async def root():
    return {"Connection Status": "200 OK"}

@app.post('/medal_prediction')
def medal_prediction(request:Data):
    data = medal_predict_data(request)
    return data


@app.post('/mould_prediction')
def mould_prediction(request: Data):
    # # Set quantity = 1
    # request.quantity = 1
    data = mould_predict_data(request)
    return data
