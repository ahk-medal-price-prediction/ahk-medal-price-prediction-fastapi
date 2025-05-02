from fastapi import FastAPI
from schemas import Data
from prediction import predict_data

app = FastAPI()


@app.get("/")
async def root():
    return {"Connection Status": "200 OK"}

@app.post('/model_prediction')
def model_prediction(request:Data):
    data = predict_data(request)
    return data