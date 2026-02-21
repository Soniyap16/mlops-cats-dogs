import time
import logging
from fastapi import Request

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ml-api")

request_count = 0

from fastapi import FastAPI, File, UploadFile
import numpy as np
import cv2
import joblib

app = FastAPI()

# load trained model
model = joblib.load("baseline_model.pkl")

IMG_SIZE = 64


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # read image
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # preprocess
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.flatten().reshape(1, -1) / 255.0

    # prediction
    pred = model.predict(img)[0]
    label = "cat" if pred == 0 else "dog"

    return {"prediction": label}
