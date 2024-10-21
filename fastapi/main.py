from fastapi import FastAPI
import logging
from pydantic import BaseModel
from logging.config import dictConfig
from logger_config import log_config
from models import Models
from fastapi import HTTPException
import os

app = FastAPI()
dictConfig(log_config)
logger = logging.getLogger("mlops")

class ImagePayload(BaseModel):
    img_b64: str

@app.get('/health')
def health():
    logger.info("Health request received.")
    return "Service is online."

@app.post("/classify/tensorflow")
def classify_tensorflow(request: ImagePayload):
    try:
        logger.info("Tensorflow request received.")
        img_array = Models()
        img_array.load_image_tf(img_b64 = request.img_b64)
        result = Models.predict_tensorflow(img_array = img_array)
        return result
        
    except Exception as e:
        message = "Server error while processing image!"
        logger.error(f"{message}: {e}", exc_info = True)
        raise HTTPException(status_code = 500, detail = message)

#@app.post("/classify/pytorch")