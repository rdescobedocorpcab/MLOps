import io
import tensorflow as tf
from tensorflow.keras.utils import load_img, img_to_array
import torch
from torchvision import transforms
from torchvision import models
from torch import nn
from PIL import Image
import base64
import numpy as np

class Models():
    def __init__(self) -> None:
        self.tf_model = tf.keras.models.load_model("C:/Users/rdescobedo/Desktop/Carpetas/MLOps/fastapi/ML-OPS-BEGINNER-STUDENT/final_model")
        checkpoint = torch.load("C:/Users/rdescobedo/Desktop/Carpetas/MLOps/fastapi/ML-OPS-BEGINNER-STUDENT/catvdog.pt", map_location = torch.device("cpu"))
        self.pytorch_model = models.densenet121(pretrained = False)
        self.pytorch_model.classifier = nn.Sequential(nn.Linear(1024, 512),
                                 nn.ReLU(),
                                 nn.Dropout(0.2),
                                 nn.Linear(512, 256),
                                 nn.ReLU(),
                                 nn.Dropout(0.1),
                                 nn.Linear(256, 2),
                                 nn.LogSoftmax(dim = 1)
                                 )
        
        self.pytorch_model.parameters = checkpoint["parameters"]
        self.pytorch_model.load_state_dict(checkpoint["state_dict"])
        self.pytorch_model.eval()
        
    def predict_tensorflow(self, img_array: np.ndarray) -> dict:
        prediction = self.tf_model.predict(img_array)
        prediction_value = prediction[0][0]
        if prediction_value > 0.5: #0-0.5 = cat & 0.5-1 = dog
            return {"class": "Dog", "value": float(prediction_value)}
        else: 
            return {"class": "Cat", "value": float(prediction_value)}
        
    def load_image_tf(self, img_b64: str) -> np.ndarray:
        img = Image.open(io.BytesIO(base64.b64decode(img_b64)))
        
        img = img.convert("RGB")
        img = img.resize((224, 224), Image.NEAREST)
        
        # convert to array
        img = img_to_array(img)
        # reshape into a single sample with 3 channels
        img = img.reshape(1, 224, 224, 3)
        # center pixel data
        img = img.astype('float32')
        img = img - [123.68, 116.779, 103.939]
        return img
