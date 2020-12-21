import os
import falcon
from tensorflow.keras.models import load_model
from predict import PredictResource
from falcon_cors import CORS    
import numpy as np 
import cv2
from GMM import model_g
from PIL import Image

api = application = falcon.API()

def load_trained_model():
    global model
    model=model_g()
    return model

predict = PredictResource(model=load_trained_model())
api.add_route('/GMM/', predict)

