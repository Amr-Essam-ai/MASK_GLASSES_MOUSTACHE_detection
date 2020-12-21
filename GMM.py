
import os

os.environ['KERAS_BACKEND'] = 'tensorflow'
import numpy as np 
import cv2
import tensorflow.keras
from  tensorflow.keras.models import load_model
import pandas as pd 
from focal_loss import SparseCategoricalFocalLoss

class model_g:
    def __init__(self):
        self.model_g=self.load_model()

    def load_model(self):
        print("load model")
        return load_model('model/GMM.h5',custom_objects={'glasses_out':SparseCategoricalFocalLoss(gamma=2.),
        'moustace_out':SparseCategoricalFocalLoss(gamma=2.),
        'mask_out':SparseCategoricalFocalLoss(gamma=2.)})

    def summary(self):
        self.model_g.summary()

    def layers(self):
        return  self.model_g.layers
    

    def pred_transform(self,pred):

        glass_label,glass_confidence = np.argmax(pred[0], axis=1),np.int16(np.max(pred[0]*100, axis=1))
        Mostatch_label,Mostatch_confidence = np.argmax(pred[1], axis=1),np.int16(np.max(pred[1]*100, axis=1))
        mask_label,mask_confidence = np.argmax(pred[2], axis=1),np.int16(np.max(pred[2]*100, axis=1))
        glass_label=np.where(glass_label==0, 'No glass', glass_label) 
        glass_label=np.where(glass_label=='1', 'glass', glass_label) 

        Mostatch_label=np.where(Mostatch_label==0, 'No Mostatch', Mostatch_label) 
        Mostatch_label=np.where(Mostatch_label=='1', 'Mostatch', Mostatch_label) 

        mask_label=np.where(mask_label==0, 'No mask', mask_label) 
        mask_label=np.where(mask_label=='1', 'mask', mask_label) 

        r={'glass':{'label':glass_label.tolist(),'confidence':glass_confidence.tolist()},
        'Mostatch':{'label':Mostatch_label.tolist(),'confidence':Mostatch_confidence.tolist()},
        'mask':{'label':mask_label.tolist(),'confidence':mask_confidence.tolist()}}
        return r 

    def predict_label(self,im):
        pred=self.model_g.predict(im)
        pred_df=self.pred_transform(pred)
        return pred_df
    
    
   
 




