from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from starlette.responses import HTMLResponse 
import re



from CSIKit.reader import get_reader
from CSIKit.util import csitools
import os
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pickle

import shutil
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Callable

from fastapi import UploadFile

class CSIMatrix(BaseModel):
    csi_matrix: list

def preProcess_data(data_array): #cleaning the data
    data=data_array.reshape(1,300,234,1)
    return data

app = FastAPI()

def my_pipeline(data): #pipeline
  X = preProcess_data(data)
  return X

def save_upload_file_tmp(upload_file: UploadFile) -> Path:
    try:
        suffix = Path(upload_file.filename).suffix
        with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            shutil.copyfileobj(upload_file.file, tmp)
            tmp_path = Path(tmp.name)
    finally:
        upload_file.file.close()
    return tmp_path

@app.get('/') #basic get view
def basic_view():
    return {"WELCOME": "GO TO /docs route, or /post or send post request to /predict "}



@app.get('/predict', response_class=HTMLResponse) #data input by forms
def take_inp():
    return '''<form method="post" enctype='multipart/form-data'> 
    <input type="file"  name="uploaded_file" value="Data to be tested"/>  
    <input type="submit"/> 
    </form>'''



# @app.post('/predict') #prediction on data
# async def predict(uploaded_file: UploadFile = File(...)): #input is from forms
#     path=save_upload_file_tmp(uploaded_file)
#     clean_text = my_pipeline(path) #cleaning and preprocessing of the texts
#     loaded_model = tf.keras.models.load_model('sentiment.h5') #load the saved model 
#     predictions = loaded_model.predict(clean_text) #predict the text
#     sentiment = int(np.argmax(predictions)) #calculate the index of max sentiment
#     probability = max(predictions.tolist()[0]) #calulate the probability
#     if sentiment==0:
#          t_sentiment = 'nomv' #set appropriate sentiment
#     elif sentiment==1:
#          t_sentiment = 'std'
#     elif sentiment==2:
#          t_sentiment='bed'
#     return { #return the dictionary for endpoint
#          "PREDICTED SENTIMENT": t_sentiment,
#          "Probability": probability
    # }


@app.post('/predict') #prediction on data
async def test(csiMatrix: CSIMatrix ): #input is from forms
    print(csiMatrix.csi_matrix[0][0], )
    data_array = np.array(csiMatrix.csi_matrix)
    print(data_array.shape)
    clean_text = my_pipeline(data_array) #cleaning and preprocessing of the texts
    loaded_model = tf.keras.models.load_model('sentiment.h5') #load the saved model 
    predictions = loaded_model.predict(clean_text) #predict the text
    sentiment = int(np.argmax(predictions)) #calculate the index of max sentiment
    probability = max(predictions.tolist()[0]) #calulate the probability
    print("sentiment:", sentiment, ", probability: ", probability)
    # if sentiment==0:
    #      t_sentiment = 'nomv' #set appropriate sentiment
    # elif sentiment==1:
    #      t_sentiment = 'std'
    # elif sentiment==2:
    #      t_sentiment='bed'
    return { #return the dictionary for endpoint
        "csi_matrix": csiMatrix.csi_matrix[0][0],
         "PREDICTED SENTIMENT": sentiment,
         "Probability": probability
    }
