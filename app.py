from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from starlette.responses import HTMLResponse 
import re

import paho.mqtt.client as paho
from paho import mqtt

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
    data=data_array.reshape(1,500,234,1)
    return data

app = FastAPI()

# setting callbacks for different events to see if it works, print the message etc.
def on_connect(client, userdata, flags, rc, properties=None):
    print("CONNACK received with code %s." % rc)

# with this callback you can see if your publish was successful
def on_publish(client, userdata, mid, properties=None):
    print("mid: " + str(mid))

# print which topic was subscribed to
def on_subscribe(client, userdata, mid, granted_qos, properties=None):
    print("Subscribed: " + str(mid) + " " + str(granted_qos))

# print message, useful for checking if it was successful
def on_message(client, userdata, msg):
    print(msg.topic + " " + str(msg.qos) + " " + str(msg.payload))

client = paho.Client(client_id="", userdata=None, protocol=paho.MQTTv5)


client.on_connect = on_connect

client.tls_set(tls_version=mqtt.client.ssl.PROTOCOL_TLS)
client.username_pw_set("fitiontest", "Fition@123")
client.connect("3da5a785b21c44fc9f3ff0d47f7b11d4.s1.eu.hivemq.cloud", 8883)

client.on_subscribe = on_subscribe
client.on_message = on_message
client.on_publish = on_publish











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
    print(csiMatrix.csi_matrix[0][0])
    data_array = np.array(csiMatrix.csi_matrix)
    print(data_array.shape)
    clean_text = my_pipeline(data_array) #cleaning and preprocessing of the texts
    loaded_model = tf.keras.models.load_model('convLSTM111.h5') #load the saved model 
    predictions = loaded_model.predict(clean_text) #predict the text
    sentiment = int(np.argmax(predictions)) #calculate the index of max sentiment
    probability = max(predictions.tolist()[0]) #calulate the probability
    print("sentiment:", sentiment, ", probability: ", probability)
    client.publish("testtopic", sentiment)
    client.loop_start()
    # client = connect_mqtt()
    # client.publish(topic, sentiment)
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


@app.get('/mqtt')
def testt():
    
    client.publish("testtopic", "hey msg")
    client.loop_start()
    return '''<h1>Hey</h1>'''

    