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
from queue import Queue

import shutil
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Callable

from fastapi import Depends, UploadFile

from datetime import datetime

from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.orm import Session
from database import SessionLocal, engine

import models
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
class CSIMatrix(BaseModel):
    csi_matrix: list

def preProcess_data(data_array): #cleaning the data
    data=data_array.reshape(1,500,234,1)
    return data

models.Base.metadata.create_all(bind=engine)


app = FastAPI()

# Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

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


loaded_model = tf.keras.models.load_model('norm1000.h5')

prev_act = 0
current_act = 0

q = Queue(maxsize = 2)
values={0:"nm",1:"sitdown", 2:"standup",3:"walking",4:"falling",5:"getintobed"}
# sentiment = 0
probability = 0

# db = get_db()

def savePrediction(db, sentiment, sentiment_text, probability, current_time):
    db_activity = models.Activity(sentiment=sentiment, sentiment_text=sentiment_text, probability='{0:.2f}'.format(probability), current_time=current_time)
    db.add(db_activity)
    db.commit()
    db.refresh(db_activity)
    return db_activity

def getActivities(db, skip: int = 0, limit: int = 100):
    return db.query(models.Activity).offset(skip).limit(limit).all()
    

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



@app.post('/predict') #prediction on data
async def predict(csiMatrix: CSIMatrix, db: Session = Depends(get_db)): #input is from request body
    global prev_act
    print(csiMatrix.csi_matrix[0][0])
    csi_array = np.array(csiMatrix.csi_matrix)
    # Check if queue is full. If it's full then wait until not full
    while (q.full()):
        continue
    q.put(csi_array)
    if(q.full()):
        data_array = np.array(list(q.queue))
        print(data_array.shape)
        clean_text = my_pipeline(data_array) #cleaning and preprocessing of the texts
        #load the saved model when app starts
        predictions = loaded_model.predict(clean_text) #predict the text
        probability = max(predictions.tolist()[0]) #calulate the probability
        sentiment = int(np.argmax(predictions)) #calculate the index of max sentiment
        sentiment_text = values[sentiment]
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        print("sentiment:", sentiment, "-", sentiment_text, ", probability: ", probability)
        if(probability>0.99):
            
            current_act = sentiment
            if prev_act != current_act:
                savePrediction(db, sentiment, sentiment_text, probability, current_time)
                client.publish("testtopic", str(sentiment)+" "+sentiment_text+"-"+'{0:.2f}'.format(probability)+" "+current_time)
            prev_act = current_act
            client.loop_start()
        q.get()
    else:
        print("Queue not full yet. Waiting for next csi array")
        sentiment_text = "WAIT"
        probability = 0
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
         "PREDICTED SENTIMENT": sentiment_text,
         "Probability": probability
    }


@app.get('/mqtt')
def testt():
    
    client.publish("testtopic", "hey msg")
    client.loop_start()
    return '''<h1>Hey</h1>'''


@app.get('/history')
def testt(db: Session = Depends(get_db)):
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    # savePrediction(db, "test", "test", 0.25, current_time)
    activities = getActivities(db)
    return activities



    