import keras
from keras.models import Model,model_from_json
import tensorflow as tf
import numpy as np
import os, sys, time
from keras.callbacks import*

class TrainHistory(keras.callbacks.Callback):
    def __init__(self):
        self.loss = None
        self.acc = None

    def on_epoch_end(self,epoch, logs={}):
        self.loss = logs.get('loss')
        self.acc = logs.get('acc')

class WorkerModeling(object):
    def __init__(self):
        self.model = None
        self.model_json=None
        self.model_weights=None
        self.best_model_weights=None
        self.val_loss = None
        self.val_acc = None
        self.loss = None
        self.acc = None
        self.history = TrainHistory()
        self.loss_list=[]
        self.acc_list=[]

    def load(self):
        self.model = model_from_json(self.model_json)
        self.model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy','mae'])

    def train(self,data,label,end_epoch):
        self.model.fit(x=data,y=label, epochs=1,callbacks=[self.history],verbose = 0)
        if end_epoch:
            self.model_weights=self.model.get_weights()
            self.loss_list.append(self.history.loss)
            self.acc_list.append(self.history.acc)

    def validate(self,data,label):
        score=self.model.evaluate(x=data,y=label,verbose = 0)
        self.val_loss=score[0]
        self.val_acc=score[1]
    
    def read(self,isTrain):
        if isTrain:
            self.model.set_weights(self.model_weights)
        else:
            self.model.set_weights(self.best_model_weights)  

    def test(self,data):
        prediction=self.model.predict(data)
        return prediction




