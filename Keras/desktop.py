from keras.models import Model,h5py
from keras.layers import (Dense,Activation,Dropout,Conv1D,MaxPooling1D,
                            BatchNormalization,Flatten,Input,Add)
from keras.optimizers import Adam
import tensorflow as tf
import pandas as pd
import numpy as np
import math
import os, sys, time
import keras.backend.tensorflow_backend as KTF
from keras.callbacks import*
#import matplotlib
#matplotlib.use("Agg")
#import matplotlib.pyplot as plt
import getopt
from sklearn.model_selection import train_test_split
import random
from sklearn.metrics import confusion_matrix,f1_score, precision_recall_fscore_support

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] ="3"

pd.set_option('display.max_columns', None)


xdata= './../input/xdata.npy'
ylabel='./../input/ydata.npy'
batch_size=8
epochs=30
lr=0.0001

class Data(object):
    def __init__(self,xdata,ydata):

        self.X=np.array([])
        self.Y=np.array([])
        self.readdata(xdata,ydata)

        self.X_train = np.array([])
        self.Y_train = np.array([])

        self.X_validation = np.array([])
        self.Y_validation = np.array([])

        self.X_test = np.array([])
        self.Y_test = np.array([])
        self.datapreposessing()

    def readdata(self,xdata,ydata):
        self.X = np.load(xdata)
        self.Y = np.load(ydata)
        #self.X = self.X[:500]
        #self.Y = self.Y[:500]
        self.Y=pd.get_dummies(self.Y).values

    def datapreposessing(self):
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split\
            (self.X, self.Y, test_size=0.2, random_state=random.seed(42))
        self.X_validation, self.X_test, self.Y_validation, self.Y_test = train_test_split(self.X_test, self.Y_test, 
                                                                        test_size=0.5, random_state=random.seed(22))

        s = self.X_train.shape
        self.X_train = np.reshape(self.X_train, (s[0], s[1] * s[2], 1))
        s = self.X_validation.shape
        self.X_validation = np.reshape(self.X_validation, (s[0], s[1] * s[2], 1))
        s = self.X_test.shape
        self.X_test = np.reshape(self.X_test, (s[0], s[1] * s[2], 1))

    def generator(self,X_train,y_train,batch_size):
        while 1:
            idx = np.arange(0 , len(X_train))
            np.random.shuffle(idx)
            for i in range(0, len(X_train), batch_size):
                k = i+batch_size
                if k>len(X_train):
                    k=len(X_train)
                batch_idx = idx[i:k]
                data_shuffle = [X_train[ ii] for ii in batch_idx]
                labels_shuffle = [y_train[ii] for ii in batch_idx]
                yield np.asarray(data_shuffle), np.asarray(labels_shuffle)


class Modeling(object):
    def __init__(self,dataset,batch_size,epochs):
        self.dataset = dataset
        self.batch_size = batch_size 
        self.epochs = epochs
        self.history = History()
        #self.model = Model()
        self.model = None
        self.model_json=None
        self.model_weights=None
        self.acc=None
        self.training_track=[]
    

    def load(self, received_model):
        self.model = model_from_json(received_model)
        adamopt=Adam(lr=lr)
        self.model.compile(loss='categorical_crossentropy', optimizer=adamopt, metrics=['acc','mae'])

    def train(self):
        config = tf.ConfigProto(allow_soft_placement=True)
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
        config.gpu_options.allow_growth=True   
        session = tf.Session(config=config)
        KTF.set_session(session)

        file_path = "model.h5"
        checkpoint = ModelCheckpoint(file_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    
        steps_per_epoch=math.ceil(len(self.dataset.X_train)/self.batch_size)
        self.history = self.model.fit_generator(self.dataset.generator(self.dataset.X_train,self.dataset.Y_train,self.batch_size), 
                                steps_per_epoch=steps_per_epoch,
                                epochs=self.epochs, verbose=2,
                                use_multiprocessing=False,callbacks=[checkpoint],
                                validation_data = (self.dataset.X_validation,self.dataset.Y_validation))
        self.model_weights=self.model.get_weights()
   
    def average_weights(self,all_weights):
        new_weights = []
        for weights_list_tuple in zip(*all_weights):
            new_weights.append([np.array(weights_).mean(axis=0) for weights_ in zip(*weights_list_tuple)])
        self.model_weights = new_weights
        
    def r_block(self,in_layer,k,f):
        x=BatchNormalization()(in_layer)
        x=Activation('relu')(x)
        x=Dropout(0.2)(x)
        x=Conv1D(64*k,f,padding='same')(x)
        x=BatchNormalization()(x)
        x=Activation('relu')(x)
        x=Dropout(0,2)(x)
        x=Conv1D(64*k,f,padding='same')(x)
        x=Add()([x,in_layer])
        return x

    def subsampling_r_block(self,in_layer,k,f):
        x=BatchNormalization()(in_layer)
        x=Activation('relu')(x)
        x=Dropout(0.2)(x)
        x=Conv1D(64*k,f,padding='same')(x)
        x=BatchNormalization()(x)
        x=Activation('relu')(x)
        x=Dropout(0,2)(x)
        x=Conv1D(64*k,1,strides=2)(x)
        pool=MaxPooling1D(1,strides=2)(in_layer)
        x=Add()([x,pool])
        return x

    def create_model(self):
        filter_size=16
        ins=Input((1300, 1))
        act1=Conv1D(64,filter_size,padding='same')(ins)
        x=BatchNormalization()(act1)
        x=Activation('relu')(x)
        x=Conv1D(64,filter_size,padding='same')(x)
        x=BatchNormalization()(x)
        x=Activation('relu')(x)
        x=Dropout(0.2)(x)
        conv2=Conv1D(64,1,strides=2)(x)
        pool1=MaxPooling1D()(act1)
        x=Add()([conv2,pool1])
        k=1
        for i in range(1,9,1):
            if i%2 ==0:
                k+=1
            x=Conv1D(64*k,filter_size,padding='same')(x)
            x=self.r_block(x,k,filter_size)
            x=self.subsampling_r_block(x,k,filter_size)
        x=BatchNormalization()(x)
        x=Activation('relu')(x)
        x=Flatten()(x)
        dense=Dense(4,activation='softmax')(x)
        self.model=Model(inputs=ins,outputs=dense)
        #self.model.summary()
        adamopt=Adam(lr=lr)
        self.model.compile(optimizer=adamopt,loss='categorical_crossentropy',metrics=['acc','mae'])


    def predict(self):
        self.model.load_weights("model.h5")
        score = self.model.predict(self.dataset.X_test)

        score =  np.argmax(score, axis = 1)
        ytrue =  np.argmax(self.dataset.Y_test, axis = 1)
        labels, counts = np.unique(ytrue, return_counts = True)

        conf_matrix = confusion_matrix(ytrue, score, labels)
        true_positive = np.diag(conf_matrix)
        false_negative = []
        false_positive = []
        for i in range(len(conf_matrix)):
            false_positive.append(int(sum(conf_matrix[:,i]) - conf_matrix[i,i]))
            false_negative.append(int(sum(conf_matrix[i,:]) - conf_matrix[i,i]))
        
        precision, recall, F1_score, support = precision_recall_fscore_support(ytrue,score, average = None)

        for i in range(len(labels)):
            precision[i] =  round(precision[i], 2)
            recall[i] =  round(recall[i], 2)
            F1_score[i] = round(F1_score[i], 2)


        label_occurrences = np.where(support !=0)
        occs = label_occurrences[0]
        metrics_values = np.vstack((labels, true_positive, false_negative,
                                    false_positive, precision[occs],
                                    recall[occs], F1_score[occs], support[occs]))
        metrics_values = np.transpose(metrics_values)
        metrics_values = pd.DataFrame(metrics_values, columns = ["Labels", "TP", "FN", "FP",
                                    "Precision", "Recall", "F1 Score", "Records by Labels"])
        print("{}".format(metrics_values))
        m_file = open('output/keras_F1_data.txt','w')
        print("{}".format(metrics_values),file=m_file)
        m_file.close()



    def stats(self):

        loss_list = [s for s in self.history.history.keys() if 'loss' in s and 'val' not in s]
        val_loss_list = [s for s in self.history.history.keys() if 'loss' in s and 'val' in s]
        acc_list = [s for s in self.history.history.keys() if 'acc' in s and 'val' not in s]
        val_acc_list = [s for s in self.history.history.keys() if 'acc' in s and 'val' in s]

        if len(loss_list) == 0:
            print('Loss is missing in history')
            return
        epochs = range( 1, len(self.history.history[loss_list[0]]) + 1 )

        m_file = open('output/keras_train_loss_data.txt','w')
        mv_file = open('output/keras_val_loss_data.txt','w')
        
    
        for l in loss_list:
            print(self.history.history[l],file=m_file)
        m_file.close()
        for l in val_loss_list:
            print(self.history.history[l],file=mv_file)
        mv_file.close()
        
        for i in epochs:
            self.training_track.append((i,self.history.history[loss_list[0]][i-1],self.history.history[val_loss_list[0]][i-1],self.history.history[acc_list[0]][i-1],self.history.history[val_acc_list[0]][i-1]))

        with open('output/keras_train_data.txt', 'w') as f:
            f.write('\n'.join('%s, %s, %s, %s, %s' % x for x in self.training_track))



def main():


    start= time.time()
    data=Data(xdata,ylabel)
    end_data= time.time()
    
    model=Modeling(data,batch_size,epochs)
    model.create_model()

    fit = time.time()
    model.train()
    end = time.time()
    model.predict()
    end_evaluate = time.time()
    model.stats()
    end_matrix=time.time()

    print('Time to load data:', end_data-start)
    print('Time to create graph:', fit-end_data)
    print('Time to fit data:', end-fit)
    print('Time to evaluate:', end_evaluate-end)
    print('Time for matrics:', end_matrix-end_evaluate)

if __name__ == '__main__':
    main()
