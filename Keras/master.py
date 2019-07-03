from keras.models import Model
from keras.layers import (Dense,Activation,Dropout,Conv1D,MaxPooling1D,
                            BatchNormalization,Flatten,Input,Add)
from keras.optimizers import Adam
import tensorflow as tf
import pandas as pd
import numpy as np
import math
import os, sys, time
from keras.callbacks import*
import getopt
from sklearn.model_selection import train_test_split
import random
from sklearn.metrics import confusion_matrix,f1_score, precision_recall_fscore_support
pd.set_option('display.max_columns', None)


class Data(object):
    def __init__(self,xdata,ydata,batch_size,size):

        self.X=np.array([])
        self.Y=np.array([])
        self.readdata(xdata,ydata)

        self.X_train = np.array([])
        self.Y_train = np.array([])

        self.X_validation = np.array([])
        self.Y_validation = np.array([])

        self.X_test = np.array([])
        self.Y_test = np.array([])
        self.batch_size = batch_size
        self.size = size
        self.datapreposessing()
      

    def readdata(self,xdata,ydata):
        self.X = np.load(xdata)
        self.Y = np.load(ydata)
        #self.X = self.X[:1000]
        #self.Y = self.Y[:1000]
        self.Y=pd.get_dummies(self.Y).values


    def datapreposessing(self):
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(self.X,
                                                                self.Y, test_size=0.2, random_state=random.seed(42))
        self.X_validation, self.X_test, self.Y_validation, self.Y_test = train_test_split(self.X_test, self.Y_test, 
                                                                        test_size=0.5, random_state=random.seed(22))

        s = self.X_train.shape
        self.X_train = np.reshape(self.X_train, (s[0], s[1] * s[2], 1))
        s = self.X_validation.shape
        self.X_validation = np.reshape(self.X_validation, (s[0], s[1] * s[2], 1))
        s = self.X_test.shape
        self.X_test = np.reshape(self.X_test, (s[0], s[1] * s[2], 1))

    def generator(self,X_train,y_train):
        while 1:
            idx = np.arange(0 , len(X_train))
            np.random.shuffle(idx)
            for i in range(0, len(X_train), self.batch_size*self.size):
                k = i+self.batch_size*self.size
                if k>len(X_train):
                    break;
                batch_idx = idx[i:k]
                data_shuffle = [X_train[ ii] for ii in batch_idx]
                labels_shuffle = [y_train[ii] for ii in batch_idx]
                yield np.asarray(data_shuffle), np.asarray(labels_shuffle)

    def getstep(self,X_train):
        return len(X_train)//(self.batch_size*self.size)


class MasterModeling(object):
    def __init__(self,dataset):
        self.dataset = dataset
        self.model = Model
        self.model_json=None
        self.model_weights=None
        self.best_model_weights=None
        self.loss=99
        self.loss_list=[]
        self.acc_list=[]
        self.val_loss_list=[]
        self.val_acc_list=[]
        self.main_file=open('output/main_data.txt','w')
        self.training_track=[]        

    def create(self,lr):
        self.create_model(lr)
        self.model_json = self.model.to_json()
        self.model_weights = self.model.get_weights()

    def average_weights(self,all_weights):
        new_weights = []
        for weights_list_tuple in zip(*all_weights):
            new_weights.append([np.array(weights_).mean(axis=0) for weights_ in zip(*weights_list_tuple)])
        self.model_weights =new_weights

    
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

    def create_model(self,lr):
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
        for i in range(1,3,1):
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
        Adamopt=Adam(lr=lr)
        #self.model.summary()
        self.model.compile(optimizer=Adamopt,loss='categorical_crossentropy',metrics=['accuracy','mae'])

    def update(self,score,times,epoch,fit):
        new_score = []
        new_score = [float(sum(col))/len(col) for col in zip(*score)]

        if new_score[0] <self.loss:
            self.loss = new_score[0]
            self.best_model_weights = self.model_weights
            self.best_epoch=epoch+1
	    self.best_time=time.time()-fit

        self.loss_list.append(new_score[2])
        self.acc_list.append(new_score[3])
        self.val_loss_list.append(new_score[0])
        self.val_acc_list.append(new_score[1])

        msg="Epoch Info:{0},Train Acc:{1:>5.4},Train Loss:{2:>5.4},Val Acc:{3:>5.4},Val Loss:{4:>5.4} --- Time:{5}s"
        epotime=time.time()-times
        print(msg.format(epoch + 1, new_score[3],new_score[2], new_score[1],new_score[0], epotime))
        self.training_track.append((epoch + 1,new_score[2],new_score[0],new_score[3],new_score[1],epotime))

    def predict(self,pred,label,ltime):
        print("-----Total-----")
        pred=np.vstack(pred)
        label=np.vstack(label)
        pred =  np.argmax(pred, axis = 1)
        label =  np.argmax(label, axis = 1)
        labels, counts = np.unique(label, return_counts = True)

        conf_matrix = confusion_matrix(label, pred, labels)
        true_positive = np.diag(conf_matrix)
        false_negative = []
        false_positive = []
        for i in range(len(conf_matrix)):
            false_positive.append(int(sum(conf_matrix[:,i]) - conf_matrix[i,i]))
            false_negative.append(int(sum(conf_matrix[i,:]) - conf_matrix[i,i]))
        self.pred_acc = f1_score(y_true=label, y_pred=pred, average='micro')
        print("test acc:",self.pred_acc)        
        precision, recall, F1_score, support = precision_recall_fscore_support(label,pred, average = None)

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
        print("testing time :",time.time()-ltime)
        self.testing_time=time.time()-ltime
        m_file = open('output/F1_data.txt','w')
        print("{}".format(metrics_values),file=m_file)
        m_file.close()
        print("-----Total------")
       




    def savestat(self):
        print("preparing time :",self.dataset_time,file=self.main_file)
        print("training time :",self.training_time,file=self.main_file)
        print("testing time :",self.testing_time,file=self.main_file)
        print("minimum appears at:",self.best_epoch,file=self.main_file)
	print("converage time:",self.best_time,file=self.main_file)
	print("test acc:",self.pred_acc,file=self.main_file) 
        self.main_file.close()
        with open('output/training_track.txt', 'w') as f:
            f.write('\n'.join('%s, %s, %s, %s, %s, %s' % x for x in self.training_track))


def mastermain(size,batch_size,lr,mode,data='./../input/xdata.npy',label='./../input/ydata.npy'):


    start=time.time()
    if mode ==1:
        data=Data(data,label,batch_size,size)
	
        train_next_batch_gen = data.generator( data.X_train, data.Y_train)
        val_next_batch_gen = data.generator( data.X_validation, data.Y_validation)
        test_next_batch_gen = data.generator( data.X_test, data.Y_test)
        train_step = data.getstep(data.X_train)
        val_step = data.getstep(data.X_validation)
        test_step = data.getstep(data.X_test)

        modeling=MasterModeling(data)
        modeling.create(lr)
        modeling.create_time=start
        modeling.dataset_time=time.time()-start
        print('Dataset preparing --- Time:',time.time()-start)
        return modeling,train_next_batch_gen,val_next_batch_gen,\
        test_next_batch_gen,train_step,val_step,test_step
    else:
        data=None
        modeling=MasterModeling(data)
        modeling.create(lr)
        modeling.create_time=start
        modeling.dataset_time=time.time()-start
        return modeling

