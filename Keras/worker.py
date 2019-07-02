import keras
from keras.models import Model,model_from_json
import tensorflow as tf
import numpy as np
import os, sys, time
from keras.callbacks import *
from keras.optimizers import Adam
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score, precision_recall_fscore_support
import pandas as pd


class TrainHistory(keras.callbacks.Callback):
    def __init__(self):
        self.loss = None
        self.acc = None

    def on_epoch_end(self,epoch, logs={}):
        self.loss = logs.get('loss')
        self.acc = logs.get('acc')

class WorkerModeling(object):
    def __init__(self,batch_size):
        self.model = None
        self.model_json=None
        self.model_weights=None
        self.best_model_weights=None
        self.val_loss = None
        self.val_acc = None
        self.batch_size=batch_size

        self.loss = 0
        self.acc = 0
        self.step=0
        self.history = TrainHistory()
        self.loss_list=[]
        self.acc_list=[]
        self.val_loss_list=[]
        self.val_acc_list=[]
        self.label=[]
        self.pred=[]
        self.step=0
        self.training_track=[]

    def data(self,xtrain_name,ytrain_name,xval_name,yval_name,xtest_name,ytest_name):
        self.x_train=np.load(xtrain_name)
        self.y_train=np.load(ytrain_name)
        self.x_valid=np.load(xval_name)
        self.y_valid=np.load(yval_name)
        self.x_test=np.load(xtest_name)
        self.y_test=np.load(ytest_name)
        self.train_next_batch_gen = self.generator( self.x_train, self.y_train)
        self.val_next_batch_gen = self.generator( self.x_valid, self.y_valid)
        self.test_next_batch_gen = self.generator( self.x_test, self.y_test)
        self.train_step = self.getstep(self.x_train)
        self.val_step = self.getstep(self.x_valid)
        self.test_step = self.getstep(self.x_test)

    def getstep(self,X_train):
        return len(X_train)//(self.batch_size)

    def generator(self,X_train,y_train):
        while 1:
            idx = np.arange(0 , len(X_train))
            np.random.shuffle(idx)
            for i in range(0, len(X_train), self.batch_size):
                k = i+self.batch_size
                if k>len(X_train):
                    break;
                batch_idx = idx[i:k]
                data_shuffle = [X_train[ ii] for ii in batch_idx]
                labels_shuffle = [y_train[ii] for ii in batch_idx]
                yield np.asarray(data_shuffle), np.asarray(labels_shuffle)

    def track(self,i,stime):
        self.training_track.append((i+1,self.loss_list[-1],self.val_loss_list[-1],self.acc_list[-1],self.val_acc_list[-1],time.time()-stime))

    def load(self,lr):
        self.model = model_from_json(self.model_json)
        Adamopt=Adam(lr=lr)
        self.model.compile(loss='categorical_crossentropy', optimizer=Adamopt, metrics=['accuracy','mae'])

    def train(self,data,label,end_epoch):
        self.model.fit(x=data,y=label, epochs=1,callbacks=[self.history],verbose = 0)
        self.loss+=self.history.loss
        self.acc+=self.history.acc
        self.step+=1
        if end_epoch:
            self.model_weights=self.model.get_weights()
            self.loss_list.append((self.loss/self.step))
            self.acc_list.append((self.acc/self.step))
            self.acc=0
            self.loss=0
            self.step=0

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
        self.pred.append(prediction)
        return prediction


    def trainstats(self,rank,host):

      
        pred=np.vstack(self.pred)
        label=np.vstack(self.label)


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
        pred_acc = f1_score(y_true=label, y_pred=pred, average='micro')
        print("work rank",rank,"test acc:",pred_acc)
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

        filename=str('output/worker/host_'+str(host)+'_rank_'+str(rank)+'_F1_score.txt') 
        wfile = open(filename,'w')

        print(" {}".format(metrics_values),file=wfile)

        wfile.close()
        with open('output/worker/host_'+str(host)+'_rank_'+str(rank)+'_training_track.txt', 'w') as f:
            f.write('\n'.join('%s, %s, %s, %s, %s, %s' % x for x in self.training_track))
