import keras
from keras.models import Model,model_from_json
import tensorflow as tf
import numpy as np
import os, sys, time
from keras.callbacks import*
#from memory_profiler import profile
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
    def __init__(self):
        self.model = None
        self.model_json=None
        self.model_weights=None
        self.best_model_weights=None
        self.val_loss = None
        self.val_acc = None

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

        self.loss = 0.0
        self.acc = 0.0
        self.history = TrainHistory()
        self.loss_list=[]
        self.acc_list=[]
        self.log_list=[]
        self.step=0

    def load(self):
        self.model = model_from_json(self.model_json)
        self.model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy','mae'])

    #@profile(precision=4,stream=open('output/memory_profiler.log','w+'))
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

    #@profile(precision=4,stream=open('output/memory_profiler.log','w+'))
    def validate(self,data,label):
        score=self.model.evaluate(x=data,y=label,verbose = 0)
        self.val_loss=score[0]
        self.val_acc=score[1]
    
    def read(self,isTrain):
        if isTrain:
            self.model.set_weights(self.model_weights)
        else:

            self.model.set_weights(self.best_model_weights)  
            #self.trainstats()


    #@profile(precision=4,stream=open('output/memory_profiler.log','w+'))
    def test(self,data):
        prediction=self.model.predict(data)
        self.pred.append(prediction)
        return prediction


    def trainstats(self,rank,host):
       
        #print("---------------")
      
        pred=np.vstack(self.pred)
        label=np.vstack(self.label)
        #print("pred",pred)
        #print("label",label)
        pred =  np.argmax(pred, axis = 1)
        label =  np.argmax(label, axis = 1)
        labels, counts = np.unique(label, return_counts = True)

        #print("pred",pred)
        #print("label",label)
        #return 0
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
        #print("work rank",rank,"\n {}".format(metrics_values))
        #print("---------------")
        filename=str('output/worker/host_'+str(host)+'_rank_'+str(rank)+'_train.txt') 
        wfile = open(filename,'w')
        filename1=str('output/worker/host_'+str(host)+'_rank_'+str(rank)+'_train_loss.txt')
        wtfile = open(filename1,'w')
        filename2=str('output/worker/host_'+str(host)+'_rank_'+str(rank)+'_val_loss.txt')
        wvfile = open(filename2,'w')

        print(self.loss_list,file=wtfile)
        print(self.val_loss_list,file=wvfile)
        print("test acc:",pred_acc,file=wfile)
        print(" {}".format(metrics_values),file=wfile)

        wvfile.close()
        wtfile.close()
        wfile.close()

