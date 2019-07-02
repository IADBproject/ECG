import tensorflow as tf
import pandas as pd
import numpy as np
import math
import os, sys, time
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
        self.model_weights=None
        self.best_model_weights=None
        self.loss=99
        self.loss_list=[]
        self.acc_list=[]
        self.val_loss_list=[]
        self.val_acc_list=[]
        self.main_file=open('output/main_data.txt','w')
        self.training_track=[] 
        self.last_improvement=0

    def average_gradients(self, tower_grads):
        new_weights = []
        for weights_list_tuple in zip(*tower_grads):
            new_weights.append([np.array(weights_).mean(axis=0) for weights_ in zip(*weights_list_tuple)])
        self.model_weights=new_weights
        print(new_weights)
        return new_weights

    def average_gradients1(self, tower_grads):
        """
        Merge the grads computations done by each GPU tower
        """
        ### First Print
        #print("\n \n")
        # print("tower_grads: {}".format(tower_grads))
        average_grads = []
        for grad_and_vars in zip(*tower_grads):
            ## Second print
            print("inintial  grad_and_vars:",grad_and_vars)
            grads = []
            for g, _ in grad_and_vars:
                ## Third Print
                #print("+ Grad by Tower: {}".format(g))
                print(g)
                if g is None:
                    pass
                else:
                    # Add 0 dimension to the gradients to represent the tower.
                    expanded_g = tf.expand_dims(g, 0)

                    # Append on a 'tower' dimension which we will average over below.
                    grads.append(expanded_g)


            # Average over the 'tower' dimension.
            grad = tf.concat(grads, 0)
            grad = tf.reduce_mean(grad, 0)
            print("grad:" ,grad)
            # Keep in mind that the Variables are redundant because they are shared
            # across towers. So .. we will just return the first tower's pointer to
            # the Variable.
            v = grad_and_vars[0][1]
            print("var:" ,v)
            grad_and_var = [grad, v]
            print("grad_and_vars:",grad_and_var)
            print("---------***-----***-------***********__________----------")
            grad_and_var=np.array(grad_and_var)
            average_grads.append(grad_and_var)
           
        self.model_weights=average_grads
        print(average_grads)
        return average_grads


    def update(self,score,times,epoch):
        new_score = []
        new_score = [float(sum(col))/len(col) for col in zip(*score)]

        if new_score[0] <self.loss:
            self.loss = new_score[0]
            self.best_model_weights = self.model_weights
            self.last_improvement=epoch+1

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
        pred_acc = f1_score(y_true=label, y_pred=pred, average='micro')
        print("test acc:",pred_acc)        
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
        self.main_file.close()
        with open('output/training_track.txt', 'w') as f:
            f.write('\n'.join('%s, %s, %s, %s, %s, %s' % x for x in self.training_track))



class WorkerModeling(object):
    def __init__(self,model,batch_size):
        self.model = model
        self.model_weights=None
        self.best_model_weights=None
        self.val_loss = None
        self.val_acc = None
        self.batch_size=batch_size
        self.best_validation_loss=999
        self.last_improvement=0


        self.loss = 0
        self.acc = 0
        self.step=0
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

    def read(self,isTrain):
        if isTrain:
            self.model.adam_op.apply_gradients(self.model_weights) 
        else:
            self.model.adam_op.apply_gradients(self.best_model_weights)  


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




