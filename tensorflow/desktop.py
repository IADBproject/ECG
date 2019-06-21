import tensorflow as tf
import pandas as pd
import numpy as np
import math
import os, sys, time
#import matplotlib
#matplotlib.use("Agg")
#import matplotlib.pyplot as plt
import getopt
from sklearn.model_selection import train_test_split
import random
from sklearn.metrics import confusion_matrix,f1_score, precision_recall_fscore_support

pd.set_option('display.max_columns', None)


xdata= './../input/xdata.npy'
ylabel='./../input/ydata.npy'
batch_size =16
epochs= 30
learning_rate= 0.001

class Data(object):
    def readdata(self,xdata,ydata):
        self.X = np.load(xdata)
        self.Y = np.load(ydata)
        #self.Y = self.Y[:500]
        #self.X = self.X[:500]
        self.Y = pd.get_dummies(self.Y).to_numpy()

    def datapreposessing(self):
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(self.X, self.Y, test_size=0.2, random_state=random.seed())
        self.X_validation, self.X_test, self.Y_validation, self.Y_test = train_test_split(self.X_test, self.Y_test, test_size=0.5, random_state=random.seed())
        s = self.X_train.shape
        self.X_train = np.reshape(self.X_train, (s[0], s[1] * s[2], 1))
        s = self.X_validation.shape
        self.X_validation = np.reshape(self.X_validation, (s[0], s[1] * s[2], 1))
        s = self.X_test.shape
        self.X_test = np.reshape(self.X_test, (s[0], s[1] * s[2], 1))

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

    def next_batch(self,data,labels,batch_size):
        num_el = data.shape[0]
        while True:
            idx = np.arange(0 , num_el)
            np.random.shuffle(idx)
            current_idx = 0
            while current_idx < num_el:
                batch_idx = idx[current_idx:current_idx+batch_size]
                current_idx += batch_size
                data_shuffle = [data[ i] for i in batch_idx]
                labels_shuffle = [labels[i] for i in batch_idx]
                yield np.asarray(data_shuffle), np.asarray(labels_shuffle)


class Modeling(object):

    def __init__(self,dataset,batch_size,epochs,learning_rate):
        self.dataset = dataset
        self.batch_size = batch_size 
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.loss_list=[]
        self.val_loss_list=[]
        self.best_validation_loss=9999
        self.reuse=False
        self.train()

            
    def r_block(self,in_layer,k,is_training):
        x = tf.layers.batch_normalization(in_layer)
        x = tf.nn.relu(x)
        x = tf.layers.dropout(x, rate=0.2, training=is_training)
        x = tf.layers.conv1d(x,64*k,16,padding='same',kernel_initializer=tf.glorot_uniform_initializer())
        x = tf.layers.batch_normalization(x)
        x = tf.nn.relu(x)
        x = tf.layers.dropout(x, rate=0.2, training=is_training)
        x = tf.layers.conv1d(x,64*k,16,padding='same',kernel_initializer=tf.glorot_uniform_initializer())
        x = tf.add(x,in_layer)
        return x

    def subsampling_r_block(self,in_layer,k,is_training):
        x = tf.layers.batch_normalization(in_layer)
        x = tf.nn.relu(x)
        x = tf.layers.dropout(x, rate=0.2, training=is_training)
        x = tf.layers.conv1d(x,64*k,16,kernel_initializer=tf.glorot_uniform_initializer(),padding='same')
        x = tf.layers.batch_normalization(x)
        x = tf.nn.relu(x)
        x = tf.layers.dropout(x, rate=0.2, training=is_training)
        x = tf.layers.conv1d(x, 64*k, 1, strides=2,kernel_initializer=tf.glorot_uniform_initializer())
        pool = tf.layers.max_pooling1d(in_layer,1,strides=2)
        x = tf.add(x,pool)
        return x

    def conv_net(self,x,is_training):
        # Define a scope for reusing the variables
        with tf.variable_scope('ConvNet', reuse=self.reuse): 

            act1 = tf.layers.conv1d(x, 64, 16, padding='same',kernel_initializer=tf.glorot_uniform_initializer())
            x = tf.layers.batch_normalization(act1)
            x = tf.nn.relu(x)

            x = tf.layers.conv1d(x, 64, 16, padding='same',kernel_initializer=tf.glorot_uniform_initializer())
            x = tf.layers.batch_normalization(x)
            x = tf.nn.relu(x)

            x = tf.layers.dropout(x, rate=0.2, training=is_training)
            x1 = tf.layers.conv1d(x, 64, 1, strides=2,kernel_initializer=tf.glorot_uniform_initializer())

            x2 = tf.layers.max_pooling1d(act1,2,strides=2)
            x = tf.add(x1,x2)

            k=1
            for i in range(1,9,1):
                if i%2 ==0:
                    k+=1
                x=tf.layers.conv1d(x,64*k,16,padding='same',kernel_initializer=tf.glorot_uniform_initializer())
                x=self.r_block(x,k,is_training)
                x=self.subsampling_r_block(x,k,is_training)

            x = tf.layers.batch_normalization(x)
            x = tf.nn.relu(x)
            x = tf.layers.flatten(x)
            out = tf.layers.dense(x, 4,kernel_initializer=tf.glorot_uniform_initializer())
        return out


    def train(self):
        tf.reset_default_graph()
        with tf.Graph().as_default() as graph:
            X = tf.placeholder(tf.float32, [None, 1300,1])
            Y = tf.placeholder(tf.float32, [None, 4])
            is_training = tf.placeholder(tf.bool, shape=())

            trainmodel = self.conv_net(X, is_training)
           
            loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=trainmodel, labels=Y))
            train_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(loss_op)
            accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(trainmodel, 1), tf.argmax(Y, 1)), tf.float32))


        n_batches = len(self.dataset.X_train)//self.batch_size
        n_valbatches = math.ceil(len(self.dataset.X_validation)/self.batch_size)
       

        with tf.Session(graph=graph) as sess:
            init = tf.group(tf.global_variables_initializer(),
                        tf.local_variables_initializer())
            sess.run(init)
            saver = tf.train.Saver()
            

            for i in range(self.epochs):
                acc,loss,val_acc,val_loss=0,0,0,0
                next_batch_gen = self.dataset.next_batch( self.dataset.X_train, self.dataset.Y_train,self.batch_size)
                estart=time.time()
                for j in range(n_batches):
                    data, label = next(next_batch_gen)
                    self.reuse=False
                    _,losst,acct=sess.run([train_op,loss_op,accuracy], 
                                    feed_dict={X: data, Y: label, is_training: True})
                    acc+=acct/n_batches
                    loss+=losst/n_batches
                #output =sess.run(trainmodel, 
                #                    feed_dict={X: data, is_training: True})
                #print("trainmodel {}".format(output))
                self.reuse=True
                for t in  range(n_valbatches):
                    j = min((t+1 )* self.batch_size, len(self.dataset.Y_validation))
                    acct,losst=sess.run([accuracy,loss_op], 
                                    feed_dict={X: self.dataset.X_validation[t* self.batch_size:j],
                                    Y: self.dataset.Y_validation[t* self.batch_size:j], is_training: False})
                    val_acc+=acct/n_valbatches
                    val_loss+=losst/n_valbatches
                
                improved_str='None'
                if val_loss < self.best_validation_loss:
                    self.best_validation_loss = val_loss
                    saver.save(sess,  "./model.ckpt")
                    improved_str = '\n Update!'

                msg = "Iter: {0}, Train Accuracy: {1:.4},Train Loss: {2:.4}, Validation Acc: {3:.4}, Validation Loss: {4:.4} ---Time: {5:.4} {6}"
                print(msg.format(i + 1, acc,loss, val_acc,val_loss,time.time()-estart, improved_str))
                self.loss_list.append(loss)
                self.val_loss_list.append(val_loss)

            saver.restore(sess, "./model.ckpt")
            pred=[]
            acc_test,loss_test=0,0
            n_tbatches = math.ceil(len(self.dataset.X_test)/self.batch_size)
            for i in  range(n_tbatches):
                    j = min((i+1 ) * self.batch_size, len(self.dataset.Y_test))
                    pre,acc,loss=sess.run([trainmodel,accuracy,loss_op], 
                                    feed_dict={X: self.dataset.X_test[i* self.batch_size:j],
                                    Y: self.dataset.Y_test[i* self.batch_size:j], is_training: False})
                    acc_test+=acc/n_valbatches
                    loss_test+=loss/n_valbatches
                    pred.append(pre)

            print('Predict loss:', loss_test)
            print('Predict accuracy:', acc_test)
            self.predict(pred)


    def predict(self,pred):

        pred=np.vstack(pred)
        pred =  np.argmax(pred, axis = 1)
        label =  np.argmax(self.dataset.Y_test, axis = 1)
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

def main():

    data=Data(xdata,ylabel)

    model=Modeling(data,batch_size,epochs,learning_rate)

if __name__ == '__main__':
    main()

