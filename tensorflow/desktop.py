import tensorflow as tf
import pandas as pd
import numpy as np
import math
import os, sys, time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import getopt
from sklearn.model_selection import train_test_split
import random
from sklearn.metrics import confusion_matrix,f1_score, precision_recall_fscore_support
pd.set_option('display.max_columns', None)


xdata= './../input/xdata.npy'
ylabel='./../input/ydata.npy'
batch_size =4
epochs= 5
learning_rate= 0.001

class Data(object):
    def readdata(self,xdata,ydata):
        self.X = np.load(xdata)
        self.Y = np.load(ydata)
        self.Y = self.Y[:500]
        self.X = self.X[:500]
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
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess=tf.Session(config=config)
        self.loss_list=[]
        self.val_loss_list=[]
        self.best_validation_loss=99
        self.train()

            
    def r_block(self,in_layer,k,is_training):
        x = tf.layers.batch_normalization(in_layer)
        x = tf.nn.relu(x)
        x = tf.layers.dropout(x, rate=0.2, training=is_training)
        x = tf.layers.conv1d(x,64*k,16,padding='same')
        x = tf.layers.batch_normalization(x)
        x = tf.nn.relu(x)
        x = tf.layers.dropout(x, rate=0.2, training=is_training)
        x = tf.layers.conv1d(x,64*k,16,padding='same')
        x = tf.add(x,in_layer)
        return x

    def subsampling_r_block(self,in_layer,k,is_training):
        x = tf.layers.batch_normalization(in_layer)
        x = tf.nn.relu(x)
        x = tf.layers.dropout(x, rate=0.2, training=is_training)
        x = tf.layers.conv1d(x,64*k,16,padding='same')
        x = tf.layers.batch_normalization(x)
        x = tf.nn.relu(x)
        x = tf.layers.dropout(x, rate=0.2, training=is_training)
        x = tf.layers.conv1d(x, 64*k, 1, strides=2)
        pool = tf.layers.max_pooling1d(in_layer,1,strides=2)
        x = tf.add(x,pool)
        return x

    def conv_net(self,x,reuse,is_training):
        # Define a scope for reusing the variables
        with tf.variable_scope('ConvNet', reuse=reuse):

            x = tf.layers.conv1d(x, 64, 16, padding='same')
            x = tf.layers.batch_normalization(x)
            act1 = tf.nn.relu(x)

            x = tf.layers.conv1d(act1, 64, 16, padding='same')
            x = tf.layers.batch_normalization(x)
            act1 = tf.nn.relu(x)
            x = tf.layers.dropout(x, rate=0.2, training=is_training)
            x1 = tf.layers.conv1d(x, 64, 1, strides=2)

            x2 = tf.layers.max_pooling1d(act1,2,strides=2)
            x = tf.add(x1,x2)

            k=1
            for i in range(1,9,1):
                if i%2 ==0:
                    k+=1
                x=tf.layers.conv1d(x,64*k,16,padding='same')
                x=self.r_block(x,k,is_training)
                x=self.subsampling_r_block(x,k,is_training)

            x = tf.layers.batch_normalization(x)
            x = tf.nn.relu(x)
            x = tf.contrib.layers.flatten(x)
            out = tf.layers.dense(x, 4)
            out = tf.nn.softmax(out)  if not is_training else out
        return out


    def train(self):

        X = tf.placeholder(tf.float32, [None, 1300,1])
        Y = tf.placeholder(tf.float32, [None, 4])

        self.trainmodel = self.conv_net(X, reuse=False, is_training=True)
       
        loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.trainmodel, labels=Y))
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        train_op = optimizer.minimize(loss_op)
                
        correct_pred = tf.equal(tf.argmax(self.trainmodel, 1), tf.argmax(Y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


        init = tf.global_variables_initializer()
        n_batches = len(self.dataset.X_train)//self.batch_size
        best_validation_accuracy=0
        saver = tf.train.Saver()

        # Launch the graph
        self.sess.run(init)
        next_batch_gen = self.dataset.next_batch( self.dataset.X_train, self.dataset.Y_train,self.batch_size)
        for i in range(self.epochs):
            estart=time.time()
            for j in range(n_batches):
                data, label = next(next_batch_gen)
                self.sess.run(train_op, feed_dict={X: data, Y: label})
                
            losst,acct=self.sess.run([loss_op,accuracy], feed_dict={X: data, Y: label})
            acc_validation,val_loss= self.predict(self.dataset.X_validation,self.dataset.Y_validation,False)
            improved_str='None'
            if val_loss < self.best_validation_loss:
                self.best_validation_loss = val_loss
                saver.save(self.sess,  "./model.ckpt")
                improved_str = '\n Update!'

            msg = "Iter: {0:>4}, Train-Batch Accuracy: {1:>6.3},Train-Batch Loss: {2:>6.3}, Validation Acc: {3:>6.3}, Validation Loss: {4:>6.3} ---Time: {5} {6}"
            print(msg.format(i + 1, acct,losst, acc_validation,val_loss,time.time()-estart, improved_str))
            self.loss_list.append(losst)
            self.val_loss_list.append(val_loss)

        saver.restore(self.sess, "./model.ckpt")
        acc_test,loss_test= self.predict(self.dataset.X_test,self.dataset.Y_test,True)
        print('Predict loss:', loss_test)
        print('Predict accuracy:', acc_test)
        self.plot()


    def predict(self,data,label,isValid):
        num=len(data)
        i = 0
        acc_list=[]
        loss_list=[]
        pred=[]
        XX = tf.placeholder(tf.float32, [None, 1300,1])
        YY = tf.placeholder(tf.float32, [None, 4])
        self.testmodel = self.conv_net(XX, reuse=True, is_training=False)
        loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.testmodel, labels=YY))
        correct_pred = tf.equal(tf.argmax(self.testmodel, 1), tf.argmax(YY, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        
        while i < num:
            j = min(i + self.batch_size, num)
            pre,acc,losst=self.sess.run([self.testmodel,accuracy,loss_op], feed_dict={XX: data[i:j],YY: label[i:j]})
            pred.append(pre)
            acc_list.append(acc)
            loss_list.append(losst)
            i = j
        acc=float(sum(acc_list))/len(acc_list)
        loss=float(sum(loss_list))/len(loss_list)

        if isValid:
            pred=np.vstack(pred)
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

        return acc,loss
        
    def plot(self):

        if len(self.loss_list) == 0:
            print('Loss is missing in history')
            return

        epochs = range( 1, len(self.loss_list) + 1 )

        plt.figure(1)
        plt.plot( epochs,
                  self.loss_list,
                  'b',
                  label = 'Training loss (' + str( str( format( self.loss_list[-1],'.5f' ) ) + ')' )
                )
        plt.plot( epochs,
                  self.val_loss_list,
                  'g',
                  label = 'Validation loss (' + str ( str( format( self.val_loss_list[-1],'.5f' ) ) + ')' )
                )

        plt.title('Loss per Epoch')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        plt.savefig( 'loss.png', bbox_inches='tight' )
        plt.close()


def main():

    data=Data(xdata,ylabel)

    model=Modeling(data,batch_size,epochs,learning_rate)

if __name__ == '__main__':
    main()

