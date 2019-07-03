from __future__ import division
import sys
import glob
import numpy as np
import pandas as pd
import tensorflow as tf
import time
import sklearn
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix , f1_score , precision_recall_fscore_support
from sklearn.preprocessing import label_binarize
import asyncio
import subprocess as sp


class CNNClassifier(object):

    def __init__(self, *args, **kwargs):
        self.lr_op = float(args[0]) 
        self.sandbox_fn = args[1] 
        self.argv_testbed = args[2] 

        self.y_TRAIN = args[3] 
        self.y_VALID= args[4]
        self.y_TEST = args[5]

        self.x_TRAIN = args[6]
        self.x_VALID = args[7]
        self.x_TEST = args[8]

        self.max_epochs = int(args[9])
        self.batch_size = int(args[10])

        self.tf_ps = args[11]
        self.tf_workers = args[12]
        self.num_ps = int(args[13])
        self.num_workers = int(args[14])
        self.job_name = args[15]
        self.task_index = int(args[16])
        self.testbed_dir = args[17]
        
        #######################################################################
        ## Local variables
        self.Y_train = np.load(str(self.sandbox_fn +self.y_TRAIN))
        self.Y_validation = np.load(str(self.sandbox_fn +self.y_VALID))
        self.Y_test = np.load(str(self.sandbox_fn +self.y_TEST))
        #self.testbed_dir = str("./testbed/")
        self.X_train =  np.load(str(self.sandbox_fn+ self.x_TRAIN))
        self.X_validation =  np.load(str(self.sandbox_fn +self.x_VALID))
        self.X_test =  np.load(str(self.sandbox_fn+self.x_TEST))
        #self.Y_train = self.Y_train[:300]
        #self.X_train = self.X_train[:300]
        #self.X_validation= self.X_validation[:50]
        #self.Y_validation= self.Y_validation[:50]
        self.tf_ps = self.tf_ps.split(",")
        self.tf_workers = self.tf_workers.split(",")
        
        ## Distributed Setup
        CNNClassifier.cluster = tf.train.ClusterSpec({
                                        "ps": self.tf_ps,
                                        "worker": self.tf_workers
                                        })
        ## Define the machine rol = input flags
        ## start a server for a specific task
        CNNClassifier.server = tf.train.Server(self.cluster,
                                                job_name = self.job_name,
                                                task_index = self.task_index)


    #######################################################################
    def create_done_queue(self, i):
        '''Queue used to signal death for i'th ps shard. Intended to have
        all workers enqueue an item onto it to signal doneness.'''
        print("******* def create_done_queue: /job:ps/task: -> {} ".format(i))
        with tf.device("/job:ps/task:%d" % (i)):
            return tf.FIFOQueue(self.num_workers, tf.int32, shared_name="done_queue"+str(i))

    def create_done_queues(self):
        print("****** def -> create_done_queues")
        return [self.create_done_queue(i) for i in range(self.num_ps)]

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

    def getstep(self,X_train):
        return (len(X_train)//(self.batch_size))            

            
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
        #with tf.variable_scope('ConvNet', reuse=tf.AUTO_REUSE): 
        with tf.variable_scope('ConvNet'):
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
            #x=tf.layers.conv1d(x,64*k,16,padding='same',kernel_initializer=tf.glorot_uniform_initializer())
            for i in range(1,3,1):
                if i%2 ==0:
                    k+=1
                x=tf.layers.conv1d(x,64*k,16,padding='same',kernel_initializer=tf.glorot_uniform_initializer())
                x=self.r_block(x,k,is_training)
                x=self.subsampling_r_block(x,k,is_training)

            x = tf.layers.batch_normalization(x)
            x = tf.nn.relu(x)
            x = tf.contrib.layers.flatten(x)
            out = tf.layers.dense(x, 4,kernel_initializer=tf.glorot_uniform_initializer())
        return out

    ## Compute fpr, tpr, fnr and tnr:
    def compute_metrics(self,cm ,y_true, y_pred):

        FalsePositive = []
        FalseNegative = []
        TrueNegative = []

        # Compute True positive
        TruePositive = np.diag(cm)

        # Compute False positive
        for i in range(len(cm)):
            FalsePositive.append(int(sum(cm[:,i]) - cm[i,i]))

        # Compute False negative
        for i in range(len(cm)):
            FalseNegative.append(int(sum(cm[i,:]) - cm[i,i]))

        # Compute True negative
        for i in range(len(cm)):
            temp = np.delete(cm, i, 0)
            temp = np.delete(temp, i, 1)
            TrueNegative.append(int(sum(sum(temp))))

        return TruePositive,  FalsePositive , FalseNegative


    ########################## Model's training #################d##############"
    def launch_training(self):
        """
        Define role for distributed processing
        """
        if self.job_name ==  "ps":
            sess = tf.Session(self.server.target)
            queue =  self.create_done_queue(self.task_index)

            # Wait intil all workers are done
            for i in range(self.num_workers):
                sess.run(queue.dequeue())
                print("ps %d recieved done %d" %(self.task_index,i))
            print("ps %d: quitting" %(self.task_index))

        elif self.job_name == "worker" :

            with tf.Graph().as_default() as cnn_graph:

                with tf.device(tf.train.replica_device_setter(
                            worker_device="/job:worker/task:%d" % self.task_index,
                            cluster=self.cluster)):

                    start_time_loaddata = time.time()
                    lr = self.lr_op

                    seed = 1
                    np.random.seed(seed)
                    tf.set_random_seed(seed)

                    with tf.variable_scope("global_step", reuse=True):
                        global_step = tf.Variable(self.batch_size)

                    x = tf.placeholder(tf.float32, shape = [None, 1300,1], name = "x")
                    y = tf.placeholder(tf.int32, shape = [None, 4], name = "labels")
                    is_training = tf.placeholder(tf.bool, shape=())

                    print("lr {}".format(lr))
                    y_pred = self.conv_net(x,is_training)

                    y_pred_soft = tf.nn.softmax(y_pred)
                    pred_max = tf.argmax(y_pred_soft,1)
                    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = y_pred, labels = y))
                    grads = tf.train.AdamOptimizer(lr).minimize(loss,
                                                        global_step=global_step)
                    accuracy = tf.reduce_mean(tf.cast(tf.equal(pred_max, tf.argmax(y, 1)), tf.float32))

                    init_op = tf.group(tf.global_variables_initializer(),
                                    tf.local_variables_initializer())

                    enq_ops = []
                    for q in self.create_done_queues():
                        qop = q.enqueue(1)
                        enq_ops.append(qop)

                train_next_batch_gen = self.generator( self.X_train, self.Y_train)
                val_next_batch_gen = self.generator( self.X_validation, self.Y_validation)
                test_next_batch_gen = self.generator( self.X_test, self.Y_test)
                train_step = self.getstep(self.X_train)
                val_step = self.getstep(self.X_validation)
                test_step = self.getstep(self.X_test)

                #########################################################
                ## Create a distributed session whit training supervisor
                saver = tf.train.Saver()
                sv = tf.train.Supervisor(is_chief=(self.task_index == 0),
                                            graph=cnn_graph,saver=saver,
                                            checkpoint_basename=str(self.testbed_dir+"/"+self.argv_testbed+"/"+"model-"+self.job_name+"-"+str(self.task_index)+".ckpt"),
                                            global_step=global_step,
                                            init_op=init_op)
                start_time = time.time()
                best_val_loss=9999
                #config = tf.ConfigProto(allow_soft_placement=True)
                #config.gpu_options.allow_growth = True
                #saver = tf.train.Saver()
                with sv.managed_session(self.server.target) as sess:
                    epoch = 0
                    listtrain_losses = []
                    listtrain_auc = []
                    listvalid_losses = []
                    listvalid_auc = []
                    time_per_epoch = []
                    last_improvement=0
                    
                    end_time = (time.time()-start_time_loaddata)
                    print("Loading data took: %s secondes " % end_time)

                    ################## Start training #########################
                    print("+++++++++++++ Start training ++++++++++++++++++++++")


                    while not sv.should_stop() and (epoch < self.max_epochs):
                        start_time_epoch = time.time()
                        print("\n --------- Epoch {} --------".format(epoch+1))

                        train_loss = 0.0
                        train_auc = 0.0

                        for i in range(train_step):
                            x_training, y_training = next(train_next_batch_gen)
                            gradients, tloss , tacc = sess.run([grads, loss,accuracy], feed_dict ={x: x_training, y: y_training, is_training :True})
                            train_loss += tloss/train_step
                            train_auc +=tacc/train_step
                        listtrain_losses.append(train_loss)
                        listtrain_auc.append(train_auc)

                        ############ Validation ########
                        valid_loss = 0.0
                        valid_auc = 0.0


                        for i in range(val_step):

                           x_valid, y_valid = next(val_next_batch_gen)
                           #Validation loss
                           vloss,vauc = sess.run([loss,accuracy], feed_dict = {x: x_valid, y: y_valid, is_training : False})

                           valid_loss += vloss/val_step
                           valid_auc += vauc/val_step

                        end_time_epoch = (time.time()-start_time_epoch)
                        time_per_epoch.append(end_time_epoch)
                        listvalid_losses.append(valid_loss)
                        listvalid_auc.append(valid_auc)

                        if valid_loss < best_val_loss:
                          # Update the best-known validation accuracy.
                           best_val_loss = valid_loss
                           saver.save(sess, str(self.testbed_dir+"/"+self.argv_testbed+"/"+"model-"+self.job_name+"-"+str(self.task_index)+".ckpt"))
                          # Set the iteration for the last improvement to current.
                           last_improvement = epoch
                           converage_time=time.time()-end_time
                           
                        epoch = epoch + 1
                        print('training loss: {} --  training acc: {}  validation loss: {} --  validation acc: {}  time :{}'.format(train_loss,train_auc, valid_loss,valid_auc,end_time_epoch))

                    track_training = np.vstack((np.arange(1,len(listtrain_losses)+1), listtrain_losses, listvalid_losses,listtrain_auc,listvalid_auc, time_per_epoch))
                    track_training = np.transpose(track_training)
                    df_track = pd.DataFrame(track_training, columns = ["epoch", "training loss", "validation loss" , "training acc", "validation acc" ,  "execution time per epoch"])

                    df_track.to_csv(str(self.testbed_dir+"/"+self.argv_testbed+"/"
                                    +self.argv_testbed+"-"+self.job_name+"-"+str(self.task_index)
                                    +"-training_track.csv"), index = False)

                    tri_end_time = (time.time()-start_time)
                    print("Execution time : %s secondes " % tri_end_time)
#                    print("End training with {} training records and {} vaidation records ".format(x_trainsamples, x_validsamples))
                    saver.restore(sess, str(self.testbed_dir+"/"+self.argv_testbed+"/"+"model-"+self.job_name+"-"+str(self.task_index)+".ckpt"))

                    print("++ Testing started......")
                    representation = str(x_training.shape[1])
                    features_name = "Dim_"+representation

                    for i in range(test_step):
                        x_test, y_test = next(test_next_batch_gen)

                        with open(str(self.testbed_dir+"/"+self.argv_testbed+"/"+self.argv_testbed+"-"+self.job_name+"-"+str(self.task_index)
                                        +"-y_test_"+features_name+".txt") , 'a') as file_ytest:
                            np.savetxt(file_ytest, y_test, fmt='%s',delimiter=',',newline='\n' )
                        file_ytest.close()

                        y_predmax = sess.run(pred_max, feed_dict = {x: x_test, is_training:False})
                        with open(str(self.testbed_dir+"/"+self.argv_testbed+"/"
                                        +self.argv_testbed+"-"+self.job_name+"-"+str(self.task_index)
                                        +"-probas_test_"+features_name+".txt") , 'a') as file_predtest:
                            np.savetxt(file_predtest, y_predmax, fmt='%s',delimiter=',',newline='\n' )
                        file_predtest.close()

                    print("++++++++++++++++++++ compute metrics ++++++++++++++++++++")
                    
                    y_test = np.loadtxt(str(self.testbed_dir+"/"+self.argv_testbed+"/"
                                    +self.argv_testbed+"-"+self.job_name+"-"+str(self.task_index)
                                    +"-y_test_"+features_name+".txt"),delimiter=',')
                    
                
                    y_pred = np.loadtxt(str(self.testbed_dir+"/"+self.argv_testbed+"/"
                                    +self.argv_testbed+"-"+self.job_name+"-"+str(self.task_index)
                                    +"-probas_test_"+features_name+".txt"),delimiter=',')
                   


                    #########################################################

                    y_test =  np.argmax(y_test, axis = 1)

                    labels, counts = np.unique(y_test, return_counts = True)

                    ## True positive, False positive, False negative computing......

                    confmatrix = confusion_matrix(y_test , y_pred, labels)
                    tp, fp , fn = self.compute_metrics(confmatrix,y_test, y_pred)

                    ## Compute metrics per class
                    precision, recall , F1_score , support = precision_recall_fscore_support(y_test, y_pred, average = None)

                    f1_weighted = f1_score(y_test, y_pred, average = "weighted")
                    f1_micro = f1_score(y_test, y_pred, average = "micro")

                    a = np.where(support !=0)
                    a = a[0]
                   

                    metrics_values = np.vstack((labels,tp, fp, fn,precision[a], recall[a], F1_score[a], support[a]))
                    metrics_values = np.transpose(metrics_values)
                    df = pd.DataFrame(metrics_values, columns = ["labels", "tp", "fp", "fn","precision" , "recall", "F1_score", "Occurence"])
                    print(df)

                    opt = "Optimizer: Adam and learning rate {}".format(lr)
                    weight_f1 = "Weighted average of F1 score : {}".format(f1_weighted)
                    micro_f1 = "Micro average of F1 score : {}".format(f1_micro)
                    end1 = "preparing time : {} secondes".format(end_time)
                    end2 = "Execution time : {} secondes".format(tri_end_time)
                    end3 = "test time : {} secondes".format(time.time()-tri_end_time)
                    updation = "minimum val loss epoch: {} ".format(last_improvement)
                    conv_time = "converage_time: {} ".format(converage_time)

                    sandbx = "\nSandbox used: {}".format(self.sandbox_fn)

                    print(weight_f1)
                    print(micro_f1)
                    print(end3)

                    infos = np.array([opt, weight_f1 , micro_f1 , end1,end2,end3,updation, conv_time,sandbx])

                    df.to_csv(str(self.testbed_dir+"/"+self.argv_testbed+"/"
                                    +self.argv_testbed+"-"+self.job_name+"-"+str(self.task_index)
                                    +"-metriques_"+features_name+".csv"), index = False)

                    with open(str(self.testbed_dir+"/"+self.argv_testbed+"/"
                                    +self.argv_testbed+"-"+self.job_name+"-"+str(self.task_index)
                                    +"-metriques_"+features_name+".csv") , 'a') as file_ytest:
                            np.savetxt(file_ytest, infos, fmt='%s',delimiter=',',newline='\n' )
                    file_ytest.close()

                    print("++ testbed_dir: {}{}".format(self.testbed_dir, self.argv_testbed))

                    ## signal to ps shards that we are done
                    for op in enq_ops:
                        sess.run(op)
                    print('-- Done! --')
                sv.stop()
            sess.close()


if __name__ == '__main__' :

    features_name = "ECG"
    sandbox = "../input/"

    tf_ps = "134.59.132.111:2222"
    tf_workers = "134.59.132.22:2222,134.59.132.23:2222"
    #tf_workers = "134.59.132.22:2222,134.59.132.23:2222,134.59.132.26:2222,134.59.132.21:2222"
    num_ps = 1
    num_workers = 2
    job_name = sys.argv[1]
    task_index = int(sys.argv[2])
    testbed_arg=sys.argv[4]
    testbed_dir=sys.argv[3]
    lr_op = 0.0001
    max_epochs=15
    batch_size=8


    y_TRAIN=str("train/ytrain-"+str(task_index+1)+".npy")
    y_VALID=str("val/yval-"+str(task_index+1)+".npy")
    y_TEST=str("test/ytest-"+str(task_index+1)+".npy")

    x_TRAIN=str("train/xtrain-"+str(task_index+1)+".npy")
    x_VALID=str("val/xval-"+str(task_index+1)+".npy")
    x_TEST=str("test/xtest-"+str(task_index+1)+".npy")

    cnn = CNNClassifier( lr_op, sandbox, testbed_arg, y_TRAIN, y_VALID, y_TEST , x_TRAIN, x_VALID, x_TEST,
                         max_epochs, batch_size,tf_ps, tf_workers, num_ps, num_workers, job_name, task_index,testbed_dir)
    cnn.launch_training()
