"""
A session executor...
"""
from typing import Sequence, NamedTuple

import tensorflow as tf
import numpy as np

import os, time, json
import multiprocessing as mp

from datamanager import Dataset, Batching
from io_functions import IO_Functions
from monitor import enerGyPU, Metrics
from sklearn.metrics import f1_score
import subprocess as sp
import logging
logger = logging.getLogger('_DiagnoseNET_')
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s: - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S')

fh = logging.FileHandler('log.txt')
fh.setLevel(logging.DEBUG)
fh.setFormatter(formatter)

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
ch.setFormatter(formatter)
PS_OPS = ['Variable', 'VariableV2', 'AutoReloadVariable']
logger.addHandler(ch)
logger.addHandler(fh)
Batch = NamedTuple("Batch", [("inputs", np.ndarray), ("targets", np.ndarray)])
BatchPath = NamedTuple("BatchPath", [("input_files", list), ("target_files", list)])

class DesktopExecution:
    """
    Implements the back-propagation algorithm ...
    Args:
        model: Is a graph object of the neural network architecture selected
    Returns:
    """

    def __init__(self, model, monitor: enerGyPU = None, datamanager: Dataset = None,
                    max_epochs: int = 10, min_loss: float = 0.2,batch_size=4) -> None:
        self.model = model
        self.data = datamanager
        self.max_epochs = max_epochs
        self.min_loss = min_loss
        self.monitor = monitor
        self.batch_size=batch_size
        self.best_validation_loss=9999

        ## Time logs
        self.time_latency: time()
        self.time_dataset: time()
        self.time_training: time()
        self.time_testing: time()
        self.time_metrics: time()


        ## Testbed and Metrics
        self.processing_mode: str
        self.training_track: list = []


    def set_monitor_recording(self) -> None:
        latency_start = time.time()
        if self.monitor == None:
            self.monitor = enerGyPU(testbed_path="../enerGyPU/testbed",
                                write_metrics=True,
                                power_recording=True,
                                platform_recording=True)

        self.monitor.generate_testbed(self.monitor.testbed_path,
                                        self.model, self.data,
                                        self.__class__.__name__,
                                        self.max_epochs)

        if self.monitor.power_recording == True: self.monitor.start_power_recording()

        if self.monitor.platform_recording == True: self.monitor.start_platform_recording(os.getpid())


        ## Time recording
        self.time_latency = time.time()-latency_start

    def set_dataset_memory(self, inputs: np.ndarray, targets: np.ndarray) -> Batch:
        dataset_start = time.time()
        self.data = Batching( batch_size=self.batch_size,valid_size=0.1, test_size=0.1)
        self.data.set_data_file(inputs, targets)
        self.data.dataset_name="ecg"
        train, valid, test = self.data.memory_batching()
        self.time_dataset = time.time()-dataset_start
        return train, valid, test

    def training_memory(self, inputs: np.ndarray, targets: np.ndarray) -> tf.Tensor:

        self.processing_mode = "memory_batching"

        train, valid, test = self.set_dataset_memory(inputs, targets)
        self.set_monitor_recording()

        ### Training Start
        training_start = time.time()
        ## Generates a Desktop Graph
        self.model.desktop_graph()


        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True


        with tf.Session(config=config, graph=self.model.cnn_graph) as sess:
            init = tf.group(tf.global_variables_initializer(),
                                tf.local_variables_initializer())
            sess.run(init)
            saver = tf.train.Saver()
            epoch: int = 0
            epoch_convergence: bin = 0
            while (epoch_convergence == 0):
                train_loss,train_acc,valid_loss,valid_acc=0,0,0,0
                epoch_start = time.time()
                self.model.reuse=False
                for i in range(len(train.inputs)):
                    train_losst, _ = sess.run([self.model.cnn_loss, self.model.cnn_grad_op],
                                    feed_dict={self.model.X: train.inputs[i],
                                                self.model.Y: train.targets[i],
                                                self.model.is_training: True})
                    train_pred = sess.run(self.model.projection_1hot,
                                    feed_dict={self.model.X: train.inputs[i],
                                                self.model.is_training: True})
                    ## F1_score from Skit-learn metrics
                    train_acct = f1_score(y_true=train.targets[i].astype(np.float),
                                            y_pred=train_pred, average='micro')
                    train_loss+=train_losst/len(train.inputs)
                    train_acc+=train_acct/len(train.inputs)

                self.model.reuse=True

                for i in range(len(valid.inputs)):
                    valid_losst = sess.run(self.model.cnn_loss,
                                    feed_dict={self.model.X: valid.inputs[i],
                                                self.model.Y: valid.targets[i],
                                                self.model.is_training: False})
                    valid_pred = sess.run(self.model.projection_1hot,
                                    feed_dict={self.model.X: valid.inputs[i],
                                                self.model.is_training: False})
                    ## F1_score from Skit-learn metrics
                    valid_acct = f1_score(y_true=valid.targets[i].astype(np.float),
                                            y_pred=valid_pred, average='micro')
                    valid_loss+=valid_losst/len(valid.inputs)
                    valid_acc+=valid_acct/len(valid.inputs)

                                
                if valid_loss < self.best_validation_loss:
                    self.best_validation_loss = valid_loss
                    saver.save(sess,  "./model.ckpt")
                    logger.info("Update!")
                epoch_elapsed = (time.time() - epoch_start)
                logger.info("Epoch {} | Train loss: {} |  Valid loss: {} | Train Acc: {} | Valid Acc: {} | Epoch_Time: {}".format(epoch,
                                                        train_loss, valid_loss, train_acc, valid_acc, np.round(epoch_elapsed, decimals=4)))
                self.training_track.append((epoch,train_loss, valid_loss, train_acc, valid_acc, np.round(epoch_elapsed, decimals=4)))
                epoch = epoch + 1

                ## While Convergence conditional
                if valid_loss <= self.min_loss or epoch == self.max_epochs:
                    epoch_convergence = 1
                    self.max_epochs=epoch
                    self.min_loss=valid_loss
                else:
                    epoch_convergence = 0
                ### end While loop
            self.time_training = time.time()-training_start

            ### Testing Starting
            testing_start = time.time()
            saver.restore(sess, "./model.ckpt")

            if len(test.inputs) != 0:
                test_pred_probas: list = []
                test_pred_1hot: list = []
                test_true_1hot: list = []

                for i in range(len(test.inputs)):
                    tt_pred_probas,tt_pred_1hot = sess.run([self.model.projection,self.model.projection_1hot],
                                                feed_dict={self.model.X: test.inputs[i],
                                                            self.model.is_training: False})

                    test_pred_probas.append(tt_pred_probas)
                    test_pred_1hot.append(tt_pred_1hot)
                    test_true_1hot.append(test.targets[i].astype(np.float))

                self.test_pred_probas = np.vstack(test_pred_probas)
                self.test_pred_1hot = np.vstack(test_pred_1hot)
                self.test_true_1hot = np.vstack(test_true_1hot)

                ## Compute the F1 Score
                self.test_f1_weighted = f1_score(self.test_true_1hot,
                                                    self.test_pred_1hot, average = "weighted")
                self.test_f1_micro = f1_score(self.test_true_1hot,
                                                    self.test_pred_1hot, average = "micro")
                logger.info("-- Test Results --")
                logger.info("F1-Score Weighted: {}".format(self.test_f1_weighted))
                logger.info("F1-Score Micro: {}".format(self.test_f1_micro))

                ## Compute_metrics by each label
                self.metrics_values = Metrics().compute_metrics(y_pred=self.test_pred_1hot,
                                                            y_true=self.test_true_1hot)
                self.time_testing = time.time()-testing_start

                ## Write metrics on testbet directory = self.monitor.testbed_exp
                if self.monitor.write_metrics == True: self.write_metrics()
            


    def write_metrics(self, testbed_path: str = 'testbed') -> None:
        """
        Uses Testbed to isolate the training metrics by experiment directory
        """
        metrics_start = time.time()

        ### Add elements to json experiment Description architecture
        eda_json = self.monitor.read_eda_json(self.monitor.testbed_exp, self.monitor.exp_id)

        ## Add values to platform_parameters
        eda_json['model_hyperparameters']['max_epochs'] = self.max_epochs

        ## Add dataset shape as number of records (inputs, targets)
        eda_json['dataset_config']['train_records'] = str(self.data.train_shape)
        eda_json['dataset_config']['valid_records'] = str(self.data.valid_shape)
        eda_json['dataset_config']['test_records'] = str(self.data.test_shape)

        ## Add values to platform_parameters
        eda_json['platform_parameters']['processing_mode'] = self.processing_mode


        ## Add values to results
        eda_json['results']['f1_score_weigted'] = self.test_f1_weighted
        eda_json['results']['f1_score_micro'] = self.test_f1_micro
        eda_json['results']['loss_validation'] = str(self.min_loss)
        eda_json['results']['time_latency'] = self.time_latency
        eda_json['results']['time_dataset'] = self.time_dataset
        eda_json['results']['time_training'] = self.time_training
        eda_json['results']['time_testing'] = self.time_testing

        ## End time metrics
        self.time_metrics = time.time()-metrics_start
        eda_json['results']['time_metrics'] = self.time_metrics

        ## Serialize the eda json and rewrite the file
        eda_json = json.dumps(eda_json, separators=(',', ': '), indent=2)
        file_path = str(self.monitor.testbed_exp+"/"+self.monitor.exp_id+"-exp_description.json")
        IO_Functions()._write_file(eda_json, file_path)


        ## End computational recording
        self.monitor.end_platform_recording()
 
        ## End power recording
        self.monitor.end_power_recording()

        sp.Popen(["mv","log.txt",str(self.monitor.testbed_exp)])
        sp.Popen(["mv","F1_data.txt",str(self.monitor.testbed_exp)])
        logger.info("Tesbed directory: {}".format(self.monitor.testbed_exp))



class MultiGPU:
    """
    Implements the back-propagation algorithm ...
    Args:
        model: Is a graph object of the neural network architecture selected
    Returns:
    """

    def __init__(self, model, monitor: enerGyPU = None,
                            datamanager: Dataset = None,
                            gpus: int = 2, max_epochs: int = 10,batch_size=4) -> None:
        self.monitor = monitor
        self.model = model
        self.data = datamanager
        self.num_gpus = gpus
        self.max_epochs = max_epochs
        self.reuse = False
        self.batch_size=batch_size
        self.best_validation_loss=9999

        ## Testbed and Metrics
        testbed_path: str = 'testbed'
        self.training_track: list = []

    def set_monitor_recording(self) -> None:
        """
        Power and performance monitoring launcher for workload characterization
        """
        latency_start = time.time()
        if self.monitor == None:
            self.monitor = enerGyPU(testbed_path="../enerGyPU/testbed",
                                write_metrics=True,
                                power_recording=True,
                                platform_recording=False)

        ## Generate ID-experiment and their testebed directory
        self.monitor.generate_testbed(self.monitor.testbed_path,
                                        self.model, self.data,
                                        self.__class__.__name__,
                                        self.max_epochs)

        ## Start power recording
        if self.monitor.power_recording == True: self.monitor.start_power_recording()

        ## Start platform recording
        if self.monitor.platform_recording == True: self.monitor.start_platform_recording(os.getpid())

        ## Get GPU availeble and set for processing
        self.idgpu = self.monitor._get_available_GPU()


        #os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
        #os.environ["CUDA_VISIBLE_DEVICES"]="5,6"
        #self.num_gpus = len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))

        ## Time recording
        self.time_latency = time.time()-latency_start


    def set_dataset_memory(self, inputs: np.ndarray, targets: np.ndarray) -> Batch:
        """
        Uses datamanager classes for splitting, batching the dataset and target selection
        """
        dataset_start = time.time()
        batch_size= self.batch_size*self.num_gpus
        self.data = Batching(batch_size =batch_size,valid_size=0.1, test_size=0.1)
        self.data.set_data_file(inputs, targets)
        self.data.dataset_name="ecg"
        train, valid, test = self.data.memory_batching()

        self.time_dataset = time.time()-dataset_start
        return train, valid, test

###############################################################################"
###############################################################################"


    def multiGPU_loss(self, y_pred: tf.Tensor, y_true: tf.Tensor) -> tf.Tensor:
        """
        """
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred)
        cross_entropy_reduce = tf.reduce_mean(cross_entropy)

        return cross_entropy_reduce

    def average_gradients(self, tower_grads):
        """
        Merge the grads computations done by each GPU tower
        """
        ### First Print
        #print("\n \n")
        # print("tower_grads: {}".format(tower_grads))
        average_grads = []
        for grad_and_vars in zip(*tower_grads):
            ## Second print
            # print("grad_and_vars: {}".format(grad_and_vars))
            grads = []
            for g, _ in grad_and_vars:
                ## Third Print
                #print("+ Grad by Tower: {}".format(g))
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

            # Keep in mind that the Variables are redundant because they are shared
            # across towers. So .. we will just return the first tower's pointer to
            # the Variable.
            v = grad_and_vars[0][1]
            grad_and_var = (grad, v)
            average_grads.append(grad_and_var)

        return average_grads


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

    def stacked(self,x,is_training):
        # Define a scope for reusing the variables
        with tf.variable_scope('ConvNet', reuse=tf.AUTO_REUSE): 

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
            x = tf.contrib.layers.flatten(x)
            out = tf.layers.dense(x, 4,kernel_initializer=tf.glorot_uniform_initializer())
        return out

    def assign_to_device(device, ps_device='/cpu:0'):
        def _assign(op):
            print(device)
            print(op)
            node_def = op if isinstance(op, tf.NodeDef) else op.node_def
            print(node_def.op )
            if node_def.op in PS_OPS:
                return "/" + ps_device
            else:
                return device

    def training_multigpu(self, inputs: np.ndarray, targets: np.ndarray) -> tf.Tensor:

        self.processing_mode = "multiGPU"
        train, valid, test = self.set_dataset_memory(inputs, targets)
        self.set_monitor_recording()
        self.gpu_batch_size=int((self.data.batch_size/self.num_gpus))

        # with tf.Graph().as_default() as self.cnn_graph:
        with tf.device('/cpu:0'):

                self.total_projection = []
                self.total_losses = []
                self.total_grads = []

                X = tf.placeholder(tf.float32, shape=(None, 1300,1), name="Inputs")
                Y = tf.placeholder(tf.float32, shape=(None, 4), name="Targets")
                self.is_training = tf.placeholder(tf.bool, shape=())

                self.adam_op = tf.train.AdamOptimizer(learning_rate=0.001)
                

                for igpu in range(self.num_gpus):
                    with tf.device('/gpu:{}'.format(igpu)):
                            # tf.variable_scope.reuse_variables()
                            # Split data between GPUs
                            _X = X[(igpu * self.gpu_batch_size):
                                      (igpu * self.gpu_batch_size) + (self.gpu_batch_size)]
                            _Y = Y[(igpu * self.gpu_batch_size):
                                      (igpu * self.gpu_batch_size) + (self.gpu_batch_size)]

                            ## Projection by Tower Model operations

                            self.reuse = False
                            self.projection = self.stacked(_X, self.is_training)
                            self.total_projection.append(self.projection)


                            # ## Loss by Tower Model operations
                            self.loss = self.multiGPU_loss(self.projection, _Y)
                            self.total_losses.append(self.loss)

                            ## Grads by Tower Model operations
                            self.grads_computation = self.adam_op.compute_gradients(self.loss)
                            # reuse_vars = True
                            self.total_grads.append(self.grads_computation)


                            print("{}".format("+"*20))
                            print("+ GPU: {}".format(igpu))
                            print("+ Split_X: {}, {}".format((igpu * self.gpu_batch_size),
                                (igpu * self.gpu_batch_size) + (self.gpu_batch_size)))
                            print("+ Tower_Projection: {}".format(self.projection.name))
                            print("{}".format("+"*20))


                with tf.device('/cpu:0'):
                    self.output1 = tf.concat(self.total_projection, axis=0)
                    self.output2 = self.total_losses
                    self.output3 = self.average_gradients(self.total_grads)
                    self.train_op = tf.group(self.adam_op.apply_gradients(self.output3))
                    self.output4 = tf.one_hot(tf.argmax(self.output1, 1), depth = 4)


        #################################################################""
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        #config.intra_op_parallelism_threads = 16

        # config.gpu_options.per_process_gpu_memory_fraction = 0.4

        with tf.Session(config=config) as sess:
            init = tf.group(tf.global_variables_initializer(),
                            tf.local_variables_initializer())
            sess.run(init)
            saver = tf.train.Saver()

            epoch: int = 0
            training_start = time.time()
            while epoch < self.max_epochs:
                epoch_start = time.time()
                train_loss,train_acc,valid_loss,valid_acc=0,0,0,0
                for i in range(len(train.inputs)):

                    ### Temporaly conditional
                    if train.inputs[i].shape[0] >= (self.data.batch_size):
                        train_grads = sess.run(self.train_op,
                                    feed_dict={X: train.inputs[i],
                                            Y: train.targets[i],
                                            self.is_training: True})

                        train_pred = sess.run(self.output4,
                                    feed_dict={X: train.inputs[i],
                                            self.is_training: True})

                        train_losst = sess.run(self.output2,
                                    feed_dict={X: train.inputs[i],
                                            Y: train.targets[i],
                                            self.is_training: True})

                        train_acct = f1_score(y_true=train.targets[i].astype(np.float),
                                            y_pred=train_pred, average='micro')
                        #print("train_loss: {}".format(train_loss))
                        train_loss+=(np.mean(train_losst))/len(train.inputs)
                        train_acc+=train_acct/len(train.inputs)

                        #print("train_grads: {}".format(train_grads))
                for i in range(len(valid.inputs)):
                    if valid.inputs[i].shape[0] >= (self.data.batch_size):

                        val_pred = sess.run(self.output4,
                                    feed_dict={X: valid.inputs[i],
                                            self.is_training: False})


                        valid_losst = sess.run(self.output2,
                                    feed_dict={X: valid.inputs[i],
                                            Y: valid.targets[i],
                                            self.is_training: False})

                        valid_acct = f1_score(y_true=valid.targets[i].astype(np.float),
                                            y_pred=val_pred, average='micro')
                        valid_loss+=(np.mean(valid_losst))/len(valid.inputs)
                        valid_acc+=valid_acct/len(valid.inputs)
                if valid_loss < self.best_validation_loss:
                    self.best_validation_loss = valid_loss
                    saver.save(sess,  "./model.ckpt")
                    logger.info(" Update!")
                epoch_elapsed = (time.time() - epoch_start)
                logger.info("Epoch {} | Train loss: {} |  Valid loss: {} | Train Acc: {} | Valid Acc: {} | Epoch_Time: {}".format(epoch,
                                                        train_loss, valid_loss, train_acc, valid_acc, np.round(epoch_elapsed, decimals=4)))
                
                epoch = epoch + 1
            self.time_training = time.time()-training_start

            ### Testing Starting
            testing_start = time.time()
            saver.restore(sess, "./model.ckpt")

            if len(test.inputs) != 0:
                test_pred_probas: list = []
                test_pred_1hot: list = []
                test_true_1hot: list = []

                for i in range(len(test.inputs)):
                    if test.inputs[i].shape[0] >= (self.data.batch_size):
                        tt_pred = sess.run(self.output4,
                                                    feed_dict={X: test.inputs[i],
                                                                self.is_training: False})
                        tt_losst = sess.run(self.output2,
                                                    feed_dict={X: test.inputs[i],
                                                                Y: test.targets[i],
                                                                self.is_training: False})

                        test_pred_1hot.append(tt_pred)
                        test_true_1hot.append(test.targets[i].astype(np.float))


                self.test_pred_1hot = np.vstack(test_pred_1hot)
                self.test_true_1hot = np.vstack(test_true_1hot)

                ## Compute the F1 Score
                self.test_f1_weighted = f1_score(self.test_true_1hot,
                                                    self.test_pred_1hot, average = "weighted")
                self.test_f1_micro = f1_score(self.test_true_1hot,
                                                    self.test_pred_1hot, average = "micro")
                logger.info("-- Test Results --")
                logger.info("F1-Score Weighted: {}".format(self.test_f1_weighted))
                logger.info("F1-Score Micro: {}".format(self.test_f1_micro))
                

                ## Compute_metrics by each label
                self.metrics_values = Metrics().compute_metrics(y_pred=self.test_pred_1hot,
                                                            y_true=self.test_true_1hot)
                self.time_testing = time.time()-testing_start

                ## Write metrics on testbet directory = self.monitor.testbed_exp
                if self.monitor.write_metrics == True: self.write_metrics()

            ## Print the sandbox
            sp.Popen(["mv","log.txt",str(self.monitor.testbed_exp)])
            sp.Popen(["mv","F1_data.txt",str(self.monitor.testbed_exp)])
            logger.info("Tesbed directory: {}".format(self.monitor.testbed_exp))


    def write_metrics(self, testbed_path: str = 'testbed') -> None:
        """
        Uses Testbed to isolate the training metrics by experiment directory
        """
        metrics_start = time.time()

        ### Add elements to json experiment Description architecture
        eda_json = self.monitor.read_eda_json(self.monitor.testbed_exp, self.monitor.exp_id)

        ## Add values to platform_parameters
        eda_json['model_hyperparameters']['max_epochs'] = self.max_epochs

        ## Add dataset shape as number of records (inputs, targets)
        eda_json['dataset_config']['train_records'] = str(self.data.train_shape)
        eda_json['dataset_config']['valid_records'] = str(self.data.valid_shape)
        eda_json['dataset_config']['test_records'] = str(self.data.test_shape)

        ## Add values to platform_parameters
        eda_json['platform_parameters']['processing_mode'] = self.processing_mode
        eda_json['platform_parameters']['gpu_id'] = self.idgpu[0]

        ## Add values to results
        eda_json['results']['f1_score_weigted'] = self.test_f1_weighted
        eda_json['results']['f1_score_micro'] = self.test_f1_micro
        eda_json['results']['time_latency'] = self.time_latency
        eda_json['results']['time_dataset'] = self.time_dataset
        eda_json['results']['time_training'] = self.time_training
        eda_json['results']['time_testing'] = self.time_testing

        ## End time metrics
        self.time_metrics = time.time()-metrics_start
        eda_json['results']['time_metrics'] = self.time_metrics

        ## Serialize the eda json and rewrite the file
        eda_json = json.dumps(eda_json, separators=(',', ': '), indent=2)
        file_path = str(self.monitor.testbed_exp+"/"+self.monitor.exp_id+"-exp_description.json")
        IO_Functions()._write_file(eda_json, file_path)


        ## End computational recording
        self.monitor.end_platform_recording()
 
        ## End power recording
        self.monitor.end_power_recording()

        logger.info("Tesbed directory: {}".format(self.monitor.testbed_exp))





