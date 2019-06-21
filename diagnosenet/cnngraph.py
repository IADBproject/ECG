import tensorflow as tf

from losses import Loss
from optimizers import Optimizer

from monitor import Metrics

class CNNGraph:

    def __init__(self, input_size_1: int,input_size_2: int, output_size: int,
                        loss: Loss,
                        optimizer: Optimizer,
                        dropout: float = 0.2) -> None:

        ## A neural network architecture:
        self.input_size_1 = input_size_1
        self.input_size_2 = input_size_2
        self.output_size = output_size
        self.loss = loss
        self.optimizer = optimizer
        self.dropout = dropout

        ## Graph object trainable parameters:
        self.cnn_graph = tf.Graph()
        self.X: tf.placeholder
        self.Y: tf.placeholder
        self.is_training: tf.placeholder
        self.projection: tf.Tensor
        self.cnn_loss: tf.Tensor
        self.cnn_grad_op: tf.Tensor
        self.reuse =False

            
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
            x = tf.contrib.layers.flatten(x)
            out = tf.layers.dense(x, 4,kernel_initializer=tf.glorot_uniform_initializer())
        return out



    def desktop_graph(self) -> tf.Tensor:
        with tf.Graph().as_default() as self.cnn_graph:
            self.X = tf.placeholder(tf.float32, shape=(None, self.input_size_1,self.input_size_2), name="Inputs")
            self.Y = tf.placeholder(tf.float32, shape=(None, self.output_size), name="Output")
            self.is_training = tf.placeholder(tf.bool, shape=())

            #for the training part
            self.projection = self.stacked(self.X,  self.is_training)
            self.cnn_loss = self.loss.desktop_loss(self, self.projection, self.Y)
            self.cnn_grad_op = self.optimizer.desktop_Grad(self.cnn_loss)

            self.max_projection = tf.argmax(self.projection, 1)
            self.projection_1hot = tf.one_hot(self.max_projection, depth = int(self.output_size))

