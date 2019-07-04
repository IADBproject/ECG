from CNNgraph import *
from utils import *
from mpi4py import MPI
import time, math
import numpy as np
import sys
from collections import Iterable



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
        modeling.create_time=start
        modeling.dataset_time=time.time()-start
        print('Dataset preparing --- Time:',time.time()-start)
        return modeling,train_next_batch_gen,val_next_batch_gen,\
        test_next_batch_gen,train_step,val_step,test_step
    else:
        data=None
        modeling=MasterModeling(data)
        modeling.create_time=start
        modeling.dataset_time=time.time()-start
        return modeling



def localdata(filebed="./../input/",sync=1,lr=0.0001,epochs = 15,batch_size = 8,dropout=0.2):
    comm = MPI.COMM_WORLD

    size = comm.Get_size()
    rank = comm.Get_rank()
    myhost = MPI.Get_processor_name()
    model = CNNGraph(input_size_1=1300,input_size_2=1, output_size=4,
                        learning_rate=lr)
    if rank==0:
        modeling = mastermain((size-1),batch_size,lr=lr,mode=0)
        print("epochs:",epochs,file=modeling.main_file)
        print("ECG MPI tensorflow with threads",(size),file=modeling.main_file)
    else:
        xtrain_name=filebed+"train/xtrain-"+str(rank)+".npy"
        xval_name=filebed+"val/xval-"+str(rank)+".npy"
        xtest_name=filebed+"test/xtest-"+str(rank)+".npy"
        ytrain_name=filebed+"train/ytrain-"+str(rank)+".npy"
        yval_name=filebed+"val/yval-"+str(rank)+".npy"
        ytest_name=filebed+"test/ytest-"+str(rank)+".npy"
        modeling=WorkerModeling(model,batch_size)
        
        modeling.data(xtrain_name,ytrain_name,xval_name,yval_name,xtest_name,ytest_name)
    modeling.model=model
    modeling.model.graph()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    fit = time.time()
    with tf.Session(config=config, graph=modeling.model.cnn_graph) as sess:
            init = tf.group(tf.global_variables_initializer(),
                        tf.local_variables_initializer())
            sess.run(init)
            saver = tf.train.Saver()
            for e in range(epochs):
                sub_time = time.time()
                acc,loss,val_acc,val_loss=0,0,0,0
                ###training
                if rank!=0:
                    for s in range(modeling.train_step):
                        data,label=next(modeling.train_next_batch_gen)
                        if s==(modeling.train_step-1):
                            grads,losst,acct=sess.run([modeling.model._grad_op,modeling.model.cnn_loss,modeling.model.accuracy], 
                                    feed_dict={modeling.model.X: data, modeling.model.Y: label, modeling.model.keep_prob: 0.2})
                        else:
                            _,losst,acct=sess.run([modeling.model.sub_grad_op,modeling.model.cnn_loss,modeling.model.accuracy],
                                            feed_dict={modeling.model.X: data, modeling.model.Y: label, modeling.model.keep_prob: 0.2})
                        acc+=acct/modeling.train_step
                        loss+=losst/modeling.train_step

                if rank==0:
                    w=[]
                    for i in range(1, size):
                        w.append(comm.recv())
                else:
                    update_weights =  grads
                    comm.send(update_weights, dest=0)
                ####validation
                if rank==0:
                    for i in range(1, size):
                        comm.send(w, dest=i)
                else:
                    _weights = comm.recv(source=0)
                    feed_dict = {}
                    modeling.model._gradients=_weights
                    for i, placeholder in enumerate(modeling.model._grad_placeholders):
                        feed_dict[placeholder] = np.stack([g[i] for g in modeling.model._gradients], axis=0).mean(axis=0)
                    modeling.update_weight=feed_dict
                    sess.run(modeling.model._train_op, feed_dict=feed_dict)

                if rank!=0:
                    for s in range(modeling.val_step):
                        data,label=next(modeling.val_next_batch_gen)
                        losst,acct=sess.run([modeling.model.cnn_loss,modeling.model.accuracy], 
                                    feed_dict={modeling.model.X: data, modeling.model.Y: label, modeling.model.keep_prob: 1.0})
                        val_acc+=acct/modeling.val_step
                        val_loss+=losst/modeling.val_step
                        

                if rank==0:
                    w=[]
                    for i in range(1, size):
                        w.append(comm.recv(source=i))
                    modeling.update(w,sub_time,e,fit)
                    for i in range(1, size):
                        (comm.send(modeling.update_flag,dest=i))
                else:
                    comm.send([val_loss,val_acc,loss,acc], dest=0)
                    modeling.training_track.append((e + 1,loss, val_loss, acc, val_acc, np.round(time.time()-sub_time, decimals=4))) 
                    update_flag = comm.recv(source=0)
                    if update_flag==True:
                        modeling.best_model_weights =modeling.update_weight
                    if val_loss < modeling.best_validation_loss:
                        modeling.best_validation_loss = val_loss
                        modeling.last_improvement = e+1 
            end = time.time()

            ####testing
            if rank==0:
                print("training time :",end-fit)
                modeling.training_time=end-fit
            else:
                sess.run(modeling.model._train_op, feed_dict=modeling.best_model_weights)
            
            if rank!=0:
                for s in range(modeling.test_step):
                    data,label=next(modeling.test_next_batch_gen)
                    pred=sess.run(modeling.model.projection_1hot, 
                                    feed_dict={modeling.model.X: data, modeling.model.keep_prob: 1.0})
                    modeling.pred.append(pred)
                    modeling.label.append(label)
    if rank==0:
        p,t=[],[]
        for i in range(1, size):
            tmp1,tmp2=comm.recv(source=i)
            p.append(tmp1)
            t.append(tmp2)
    else:
        comm.send([modeling.pred,modeling.label], dest=0)


    #####output
    if rank==0:
        p=np.vstack(p)
        t=np.vstack(t)
        modeling.predict(p,t,end)
        modeling.savestat()
    else:
        modeling.trainstats(rank,myhost)


def masterdata(sync=1,lr=0.0001,epochs = 15,batch_size = 8):
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    myhost = MPI.Get_processor_name()
    if rank==0:
        modeling,train_next_batch_gen,val_next_batch_gen,test_next_batch_gen,train_step,val_step,\
        test_step=mastermain((size-1),batch_size,lr=lr,mode=1)

        print("epochs:",epochs,file=modeling.main_file)
        print("ECG MPI  with threads",(size),file=modeling.main_file)
    model = CNNGraph(input_size_1=1300,input_size_2=1, output_size=4,
                        learning_rate=lr)
    if rank==0:
        for i in range(1, size):
            comm.send([train_step,val_step,test_step], dest=i)
    else:
        modeling=WorkerModeling(model,batch_size)
        train_step,val_step,test_step=comm.recv(source=0)
    modeling.model=model
    modeling.model.graph()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    fit = time.time()
    with tf.Session(config=config, graph=modeling.model.cnn_graph) as sess:
            init = tf.group(tf.global_variables_initializer(),
                        tf.local_variables_initializer())
            sess.run(init)
            saver = tf.train.Saver()
            for e in range(epochs):
                sub_time = time.time()
                acc,loss,val_acc,val_loss=0,0,0,0
                for s in range(train_step):
                    if rank==0:
                        data,label=next(train_next_batch_gen)
                        for i in range(1,size):
                            k=i*batch_size
                            comm.send([data[(i-1)*batch_size:k],label[(i-1)*batch_size:k]], dest=i)
                    else:

                        data,label=comm.recv(source=0)
                        if s==(train_step-1):
                            grads,losst,acct=sess.run([modeling.model._grad_op,modeling.model.cnn_loss,modeling.model.accuracy], 
                                            feed_dict={modeling.model.X: data, modeling.model.Y: label, modeling.model.keep_prob: 0.2})
                        else:
                            _,losst,acct=sess.run([modeling.model.sub_grad_op,modeling.model.cnn_loss,modeling.model.accuracy],
                                            feed_dict={modeling.model.X: data, modeling.model.Y: label, modeling.model.keep_prob: 0.2})
                        acc+=acct/train_step
                        loss+=losst/train_step

                if rank==0:
                    w=[]
                    for i in range(1, size):
                        w.append(comm.recv())
                else:
                    update_weights =  grads
                    comm.send(update_weights, dest=0)


                ####validation
                if rank==0:
                    for i in range(1, size):
                        comm.send(w, dest=i)
                else:
                    _weights = comm.recv(source=0)
                    feed_dict = {}
                    modeling.model._gradients=_weights
                    for i, placeholder in enumerate(modeling.model._grad_placeholders):
                        feed_dict[placeholder] = np.stack([g[i] for g in modeling.model._gradients], axis=0).mean(axis=0)
                    modeling.update_weight=feed_dict
                    sess.run(modeling.model._train_op, feed_dict=feed_dict)
                val_loss,val_acc=0,0
                for s in range(val_step):
                    if rank==0:
                        data,label=next(val_next_batch_gen)
                        for i in range(1,size):
                            k=i*batch_size
                            comm.send([data[(i-1)*batch_size:k],label[(i-1)*batch_size:k]], dest=i)
                    else:
                        data,label=comm.recv(source=0)
                        losst,acct=sess.run([modeling.model.cnn_loss,modeling.model.accuracy], 
                                        feed_dict={modeling.model.X: data, modeling.model.Y: label, modeling.model.keep_prob: 1.0})
                        val_acc+=acct/val_step
                        val_loss+=losst/val_step

                if rank==0:
                    w=[]
                    for i in range(1, size):
                        w.append(comm.recv(source=i))
                    modeling.update(w,sub_time,e,fit)
                    for i in range(1, size):
                        (comm.send(modeling.update_flag,dest=i))
                else:
                    comm.send([val_loss,val_acc,loss,acc], dest=0)
                    modeling.training_track.append((e + 1,loss, val_loss, acc, val_acc, np.round(time.time()-sub_time, decimals=4))) 
                    update_flag = comm.recv(source=0)
                    if update_flag==True:
                        modeling.best_model_weights =modeling.update_weight
                    if val_loss < modeling.best_validation_loss:
                        modeling.best_validation_loss = val_loss
                        modeling.last_improvement = e+1 
            end = time.time()

            ####testing
            if rank==0:
                print("training time :",end-fit)
                modeling.training_time=end-fit
            else:
                sess.run(modeling.model._train_op, feed_dict=modeling.best_model_weights)

            p,t=[],[]
            for s in range(test_step):
                if rank==0:
                    data,label=next(test_next_batch_gen)
                    for i in range(1,size):
                        k=i*batch_size
                        comm.send([data[(i-1)*batch_size:k],label[(i-1)*batch_size:k]], dest=i)
                else:
                    data,label=comm.recv(source=0)
                    pred=sess.run(modeling.model.projection_1hot, 
                                            feed_dict={modeling.model.X: data, modeling.model.keep_prob: 1.0})
                    modeling.pred.append(pred)
                    modeling.label.append(label)

                if rank==0:
                    for i in range(1, size):
                        tmp1,tmp2=comm.recv(source=i)
                        p.append(tmp1)
                        t.append(tmp2)
                else:
                    comm.send([pred,label], dest=0)


    #####output
    if rank==0:
        modeling.predict(p,t,end)
        modeling.savestat()
    else:
        modeling.trainstats(rank,myhost)

if __name__ == '__main__':
    mode=sys.argv[3]
    if mode ==  "0":
        masterdata()
    else:
        localdata()
