from master import *
from worker import *
from mpi4py import MPI
import time, math
import numpy as np
import sys
from collections import Iterable

def localdata(filebed="./../input/",sync=1,lr=0.0001,epochs = 15,batch_size = 8):
    comm = MPI.COMM_WORLD

    size = comm.Get_size()
    rank = comm.Get_rank()
    myhost = MPI.Get_processor_name()
   
    if rank==0:
        modeling = mastermain((size-1),batch_size,lr=lr,mode=0)
        print("epochs:",epochs,file=modeling.main_file)
        print("ECG MPI  with threads",(size),file=modeling.main_file)
    else:
        xtrain_name=filebed+"train/xtrain-"+str(rank)+".npy"
        xval_name=filebed+"val/xval-"+str(rank)+".npy"
        xtest_name=filebed+"test/xtest-"+str(rank)+".npy"
        ytrain_name=filebed+"train/ytrain-"+str(rank)+".npy"
        yval_name=filebed+"val/yval-"+str(rank)+".npy"
        ytest_name=filebed+"test/ytest-"+str(rank)+".npy"
    
    if rank==0:
        for i in range(1, size):
            comm.send(modeling.model_json, dest=i)
    else:
        modeling=WorkerModeling(batch_size)
        modeling.model_json=comm.recv(source=0)
        modeling.load(lr)
        modeling.data(xtrain_name,ytrain_name,xval_name,yval_name,xtest_name,ytest_name)
    
    fit = time.time()
    for e in range(epochs):
        sub_time = time.time()
        if rank==0:
            #sub_time= time.time()
            for i in range(1, size):
                comm.send(modeling.model_weights, dest=i)
        else:
            modeling.model_weights = comm.recv(source=0)
            modeling.read(True)

        #if rank==(size-1):
        #    print("pass weights timing",time.time()-sub_time)

        ###training
        print(rank,"- train")
        if rank!=0:
            for s in range(modeling.train_step):
                data,label=next(modeling.train_next_batch_gen)
                end_epoch = True if s == (modeling.train_step-1) else False
                modeling.train(data,label,end_epoch)

        if rank==0:
            w=[]
            for i in range(1, size):
                w.append(comm.recv())
            modeling.average_weights(w)
        else:
            update_weights =  modeling.model_weights
            comm.send(update_weights, dest=0)

        print(rank,"- val")
        ####validation
        if rank==0:
            for i in range(1, size):
                comm.send(modeling.model_weights, dest=i)
        else:
            modeling.model_weights = comm.recv(source=0)
            modeling.read(True)

        val_loss,val_acc=0,0
        if rank!=0:
            for s in range(modeling.val_step):
                data,label=next(modeling.val_next_batch_gen)
                modeling.validate(data,label)
                val_loss+=modeling.val_loss/modeling.val_step
                val_acc+=modeling.val_acc/modeling.val_step

        print(rank,"- epoch end")
        if rank==0:
            w=[]
            for i in range(1, size):
                w.append(comm.recv(source=i))
            modeling.update(w,sub_time,e)
        else:
            comm.send([val_loss,val_acc,modeling.loss_list[-1],modeling.acc_list[-1]], dest=0)
            modeling.val_loss_list.append(val_loss)
            modeling.val_acc_list.append(val_acc)
            modeling.track(e,sub_time)
    end = time.time()

    ####testing
    if rank==0:
        print("training time :",end-fit)
        modeling.training_time=end-fit
        for i in range(1, size):
            comm.send(modeling.best_model_weights, dest=i)
    else:
        modeling.best_model_weights = comm.recv(source=0)
        modeling.read(False)

    
    if rank!=0:
        for s in range(modeling.test_step):
            data,label=next(modeling.test_next_batch_gen)
            pred = modeling.test(data)
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


def masterdata(sync=1,lr=0.0001,epochs = 5,batch_size = 8):
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    myhost = MPI.Get_processor_name()
    if rank==0:
        modeling,train_next_batch_gen,val_next_batch_gen,test_next_batch_gen,train_step,val_step,\
        test_step=mastermain((size-1),batch_size,lr=lr,mode=1)
        #print(train_step,val_step,test_step)
        print("epochs:",epochs,file=modeling.main_file)
        print("ECG MPI  with threads",(size),file=modeling.main_file)
    #atime=time.time()
    if rank==0:
        for i in range(1, size):
            comm.send([modeling.model_json,train_step,val_step,test_step], dest=i)
    else:
        modeling=WorkerModeling(batch_size)
        modeling.model_json,train_step,val_step,test_step=comm.recv(source=0)
        modeling.load(lr)
    #if rank==(size-1):
    #    print("pass model timing",time.time()-atime)
    fit = time.time()

    for e in range(epochs):
        sub_time = time.time()
        if rank==0:
            sub_time= time.time()
            for i in range(1, size):
                comm.send(modeling.model_weights, dest=i)
        else:
            modeling.model_weights = comm.recv(source=0)
            modeling.read(True)

        #if rank==(size-1):
        #    print("pass weights timing",time.time()-sub_time)

        ###training
        for s in range(train_step):
            if rank==0:
                data,label=next(train_next_batch_gen)
                for i in range(1,size):
                    k=i*batch_size
                    comm.send([data[(i-1)*batch_size:k],label[(i-1)*batch_size:k]], dest=i)
            else:
                end_epoch = True if s == (train_step-1) else False
                data,label=comm.recv(source=0)
                modeling.train(data,label,end_epoch)

        if rank==0:
            w=[]
            for i in range(1, size):
                w.append(comm.recv())
            modeling.average_weights(w)
        else:
            update_weights =  modeling.model_weights
            comm.send(update_weights, dest=0)


        ####validation
        if rank==0:
            for i in range(1, size):
                comm.send(modeling.model_weights, dest=i)
        else:
            modeling.model_weights = comm.recv(source=0)
            modeling.read(True)

        val_loss,val_acc=0,0
        for s in range(val_step):
            if rank==0:
                data,label=next(val_next_batch_gen)
                for i in range(1,size):
                    k=i*batch_size
                    comm.send([data[(i-1)*batch_size:k],label[(i-1)*batch_size:k]], dest=i)
            else:
                data,label=comm.recv(source=0)
                if len(data)!=0:
                    modeling.validate(data,label)
                    val_loss+=modeling.val_loss/val_step
                    val_acc+=modeling.val_acc/val_step
                else:
                    print("rank:",rank," skip val data")

        if rank==0:
            w=[]
            for i in range(1, size):
                w.append(comm.recv(source=i))
            modeling.update(w,sub_time,e)
        else:
            comm.send([val_loss,val_acc,modeling.loss_list[-1],modeling.acc_list[-1]], dest=0)
            modeling.val_loss_list.append(val_loss)
            modeling.val_acc_list.append(val_acc)
            modeling.track(e)
    end = time.time()

    ####testing
    if rank==0:
        print("training time :",end-fit)
        modeling.training_time=end-fit
        #print("training time :",end-fit,file=main_file)
        for i in range(1, size):
            comm.send(modeling.best_model_weights, dest=i)
    else:
        modeling.best_model_weights = comm.recv(source=0)
        modeling.read(False)

    p,t=[],[]
    for s in range(test_step):
        if rank==0:
            data,label=next(test_next_batch_gen)
            for i in range(1,size):
                k=i*batch_size
                comm.send([data[(i-1)*batch_size:k],label[(i-1)*batch_size:k]], dest=i)
        else:
            data,label=comm.recv(source=0)
            pred = modeling.test(data)
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
        #modeling.predictstats(rank)

if __name__ == '__main__':
    mode=sys.argv[3]
    if mode == 0:
        masterdata()
    else:
        localdata()
