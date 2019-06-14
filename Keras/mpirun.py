from master import *
from worker import *
from mpi4py import MPI
import time, math
from collections import Iterable



if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    epochs = 30
    batch_size = 4

    
    if rank==0:
        main_file = open('output/main_data.txt','w')
        modeling,train_next_batch_gen,val_next_batch_gen,test_next_batch_gen,train_step,val_step,\
        test_step=mastermain((size-1),batch_size)

    if rank==0:
        for i in range(1, size):
            comm.send([modeling.model_json,train_step,val_step,test_step], dest=i)
    else:
        modeling=WorkerModeling()
        modeling.model_json,train_step,val_step,test_step=comm.recv(source=0)
        modeling.load()

    fit = time.time()

    for e in range(epochs):
        if rank==0:
            sub_time= time.time()
            for i in range(1, size):
                comm.send(modeling.model_weights, dest=i)
        else:
            modeling.model_weights = comm.recv(source=0)
            modeling.read(True)

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

        if rank==0:
            w=[]
            for i in range(1, size):
                w.append(comm.recv(source=i))
            modeling.update(w,sub_time,e)
        else:
            comm.send([val_loss,val_acc,modeling.history.loss,modeling.history.acc], dest=0)

    end = time.time()

    ####testing
    if rank==0:
        print("training time :",end-fit)
        print("training time :",end-fit,file=main_file)
        for i in range(1, size):
            comm.send(modeling.best_model_weights, dest=i)
    else:
        modeling.best_model_weights = comm.recv(source=0)
        modeling.read(False)

    for s in range(test_step):
        if rank==0:
            data,label=next(test_next_batch_gen)
            for i in range(1,size):
                k=i*batch_size
                comm.send([data[(i-1)*batch_size:k],label[(i-1)*batch_size:k]], dest=i)
        else:
            data,label=comm.recv(source=0)
            pred = modeling.test(data)

        if rank==0:
            p,t=[],[]
            for i in range(1, size):
                tmp1,tmp2=comm.recv(source=i)
                p.append(tmp1)
                t.append(tmp2)
        else:
            comm.send([pred,label], dest=0)


    #####output
    if rank==0:
        modeling.predict(p,t,end)
        w=[]
        for i in range(1, size):
            w.append(comm.recv(source=i))
        modeling.savestat(w)
        main_file.close()
    else:
        comm.send(modeling.loss_list, dest=0)

