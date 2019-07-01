import numpy as np
import pandas as pd 
import time 
import sys
from sklearn.model_selection import train_test_split
import random

def all_np(arr):
    arr = np.array(arr)
    key = np.unique(arr)
    result = {}
    for k in key:
        mask = (arr == k)
        arr_new = arr[mask]
        v = arr_new.size
        result[k] = v
    return result

def _write_file( data, dir_, num_samples, name,num_workers):

    num_file = 1
    new_sdw_factor = (num_samples//int(num_workers))
    for i in range(len(data)):
        if i % new_sdw_factor == 0:
            if num_file == int(num_workers): 
                np.save(str(dir_) + str(num_file)+str('.npy'),data[i:num_samples])
            elif num_file < int(num_workers):
            	np.save(str(dir_) + str(num_file)+str('.npy'),data[i:(i+new_sdw_factor)])
            num_file += 1

def datamain(nsplit):
    embd_dir=("./../input")
    x=np.load("./../input/xdata.npy")
    y=np.load("./../input/ydata.npy")
    nsplit =int(nsplit)-1
    print(all_np(y))

    y=pd.get_dummies(y).values

    X_train, X_temp, y_train, y_temp = train_test_split(x,y, train_size=0.8, random_state=random.seed(42))
    X_valid, X_test, y_valid, y_test = train_test_split(X_temp,y_temp, train_size=0.5, random_state=random.seed(22))
    print("dat---")
    dir_X_train = str(embd_dir+"/train/xtrain-")
    dir_y_train = str(embd_dir+"/train/ytrain-")
    dir_X_valid = str(embd_dir+"/val/xval-")
    dir_y_valid = str(embd_dir+"/val/yval-")
    dir_X_test = str(embd_dir+"/test/xtest-")
    dir_y_test = str(embd_dir+"/test/ytest-")

    _write_file(X_train, dir_X_train, len(X_train), "X_training",nsplit)
    _write_file(y_train, dir_y_train, len(y_train), "y_training",nsplit)
    _write_file(X_valid, dir_X_valid, len(X_valid), "X_valid",nsplit)
    _write_file(y_valid, dir_y_valid, len(y_valid), "y_valid",nsplit)
    print("val---")
    _write_file(X_test, dir_X_test, len(X_test), "X_test",nsplit)
    _write_file(y_test, dir_y_test, len(y_test), "y_test",nsplit)
  
if __name__ == "__main__":
    datamain(sys.argv[1])
