#test.py
import numpy as np
import pandas as pd

from losses import CrossEntropy
from optimizers import Adam
from cnngraph import CNNGraph
from executors import DesktopExecution
import time
execution_start = time.time()
file_dir = "./../input/"
inputs = np.load(file_dir+'xdata.npy')
targets = np.load(file_dir+'ydata.npy')
targets=pd.get_dummies(targets).values 

model = CNNGraph(input_size_1=1300,input_size_2=1, output_size=4,
                        loss=CrossEntropy,
                        optimizer=Adam(lr=0.0001))

projection = DesktopExecution(model, max_epochs=30,batch_size=8)
projection.training_memory(inputs, targets)

print("Execution Time: {}".format((time.time()-execution_start)))
