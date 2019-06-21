"""
Data manager its adjusted for training a medical diagnostic workflow inside the hospitals.
It provides an isolation to fine-tuning different neural network hyper-parameters.
"""

import collections
from typing import Iterator, NamedTuple, List

import glob, os.path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from io_functions import IO_Functions

import logging
logger = logging.getLogger('_DiagnoseNET_')

DataSplit = collections.namedtuple('DataSplit', 'name inputs targets')
Batch = NamedTuple("Batch", [("inputs", np.ndarray), ("targets", np.ndarray)])
BatchPath = NamedTuple("BatchPath", [("input_files", list), ("target_files", list)])

class Dataset:
    """
    A desktop manager is provided for memory or disk training.
    """
    def __init__(self) -> None:
        self.inputs: np.ndarray
        self.targets: np.ndarray

        self.dataset_name: str
        self.dataset_path: str
        self.inputs_name: str
        self.labels_name: str

    def set_data_file(self, inputs: np.ndarray, targets: np.ndarray) -> None:
        """
        Set data file
        """
        try:
            ## Convert a pandas dataframe (df) to a numpy ndarray
            self.inputs = inputs.values
            self.targets = targets.values
        except AttributeError:
            if 'numpy' in str(type(inputs)):
                self.inputs = inputs
                self.targets = targets
            elif 'list' in str(type(inputs)):
                ## Convert a list to a numpy ndarray
                self.inputs = pd.DataFrame(inputs)
                self.inputs = np.asarray(self.inputs[0].str.split(',').tolist())
                self.targets = pd.DataFrame(targets)
                self.targets = np.asarray(self.targets[0].str.split(',').tolist())
            else:
                raise AttributeError("set_data_file(inputs, targets) requires: numpy, pandas or list ")


class Splitting(Dataset):
    """
    split dataset in Training, Valid and test,
    and split into mini-batches give a size factor.
    """
    def __init__(self, valid_size: float = None, test_size: float = None) -> None:
        super().__init__()
        self.valid_size = valid_size
        self.test_size = test_size
        ## Datamanager
        self.train: DataSplit
        self.valid: DataSplit
        self.test: DataSplit

        ## Dataset shape as number of records (inputs, targets)
        self.train_shape: str = None
        self.valid_shape: str = None
        self.test_shape: str = None


    def dataset_split(self) -> None:
        """
        split dataset in Training, Valid and test,
        and split into mini-batches give a size factor.
        """
        try:
            prop1=self.valid_size+self.test_size
            prop2=(self.test_size*1)/(self.valid_size+self.test_size)
        except TypeError:
            logger.warning('!!! Splitting proportions are missing !!!')
            self.valid_size = 0.1; self.test_size = 0.10
            logger.warning('!!! Set valid_size={} and test_size={} by the fault !!!'.format(self.valid_size, self.test_size))
            prop1=self.valid_size+self.test_size
            prop2=(self.test_size*1)/(self.valid_size+self.test_size)

        X_train, X_temp, y_train, y_temp = train_test_split(self.inputs, self.targets, test_size=prop1)
        X_valid, X_test, y_valid, y_test = train_test_split(X_temp,y_temp, test_size=prop2)

        ## DataSplit instances
        self.train = DataSplit(name='train', inputs=X_train, targets=y_train)
        self.valid = DataSplit(name='valid', inputs=X_valid, targets=y_valid)
        self.test = DataSplit(name='test', inputs=X_test, targets=y_test)

        self.train_shape = (len(self.train.inputs), len(self.train.targets))
        self.valid_shape = (len(self.valid.inputs), len(self.valid.targets))
        self.test_shape = (len(self.test.inputs), len(self.test.targets))

        logger.info('---------------------------------------------------------')
        logger.info('++ Dataset Split:  (Inputs, Targets) ++')
        logger.info('-- Train records:  {} --'.format(self.train_shape)) #len(self.train.inputs), len(self.train.targets)))
        logger.info('-- Valid records:  {} --'.format(self.valid_shape))  #len(self.valid.inputs), len(self.valid.targets)))
        logger.info('-- Test records:   {} --'.format(self.test_shape))  #len(self.test.inputs), len(self.test.targets)))


class Batching(Splitting):
    """
    Write micro-batches by a size split factor
    for each part of the data, Training, Valid, Test
    """
    def __init__(self, valid_size: float = None, test_size: float = None,
                        batch_size: int = 1000):
        super().__init__(valid_size, test_size)
        self.batch_size = batch_size

    def batching(self, data: DataSplit) -> List[Batch]:
        """
        inputs data: has the next format:
            DataSplit = collections.namedtuple
        outputs batches: are write in the next format:
            List[Batch] = List[NamedTuple[np.ndarray, np.ndarray]]
        """
        batch_index = np.arange(0, len(data.inputs), self.batch_size)
        if self.shuffle:
            np.random.shuffle(batch_index)

        batches_inputs = []
        batches_targets = []
        for start in batch_index:
            end = start + self.batch_size
            batches_inputs.append(data.inputs[start:end])
            batches_targets.append(data.targets[start:end])
        return Batch(batches_inputs, batches_targets)

    def memory_batching(self, shuffle: bool = True) -> List[Batch]:
        """
        outputs batches: are write in the next format:
            List[Batch] = List[NamedTuple[np.ndarray, np.ndarray]]
        """
        self.shuffle = shuffle

        ## Splittting the Dataset
        self.dataset_split()

        train_batches = self.batching(self.train)
        valid_batches = self.batching(self.valid)
        test_batches = self.batching(self.test)
        return train_batches, valid_batches, test_batches

class MultiTask(Batching):
    """
    A multi task label set the
    """
    def __init__(self, dataset_name: str = 'PMSI_ICU_W1',
                        valid_size: float = None, test_size: float = None,
                        batch_size: int = 1000,
                        target_name: str = 'Y11',
                        target_start: int = 0, target_end: int = 14) -> None:
        super().__init__(valid_size, test_size, batch_size)
        self.dataset_name = dataset_name
        self.target_name = target_name
        self.target_start = target_start
        self.target_end = target_end

    def memory_target_splitting(self, batches: List[Batch]) -> List[Batch]:
        """
        Inputs and output batches: Are write in the next format:
            List[Batch] = List[NamedTuple[np.ndarray, np.ndarray]]
        """
        batches_inputs = []
        batches_targets = []
        for i in range(len(batches.inputs)):
            batches_targets.append(batches.targets[i][:,self.target_start:self.target_end])
            batches_inputs.append(batches.inputs[i])
        return Batch(batches_inputs, batches_targets)

    def memory_one_target(self) -> None:
        """
        This function splits and write the label selected from a multi-labels file.
        In which each label has been write-in one-hot encode.
        """

        train_batches, valid_batches, test_batches = self.memory_batching()
        train_ = self.memory_target_splitting(train_batches)
        valid_ = self.memory_target_splitting(valid_batches)
        test_ = self.memory_target_splitting(test_batches)
        return train_, valid_, test_

    def disk_target_splitting(self, splitting: str) -> BatchPath:
        """
        output batch path:
        """
        ## Defining split directories
        splitting_path = str(self.split_path+"/data_"+splitting)

        target_path = sorted(glob.glob(splitting_path+"/y-"+self.dataset_name+"-"+self.target_name+"-*.txt"))
        if not target_path:
            y_files = sorted(glob.glob(splitting_path+"/y-*.txt"))

            for f in y_files:
                data = IO_Functions()._read_file(f)
                # ## Convert list in a numpy matrix
                data = pd.DataFrame(data)
                data = data[0].str.split(',').tolist()

                ## set the label selected
                label = np.asarray(data)[:,self.target_start:self.target_end]

                ## Set file_name
                f_names = f.split('-')
                path_to_file = str(splitting_path+"/y-"+self.dataset_name+"-"+self.target_name+"-"+f_names[-1])

                ## Write new label file
                # if path_to_file not in files:
                with open(path_to_file, 'a') as file_:
                    np.savetxt(file_, label, fmt='%s',delimiter=',',newline='\n' )
                    file_.close()

        target_files = sorted(glob.glob(splitting_path+"/y-"+self.dataset_name+"-"+self.target_name+"-*.txt"))
        input_files = sorted(glob.glob(splitting_path+"/X-*.txt"))

        return BatchPath(input_files, target_files)

    def disk_one_target(self) -> BatchPath:
        """
        This function splits and write the label selected from a multi-labels file.
        In which each label has been write-in one-hot encode.
        """
        ## Baching the dataset
        self.disk_batching()

        train_path = self.disk_target_splitting("training")
        valid_path = self.disk_target_splitting("valid")
        test_path = self.disk_target_splitting("test")

        return train_path, valid_path, test_path
