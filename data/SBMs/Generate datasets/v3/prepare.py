# Block 1
import numpy as np
import torch
import pickle
import time
import os
import matplotlib.pyplot as plt





# Block 2
import pickle

from data.SBMs import SBMsDatasetDGL 

from data.data import LoadData
from torch.utils.data import DataLoader
from data.SBMs import SBMsDataset



# Block 3
class DotDict(dict):
    def __init__(self, **kwds):
        self.update(kwds)
        self.__dict__ = self

# Block 4
start = time.time()

DATASET_NAME = 'SBM_PATTERN'
dataset = SBMsDatasetDGL(DATASET_NAME) 

print('Time (sec):',time.time() - start) 


# Block 5
print(len(dataset.train))
print(len(dataset.val))
print(len(dataset.test))

print(dataset.train[0])
print(dataset.val[0])
print(dataset.test[0])


# Block 6
start = time.time()

with open('data/SBMs/SBM_PATTERN.pkl','wb') as f:
        pickle.dump([dataset.train,dataset.val,dataset.test],f)
        
print('Time (sec):',time.time() - start) 


# Block 7
DATASET_NAME = 'SBM_PATTERN'
dataset = LoadData(DATASET_NAME) 
trainset, valset, testset = dataset.train, dataset.val, dataset.test



# Block 8
start = time.time()

batch_size = 10
collate = SBMsDataset.collate
train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, collate_fn=collate)

print('Time (sec):',time.time() - start) 