#https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly

import numpy as np
import os
import tensorflow.keras as keras
from tensorflow.python.keras.utils.data_utils import Sequence
from sklearn import preprocessing
import h5py

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    #These are defaults, but we can set these dynamically based on input data later
    def __init__(self, h5dataset, batch_size, shuffle=True):
        'Initialization'
        self.h5dataset = h5dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(self.h5dataset['derived_counts'].shape[0] // self.batch_size)

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        #list_IDs_temp = [self.h5dataset['samples'][k] for k in indexes]

        # Generate data
        X = self.__data_generation(indexes)

        return X

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(self.h5dataset['derived_counts'].shape[0])
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, indexes):
        'Generates data containing batch_size samples' 
        # Initialization
        X = np.empty((self.batch_size, self.h5dataset['derived_counts'].shape[1])) 

        # Generate data
        for i, j in zip(range(self.batch_size), indexes):
            # Store dc
            X[i,] = self.h5dataset['derived_counts'][j,]

        y = None

        return X, y
