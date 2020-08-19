#https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly

import numpy as np
import os
import tensorflow.keras as keras
from tensorflow.python.keras.utils.data_utils import Sequence
from sklearn import preprocessing
from PIL import Image

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    #These are defaults, but we can set these dynamically based on input data later
    def __init__(self, list_IDs, labels_dict, datadir, dim, batch_size,
                 n_classes, shuffle=True, fileprefix="", filesuffix="", scaler=None):
        'Initialization'
        self.dim = dim
        self.ndim = len(dim)
        self.batch_size = batch_size
        self.labels_dict = labels_dict
        self.list_IDs = list_IDs
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()
        self.datadir = datadir
        self.fileprefix = fileprefix
        self.filesuffix = filesuffix
        self.scaler = scaler

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim)) #, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data

        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            temp = np.load(os.path.join(self.datadir, str(self.fileprefix + ID + self.filesuffix + ".npy")))
            if not self.scaler:
                X[i,] = temp[:, 1:]
            else:
                X[i,] = self.scaler.transform(temp[:, 1:])

            #Generate Image
            #img = Image.fromarray(temp, 'L')
            #img.save(os.path.join(self.datadir, 'images', str(ID + '.jpg')))
            
            # Store class
            y[i] = self.labels_dict[ID]

        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)

def doubleGenerator(cat_gen, num_gen):
    while True:
        for (cat_x, cat_y), (num_x, _) in zip(cat_gen, num_gen):
            yield [cat_x, num_x], cat_y