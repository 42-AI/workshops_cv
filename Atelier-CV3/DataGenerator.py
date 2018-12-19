import numpy as np
from skimage.io import imread
from skimage.transform import resize
import urllib
import numpy as np
import cv2
import keras
from imgaug import augmenters as iaa

# seq = iaa.Sequential([
#   iaa.Fliplr(0.5, name="Flipper"),
#   iaa.GaussianBlur((0, 3.0), name="GaussianBlur"),
#   iaa.Dropout(0.02, name="Dropout"),
#   iaa.AdditiveGaussianNoise(scale=0.01*255, name="MyLittleNoise"),
#   iaa.AdditiveGaussianNoise(loc=32, scale=0.0001*255, name="SomeOtherNoise"),
#   iaa.Affine(translate_px={"x": (-40, 40)}, name="Affine")
# ])

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, labels, batch_size=32, dim=(1,150,150), n_channels=1,
                 n_classes=10, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

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
        X, y = self.__data_generation(indexes)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
    
    def read_and_resize(self, filepath):
        img = cv2.imread(filepath)
        res = resize(img, (224, 224), preserve_range=True, mode='reflect')
        return np.expand_dims(res, 0)
            
    def __data_generation(self, indexes):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)

        X = [self.read_and_resize(self.list_IDs[i])
             for i in indexes]
        y = self.labels[indexes]
        X = np.vstack(X)
#         return seq.augment_images(X), y
        return (X, y)