import numpy as np
import scipy.io
from tensorflow.keras.utils import to_categorical, Sequence
from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import to_categorical
class DataGenerator(Sequence):
    'Generates data for Keras'

    def __init__(self, pair, class_map, batch_size=16, dim=(100, 100, 3), shuffle=True):
        'Initialization'
        self.dim = dim
        self.pair = pair
        self.class_map = class_map
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.pair) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs_temp = [k for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.pair))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        # Initialization
        batch_imgs_pre = list()
        batch_imgs_post = list()
        batch_labels = list()

        # Generate data
        for i in list_IDs_temp:
            # Store sample
            # print (self.pair[i][0])
            img = image.load_img(self.pair[i][0], target_size=(100, 100))
            img = image.img_to_array(img)
            batch_imgs_pre.append((img/255))
            img = image.load_img(self.pair[i][1], target_size=(100, 100))
            img = image.img_to_array(img)
            batch_imgs_post.append((img/255))
            img = image.load_img(self.pair[i][2], target_size=(100, 100), color_mode="grayscale")
            img = image.img_to_array(img)
            batch_labels.append(np.where(img==255, 1, img) )
        #print (batch_imgs_pre.shape,batch_imgs_post.shape, batch_labels.shape)
        return [(np.array(batch_imgs_pre).astype("float32"))/255, (np.array(batch_imgs_post).astype("float32"))/255], np.array(batch_labels)
