from tensorflow.keras.regularizers import l2
import os

from tensorflow.keras.layers import Dense, MaxPooling2D, Convolution2D, UpSampling2D, AveragePooling2D, Activation, \
    Lambda, \
    BatchNormalization, GlobalAveragePooling1D, GlobalMaxPooling1D, Input, Flatten, concatenate, Dropout
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from matplotlib import pyplot as plt
import pickle as c
from sklearn.preprocessing import LabelEncoder
import json
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from tensorflow.keras.applications.resnet50 import ResNet50

# from keras.optimizers import SGD
os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICE'] = '"'
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import backend as K
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense, MaxPooling2D, Convolution2D, Input, Flatten, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg16 import VGG16
import numpy as np

input_shape = (100, 100, 1)
input_a = Input(shape=input_shape)
input_b = Input(shape=input_shape)

num_classes = 2
epochs = 10
temp = 5
lambda_const = 0.2
from tensorflow.keras.optimizers import SGD

opt = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
rms = RMSprop(lr=0.00001)
weight_decay = 1E-4
from scipy import ndimage, misc
from scipy import ndimage, misc
import numpy as np
from tensorflow.keras.losses import categorical_crossentropy as logloss
from tensorflow.keras import backend as K


def softmax(x):
    return 1.0 / (1.0 + np.exp(-x))


def euclidean_distance(vects):
    x, y = vects
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)


def contrastive_loss(y_true, y_pred):
    margin = 1
    square_pred = K.square(y_pred)
    margin_square = K.square(K.maximum(margin - y_pred, 0))
    return K.mean(y_true * square_pred + (1 - y_true) * margin_square)


def compute_accuracy(y_true, y_pred):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    pred = y_pred.ravel() < 0.5
    return np.mean(pred == y_true)


def accuracy(y_true, y_pred):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    return K.mean(K.equal(y_true, K.cast(y_pred < 0.5, y_true.dtype)))


from tensorflow.keras.applications.vgg16 import VGG16

from tensorflow.keras import regularizers


def dice_coef(y_true, y_pred, smooth=1, weight=0.5):
    """
    加权后的dice coefficient
    """
    y_true = y_true[:, :, :, -1]  # y_true[:, :, :, :-1]=y_true[:, :, :, -1] if dim(3)=1 等效于[8,256,256,1]==>[8,256,256]
    y_pred = y_pred[:, :, :, -1]
    intersection = K.sum(y_true * y_pred)
    union = K.sum(y_true) + weight * K.sum(y_pred)
    # K.mean((2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth))
    return ((2. * intersection + smooth) / (union + smooth))  # not working better using mean


def euclidean_distance(vects):
    x, y = vects
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))


def small_FCN_siamese():
    image_a = Input(shape=(100, 100, 3))
    image_b = Input(shape=(100, 100, 3))
    base_model = VGG16(weights='imagenet', include_top=False)
    for layer in base_model.layers:
        layer.trainable = False
    processed_a = base_model(image_a)
    processed_b = base_model(image_b)

    feat_1 = Convolution2D(128, (3, 3), activation='relu', name='fc1', padding='valid')(processed_a)

    feat_2 = Convolution2D(128, (3, 3), activation='relu', name='fc11', padding='valid')(processed_b)

    l2distance = Lambda(euclidean_distance,
                        output_shape=eucl_dist_output_shape)([feat_1, feat_2])

    u4 = UpSampling2D(size=(100, 100))(l2distance)
    u4 = Convolution2D(1, (1, 1))(u4)
    logits = Activation('sigmoid')(u4)
    siamese = Model([image_a, image_b], logits)
    siamese.summary()
    siamese.compile(optimizer=Adam(lr=1e-4),
                    loss=['binary_crossentropy'], metrics=['accuracy'])
    return siamese


def Large_FCN_siamese():
    image_a = Input(shape=(100, 100, 3))
    image_b = Input(shape=(100, 100, 3))
    base_model = ResNet50(weights='imagenet', include_top=False)
    for layer in base_model.layers:
        layer.trainable = False
    processed_a = base_model(image_a)
    processed_b = base_model(image_b)

    feat_1 = Convolution2D(128, (2, 2), activation='relu', name='fc1', padding='valid')(processed_a)
    feat_1 = MaxPooling2D(pool_size=(2, 2))(feat_1)

    feat_2 = Convolution2D(128, (2, 2), activation='relu', name='fc11', padding='valid')(processed_b)
    feat_2 = MaxPooling2D(pool_size=(2, 2))(feat_2)
    l2distance = Lambda(euclidean_distance,
                        output_shape=eucl_dist_output_shape)([feat_1, feat_2])

    u4 = UpSampling2D(size=(100, 100))(l2distance)
    u4 = Convolution2D(1, (1, 1))(u4)
    logits = Activation('sigmoid')(u4)
    siamese = Model([image_a, image_b], logits)
    siamese.summary()
    siamese.compile(optimizer=Adam(lr=1e-4),
                    loss=['binary_crossentropy'], metrics=['accuracy'])


    return siamese
