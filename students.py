import os
import tensorflow.keras
import numpy as np
from tensorflow.keras import backend as K
from tensorflow.keras.metrics import categorical_accuracy
from tensorflow.keras.layers import Dense, MaxPooling2D, Convolution2D, UpSampling2D, Add,AveragePooling2D,\
    Activation, Lambda, BatchNormalization, GlobalAveragePooling1D, GlobalMaxPooling1D, Input, Flatten, concatenate, Dropout
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from matplotlib import pyplot as plt
from tensorflow.keras import Sequential, Model
from tensorflow.keras.losses import categorical_crossentropy as logloss
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.initializers import glorot_uniform
num_classes = 2

from scipy.special import logit
def sigmoid(x, derivative=False):
  return x*(1-x) if derivative else 1/(1+np.exp(-x))

def knowledge_distillation_loss_withBE(y_true, y_pred, beta=0.1):
    # Extract the groundtruth from dataset and the prediction from teacher model
    y_true, y_pred_teacher = y_true[:, :1], y_true[:, 1:]

    # Extract the prediction from student model
    y_pred, y_pred_stu = y_pred[:, :1], y_pred[:, 1:]

    loss = beta * binary_crossentropy(y_true, y_pred) + (1 - beta) * binary_crossentropy(y_pred_teacher, y_pred_stu)

    return loss

def categorical_crossentropy(y_true, y_pred):
    y_true = y_true[:, :num_classes]
    y_pred = y_pred[:, :num_classes]
    return logloss(y_true, y_pred)


def top_5_accuracy(y_true, y_pred):
    y_true = y_true[:, :num_classes]
    y_pred = y_pred[:, :num_classes]
    return top_k_categorical_accuracy(y_true, y_pred)


def accuracy(y_true, y_pred):
    y_true = y_true[:, :num_classes]
    y_pred = y_pred[:, :num_classes]
    return categorical_accuracy(y_true, y_pred)


def knowledge_distillation_loss(y_true, y_pred, lambda_const, temp):
    # split in
    #    onehot hard true targets
    #    logits from xception
    y_true, logits = y_true[:, :num_classes], y_true[:, num_classes:]

    # convert logits to soft targets
    y_soft = K.softmax(logits / temp)

    # split in 
    #    usual output probabilities
    #    probabilities made softer with temperature
    y_pred, y_pred_soft = y_pred[:, :num_classes], y_pred[:, num_classes:]

    return lambda_const * logloss(y_true, y_pred) + logloss(y_soft, y_pred_soft)


def euclidean_distance(vects):
    x, y = vects
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)


def mini_vgg_siamese_network():
    cnn = Sequential()
    cnn.add(Convolution2D(64, (3, 3), padding='same',
                          input_shape=(100, 100, 3)))
    cnn.add(Convolution2D(64, (3, 3), activation='relu', padding='same'))
    cnn.add(MaxPooling2D((2, 2), strides=(2, 2)))
    cnn.add(Convolution2D(128, (3, 3), activation='relu', padding='same'))
    cnn.add(Convolution2D(128, (3, 3), activation='relu', padding='same'))
    cnn.add(MaxPooling2D((2, 2), strides=(2, 2)))
    cnn.add(Convolution2D(256, (3, 3), activation='relu', padding='same'))
    cnn.add(Convolution2D(256, (3, 3), activation='relu', padding='same'))
    cnn.add(Convolution2D(256, (3, 3), activation='relu', padding='same'))
    cnn.add(MaxPooling2D((2, 2), strides=(2, 2)))
    # cnn.add(Flatten())
    # cnn.add(Dense(128, activation='relu'))
    # cnn.add(Dense(64, activation='relu', name='feature_layer'))
    image_a = Input(shape=(100, 100, 3))
    image_b = Input(shape=(100, 100, 3))
    processed_a = cnn(image_a)
    processed_b = cnn(image_b)
    for layer in cnn.layers:
        layer.trainable = True
    feat_1 = Convolution2D(64, (3, 3), activation='relu', name='fc1', padding='valid')(processed_a)
    feat_1 = MaxPooling2D(pool_size=(10, 10))(feat_1)
    feat_2 = Convolution2D(64, (3, 3), activation='relu', name='fc11', padding='valid')(processed_b)
    feat_2 = MaxPooling2D(pool_size=(10, 10))(feat_2)
    l2distance = Lambda(euclidean_distance,
                        output_shape=eucl_dist_output_shape)([feat_1, feat_2])
    u4 = UpSampling2D(size=(100, 100))(l2distance)
    u4 = Convolution2D(1, (1, 1))(u4)
    logits = Activation('sigmoid')(u4)
    student = Model([image_a, image_b], logits)
    student.summary()
    student.compile(optimizer=Adam(lr=1e-4),
                    loss=['binary_crossentropy'], metrics=['accuracy'])
    return student


def KD_mini_vgg_siamese_network( lambda_const):
    cnn = Sequential()
    cnn.add(Convolution2D(64, (3, 3), padding='same',
                          input_shape=(100, 100, 3)))
    cnn.add(Convolution2D(64, (3, 3), activation='relu', padding='same'))
    cnn.add(MaxPooling2D((2, 2), strides=(2, 2)))
    cnn.add(Convolution2D(128, (3, 3), activation='relu', padding='same'))
    cnn.add(Convolution2D(128, (3, 3), activation='relu', padding='same'))
    cnn.add(MaxPooling2D((2, 2), strides=(2, 2)))
    cnn.add(Convolution2D(256, (3, 3), activation='relu', padding='same'))
    cnn.add(Convolution2D(256, (3, 3), activation='relu', padding='same'))
    cnn.add(Convolution2D(256, (3, 3), activation='relu', padding='same'))
    cnn.add(MaxPooling2D((2, 2), strides=(2, 2)))
    # cnn.add(Flatten())
    # cnn.add(Dense(128, activation='relu'))
    # cnn.add(Dense(64, activation='relu', name='feature_layer'))
    image_a = Input(shape=(100, 100, 3))
    image_b = Input(shape=(100, 100, 3))
    processed_a = cnn(image_a)
    processed_b = cnn(image_b)
    feat_1 = Convolution2D(64, (3, 3), activation='relu', name='fc1', padding='valid')(processed_a)
    feat_1 = MaxPooling2D(pool_size=(10, 10))(feat_1)
    feat_2 = Convolution2D(64, (3, 3), activation='relu', name='fc11', padding='valid')(processed_b)
    feat_2 = MaxPooling2D(pool_size=(10, 10))(feat_2)

    # output_shape=eucl_dist_output_shape)([processed_a, processed_b])
    # Concatenate
    l2distance = Lambda(euclidean_distance,
                        output_shape=eucl_dist_output_shape)([feat_1, feat_2])
    u4 = UpSampling2D(size=(100, 100))(l2distance)
    predictions = Convolution2D(2, (1, 1))(u4)
    #logits = Activation('sigmoid')(u4)
    #logits_T = Lambda(lambda x: x / temp)(logits)
   # probs_T = Activation('sigmoid')(logits_T)
    #predictions = concatenate([logits, probs_T])
    student = Model([image_a, image_b], predictions)
    student.summary()
    student.compile(optimizer='adam',
                    loss=lambda y_true, y_pred: knowledge_distillation_loss_withBE(y_true, y_pred, lambda_const),
                    metrics=[accuracy]
                    )
    return student


# Resuidal block BN -> relu -> conv -> bn -> relu -> conv
def res_block(x, filters):
    bn1 = BatchNormalization()(x)
    act1 = Activation('relu')(bn1)
    conv1 = Convolution2D(filters, (3, 3), data_format='channels_last', strides=(2, 2), padding='same',
                   kernel_initializer=glorot_uniform(seed=0))(act1)
    # print('conv1.shape', conv1.shape)
    bn2 = BatchNormalization()(conv1)
    act2 = Activation('relu')(bn2)
    conv2 = Convolution2D(filters, (3, 3), data_format='channels_last', strides=(1, 1), padding='same',
                   kernel_initializer=glorot_uniform(seed=0))(act2)
    # print('conv2.shape', conv2.shape)
    residual = Convolution2D(1, (1, 1), strides=(1, 1), data_format='channels_last')(conv2)

    x = Convolution2D(filters, (3, 3), data_format='channels_last', strides=(2, 2), padding='same',
               kernel_initializer=glorot_uniform(seed=0))(x)
    # print('x.shape', x.shape)
    out = Add()([x, residual])

    return out

def mini_ResNet_siamese_network():
    input1 = Input(shape=(100, 100, 3))
    res1 = res_block(input1, 64)
    res2 = res_block(res1, 128)
    res3 = res_block(res2, 256)
    res4 = res_block(res3, 512)
    # Classifier block
    act1 = Activation('relu')(res4)
    # flatten1 = Flatten()(act1)
    # dense1 = Dense(512)(flatten1)
    cnn = Model(inputs=input1, outputs=act1)

    image_a = Input(shape=(100, 100, 3))
    image_b = Input(shape=(100, 100, 3))
    processed_a = cnn(image_a)
    processed_b = cnn(image_b)
    feat_1 = Convolution2D(64, (3, 3), activation='relu', name='fc1', padding='valid')(processed_a)
    feat_1 = MaxPooling2D(pool_size=(5, 5))(feat_1)
    feat_2 = Convolution2D(64, (3, 3), activation='relu', name='fc11', padding='valid')(processed_b)
    feat_2 = MaxPooling2D(pool_size=(5, 5))(feat_2)
    l2distance = Lambda(euclidean_distance,
                        output_shape=eucl_dist_output_shape)([feat_1, feat_2])
    u4 = UpSampling2D(size=(100, 100))(l2distance)
    u4 = Convolution2D(1, (1, 1))(u4)
    logits = Activation('sigmoid')(u4)
    student = Model([image_a, image_b], logits)
    student.summary()
    student.compile(optimizer=Adam(lr=1e-4),
                    loss=['binary_crossentropy'], metrics=['accuracy'])
    return student

def KD_mini_ResNet_siamese_network(lambda_const):
    input1 = Input(shape=(100, 100, 3))
    res1 = res_block(input1, 64)
    res2 = res_block(res1, 128)
    res3 = res_block(res2, 256)
    res4 = res_block(res3, 512)
    # Classifier block
    act1 = Activation('relu')(res4)
    # flatten1 = Flatten()(act1)
    # dense1 = Dense(512)(flatten1)
    cnn = Model(inputs=input1, outputs=res4)
    image_a = Input(shape=(100, 100, 3))
    image_b = Input(shape=(100, 100, 3))
    processed_a = cnn(image_a)
    processed_b = cnn(image_b)
    feat_1 = Convolution2D(64, (3, 3), activation='relu', name='fc1', padding='valid')(processed_a)
    feat_1 = MaxPooling2D(pool_size=(5, 5))(feat_1)
    feat_2 = Convolution2D(64, (3, 3), activation='relu', name='fc11', padding='valid')(processed_b)
    feat_2 = MaxPooling2D(pool_size=(5, 5))(feat_2)
    l2distance = Lambda(euclidean_distance,
                        output_shape=eucl_dist_output_shape)([feat_1, feat_2])
    u4 = UpSampling2D(size=(100, 100))(l2distance)
    u4 = Convolution2D(2, (1, 1))(u4)
    logits = Activation('sigmoid')(u4)
    #logits_T = Lambda(lambda x: x / temp)(logits)
    #probs_T = Activation('sigmoid')(logits_T)
    #predictions = concatenate([logits, probs_T])
    student = Model([image_a, image_b], logits)
    student.summary()
    student.compile(optimizer='adam',
                    loss=lambda y_true, y_pred: knowledge_distillation_loss_withBE(y_true, y_pred, lambda_const),
                    metrics=[accuracy, categorical_crossentropy]
                    )
    return student
