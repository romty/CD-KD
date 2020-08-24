import numpy as np 

from tensorflow.keras.preprocessing import image
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications.resnet50 import preprocess_input
import data_generator as dg

##############################################################
from sklearn.utils import shuffle
import pandas

def make_train_pair(start, end):
    pairs = []
    for i in range(start, end):
        pairs.append([('LEVIR-CD/train/A/train_' + str(i) + '.png'), ('LEVIR-CD/train/B/train_' + str(i) + '.png'), ('LEVIR-CD/train/label/train_' + str(i) + '.png')])
    return pairs
def make_val_pair(start, end):
    pairs = []
    for i in range(start, end):
        pairs.append([('LEVIR-CD/val/A/val_' + str(i) + '.png'), ('LEVIR-CD/val/B/val_' + str(i ) + '.png'), ('LEVIR-CD/val/label/val_' + str(i) + '.png')])
    return pairs
import matplotlib.pyplot as plt
from random import shuffle
train_pair = make_train_pair(1, 445)
val_pair = make_val_pair(1, 64)

#############################################################
from random import sample, choice
temp = choice(train_pair)
print (temp[0], "GAB", temp[1])
tempx = choice(val_pair)
print (tempx[0], "GAB", tempx[1])
img = image.load_img(tempx[2], target_size=(100, 100), color_mode="grayscale")

plt.figure(figsize=(10,10))
plt.subplot(121)
plt.imshow(img, cmap='jet')
#plt.show()

##########################################################################
class_map=2
train_generator = dg.DataGenerator(train_pair,class_map,batch_size=10, dim=(100,100,3) ,shuffle=True)
train_steps = train_generator.__len__()
print ('train_steps:', train_steps)

val_generator = dg.DataGenerator(val_pair, class_map, batch_size=10, dim=(100,100,3) ,shuffle=True)
val_steps = val_generator.__len__()
print ('val_steps:', val_steps)
############################################################################
import teachers
small_teacher_model = teachers.small_FCN_siamese()
small_teacher_model.fit_generator(train_generator , steps_per_epoch=train_steps ,epochs=2,validation_data=val_generator,validation_steps=val_steps)
small_teacher_model.save_weights('small_teacher_model.h5')
##########################################################################
large_teacher_model = teachers.Large_FCN_siamese()
large_teacher_model.fit_generator(train_generator , steps_per_epoch=train_steps ,epochs=2,validation_data=val_generator,validation_steps=val_steps)
large_teacher_model.save_weights('large_teacher_model.h5')