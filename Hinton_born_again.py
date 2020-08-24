import numpy as np

from tensorflow.keras.preprocessing import image
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications.resnet50 import preprocess_input
import data_generator as dg
from tensorflow.keras.models import Model
##############################################################
from sklearn.utils import shuffle
import pandas

temp = 3
lambda_const = 0.9

import ABCD_db
from random import shuffle

train_pair = ABCD_db.make_train_pair(1, 446)
val_pair = ABCD_db.make_val_pair(1, 65)
##########################################################################
train_x_pre, train_x_post, train_label = ABCD_db.load_ABCD(train_pair)
val_x_pre, val_x_post, val_label = ABCD_db.load_ABCD(val_pair)

print (val_x_pre.shape, val_x_post.shape, val_label.shape)
#####################################################################################################
import teachers

teacher_model = teachers.small_FCN_siamese()
teacher_model.load_weights('small_teacher_model.h5')

print (" --------------- Finish loading Small Teacher ---------------")


##################################### EVAL orignial student #########################################
# loss, accuracy = teacher_model.evaluate([train_x_pre, train_x_post], train_label)
# print('teacher  training accuracy:', loss, accuracy)
# loss, accuracy = teacher_model.evaluate([test_x_pre, test_x_post], test_label)
# print('teacher  testing accuracy', loss, accuracy)
###########################################################################
def sigmoid(x, derivative=False):
    return x * (1 - x) if derivative else 1 / (1 + np.exp(-x))


###########################################################################
teacher_WO_Softmax = Model(teacher_model.input, teacher_model.layers[-1].output)
teacher_train_logits = teacher_WO_Softmax.predict([train_x_pre, train_x_post])
teacher_val_logits = teacher_WO_Softmax.predict([val_x_pre, val_x_post])

############################################################################

Y_train_soft = sigmoid(teacher_train_logits / temp)

Y_val_soft = sigmoid(teacher_val_logits / temp)

# y_valid = np.vstack((y_valid, y_knowledge_valid)).T
Y_train_new = np.concatenate([train_label, Y_train_soft], axis=-1)

Y_val_new = np.concatenate([val_label, Y_val_soft], axis=-1)
print ('Y_train_new', Y_train_new.shape, 'Y_val_new', Y_val_new.shape)
print (" --------------- collect soft logist ---------------")
#######################################orignial student #######################################

#######################################orignial small student #######################################
import students

print (" --------------- Small orignial student loading ---------------")
# vgg_student_model = students.mini_vgg_siamese_network()
# vgg_student_model.fit([train_x_pre, train_x_post], train_label, batch_size=10, epochs=1,
#     validation_data = ([val_x_pre, val_x_post], val_label), verbose = 1)
##################################### EVAL orignial student #########################################
#####################################################################################################
# loss, accuracy = student_model.evaluate([train_x_pre, train_x_post], train_label)
# print('orignial student training accuracy:', loss, accuracy)
# loss, accuracy = student_model.evaluate([test_x_pre, test_x_post], test_label)
# print('orignial student testing accuracy', loss, accuracy)
#####################################################################################################
print (" --------------- Large orignial student loading ---------------")
# resnet_student_model = students.mini_ResNet_siamese_network()
# resnet_student_model.fit([train_x_pre, train_x_post], train_label, batch_size=10, epochs=1,
#     validation_data = ([val_x_pre, val_x_post], val_label), verbose = 1)

#######################################kd student #######################################

# print (" ---------------Hinton -Born Again- small student loading ---------------")
# vgg_kd_student_model = students.KD_mini_vgg_siamese_network(lambda_const)
# print (train_x_pre.shape, train_x_post.shape, Y_train_new.shape)
# print (val_x_pre.shape, val_x_post.shape, Y_val_new.shape)
# vgg_kd_student_model.fit([train_x_pre, train_x_post], Y_train_new, batch_size=10, epochs=1,
#                          validation_data=([val_x_pre, val_x_post], Y_val_new), verbose=1)
##################################### EVAL kd student #########################################	
# print (teacher_train_logits.shape, teacher_test_logits.shape)
# loss, accuracy = kd_student_model.evaluate([train_x_pre, train_x_post], Y_train_new)
# print('KD student training accuracy:', loss, accuracy)
# loss, accuracy = kd_student_model.evaluate([test_x_pre, test_x_post], Y_test_new)
# print('KD student testing accuracy', loss, accuracy)

############################################################################################
print (" ---------------Hinton -Born Again- large student loading ---------------")
resnet_kd_student_model = students.KD_mini_ResNet_siamese_network(lambda_const)
resnet_kd_student_model.fit([train_x_pre, train_x_post], Y_train_new, batch_size=10, epochs=1,
                     validation_data=([val_x_pre, val_x_post], Y_val_new), verbose=1)
