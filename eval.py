
import ABCD_db
import numpy as np
from tensorflow.keras.preprocessing import image
from random import shuffle
import matplotlib.pyplot as plt
train_pair = ABCD_db.make_train_pair(1, 446)
val_pair = ABCD_db.make_val_pair(1, 65)
test_pair = ABCD_db.make_test_pair(1, 129)

##########################################################################
import teachers
print (" ---------------  loading Small Teacher ---------------")
small_teacher_model = teachers.small_FCN_siamese()
small_teacher_model.load_weights('small_teacher_model.h5')
print (" ---------------  loading large Teacher ---------------")
large_teacher_model = teachers.Large_FCN_siamese()
large_teacher_model.load_weights('large_teacher_model.h5')
###########################################################################
for row in test_pair:  # len(test_pair)
    img = image.load_img(row[0], target_size=(100, 100))
    batch_img_pre = image.img_to_array(img)
    batch_img_pre = np.expand_dims(batch_img_pre, 0)
    #
    img = image.load_img(row[1], target_size=(100, 100))
    batch_img_post = image.img_to_array(img)
    batch_img_post= np.expand_dims(batch_img_post, 0)
    img = image.load_img(row[2], target_size=(100, 100), color_mode="grayscale")
    cd_groundtruth = image.img_to_array(img)
    cd_groundtruth= (np.where(cd_groundtruth == 255, 1, cd_groundtruth))

    small_cd_predict= small_teacher_model.predict([batch_img_pre,batch_img_post ])
    large_cd_predict = large_teacher_model.predict([batch_img_pre,batch_img_post ])

    fig, axs = plt.subplots(2, 2, figsize=(9, 4.5), tight_layout=True)
    cd_groundtruth=np.squeeze(cd_groundtruth, axis=-1)
    axs[0,0].imshow(cd_groundtruth, interpolation='bicubic', cmap='Greys')
    axs[0,0].set_title('cd_groundtruth')
    small_mask= np.squeeze(small_cd_predict[0], axis=-1)
    large_mask = np.squeeze(large_cd_predict[0], axis=-1)
    axs[0,1].imshow(small_mask, interpolation='bicubic', cmap='Greys')
    axs[0,1].set_title('teacher_groundtruth')
    axs[1,0].imshow(large_mask, interpolation='bicubic', cmap='Greys')
    axs[1,0].set_title('large teacher_groundtruth')
    plt.show()