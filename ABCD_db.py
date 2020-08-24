
import numpy as np 
from tensorflow.keras.preprocessing import image
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications.resnet50 import preprocess_input
from sklearn.utils import shuffle


##########################################################################    
def load_ABCD(train_split):
    batch_imgs_pre = list()
    batch_imgs_post = list()
    batch_labels = list()
    for row in train_split:
            #img1 = 'ABCD/' + train_split.iloc[row,0]
            #img2 = 'ABCD/' + train_split.iloc[row,1]
            #label = 0 if train_split.iloc[row,2] == 'N' else 1
            img = image.load_img(row[0], target_size=(100, 100))
            x = image.img_to_array(img)
            batch_imgs_pre.append(x)
            #
            img = image.load_img(row[1], target_size=(100, 100))
            x = image.img_to_array(img)
            batch_imgs_post.append(x)
            
            img = image.load_img(row[2], target_size=(100, 100), color_mode="grayscale")
            img = image.img_to_array(img)
            batch_labels.append(np.where(img==255, 1, img) )
    
    print('start collecting train- images')
    return  (np.array(batch_imgs_pre).astype("float32"))/255, (np.array(batch_imgs_post).astype("float32"))/255, np.array(batch_labels)
##########################################################################    
def make_train_pair(start, end):
    pairs = []
    for i in range(start, end):
        pairs.append(
            [('LEVIR-CD/train/A/train_' + str(i) + '.png'), ('LEVIR-CD/train/B/train_' + str(i) + '.png'),
             ('LEVIR-CD/train/label/train_' + str(i) + '.png')])
    return pairs


def make_val_pair(start, end):
    pairs = []
    for i in range(start, end):
        pairs.append([('LEVIR-CD/val/A/val_' + str(i) + '.png'), ('LEVIR-CD/val/B/val_' + str(i) + '.png'),
                      ('LEVIR-CD/val/label/val_' + str(i) + '.png')])
    return pairs


def make_test_pair(start, end):
    pairs = []
    for i in range(start, end):
        pairs.append([('LEVIR-CD/test/A/test_' + str(i) + '.png'), ('LEVIR-CD/test/B/test_' + str(i) + '.png'),
                      ('LEVIR-CD/test/label/test_' + str(i) + '.png')])
    return pairs

