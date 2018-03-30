#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.

"""

import os
import cv2
import random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

#config part
#------------------------------------------
#data
#train_lable_path='/data6/zhangyu/base/Annotations/label.csv'
#train_image_path='/data6/zhangyu/base/'
train_lable_path='/data6/chendaqin/base/Annotations/label.csv'
train_image_path='/data6/chendaqin/base/'
image_size=299
#training
lr=1e-3
loss='categorical_crossentropy'
metrics=['accuracy']
batch_size=32
epochs=50
test_size=0.12
#model
save_model_path='./models/'
#------------------------------------------
import tensorflow as tf
os.environ["CUDA_VISIBLE_DEVICES"]="6,7"
tf_config=tf.ConfigProto()
tf_config.gpu_options.allow_growth=True
tf_config.gpu_options.per_process_gpu_memory_fraction=0.8

df_train = pd.read_csv(train_lable_path, header=None)
df_train.columns = ['image_id', 'class', 'label']

classes = ['collar_design_labels', 'neckline_design_labels', 'skirt_length_labels', 
           'sleeve_length_labels', 'neck_design_labels', 'coat_length_labels', 'lapel_design_labels', 
           'pant_length_labels']


cur_class = classes[2]#skirt_length_labels
df_load = df_train[(df_train['class'] == cur_class)].copy()#fetch the class wanted
df_load.reset_index(inplace=True)
del df_load['index']

print('{0}: {1}'.format(cur_class, len(df_load)))



n = len(df_load)
n_class = len(df_load['label'][0])

X = np.zeros((n, image_size, image_size, 3), dtype=np.uint8)
y = np.zeros((n, n_class), dtype=np.uint8)

for i in range(n):    
    tmp_label = df_load['label'][i]
    if len(tmp_label) > n_class:
        print(df_load['image_id'][i])
    X[i] = cv2.resize(cv2.imread(train_image_path+'{0}'.format(df_load['image_id'][i])), (image_size, image_size))
    y[i][tmp_label.find('y')] = 1
    

from keras.layers import *
from keras.models import *
from keras.callbacks import *
from keras.optimizers import *
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.regularizers import *
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.image import *

cnn_model = InceptionResNetV2(include_top=False, input_shape=(image_size, image_size, 3), weights='imagenet')
inputs = Input((image_size, image_size, 3))

x = inputs
x = Lambda(preprocess_input, name='preprocessing')(x)
x = cnn_model(x)
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
x = Dense(n_class, activation='softmax', name='softmax')(x)

model = Model(inputs, x)   

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=test_size, random_state=42)


adam = Adam(lr=lr)
prefix_cls = cur_class.split('_')[0]

model.compile(optimizer=adam,
              loss=loss,
              metrics=metrics)

checkpointer = ModelCheckpoint(filepath=save_model_path+'{0}.best.h5'.format(prefix_cls), verbose=1, 
                               save_best_only=True, period=2)



datagen = ImageDataGenerator(


    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

# compute quantities required for featurewise normalization
# (std, mean, and principal components if ZCA whitening is applied)
datagen.fit(X_train)

# fits the model on batches with real-time data augmentation:
model.fit_generator(datagen.flow(X_train, y_train, batch_size=batch_size),
                    steps_per_epoch=len(X_train)/batch_size, epochs=epochs,
                    validation_data=(X_valid,y_valid),
                    callbacks=[EarlyStopping(patience=3), checkpointer],
                    shuffle=True)

# here's a more "manual" example
#for e in range(epochs):
#    print('Epoch', e)
#    batches = 0
#    for x_batch, y_batch in datagen.flow(X_train, y_train, batch_size=batch_size):
#        loss = model.fit(x_batch, y_batch,validation_split=0.1)
#        batches += 1
#        if batches >= len(X_train) / batch_size:
            # we need to break the loop by hand because
            # the generator loops indefinitely
#            break
        
        
#h = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, 
 #             callbacks=[EarlyStopping(patience=3), checkpointer], 
  #            shuffle=True, 
   #           validation_split=0.1)


