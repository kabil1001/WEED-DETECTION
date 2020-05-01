# -*- coding: utf-8 -*-
"""
Created on Sat May  2 00:06:48 2020

@author: kxj133
"""

import os
import numpy as np
import keras
keras.__version__
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense, Activation
from keras.layers.convolutional import Convolution2D, MaxPooling2D
import matplotlib.pyplot as plt

train_dir = "C:/Users/kxj133/Desktop/image_classification/segmented_image/train"
test_dir = "C:/Users/kxj133/Desktop/image_classification/segmented_image/test"
validation_dir  = "C:/Users/kxj133/Desktop/image_classification/segmented_image/validation"

"""Parameters"""

img_width, img_height = 150, 150
nb_filters1 = 32
nb_filters2 = 64
nb_filters3 = 128
nb_filters4 = 128

conv1_size = 3
conv2_size = 2
conv3_size = 2
conv4_size = 2
pool_size = 2
classes_num = 4
lr = 0.0004
dropout_value = 0.5

model = Sequential()

model.add(Convolution2D(nb_filters1, conv1_size, 
                        conv1_size, border_mode ="same",
                        input_shape=(img_width, img_height, 3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))

model.add(Convolution2D(nb_filters2, conv2_size, conv2_size, border_mode ="same"))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(pool_size, pool_size), dim_ordering='th'))

model.add(Convolution2D(nb_filters3, conv3_size, conv3_size, border_mode ="same"))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(pool_size, pool_size), dim_ordering='th'))

model.add(Convolution2D(nb_filters4, conv4_size, conv4_size, border_mode ="same"))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(pool_size, pool_size), dim_ordering='th'))

model.add(Flatten())

model.add(Dropout(0.5))

model.add(Dense(512))

model.add(Activation("relu"))

model.add(Dense(classes_num, activation='softmax'))

model.compile(loss='categorical_crossentropy',optimizer=optimizers.RMSprop(lr=lr),
              metrics=['accuracy'])

#ImageDataGenerator generates batches of tensor image data with real-time data augmentation. The data will be looped over (in batches).
#Rather than performing the operations on your memory, the API is designed to be iterated by the deep learning model fitting process, 
#creating augmented image data for you just-in-time.

train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(150, 150),
        color_mode="rgb",
        batch_size=92,
        class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size=(150, 150),
        color_mode="rgb",
        batch_size=31,
        class_mode='categorical')

"""
log_dir = './tf-log/'
tb_cb = callbacks.TensorBoard(log_dir=log_dir, histogram_freq=0)
cbks = [tb_cb]"""

training_samples = sum(len(files) for _, _, files in os.walk(train_dir))
batch_size_training_generator=92
validation_samples =sum(len(files) for _, _, files in os.walk(validation_dir))
batch_size_validation_generator=31

model_verbosity = model.fit_generator(
      train_generator,
      steps_per_epoch=np.ceil(training_samples/batch_size_training_generator),
      epochs=15,
      validation_data=validation_generator,
      validation_steps=np.ceil(validation_samples/batch_size_validation_generator))


### SAVING THE FINAL MODEL IN THE LOCAL FILE
model.save('final_model/model_weedcrops.h5')

acc = model_verbosity.history['acc']
val_acc = model_verbosity.history['val_acc']
loss = model_verbosity.history['loss']
val_loss = model_verbosity.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, ',', label='Training accuracy',color = 'teal')
plt.plot(epochs, val_acc, '.', label='Validation accuracy',color = 'teal')
plt.title('Training and validation accuracy',color = 'black')
plt.xlabel("Number Of Epochs")
plt.ylabel("Model Accuracy")
plt.legend()
plt.savefig("model_charts\training_validation_accuracy.jpeg")

plt.figure()

plt.plot(epochs, loss, ',', label='Training loss',color = 'teal')
plt.plot(epochs, val_loss, '.', label='Validation loss',color = 'teal')
plt.title('Training and validation loss',color = 'black')
plt.xlabel("Number Of Epochs")
plt.ylabel("Number of Epochs")
plt.legend()
plt.savefig("model_charts\training_validation_loss.jpeg")

plt.show()