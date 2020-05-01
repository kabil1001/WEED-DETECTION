# -*- coding: utf-8 -*-
"""
Created on Sat May  2 01:10:38 2020

@author: kxj133
"""
from keras import models
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
from PIL import Image
from skimage import transform


test_dir = "C:/Users/kxj133/Desktop/image_classification/segmented_image/test"

classifier = models.load_model("model saved location")

test_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(150, 150),
        color_mode="rgb",
        batch_size=31,
        class_mode='categorical')

test_samples = sum(len(files) for _, _, files in os.walk(test_dir))

batch_size_test=31

score= classifier.evaluate_generator(test_generator, 
                                steps = np.ceil(test_samples/batch_size_test))

print("\nTest accuracy for the CNN classifier : %.1f%%" % (100.0 * score[1]))



#### PREDICTING FOR SINGLE IMAGE

## IMAGE PRE PROCESSINF FUNCTION TO LOAD THE IMAGE DATA IN THE MODEL

def load(filename):
   np_image = Image.open(filename) #Open the image
   np_image = np.array(np_image).astype('float32')/255 
   np_image = transform.resize(np_image, (150, 150, 3)) 
   np_image = np.expand_dims(np_image, axis=0)
   return np_image

label_map = (test_generator.class_indices)

print (label_map)

image_dir = "C:/Users/kxj133/Desktop/image_classification/segmented_image/test/weed/1014.tif"
image_to_predict = load(image_dir)
result = classifier.predict(image_to_predict)
result= np.around(result,decimals=3)
result=result*100

print (result)