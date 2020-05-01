# -*- coding: utf-8 -*-
"""
Created on Fri May  1 22:43:43 2020

@author: kxj133
"""

import os
import shutil
import numpy as np

base_dir = 'C:/Users/kxj133/Desktop/image_classification/segmented_image'
os.makedirs(base_dir, exist_ok=True)

# Directorio para nuestro train, validation y test
train_dir = os.path.join(base_dir, 'train')
os.makedirs(train_dir, exist_ok=True)

validation_dir = os.path.join(base_dir, 'validation')
os.makedirs(validation_dir, exist_ok=True)

test_dir = os.path.join(base_dir, 'test')
os.makedirs(test_dir, exist_ok=True)

#Directorio para training de grass,soil,soybean, weed
train_grass_dir = os.path.join(train_dir, 'grass')
os.makedirs(train_grass_dir, exist_ok=True)

train_soil_dir = os.path.join(train_dir, 'soil')
os.makedirs(train_soil_dir, exist_ok=True)

train_soybean_dir = os.path.join(train_dir, 'soybean')
os.makedirs(train_soybean_dir, exist_ok=True)

train_weed_dir = os.path.join(train_dir, 'weed')
os.makedirs(train_weed_dir, exist_ok=True)

#Directorio para validation de grass,soil,soybean, weed
validation_grass_dir = os.path.join(validation_dir, 'grass')
os.makedirs(validation_grass_dir, exist_ok=True)

validation_soil_dir = os.path.join(validation_dir, 'soil')
os.makedirs(validation_soil_dir, exist_ok=True)

validation_soybean_dir = os.path.join(validation_dir, 'soybean')
os.makedirs(validation_soybean_dir, exist_ok=True)

validation_weed_dir = os.path.join(validation_dir, 'weed')
os.makedirs(validation_weed_dir, exist_ok=True)

#Directorio para test de grass,soil,soybean, weed
test_grass_dir = os.path.join(test_dir, 'grass')
os.makedirs(test_grass_dir, exist_ok=True)

test_soil_dir = os.path.join(test_dir, 'soil')
os.makedirs(test_soil_dir, exist_ok=True)

test_soybean_dir = os.path.join(test_dir, 'soybean')
os.makedirs(test_soybean_dir, exist_ok=True)

test_weed_dir = os.path.join(test_dir, 'weed')
os.makedirs(test_weed_dir, exist_ok=True)

## DEFINING THE DIRECTORY FOR THE ORIGINAL DIRECTORY

original_dir_weed ='C:/Users/kxj133/Desktop/image_classification/dataset/broadleaf'
original_dir_grass = 'C:/Users/kxj133/Desktop/image_classification/dataset/grass'
original_dir_soil = 'C:/Users/kxj133/Desktop/image_classification/dataset/soil'
original_dir_soybean = 'C:/Users/kxj133/Desktop/image_classification/dataset/soybean'

### SEGMENTING THE TRAIN, VALIDATION AND TEST IMAGES
### TRAINING DATESET WOULD HAVE 70 PERCENT OF DATASET AND 15 PERCENT FOR VALIDATION & TEST
soil_total_images = len(os.listdir(original_dir_soil))
weed_total_images = len(os.listdir(original_dir_weed))
grass_total_images = len(os.listdir(original_dir_grass))
soybean_total_images = len(os.listdir(original_dir_soybean))

### COPYING 70 PERCENT OF SOIL DATASET TO SOIL TRAIN FOLDER
fnames = ['{}.tif'.format(i) for i in range(1,int(np.ceil(soil_total_images*0.70)))]
for fname in fnames:
    src = os.path.join(original_dir_soil, fname)
    dst = os.path.join(train_soil_dir, fname)
    shutil.copyfile(src, dst)

### COPYING THE 15 PERCENT OF SOIL DATASET TO SOIL VALIDATION FOLDER
fnames= ['{}.tif'.format(i) for i in range(int(np.ceil(soil_total_images*0.70)), int(np.ceil(soil_total_images*0.85)))]
for fname in fnames:
    src = os.path.join(original_dir_soil, fname)
    dst = os.path.join(validation_soil_dir, fname)
    shutil.copyfile(src, dst)
    
# COPYING 15 PERCENT OF SOIL DATASET TO SOIL TEST FOLDER
fnames = ['{}.tif'.format(i) for i in range(int(np.ceil(soil_total_images*0.85)), int(np.ceil(soil_total_images)))]
for fname in fnames:
    src = os.path.join(original_dir_soil, fname)
    dst = os.path.join(test_soil_dir, fname)
    shutil.copyfile(src, dst)
    
####SOYBEAN####
### COPYING 70 PERCENT OF SOIL DATASET TO SOYBEAN TRAIN FOLDER
fnames = ['{}.tif'.format(i) for i in range(1,int(np.ceil(soybean_total_images*0.70)))]
for fname in fnames:
    src = os.path.join(original_dir_soybean, fname)
    dst = os.path.join(train_soybean_dir, fname)
    shutil.copyfile(src, dst)

### COPYING THE 15 PERCENT OF SOYBEAN DATASET TO SOYBEAN VALIDATION FOLDER
fnames =  ['{}.tif'.format(i) for i in range(int(np.ceil(soybean_total_images*0.70)), int(np.ceil(soybean_total_images*0.85)))]
for fname in fnames:
    src = os.path.join(original_dir_soybean, fname)
    dst = os.path.join(validation_soybean_dir, fname)
    shutil.copyfile(src, dst)
    
# COPYING 15 PERCENT OF SOYBEAN DATASET TO SOYBEAN TEST FOLDER
fnames =  ['{}.tif'.format(i) for i in range(int(np.ceil(soybean_total_images*0.85)), soybean_total_images)]
for fname in fnames:
    src = os.path.join(original_dir_soybean, fname)
    dst = os.path.join(test_soybean_dir, fname)
    shutil.copyfile(src, dst)
    
####GRASS####
### COPYING 70 PERCENT OF GRASS DATASET TO GRASS TRAIN FOLDER
fnames = ['{}.tif'.format(i) for i in range(1,int(np.ceil(grass_total_images*0.70)))]
for fname in fnames:
    src = os.path.join(original_dir_grass, fname)
    dst = os.path.join(train_grass_dir, fname)
    shutil.copyfile(src, dst)

### COPYING THE 15 PERCENT OF GRASS DATASET TO GRASS VALIDATION FOLDER
fnames = ['{}.tif'.format(i) for i in range(int(np.ceil(grass_total_images*0.70)), int(np.ceil(grass_total_images*0.85)))]
for fname in fnames:
    src = os.path.join(original_dir_grass, fname)
    dst = os.path.join(validation_grass_dir, fname)
    shutil.copyfile(src, dst)
    
### COPYING THE 15 PERCENT OF GRASS DATASET TO GRASS TEST FOLDER
fnames = ['{}.tif'.format(i) for i in range(int(np.ceil(grass_total_images*0.85)), int(np.ceil(grass_total_images)))]
for fname in fnames:
    src = os.path.join(original_dir_grass, fname)
    dst = os.path.join(test_grass_dir, fname)
    shutil.copyfile(src, dst)
    
####WEED####
### COPYING THE 70 PERCENT OF WEED DATASET TO WEED TRAIN FOLDER
fnames = ['{}.tif'.format(i) for i in range(1,int(np.ceil(weed_total_images*0.70)))]
for fname in fnames:
    src = os.path.join(original_dir_weed, fname)
    dst = os.path.join(train_weed_dir, fname)
    shutil.copyfile(src, dst)

### COPYING THE 15 PERCENT OF WEED DATASET TO WEED VALIDATION FOLDER
fnames = ['{}.tif'.format(i) for i in range(int(np.ceil(weed_total_images*0.70)), int(np.ceil(weed_total_images*0.85)))]
for fname in fnames:
    src = os.path.join(original_dir_weed, fname)
    dst = os.path.join(validation_weed_dir, fname)
    shutil.copyfile(src, dst)
    
# Copiamos las siguientes 238 a test_weed_dir
fnames = ['{}.tif'.format(i) for i in range(int(np.ceil(weed_total_images*0.85)), weed_total_images)]
for fname in fnames:
    src = os.path.join(original_dir_weed, fname)
    dst = os.path.join(test_weed_dir, fname)
    shutil.copyfile(src, dst)