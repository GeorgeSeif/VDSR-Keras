#!/usr/bin/env python

from __future__ import print_function
import keras
from keras import backend as K
from keras.models import Sequential, Model
from keras.layers import Dense, Activation
from keras.layers import Conv2D, MaxPooling2D, Input, Merge, ZeroPadding2D, merge, add
import tensorflow as tf
from keras.models import load_model
from keras import optimizers
from keras import losses
from keras.optimizers import SGD, Adam
from keras.callbacks import ModelCheckpoint

import os, glob, sys, threading
import scipy.io
from scipy import ndimage, misc
import numpy as np
import re
import math

DATA_PATH = "./data/train/"
IMG_SIZE = (41, 41, 1)
BATCH_SIZE = 64
EPOCHS = 100
TRAIN_SCALES = [2, 3, 4]
VALID_SCALES = [4]

def tf_log10(x):
  numerator = tf.log(x)
  denominator = tf.log(tf.constant(10, dtype=numerator.dtype))
  return numerator / denominator

def load_images(directory):
	images = []
	for root, dirnames, filenames in os.walk(directory):
	    for filename in filenames:
	        if re.search("\.(jpg|jpeg|png|bmp|tiff)$", filename):
	            filepath = os.path.join(root, filename)
	            image = ndimage.imread(filepath, mode="L")
	            images.append(image)
	            
	images = np.array(images)
	array_shape = np.append(images.shape[0:3], 1)
	images = np.reshape(images, (array_shape))

	return images

def get_image_list(data_path, scales=[2, 3, 4]):
	l = glob.glob(os.path.join(data_path,"*"))
	# print(len(l))
	l = [f for f in l if re.search("^\d+.mat$", os.path.basename(f))]
	# print(len(l))
	train_list = []
	for f in l:
		if os.path.exists(f):	
			for i in range(len(scales)):
				scale = scales[i]
				string_scale = "_" + str(scale) + ".mat"
				if os.path.exists(f[:-4]+string_scale): train_list.append([f, f[:-4]+string_scale])
	return train_list

def get_image_batch(train_list, offset):
	target_list = train_list[offset:offset+BATCH_SIZE]
	input_list = []
	gt_list = []
	cbcr_list = []
	for pair in target_list:
		input_img = scipy.io.loadmat(pair[1])['patch']
		gt_img = scipy.io.loadmat(pair[0])['patch']
		input_list.append(input_img)
		gt_list.append(gt_img)
	input_list = np.array(input_list)
	input_list.resize([BATCH_SIZE, IMG_SIZE[0], IMG_SIZE[1], 1])
	gt_list = np.array(gt_list)
	gt_list.resize([BATCH_SIZE, IMG_SIZE[0], IMG_SIZE[1], 1])
	return input_list, gt_list

class threadsafe_iter:
    """Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    """
    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def next(self):
        with self.lock:
            return self.it.next()


def threadsafe_generator(f):
    """A decorator that takes a generator function and makes it thread-safe.
    """
    def g(*a, **kw):
        return threadsafe_iter(f(*a, **kw))
    return g

@threadsafe_generator
def image_gen(target_list):
	while True:
		for step in range(len(target_list)//BATCH_SIZE):
			offset = step*BATCH_SIZE
			batch_x, batch_y = get_image_batch(target_list, offset)
			yield (batch_x, batch_y)


def PSNR(y_true, y_pred):
	max_pixel = 1.0
	return 10.0 * tf_log10((max_pixel ** 2) / (K.mean(K.square(y_pred - y_true)))) 



# Get the training and testing data
train_list = get_image_list("./data/train/", scales=TRAIN_SCALES)

test_list = get_image_list("./data/test/Set5/", scales=VALID_SCALES)


input_img = Input(shape=IMG_SIZE)

model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(input_img)
model_0 = Activation('relu')(model)

for _ in range(19):  # residual block
	model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model_0)
	model = Activation('relu')(model)
	model_0 = add()([model, model_0])

model = Conv2D(1, (3, 3), padding='same', kernel_initializer='he_normal')(model)

res_img = model
output_img = add([res_img, input_img])

model = Model(input_img, output_img)

# model.load_weights('./checkpoints/weights-improvement-20-26.93.hdf5')

adam = Adam(lr=0.00001)
sgd = SGD(lr=1e-5, momentum=0.9, decay=1e-4, nesterov=False)
model.compile(adam, loss='mse', metrics=[PSNR, "accuracy"])

model.summary()

filepath="./checkpoints/weights-improvement-{epoch:02d}-{PSNR:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor=PSNR, verbose=1, mode='max')
callbacks_list = [checkpoint]

model.fit_generator(image_gen(train_list), steps_per_epoch=len(train_list) // BATCH_SIZE,  \
					validation_data=image_gen(test_list), validation_steps=len(train_list) // BATCH_SIZE, \
					epochs=EPOCHS, workers=8, callbacks=callbacks_list)

print("Done training!!!")

print("Saving the final model ...")

model.save('vdsr_model.h5')  # creates a HDF5 file 
del model  # deletes the existing model