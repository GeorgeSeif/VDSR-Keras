from keras.models import load_model
from keras.models import Sequential, Model
from keras.layers import Dense, Activation
from keras.layers import Conv2D, MaxPooling2D, Input, Merge, ZeroPadding2D, merge
from keras.preprocessing import image
from scipy.misc import imsave, imread, imresize, toimage
import numpy as np
import matplotlib.pyplot as plt

img_shape = (41, 41, 1)

input_img = Input(shape=(img_shape))

model = Conv2D(64, (3, 3), padding='same', name='conv1')(input_img)
model = Activation('relu', name='act1')(model)
model = Conv2D(64, (3, 3), padding='same', name='conv2')(model)
model = Activation('relu', name='act2')(model)
model = Conv2D(64, (3, 3), padding='same', name='conv3')(model)
model = Activation('relu', name='act3')(model)
model = Conv2D(64, (3, 3), padding='same', name='conv4')(model)
model = Activation('relu', name='act4')(model)
model = Conv2D(64, (3, 3), padding='same', name='conv5')(model)
model = Activation('relu', name='act5')(model)

model = Conv2D(64, (3, 3), padding='same', name='conv6')(model)
model = Activation('relu', name='act6')(model)
model = Conv2D(64, (3, 3), padding='same', name='conv7')(model)
model = Activation('relu', name='act7')(model)
model = Conv2D(64, (3, 3), padding='same', name='conv8')(model)
model = Activation('relu', name='act8')(model)
model = Conv2D(64, (3, 3), padding='same', name='conv9')(model)
model = Activation('relu', name='act9')(model)
model = Conv2D(64, (3, 3), padding='same', name='conv10')(model)
model = Activation('relu', name='act10')(model)

model = Conv2D(64, (3, 3), padding='same', name='conv11')(model)
model = Activation('relu', name='act11')(model)
model = Conv2D(64, (3, 3), padding='same', name='conv12')(model)
model = Activation('relu', name='act12')(model)
model = Conv2D(64, (3, 3), padding='same', name='conv13')(model)
model = Activation('relu', name='act13')(model)
model = Conv2D(64, (3, 3), padding='same', name='conv14')(model)
model = Activation('relu', name='act14')(model)
model = Conv2D(64, (3, 3), padding='same', name='conv15')(model)
model = Activation('relu', name='act15')(model)

model = Conv2D(64, (3, 3), padding='same', name='conv16')(model)
model = Activation('relu', name='act16')(model)
model = Conv2D(64, (3, 3), padding='same', name='conv17')(model)
model = Activation('relu', name='act17')(model)
model = Conv2D(64, (3, 3), padding='same', name='conv18')(model)
model = Activation('relu', name='act18')(model)
model = Conv2D(64, (3, 3), padding='same', name='conv19')(model)
model = Activation('relu', name='act19')(model)
model = Conv2D(1, (3, 3), padding='same', name='conv20')(model)
model = Activation('relu', name='act20')(model)
res_img = model

output_img = merge([res_img, input_img])

model = Model(input_img, output_img)

model.load_weights('vdsr_model_edges.h5')

img = image.load_img('./patch.png', grayscale=True, target_size=(41, 41, 1))
x = image.img_to_array(img)
x = x.astype('float32') / 255
x = np.expand_dims(x, axis=0)

pred = model.predict(x)

test_img = np.reshape(pred, (41, 41))

imsave('test_img.png', test_img)