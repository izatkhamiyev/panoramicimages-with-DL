#Dias Issa, CNN approach for image stitching

'''
truth_gray = (truth[0:7000,:,:,0]*0.299 + truth[0:7000,:,:,1]*0.587 + truth[0:7000,:,:,2]*0.114)
np.save('drive/My Drive/Colab Notebooks/Panoramic Dataset/stitched_imgs_gray.npy', truth_gray)

truth_gray = (truth[7000:,:,:,0]*0.299 + truth[7000:,:,:,1]*0.587 + truth[7000:,:,:,2]*0.114)
np.save('drive/My Drive/Colab Notebooks/Panoramic Dataset/stitched_imgs_test_gray.npy', truth_gray)

left_img_gray = (left_img[0:7000,:,:,0]*0.299 + left_img[0:7000,:,:,1]*0.587 + left_img[0:7000,:,:,2]*0.114)
np.save('drive/My Drive/Colab Notebooks/Panoramic Dataset/left_img_gray.npy', left_img_gray)

left_img_gray = (left_img[7000:,:,:,0]*0.299 + left_img[7000:,:,:,1]*0.587 + left_img[7000:,:,:,2]*0.114)
np.save('drive/My Drive/Colab Notebooks/Panoramic Dataset/left_img_test_gray.npy', left_img_gray)

right_img_gray = (right_img[0:7000,:,:,0]*0.299 + right_img[0:7000,:,:,1]*0.587 + right_img[0:7000,:,:,2]*0.114)
np.save('drive/My Drive/Colab Notebooks/Panoramic Dataset/right_img_gray.npy', right_img_gray)

right_img_gray = (right_img[7000:,:,:,0]*0.299 + right_img[7000:,:,:,1]*0.587 + right_img[7000:,:,:,2]*0.114)
np.save('drive/My Drive/Colab Notebooks/Panoramic Dataset/right_img_test_gray.npy', right_img_gray)
'''
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from matplotlib.pyplot import specgram
import keras


from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding, Activation
from keras.layers import LSTM
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Input, Flatten, Dropout#, Activation
from keras.layers import Conv1D, MaxPooling1D, AveragePooling1D
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import confusion_matrix

from PIL import Image
import sys
import numpy
import matplotlib.pyplot as plt
import numpy as np
import os

from keras import regularizers

import os
import pandas as pd

#PATH to directory
PATH = "./dataset/"

truth_ts = np.load(PATH + 'stitched_imgs_test_gray.npy')
truth_tr = np.load(PATH + 'stitched_imgs_gray.npy')

left_img_ts = np.load(PATH + 'left_img_test_gray.npy')
left_img_tr = np.load(PATH + 'left_img_gray.npy')


right_img_ts = np.load(PATH + 'right_img_test_gray.npy')
right_img_tr = np.load(PATH + 'right_img_gray.npy')

train_arr = np.concatenate((left_img_tr, right_img_tr), axis=2)

test_arr =  np.concatenate((left_img_ts, right_img_ts), axis=2)

y_train = truth_tr

y_test = truth_ts

del right_img_tr
del right_img_ts
del left_img_tr
del left_img_ts

import gc
gc.collect()

img = Image.fromarray(train_arr[1])
imgplot = plt.imshow(img)
plt.show()

train_arr.shape

for i in range(0,10):
  img = Image.fromarray(train_arr[i])
  imgplot = plt.imshow(img)
  plt.show()

print(train_arr.shape)

train_arr = np.expand_dims(train_arr, axis = 3)
y_train = np.expand_dims(y_train, axis = 3)

test_arr = np.expand_dims(test_arr, axis = 3)
y_test = np.expand_dims(y_test, axis = 3)

print(train_arr.shape, test_arr.shape)
print(y_train.shape, y_test.shape)

from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Conv2DTranspose, Reshape, MaxPooling2D

#mp1 = MaxPooling2D(pool_size=(2, 2))
#mp2 = MaxPooling2D(pool_size=(2, 2))
model = Sequential()
model.add(Conv2D(32, kernel_size=3,
                 activation='relu',
                 input_shape=(256, 512, 1)))

model.add(Conv2D(64, kernel_size=3, activation='relu'))

#model.add(mp1)
model.add(Conv2D(128, kernel_size=3, activation='relu'))
model.add(Conv2D(256, kernel_size=3, activation='relu'))
#model.add(mp2)

#model.add(DePool2D(mp2, size=(2,2)))
model.add(Conv2DTranspose(128, kernel_size=3, activation='relu'))
model.add(Conv2DTranspose(64, kernel_size=3, activation='relu'))
#model.add(DePool2D(mp1, size=(2,2)))
model.add(Conv2DTranspose(32, kernel_size=3, activation='relu'))

model.add(Conv2DTranspose(1, kernel_size=3, activation='relu'))

model.summary();

opt = keras.optimizers.SGD(lr=0.001, decay=1e-6, nesterov=True)
opt1 = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

model.compile(loss='mean_squared_error', optimizer=opt1, metrics=['accuracy'])

with tf.device('/gpu:0'):
  cnnhistory = model.fit(train_arr, y_train, batch_size=64, epochs=100, validation_data=(test_arr, y_test))

plt.plot(cnnhistory.history['loss'])
plt.plot(cnnhistory.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

model.compile(loss='mean_squared_error', optimizer=opt1, metrics=['accuracy'])
score = model.evaluate(test_arr, y_test, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))

model_name = 'imgPanoramicImgMatrix100.h5'
save_dir = os.path.join(PATH, 'saved_models')
# Save model and weights
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
model.save(model_path)
print('Saved trained model at %s ' % model_path)

import json
model_json = model.to_json()
with open(PATH + "saved_models/imgPanoramicImgMatrix100.json", "w") as json_file:
    json_file.write(model_json)

	'''
# loading json and creating model
from keras.models import model_from_json
json_file = open('saved_models/imgPanoramicImgMatrix100.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights(PATH + "saved_models/imgPanoramicImgMatrix100.h5")
print("Loaded model from disk")
 
# evaluate loaded model on test data
loaded_model.compile(loss='mean_squared_error', optimizer=opt1, metrics=['accuracy'])
score = loaded_model.evaluate(test_arr, y_test, verbose=0)
print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))
'''
livepreds = model.predict(test_arr, 
                         batch_size=256, 
                         verbose=1)

print(livepreds.shape)

from PIL import Image
import numpy as np

for i in y_test[0:5]:
  pred_img = np.squeeze(i, axis=2)
  print(pred_img.shape)
  img = Image.fromarray(pred_img)
  imgplot = plt.imshow(img)
  plt.show()

from PIL import Image
import numpy as np

for i in livepreds[0:5]:
  pred_img = np.squeeze(i, axis=2)
  print(pred_img.shape)
  img = Image.fromarray(pred_img)
  imgplot = plt.imshow(img)
  plt.show()