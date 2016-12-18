import numpy as np
import cv2
import csv
import json
import os.path
import pickle
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Activation, Convolution2D, Flatten, Lambda, Dropout, BatchNormalization, ZeroPadding2D, MaxPooling2D
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping
import tensorflow as tf
import sys
from scipy.misc import imresize

# config
nb_epoch = 50
lr = 0.000005 #0.0001
dropout = 0.5
csvpath='saved-model-1/driving_log-1.1.csv'
image_folder='saved-model-1/IMG-1'
from_json = False

# Load data

driving_log = []
num_images = 0
with open(csvpath,'r') as f:
	datareader = csv.reader(f,delimiter=',')
	for row in datareader:
		driving_log.append(row)
		num_images += 1

# override num images
num_images = 10 # no recovery driving


# use csv data to set X to the images and y to the steering angles
# for labels y_train, this is by initializing an array of the right length and updating values
# for feature data X_train, this is by contatenating images and reshaping into an array of images

y_train = np.zeros(num_images)

for i,row in enumerate(driving_log):
	filepath = image_folder + '/' + driving_log[i][0].rsplit('/')[-1]
	image = cv2.imread(filepath)
	image = imresize(image, (32,64,3))[12:,:,:]
	if(i % 100) == 0:
		print('Images read: ',i)
	if(i==0):
		images_concatenated = image
	elif(i<num_images):
		images_concatenated = np.concatenate((images_concatenated,image), axis=0)
	else:
		break
	y_train[i] = driving_log[i][3]

images_concatenated = np.concatenate((images_concatenated,np.fliplr(images_concatenated)), axis=0)
X_train = images_concatenated.reshape(-1,20,64,3)
print(X_train.shape)
sys.exit
y_train = np.concatenate((y_train,-1*y_train), axis=0)

# normalize

def normalize(X):
    X = X - np.mean(X)
    a,b,xmin,xmax = -0.5,0.5,np.min(X),np.max(X)
    return a+(X-xmin)*(b-a)/(xmax-xmin)

X_train = normalize(X_train)

# shuffle

def shuffle_in_unison(a, b):
    assert len(a) == len(b)
    shuffled_a = np.empty(a.shape, dtype=a.dtype)
    shuffled_b = np.empty(b.shape, dtype=b.dtype)
    permutation = np.random.permutation(len(a))
    for old_index, new_index in enumerate(permutation):
        shuffled_a[new_index] = a[old_index]
        shuffled_b[new_index] = b[old_index]
    return shuffled_a, shuffled_b

X_train,y_train = shuffle_in_unison(X_train, y_train)

# model

img_shape = (20,64,3)

if(from_json):
	with open('model.json', 'r') as jfile:
		model = model_from_json(json.load(jfile))
	model.load_weights('model.h5')

else:
	model = Sequential()
	model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_1', input_shape=(20,64,3)))
	model.add(Dropout(dropout))
	model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_2'))
	model.add(Flatten())
	model.add(Dropout(dropout))
	model.add(Dense(100, activation='relu'))
	model.add(Dense(50, activation='relu'))
	model.add(Dense(10, activation='relu'))
	model.add(Dropout(dropout))
	model.add(Dense(1,activation='tanh'))
	model.summary()

my_adam = Adam(lr=lr)
model.compile(optimizer=my_adam,loss='mse')

# Model will save the weights whenever validation loss improves
checkpoint = ModelCheckpoint(filepath = 'model.h5', verbose = 1, save_best_only=True, monitor='val_loss')

# Discontinue training when validation loss fails to decrease
callback = EarlyStopping(monitor='val_loss', patience=2, verbose=1)

model.fit(X_train, y_train, nb_epoch=nb_epoch,validation_split=0.1, shuffle=True, callbacks=[checkpoint, callback])


predictions = model.predict(X_train[0:200])
print('predictions: ', predictions)

# save model

json_string = model.to_json()
with open('model.json','w') as f:
	json.dump(json_string,f,ensure_ascii=False)
# model.save_weights('model.h5') # checkpointing weights
