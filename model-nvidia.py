import numpy as np
import cv2
import csv
import json
import os.path
import pickle
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Activation, Convolution2D, Flatten, Lambda, Dropout, BatchNormalization, ZeroPadding2D, MaxPooling2D
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, LambdaCallback
import tensorflow as tf
import sys
from scipy.misc import imresize
import threading

# config
nb_epoch = 10
samples_per_epoch = 4096
lr = 0.0001 #0.0001
dropout = 0.5
csvpath='Archive/driving_log-carnd.csv'
image_folder='Archive/IMG-carnd'
from_json = False
t_flip = 1.0 # threshold for flipping
t_angle = 0.2 # threshold of |angle| to keep, else discard...
t_keep = 0.5 # threshold to keep small angle
use_side_cameras = False

# Load driving log

driving_log = []
with open(csvpath,'r') as f:
	datareader = csv.reader(f,delimiter=',')
	for row in datareader:
		driving_log.append(row)

# override num images to use

def get_preprocessed_row(driving_log):
	i = np.random.randint(len(driving_log))
	rv_flip = np.random.uniform()
	if(use_side_cameras == True):
		camera = np.random.randint(3)
	else:
		camera = 0

	filepath = image_folder + '/' + driving_log[i][camera].rsplit('/')[-1]
	image = cv2.imread(filepath)
	image = imresize(image, (100,200,3))[34:,:,:]

	label = float(driving_log[i][3])
	if(camera == 1):
		label = label + 0.2
	elif(camera == 2):
		label = label - 0.2

	if(rv_flip > t_flip):
		image = np.fliplr(image)
		label = -1 * label
	
	return image, label

def createBatchGenerator(driving_log,batch_size=32):
	batch_images = np.zeros((batch_size, 66, 200, 3))
	batch_steering = np.zeros(batch_size)
	while 1:
		for i in range(batch_size):
			keep_flag = False
			while(keep_flag==False):
				rv_keep = np.random.uniform()
				x,y = get_preprocessed_row(driving_log)
				if(abs(y) > t_angle or rv_keep > t_keep):
					keep_flag=True
			batch_images[i]=x
			batch_steering[i]=y
		yield batch_images, batch_steering

# model

if(from_json):
	with open('model.json', 'r') as jfile:
		model = model_from_json(json.load(jfile))
	model.load_weights('model.h5')

else:
	model = Sequential()
	# model.add(Lambda(lambda x: normalize(x), input_shape=(66,200,3)))
	# model.add(Dropout(dropout))
	model.add(Lambda(lambda x: x/127.5 - 1., input_shape=(66,200,3)))
	model.add(Convolution2D(24, 5, 5, activation='elu', subsample=(2,2)))
	model.add(Convolution2D(36, 5, 5, activation='elu', subsample=(2,2)))
	model.add(Convolution2D(48, 5, 5, activation='elu', subsample=(2,2)))
	model.add(Convolution2D(64, 3, 3, activation='elu'))
	model.add(Convolution2D(64, 3, 3, activation='elu'))
	model.add(Flatten())
	model.add(Dense(1164, activation='elu'))
	model.add(Dense(100, activation='elu'))
	model.add(Dense(50, activation='elu'))
	model.add(Dense(10, activation='elu'))
	model.add(Dense(1))
	# model.add(Dense(1,activation='tanh'))
	model.summary()

my_adam = Adam(lr=lr)
model.compile(optimizer=my_adam,loss='mse')

# Model will save the weights whenever validation loss improves
checkpoint = ModelCheckpoint(filepath = 'model.h5', verbose = 1, save_best_only=False, monitor='loss')

# Discontinue training when validation loss fails to decrease
callback = EarlyStopping(monitor='val_loss', patience=2, verbose=1)

# change probability of keeping small angles
def change_t_keep():
	print('hello from callback')
myCallback = LambdaCallback(on_epoch_begin=change_t_keep())

# model.fit(X_train, y_train, nb_epoch=nb_epoch,validation_split=0.1, shuffle=True, callbacks=[checkpoint, callback])
model.fit_generator(createBatchGenerator(driving_log), samples_per_epoch = samples_per_epoch, nb_epoch = nb_epoch,
	verbose=1, callbacks=[checkpoint,myCallback])

# , validation_data=createBatchGenerator(driving_log),
	# nb_val_samples=10, class_weight=None, pickle_safe=False

# predictions = model.predict(X_train[0:200])
# print('predictions: ', predictions)

# save model

json_string = model.to_json()
with open('model.json','w') as f:
	json.dump(json_string,f,ensure_ascii=False)
