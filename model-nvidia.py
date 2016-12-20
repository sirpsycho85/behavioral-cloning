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
from_json = False
from_model='model-2'
to_model = 'model3'
csvpath='Archive/driving_log-carnd.csv'
image_folder='Archive/IMG-carnd'
lr = 0.0001 #0.001

nb_sessions = 10
nb_epoch = 1
samples_per_epoch = 20000
dropout = 0.5
t_flip = 0.5 # threshold for flipping
t_angle = 0.15 # threshold of |angle| to keep, else discard...
use_side_cameras = True
angle_multiplier = 1
side_camera_added_angle = 0.25

# Load driving log

driving_log = []
with open(csvpath,'r') as f:
	datareader = csv.reader(f,delimiter=',')
	for row in datareader:
		driving_log.append(row)

# preprocessing

def augment_brightness_camera_images(image):
    image1 = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    random_bright = .25+np.random.uniform()
    image1[:,:,2] = image1[:,:,2]*random_bright
    image1 = cv2.cvtColor(image1,cv2.COLOR_HSV2RGB)
    return image1

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
		label = label + side_camera_added_angle
	elif(camera == 2):
		label = label - side_camera_added_angle

	label = label * angle_multiplier

	if(rv_flip > t_flip):
		image = np.fliplr(image)
		label = -1 * label
	
	image = augment_brightness_camera_images(image)

	return image, label

def createBatchGenerator(driving_log,batch_size=256):
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

def createBatchGeneratorValidation(driving_log,batch_size=256):
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
from_model_json_path = from_model+'.json'
from_model_h5_path = from_model+'.h5'
to_model_path = to_model+'/' + to_model

if(from_json):
	with open(from_model_json_path, 'r') as jfile:
		model = model_from_json(json.load(jfile))
	model.load_weights(from_model_h5_path)
	print('tuning existing model - loaded model weights')

else:
	model = Sequential()
	model.add(Lambda(lambda x: x/127.5 - 1., input_shape=(66,200,3)))
	model.add(Convolution2D(24, 5, 5, activation='elu', subsample=(2,2)))
	model.add(Convolution2D(36, 5, 5, activation='elu', subsample=(2,2)))
	model.add(Convolution2D(48, 5, 5, activation='elu', subsample=(2,2)))
	model.add(Dropout(dropout))
	model.add(Convolution2D(64, 3, 3, activation='elu'))
	model.add(Convolution2D(64, 3, 3, activation='elu'))
	model.add(Dropout(dropout))
	model.add(Flatten())
	model.add(Dense(1164, activation='elu'))
	model.add(Dense(100, activation='elu'))
	model.add(Dense(50, activation='elu'))
	model.add(Dense(10, activation='elu'))
	model.add(Dense(1))
	model.summary()

for i in range(nb_sessions):
	t_keep = 1/(i+1)
	# t_keep = 1 - i/nb_sessions
	print('t_keep = ',t_keep)

	# if it's not first iteration, load the saved model from last iteration
	if(i>0):
		with open(from_model_json_path, 'r') as jfile:
			model = model_from_json(json.load(jfile))
		model.load_weights(from_model_h5_path)
		print('loaded weights')
	
	# compile model
	my_adam = Adam(lr=lr)
	model.compile(optimizer=my_adam,loss='mse')

	# Model will save the weights whenever validation loss improves
	checkpoint = ModelCheckpoint(filepath = to_model_path + '-epoch-' + str(i) + '.h5', verbose = 1, save_best_only=False, monitor='val_loss')

	# Discontinue training when validation loss fails to decrease
	earlyStop = EarlyStopping(monitor='val_loss', patience=2, verbose=1)

	model.fit_generator(createBatchGenerator(driving_log), samples_per_epoch = samples_per_epoch, nb_epoch = nb_epoch,
		verbose=1, callbacks=[checkpoint,earlyStop], validation_data=createBatchGeneratorValidation(driving_log), nb_val_samples = 100)
	
	# save model
	json_string = model.to_json()
	with open(to_model_path + '.json','w') as f:
		json.dump(json_string,f,ensure_ascii=False)
