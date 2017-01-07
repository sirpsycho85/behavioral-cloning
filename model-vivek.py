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
to_model = 'model16'
csvpath='Archive/driving_log-carnd.csv'
image_folder='Archive/IMG-carnd'
lr = 0.0001

from_json = False
from_model='model5'
from_epoch='5'

nb_sessions = 8
nb_epoch = 1
samples_per_epoch = 20000 #20000
dropout = 0.5
t_flip = 0.5 # threshold for flipping
t_angle = 0.15 # threshold of |angle| to keep, else discard...
use_side_cameras = True
angle_multiplier = 1
side_camera_added_angle = 0.25
trans_range = 80
trans_range_y = 0 #40?

cols=320
rows=160
cols_resized=64
rows_resized=64


# Load driving log

driving_log = []
with open(csvpath,'r') as f:
	datareader = csv.reader(f,delimiter=',')
	for row in datareader:
		driving_log.append(row)

# preprocessing

def add_random_shadow(image):
	top_y = 320*np.random.uniform()
	top_x = 0
	bot_x = 160
	bot_y = 320*np.random.uniform()
	image_hls = cv2.cvtColor(image,cv2.COLOR_RGB2HLS)
	shadow_mask = 0*image_hls[:,:,1]
	X_m = np.mgrid[0:image.shape[0],0:image.shape[1]][0]
	Y_m = np.mgrid[0:image.shape[0],0:image.shape[1]][1]
	shadow_mask[((X_m-top_x)*(bot_y-top_y) -(bot_x - top_x)*(Y_m-top_y) >=0)]=1
	if np.random.randint(2)==1:
		random_bright = .5
		cond1 = shadow_mask==1
		cond0 = shadow_mask==0
		if np.random.randint(2)==1:
			image_hls[:,:,1][cond1] = image_hls[:,:,1][cond1]*random_bright
		else:
			image_hls[:,:,1][cond0] = image_hls[:,:,1][cond0]*random_bright    
	image = cv2.cvtColor(image_hls,cv2.COLOR_HLS2RGB)
	return image

def trans_image(image,steer,trans_range):
    tr_x = trans_range*np.random.uniform()-trans_range/2
    steer_ang = steer + tr_x/trans_range*2*.2
    tr_y = trans_range_y*np.random.uniform()-trans_range_y/2
    Trans_M = np.float32([[1,0,tr_x],[0,1,tr_y]])
    image_tr = cv2.warpAffine(image,Trans_M,(cols,rows))
    return image_tr,steer_ang

def augment_brightness_camera_images(image):
    image1 = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    random_bright = .25+np.random.uniform()
    image1[:,:,2] = image1[:,:,2]*random_bright
    image1 = cv2.cvtColor(image1,cv2.COLOR_HSV2RGB)
    return image1

def get_unprocessed_row(driving_log, i=None):
	if(i == None):
		i = np.random.randint(len(driving_log))
	camera = 0
	filepath = image_folder + '/' + driving_log[i][camera].rsplit('/')[-1]
	image = cv2.imread(filepath)
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	label = float(driving_log[i][3])
	return image, label

def get_preprocessed_row(driving_log, i=None):
	if(i==None):
		i = np.random.randint(len(driving_log))
	rv_flip = np.random.uniform()
	if(use_side_cameras == True):
		camera = np.random.randint(3)
	else:
		camera = 0

	filepath = image_folder + '/' + driving_log[i][camera].rsplit('/')[-1]
	image = cv2.imread(filepath)

	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

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

	image,label = trans_image(image,label,trans_range)

	# image = add_random_shadow(image)

	image = image[32:135,:,:]
	image = imresize(image, (64,64,3))

	return image, label

def createBatchGenerator(driving_log,batch_size=256):
	batch_images = np.zeros((batch_size, rows_resized, cols_resized, 3))
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
	batch_images = np.zeros((batch_size, rows_resized, cols_resized, 3))
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
from_model_json_path = from_model + '/' + from_model+'.json'
from_model_h5_path = from_model + '/' + from_model + '-epoch-' + from_epoch + '.h5'
to_model_path = to_model + '/' + to_model

if(from_json):
	with open(from_model_json_path, 'r') as jfile:
		model = model_from_json(json.load(jfile))
	model.load_weights(from_model_h5_path)
	print('tuning existing model - loaded model weights')

else:
	model = Sequential()
	model.add(Lambda(lambda x: x/127.5 - 1., input_shape=(rows_resized,cols_resized,3)))
	
	model.add(Convolution2D(3, 1, 1))
	
	model.add(Convolution2D(32, 3, 3, activation='elu'))
	model.add(Convolution2D(32, 3, 3, activation='elu', subsample=(2,2)))
	model.add(Dropout(dropout))
	
	model.add(Convolution2D(64, 3, 3, activation='elu'))
	model.add(Convolution2D(64, 3, 3, activation='elu', subsample=(2,2)))
	model.add(Dropout(dropout))

	model.add(Convolution2D(128, 3, 3, activation='elu'))
	model.add(Convolution2D(128, 3, 3, activation='elu', subsample=(2,2)))
	model.add(Dropout(dropout))

	model.add(Flatten())
	model.add(Dense(512, activation='elu'))
	model.add(Dense(64, activation='elu'))
	model.add(Dense(16))
	model.add(Dense(1))
	model.summary()

# validate there's no directory
if os.path.exists(to_model):
	print('directory already exists')
	sys.exit()
else:
	os.makedirs(to_model)

for i in range(nb_sessions):
	t_keep = 1/(i+1)
	# t_keep = 1 - i/nb_sessions
	print('t_keep = ',t_keep)

	# if it's not first iteration, load the saved model from last iteration
	if(i>0):
		print('loading model from file: ',from_model_json_path)
		with open(from_model_json_path, 'r') as jfile:
			model = model_from_json(json.load(jfile))
		print('loading weights from file: ',from_model_h5_path)
		model.load_weights(from_model_h5_path)
		print('loaded')
	
	# compile model
	my_adam = Adam(lr=lr)
	model.compile(optimizer=my_adam,loss='mse')

	# Model will save the weights whenever validation loss improves
	from_model_h5_path = to_model_path + '-epoch-' + str(i) + '.h5'
	checkpoint = ModelCheckpoint(filepath = from_model_h5_path, verbose = 1, save_best_only=False, monitor='val_loss')

	# Discontinue training when validation loss fails to decrease
	earlyStop = EarlyStopping(monitor='val_loss', patience=2, verbose=1)

	model.fit_generator(createBatchGenerator(driving_log), samples_per_epoch = samples_per_epoch, nb_epoch = nb_epoch,
		verbose=1, callbacks=[checkpoint,earlyStop], validation_data=createBatchGeneratorValidation(driving_log), nb_val_samples = 100)
	
	# save model
	from_model_json_path = to_model_path + '.json'
	json_string = model.to_json()
	with open(from_model_json_path,'w') as f:
		json.dump(json_string,f,ensure_ascii=False)
