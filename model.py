import numpy as np
import cv2
import csv
import json
import os.path
import pickle
from keras.models import Sequential
from keras.layers import Dense, Activation, Convolution2D, Flatten, Lambda
from keras.optimizers import Adam
import tensorflow as tf

from scipy.misc import imresize

# config
nb_epoch = 50
lr = 0.001

# Load and preprocess data

# if pickled training data already exists, unpickle. Otherwise use the log to read training data and pickle it
if(os.path.exists('training_data.p')):
	training_data = pickle.load(open('training_data.p','rb'))
	X_train, y_train = training_data['X_train'], training_data['y_train']
else:
	# import csv into list of lists of strings, cells accessible as data[][]
	with open('driving_log.csv','r') as f:
		datareader = csv.reader(f,delimiter=',')
		driving_log = []
		for row in datareader:
			driving_log.append(row)

	# use csv data to set X to the images and y to the steering angles
	# for labels y_train, this is by initializing an array of the right length and updating values
	# for feature data X_train, this is by contatenating images and reshaping into an array of images
	num_images = 1725 #1725 total
	y_train = np.zeros(num_images)

	for i,row in enumerate(driving_log):
		image = cv2.imread(driving_log[i][0])
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

	X_train = images_concatenated.reshape(-1,20,64,3)

	# normalize: mean zero and range -0.5 to 0.5
	def normalize(X):
	    X = X - np.mean(X)
	    a,b,xmin,xmax = -0.5,0.5,np.min(X),np.max(X)
	    return a+(X-xmin)*(b-a)/(xmax-xmin)

	X_train = normalize(X_train)

	# pickle
#	training_data = {'X_train': X_train, 'y_train': y_train}
#	pickle.dump(training_data, open('training_data.p','wb'))

# model

img_shape = (20,64,3)

model = Sequential([
		Convolution2D(nb_filter=32,nb_row=5,nb_col=5,input_shape=img_shape),
		Convolution2D(32, 5, 5, subsample=(2, 2), border_mode="same"),
		Convolution2D(32, 5, 5, subsample=(2, 2), border_mode="same"),
        Flatten(),
        Dense(32,activation='tanh'),
        Dense(1,activation='tanh')
    ])

my_adam = Adam(lr=lr)
model.summary()
model.compile(optimizer=my_adam,loss='mse')
model.fit(X_train, y_train, nb_epoch=nb_epoch,validation_split=0.05, shuffle=True)

# save model

json_string = model.to_json()
with open('model.json','w') as f:
	json.dump(json_string,f,ensure_ascii=False)
model.save_weights('model.h5')
