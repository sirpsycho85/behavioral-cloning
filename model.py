import numpy as np
import cv2
import csv
import json
import os.path
import pickle
from keras.models import Sequential
from keras.layers import Dense, Activation, Convolution2D, Flatten, Lambda, Dropout, BatchNormalization
from keras.optimizers import Adam
import tensorflow as tf

from scipy.misc import imresize

# config
nb_epoch = 50
lr = 0.001
dropout = 0

# Load and preprocess data

# if pickled training data already exists, unpickle. Otherwise use the log to read training data and pickle it
# if(os.path.exists('training_data.p')):
# 	training_data = pickle.load(open('training_data.p','rb'))
# 	X_train, y_train = training_data['X_train'], training_data['y_train']
# else:
	
# import csv into list of lists of strings, cells accessible as data[][]
driving_log = []
num_images = 0
with open('driving_log.csv','r') as f:
	datareader = csv.reader(f,delimiter=',')
	for row in datareader:
		driving_log.append(row)
		num_images += 1

# use csv data to set X to the images and y to the steering angles
# for labels y_train, this is by initializing an array of the right length and updating values
# for feature data X_train, this is by contatenating images and reshaping into an array of images

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

model = Sequential()
model.add(BatchNormalization(axis=1, input_shape=(20,64,3)))
model.add(Convolution2D(16, 3, 3, border_mode='valid', subsample=(2,2), activation='relu'))
model.add(Convolution2D(24, 3, 3, border_mode='valid', subsample=(1,2), activation='relu'))
model.add(Convolution2D(36, 3, 3, border_mode='valid', activation='relu'))
model.add(Convolution2D(48, 2, 2, border_mode='valid', activation='relu'))
model.add(Convolution2D(48, 2, 2, border_mode='valid', activation='relu'))
model.add(Flatten())
model.add(Dense(512))
model.add(Dropout(dropout))
model.add(Activation('relu'))
model.add(Dense(10))
model.add(Activation('relu'))
model.add(Dense(1))


#model = Sequential([
#		Convolution2D(16, 3, 3, border_mode='valid', subsample=(2,2), activation='relu',input_shape=img_shape),
#		Convolution2D(24, 3, 3, border_mode='valid', subsample=(1,2), activation='relu'),
#		Convolution2D(36, 3, 3, border_mode='valid', activation='relu'),
#		Convolution3D(48, 2, 2, border_mode='valid', activation='relu'),
#		Convolution2D(48, 2, 2, border_mode='valid', activation='relu'),
#		Convolution2D(64, 5, 5, subsample=(2, 2), border_mode="same"),
#		Dropout(dropout),
#		Convolution2D(64, 5, 5, subsample=(2, 2), border_mode="same"),
#		Dropout(dropout),
 #       Flatten(),
  #      Dense(128,activation='tanh'),
   #     Dropout(dropout),
   #     Dense(1,activation='tanh')
   # ])

my_adam = Adam(lr=lr)
model.summary()
model.compile(optimizer=my_adam,loss='mse')
model.fit(X_train, y_train, nb_epoch=nb_epoch,validation_split=0.1, shuffle=True)

# save model

json_string = model.to_json()
with open('model.json','w') as f:
	json.dump(json_string,f,ensure_ascii=False)
model.save_weights('model.h5')
