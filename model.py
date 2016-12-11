import numpy as np
import cv2
import csv
import json
import os.path
import pickle
from keras.models import Sequential
from keras.layers import Dense, Activation, Convolution2D, Flatten, Lambda, Dropout, BatchNormalization, ZeroPadding2D, MaxPooling2D
from keras.optimizers import Adam
import tensorflow as tf

from scipy.misc import imresize

# config
nb_epoch = 50
lr = 2e-3 #0.00005
dropout = 0.5

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

# override num images
num_images = 1000


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

images_concatenated = np.concatenate((images_concatenated,np.fliplr(images_concatenated)), axis=0)
X_train = images_concatenated.reshape(-1,20,64,3)
y_train = np.concatenate((y_train,-1*y_train), axis=0)

# normalize: mean zero and range -0.5 to 0.5
def normalize(X):
    X = X - np.mean(X)
    a,b,xmin,xmax = -0.5,0.5,np.min(X),np.max(X)
    return a+(X-xmin)*(b-a)/(xmax-xmin)

X_train = normalize(X_train)

print('max y_train = ',max(y_train))

	# pickle
#	training_data = {'X_train': X_train, 'y_train': y_train}
#	pickle.dump(training_data, open('training_data.p','wb'))

# model

img_shape = (20,64,3)

model = Sequential()
#model.add(BatchNormalization(axis=1, input_shape=(20,64,3)))
model.add(BatchNormalization(input_shape=(20,64,3)))
# model.add(Convolution2D(24, 5, 5, border_mode='valid', activation='relu',input_shape=(20,64,3)))
# # model.add(BatchNormalization())
# model.add(Convolution2D(36, 5, 5, border_mode='valid', activation='relu'))
# # model.add(BatchNormalization())
# model.add(Convolution2D(48, 3, 3, border_mode='valid', activation='relu'))
# # model.add(BatchNormalization())
# model.add(Convolution2D(64, 3, 3, border_mode='valid', activation='relu'))
# # model.add(BatchNormalization())
# model.add(Convolution2D(64, 3, 3, border_mode='valid', activation='relu'))
# model.add(BatchNormalization())


model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_1', input_shape=(20,64,3)))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_2'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_1'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_2'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

# model.add(ZeroPadding2D((1, 1)))
# model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_1'))
# model.add(ZeroPadding2D((1, 1)))
# model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_2'))
# model.add(ZeroPadding2D((1, 1)))
# model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_3'))
# model.add(MaxPooling2D((2, 2), strides=(2, 2)))

# model.add(ZeroPadding2D((1, 1)))
# model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_1'))
# model.add(ZeroPadding2D((1, 1)))
# model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_2'))
# model.add(ZeroPadding2D((1, 1)))
# model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_3'))
# model.add(MaxPooling2D((2, 2), strides=(2, 2)))

# model.add(ZeroPadding2D((1, 1)))
# model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_1'))
# model.add(ZeroPadding2D((1, 1)))
# model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_2'))
# model.add(ZeroPadding2D((1, 1)))
# model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_3'))
# model.add(MaxPooling2D((2, 2), strides=(2, 2)))


model.add(Flatten())
# model.add(Dense(1164))
model.add(Dropout(dropout))
# model.add(Dense(100))
# model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1,activation='tanh'))


my_adam = Adam(lr=lr)
model.summary()
model.compile(optimizer=my_adam,loss='mse')
model.fit(X_train, y_train, nb_epoch=nb_epoch,validation_split=0.1, shuffle=True)

# save model

json_string = model.to_json()
with open('model.json','w') as f:
	json.dump(json_string,f,ensure_ascii=False)
model.save_weights('model.h5')
