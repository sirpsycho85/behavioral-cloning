# Project 3, "Behavioral Cloning", in Udacity's Self-Driving Car Nano-degree

# Overview
The goal of this project was to use camera data from a simulated car to set the apropriate steering angle for the car and successfully drive around a course.
- The final model uses a slightly modified architecture from nvidia for end-to-end self-driving (https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf).
- The raw data came from Udacity, capturing about 8000 points in time on Track 1, using three cameras for each (center, left, and right).
- A lot of augmentation was applied to the data, as noted further below, per the advice of Vivek Yadav, another student in the course (https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9#.u75jlyg06).

# Initial approach
I initially tried training using my own unmodified data. However, this proved to be ineffective. In particular, the car would tend to overtrain initially on batches of small-angle or large-angle data, and go either straight ahead of immediately turn. In either case, it did not have the ability to recover once it started to turn towards one of the lane lines. I did not want to take the approach of doing additional training on "problem spots", as I wanted to take an approach that wasn't track-specific. Per the suggestion of Vivek Yadav, I took the the method of augmentation - creating additional images from my existing images to represent a more varied set of data.

# Architecture
I tried a few architectures, but eventually settled on the nvidia one as it was proven and I wanted to remove bad architecture as a variable. The layers in detail are, as conveniently printed by Keras' model summary feature:
____________________________________________________________________________________________________
Layer (type)                     Output Shape          Param #     Connected to                     
====================================================================================================
lambda_1 (Lambda)                (None, 66, 200, 3)    0           lambda_input_1[0][0]             
____________________________________________________________________________________________________
convolution2d_1 (Convolution2D)  (None, 31, 98, 24)    1824        lambda_1[0][0]                   
____________________________________________________________________________________________________
convolution2d_2 (Convolution2D)  (None, 14, 47, 36)    21636       convolution2d_1[0][0]            
____________________________________________________________________________________________________
convolution2d_3 (Convolution2D)  (None, 5, 22, 48)     43248       convolution2d_2[0][0]            
____________________________________________________________________________________________________
dropout_1 (Dropout)              (None, 5, 22, 48)     0           convolution2d_3[0][0]            
____________________________________________________________________________________________________
convolution2d_4 (Convolution2D)  (None, 3, 20, 64)     27712       dropout_1[0][0]                  
____________________________________________________________________________________________________
convolution2d_5 (Convolution2D)  (None, 1, 18, 64)     36928       convolution2d_4[0][0]            
____________________________________________________________________________________________________
dropout_2 (Dropout)              (None, 1, 18, 64)     0           convolution2d_5[0][0]            
____________________________________________________________________________________________________
flatten_1 (Flatten)              (None, 1152)          0           dropout_2[0][0]                  
____________________________________________________________________________________________________
dense_1 (Dense)                  (None, 1164)          1342092     flatten_1[0][0]                  
____________________________________________________________________________________________________
dense_2 (Dense)                  (None, 100)           116500      dense_1[0][0]                    
____________________________________________________________________________________________________
dense_3 (Dense)                  (None, 50)            5050        dense_2[0][0]                    
____________________________________________________________________________________________________
dense_4 (Dense)                  (None, 10)            510         dense_3[0][0]                    
____________________________________________________________________________________________________
dense_5 (Dense)                  (None, 1)             11          dense_4[0][0]                    
====================================================================================================
Total params: 1595511

# Augmentation
Without augmentation, the car was unable to recover back to the center of the road once it started heading at an incorrect angle. The goal of augmentation was to train the car in more situations it would encounter as it drove itself. The following methods were used:

## Cropping
I cropped from both the top and bottom of the image, as this mostly consisted of sky and car hood. Presumably the model would ignore the hood, but I suspect it would take longer to train as the model needs to learn this. The sky changes from location to location, but should have no effect on the steering angle, so I think it's important to crop out.

## Image flipping
The car is trained on a track with mostly left turns, so 50% of the time I flip the image to get an equal number of left and right turns in my training data.

## Brightness
Randomizing the brightness was used to try and make the training data cover similar situations at different levels of brightness, as there are shadows on the road and different times of day.

## Side cameras and translation
Instead of just using the center camera, the side cameras were used as an example of a car being out of position. When using a side camera, the steering angle label should be adjusted for what it should be if the car was actually displaced to the side. E.g. if using the right camera, you should compensate the steering with some additional turn to the left.

Translating the image and adjusting the steering angle is another way approach to generate recovery data. I did not do a deep dive of translation vs side cameras, but I imagine that side cameras give a complete image of the road, but are more limited because they are in a fixed position, while translation necessarily cuts off part of the image, but allows for recovery training from a wide range of shifts. I used both in the final model.

# Training
For each mini-batch, I would generate a set of images using the augmentation techniques above. Then, a percentage of the time p, I would discard the image if the angle was below a minimum angle threshold (a hyperparameter which I eventually set to 0.15). p itself was reduced between each epoch, so for the first epoch I always discarded a sample with a small angle, but with each subsequent epoch I was more and more likely to use it. This approach significantly increased the model's ability to recover in spots that were challenging on the course, such as sharp turns.

After each epoch I saved the weights and later tested the results. The best-performing one is included in the repo.

I used an adam optimizer and mean squared error as my cost function (error between labeled angle and predicted angle).

# Generalization
I tran the car on track 2 as well. At throttle = 2 it could not take the hill, at throttle = 3 it crashed quickly. I started binary searching for a good throttle level in between. At 2.75, I got all the way to the roadblock!

# Discarded ideas
There were other ideas that I tried but eventually did not use.
- Just multiplying all of the angles by a factor (another hyperparameter), but this proved less effective than translations.
- Different architectures, including my own and one that Vivek Yadav suggested. These did not perform as well as nvidia in combination with the above augemntation and training approach.
- Dropout was sometimes used. In the end, my very best model did not use dropout during training.
- Tuning a model - I wanted to see if I could get it working by training at once on a large set of data, rather than tuning a model as I encountered problem spots in the road.

# Future improvements
I would love to try and control the throttle as well as the angle, which is more similar to a real self-driving-car. Other than that, I would personally want to focus more on understanding in greater depth why certain approaches work and the mathematics behind deep learning, not just to try more things out :)
