import time
import tensorflow as tf 
import cv2
import csv
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Flatten , Dense , Activation, Lambda, Convolution2D, MaxPooling2D, Cropping2D, Dropout
from generator import generator
import matplotlib.pyplot as plt
plt.switch_backend('agg')
print('Importing dependencies done')

# Reading the images path from a csv file and using image path to read the image using mpimg.imread 
# and stacking all the images in an array

data_file = './data/driving_log.csv'
path_to_image_data = './data/IMG/'
image_shape =(160,320,3)
correction_factor = 0.2 # this factor is usede while evaluating steering angles for left and right image

# opening csv file and saving rows of csv file in samples

samples = []
with open(data_file,'r') as f:
    reader = csv.reader(f)
    for row in reader:
        samples.append(row)
samples = samples[1:] # skipping headers

# 20 % data is used for validation
train_samples, validation_samples = train_test_split(samples, test_size = 0.2) 

# this is batch size for generator function
BATCH_SIZE = 64 
EPOCH = 5

# Using generator function to efectivey use memory while converting list of image data  into np.arrays 
train_generator = generator(train_samples, BATCH_SIZE, path_to_image_data, correction_factor)
validation_generator = generator(validation_samples,BATCH_SIZE, path_to_image_data, correction_factor)

# Neural Network architecture
model = Sequential()
#Adding cropping to ignore top portion of image which include trees etc 
# and bottom portion which include car dasboard
model.add(Cropping2D(cropping =((70,25),(0,0)),input_shape = image_shape))
model.add(Lambda(lambda x: (x/255.0)-0.5))     #Normalization

model.add(Convolution2D(24,5,2,subsample =(2,2),activation ='relu'))
model.add(Convolution2D(36,5,2,subsample =(2,2),activation ='relu'))
model.add(Convolution2D(48,5,2,subsample =(2,2),activation ='relu'))
model.add(Convolution2D(64,3,1,activation ='relu'))
model.add(Convolution2D(64,3,1,activation ='relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))
model.compile(loss ='mse' , optimizer = 'adam') # using means square as loss funcrion and adam optimizer

history_object = model.fit_generator(train_generator, samples_per_epoch = 4*len(train_samples), \
    validation_data = validation_generator, nb_val_samples = 4*len(validation_samples), nb_epoch = EPOCH, \
    verbose =1)

### print the keys contained in the history object
print(history_object.history.keys())
### plot the training and validation loss for each epoch

plt.figure(1)
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.savefig('./results.png')


# Saving the model
print('saving Model')
model.save('model.h5')
print('Model saved')











