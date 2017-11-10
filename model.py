import time
import tensorflow as tf 
import cv2
import csv
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Flatten , Dense , Activation, Lambda, Convolution2D, MaxPooling2D
from generator import generator
import matplotlib.pyplot as plt

print('Importing dependencies done')

# Reading the images path from a csv file and using image path to read the image using cv2.imread 
# and stacking all the images in an array

data_file = './data/driving_log.csv'
#lines = []
# with open (data_file,'r') as f:
#     reader = csv.reader(f)
#     for line in reader:
#         lines.append(line)


# images = []
# steering =[]
# for line in lines[1:]:      # skipping the header by starting index from 1 instead of 0
#     image_path = './data/'+line[0]
#     #print(cv2.imread(image_path).shape)
#     images.append(cv2.imread(image_path))
#     angle = float(line[3])
#     steering.append(angle)
# print('No. of centre Images = {}'.format(len(images)))
# print('No. of steering angle points = {}'.format(len(steering)))
# image_shape = images[0].shape
# print('Image_size = {}'.format(image_shape))

#X_train = np.array(images)
#y_train = np.array(angle)

image_shape =(160,320,3)
samples = []
with open(data_file,'r') as f:
    reader = csv.reader(f)
    for row in reader:
        samples.append(row)
samples = samples[1:] # skipping headers
train_samples, validation_samples = train_test_split(samples, test_size = 0.2)
BATCH_SIZE = 20 
EPOCH = 5

train_generator = generator(train_samples, BATCH_SIZE)
validation_generator = generator(validation_samples,BATCH_SIZE)

# Neural Network architecture
model = Sequential()
model.add(Lambda(lambda x: (x/255.0)-0.5, input_shape = image_shape))
model.add(Convolution2D(32,3,3))
model.add(MaxPooling2D((2,2)))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(32))
model.add(Activation('relu'))
model.add(Dense(1))
model.compile(loss ='mse' , optimizer = 'adam')

#history_object = model.fit(_train , validation_split = 0.2, shuffle = True ,nb_epoch = EPOCH)
history_object = model.fit_generator(train_generator, samples_per_epoch = len(train_samples), \
    validation_data = validation_generator, nb_val_samples = len(validation_samples), nb_epoch = EPOCH, \
    verbose =1)
### print the keys contained in the history object
#print(history_object.history.keys())

### plot the training and validation loss for each epoch
#plt.plot(history_object.history['loss'])
#plt.plot(history_object.history['val_loss'])
#plt.title('model mean squared error loss')
#plt.ylabel('mean squared error loss')
#plt.xlabel('epoch')
#plt.legend(['training set', 'validation set'], loc='upper right')
#plt.show()
print('saving')
model.save('model.h5')
print('saved')










