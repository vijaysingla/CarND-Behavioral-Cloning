import cv2
from sklearn.utils import shuffle
import numpy as np 
import matplotlib.image as mpimg
def generator(samples, batch_size):
    num_samples = len(samples)
    while 1:
        shuffle(samples)
        for offset in range(0,num_samples,batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []

            for list in batch_samples:
                image_path = './data/IMG/'+ list[0].split('/')[-1]
                #print(image_path)
                left_image_path = './data/IMG/'+ list[1].split('/')[-1]
                #print(left_image_path)
                right_image_path = './data/IMG/'+ list[2].split('/')[-1]

                centre_image = mpimg.imread(image_path)
                #[:,:,::-1]
                flip_image = np.fliplr(centre_image)

                left_image =mpimg.imread(left_image_path)
                #print(left_image)
                right_image = mpimg.imread(right_image_path)
                centre_angle = float(list[3])
                flip_angle = -centre_angle
                flip_left =  np.fliplr(left_image)
                flip_right = np.fliplr (right_image)

                steering_left = centre_angle + 0.2
                steering_right = centre_angle - 0.2 
                steering_flip_left = -steering_left
                steering_flip_right = -steering_right
                images.append(centre_image)
                images.append(left_image)
                images.append(right_image)
                #images.append(flip_image)
                #images.append(flip_left)
                #images.append(flip_right)
                angles.append(centre_angle)
                angles.append(steering_left)
                angles.append(steering_right)
                #angles.append(flip_angle)
                #angles.append(steering_flip_left)
                #angles.append(steering_flip_right)


            X_train = np.array(images)
            Y_train = np.array(angles)
            yield shuffle(X_train,Y_train)

