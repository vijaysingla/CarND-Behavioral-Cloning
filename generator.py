import cv2
from sklearn.utils import shuffle
import numpy as np 
import matplotlib.image as mpimg

def generator(samples, batch_size, path_image_data = './data/IMG/',correction_factor = 0.15):
    num_samples = len(samples)
    while 1:# Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0,num_samples,batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []

            for list in batch_samples:
                #getting path to centre , left and rigth camera images
                image_path = path_image_data + list[0].split('/')[-1]
                left_image_path = path_image_data + list[1].split('/')[-1]
                right_image_path = path_image_data + list[2].split('/')[-1]

                # reading center, left , and right images 
                centre_image = mpimg.imread(image_path)
                left_image =mpimg.imread(left_image_path)
                right_image = mpimg.imread(right_image_path)

                # flipping centre image to augment data
                flip_image = np.fliplr(centre_image) 

                # creating adjusting steering for right , left cameras and flipped centre image
                centre_angle = float(list[3])
                flip_angle = -centre_angle
                steering_left = centre_angle + correction_factor
                steering_right = centre_angle - correction_factor


                # appending  the aboves images  to list named images
                images.append(centre_image)
                images.append(left_image)
                images.append(right_image)
                images.append(flip_image)

                # appending  the aboves angles to list named images
                angles.append(centre_angle)
                angles.append(steering_left)
                angles.append(steering_right)
                angles.append(flip_angle)


            # converting list into array 
            X_train = np.array(images)
            Y_train = np.array(angles)
            yield shuffle(X_train,Y_train)

