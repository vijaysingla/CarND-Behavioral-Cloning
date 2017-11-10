import cv2
from sklearn.utils import shuffle
import numpy as np 

def generator(samples, batch_size):
    num_samples = len(samples)
    while 1:
        shuffle(samples)
        for offset in range(0,num_samples,batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []

            for list in batch_samples:
                image_path = './data/'+ list[0]
                centre_image = cv2.imread(image_path)
                centre_angle = float(list[3])
                images.append(centre_image)
                angles.append(centre_angle)

            X_train = np.array(images)
            Y_train = np.array(angles)
            yield shuffle(X_train,Y_train)

