import csv
import cv2
import numpy as np
import random

import os

def LoadData():
    images = []
    measurements = []
    rootdir='./data/'
    dirs = [os.path.join(rootdir,eachdir) for eachdir in os.listdir(rootdir) if os.path.isdir(os.path.join(rootdir,eachdir))]
    for dir in dirs:
        pre_num = len(images)
        LoadDir(dir, images, measurements)
        print ("load ", len(images) - pre_num, " from ", dir)
    print("image num=", len(images))
    return images, measurements

 
def LoadDir(path, images, measurements):
    lines = []
    with open(path+'/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)

    for n, line in enumerate(lines):
        try:
            source_path = line[0]
            filename = source_path.split('/')[-1]
            current_path = getActualPath(path, line[0])
            steering_center = float(line[3])
            speed = float(line[6])
            #ignore straight in recover
            #if path.find('recover') >=0:
            if abs(steering_center) < 0.1 and random.random() >0.3:
                continue
            #ignore the car almost not moving cases
            if speed < 0.5:
                continue 

            addImage(current_path, steering_center, images, measurements)
            #recover no need side camera
            #if path.find('recover') >=0:
            #    continue
            correction = 0.25 # this is a parameter to tune
            steering_left = steering_center + correction
            steering_right = steering_center - correction
            left_img_path = getActualPath(path, line[1])
            right_img_path = getActualPath(path, line[2])
            addImage(left_img_path, steering_left, images, measurements)
            addImage(right_img_path, steering_right, images, measurements)

        except IndexError:
            print(n, line)

def getActualPath(path, source_path):
    filename = source_path.split('/')[-1]
    return (path + '/IMG/' + filename)

def addImage(path, steering, images, measurements):
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    brightness = .25 + np.random.uniform()
    image[:,:,2] = image[:,:,2] * brightness
    image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)    
    measurements.append(steering)
    images.append(image)
    image_flipped = np.fliplr(image)
    steering_flipped = -steering
    images.append(image_flipped)
    measurements.append(steering_flipped)

imgs, measures = LoadData()
    
X_train = np.array(imgs)
y_train = np.array(measures)

from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

import tensorflow as tf

print("tensorflow lib locatoin:", tf.__file__)
def NividiaNet():
    model = Sequential()
    model.add(Lambda(lambda x: (x / 127.5) - 1.0, input_shape=(160,320,3)))
    model.add(Cropping2D(cropping=((70,10),(0,0))))
    model.add(Convolution2D(24,5,5, subsample=(2,2), activation='relu'))
    model.add(Dropout(0.1))
    model.add(Convolution2D(36,5,5, subsample=(2,2), activation='relu'))
    model.add(Dropout(0.2))
    model.add(Convolution2D(48,5,5, subsample=(2,2), activation='relu'))
    model.add(Dropout(0.2))
    model.add(Convolution2D(64,3,3, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Convolution2D(64,3,3, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dropout(0.3))
    model.add(Dense(100))
    model.add(Dropout(0.4))
    model.add(Dense(50))
    model.add(Dropout(0.5))
    model.add(Dense(10))
    model.add(Dropout(0.6))
    model.add(Dense(1))
    return model

    
def AlexNet():
    model = Sequential()
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
    model.add(Cropping2D(cropping=((70,25),(0,0))))
    model.add(Convolution2D(64, 3, 11, 11, border_mode='full'))
    model.add(BatchNormalization((64,226,226)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(poolsize=(3, 3)))

    model.add(Convolution2D(128, 64, 7, 7, border_mode='full'))
    model.add(BatchNormalization((128,115,115)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(poolsize=(3, 3)))

    model.add(Convolution2D(192, 128, 3, 3, border_mode='full'))
    model.add(BatchNormalization((128,112,112)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(poolsize=(3, 3)))

    model.add(Convolution2D(256, 192, 3, 3, border_mode='full'))
    model.add(BatchNormalization((128,108,108)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(poolsize=(3, 3)))

    model.add(Flatten())
    model.add(Dense(12*12*256, 4096, init='normal'))
    model.add(BatchNormalization(4096))
    model.add(Activation('relu'))
    model.add(Dense(4096, 4096, init='normal'))
    model.add(BatchNormalization(4096))
    model.add(Activation('relu'))
    model.add(Dense(4096, 1000, init='normal'))
    model.add(BatchNormalization(1000))
    model.add(Activation('softmax'))
    return model


model = NividiaNet()
model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.1, shuffle=True, nb_epoch=4)

model.save('model.h5')


