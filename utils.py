import sys
import os
import cv2
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import sklearn

from keras.models import Sequential
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.core import Activation, Dropout, Flatten, Dense, Lambda
from keras.layers import ELU
from keras.optimizers import Adam
from keras.layers import Input
from keras import regularizers
from keras.layers.normalization import BatchNormalization
import keras.backend.tensorflow_backend as KTF
from keras.models import Model

def single_optical_flow_dense(old_image_gray, current_image_gray):
    """
    input: old_image_gray, current_image_gray (gray images)
    * calculates optical flow magnitude and angle and places it into HSV image
    * Set the hue to the angles returned from computing the flow params
    * set the value to the magnitude returned from computing the flow params
    * Convert from HSV to RGB and return RGB image with same size as original image
    """
    # Flow Parameters
    # flow = None
    pyr_scale = 0.5
    levels = 1
    winsize = 12
    iterations = 2
    poly_n = 5
    poly_sigma = 1.3
    extra = 0
    flow = cv2.calcOpticalFlowFarneback(old_image_gray, current_image_gray,None, #flow_mat 
                                        pyr_scale,levels,winsize, iterations, poly_n, poly_sigma, extra)

    return flow_to_rgb(flow)


def flow_to_rgb(flow):
    h, w = flow.shape[:2]
    fx, fy = flow[:,:,0], flow[:,:,1]
    ang = np.arctan2(fy, fx) + np.pi
    v = np.sqrt(fx*fx+fy*fy)
    hsv = np.zeros((h, w, 3), np.uint8)
    hsv[...,0] = ang*(180/np.pi/2)
    hsv[...,1] = 255
    hsv[...,2] = np.minimum(v*4, 255)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return rgb

def crop_image(image):
    image = image[90:320,:]
    #image = np.resize(image, (110, 220))
    return image

def read_img_to_gray(img_path):
    img = cv2.imread(img_path)
    img = crop_image(img)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    return img

def batch_generator(data,batch_size=16):

    sample_image = cv2.imread('./data/training_frames/0.jpg')
    cropped = crop_image(sample_image)
    x_batch = np.zeros((batch_size,cropped.shape[0],cropped.shape[1],cropped.shape[2]))
    y_batch = np.zeros(batch_size)
    counter = 0
    while True:
        data_point = data.iloc[counter]
        curPath = data_point['curImgPath']
        nxtPath = data_point['nxtImgPath']
        y = data_point['avg_speed']
        img1 = read_img_to_gray(curPath)
        img2 = read_img_to_gray(nxtPath)
        flow_dense = single_optical_flow_dense(img1,img2)

        x_batch[counter%batch_size] = flow_dense
        y_batch[counter%batch_size] = y
        counter += 1
        if counter%batch_size == 0:
            yield x_batch,y_batch
        counter = counter%len(data)

N_img_height = 230
N_img_width = 640
N_img_channels = 3
def nvidia_model():
    inputShape = (N_img_height, N_img_width, N_img_channels)

    model = Sequential()
    # normalization    
    # perform custom normalization before lambda layer in network
    model.add(Lambda(lambda x: x/ 127.5 - 1, input_shape = inputShape))

    model.add(Convolution2D(24, (5, 5), 
                            strides=(2,2), 
                            padding = 'valid',
                            kernel_initializer = 'he_normal',
                            name = 'conv1'))
    
    
    model.add(ELU())    
    model.add(Convolution2D(36, (5, 5), 
                            strides=(2,2), 
                            padding = 'valid',
                            kernel_initializer = 'he_normal',
                            name = 'conv2'))
    
    model.add(ELU())    
    model.add(Convolution2D(48, (5, 5), 
                            strides=(2,2), 
                            padding = 'valid',
                            kernel_initializer = 'he_normal',
                            name = 'conv3'))
    model.add(ELU())
    model.add(Dropout(0.5))
    model.add(Convolution2D(64, (3, 3), 
                            strides = (1,1), 
                            padding = 'valid',
                            kernel_initializer = 'he_normal',
                            name = 'conv4'))
    
    model.add(ELU())              
    model.add(Convolution2D(64, (3, 3), 
                            strides= (1,1), 
                            padding = 'valid',
                            kernel_initializer = 'he_normal',
                            name = 'conv5'))
              
              
    model.add(Flatten(name = 'flatten'))
    model.add(ELU())
    model.add(Dense(100, kernel_initializer = 'he_normal', name = 'fc1'))
    model.add(ELU())
    model.add(Dense(50, kernel_initializer = 'he_normal', name = 'fc2'))
    model.add(ELU())
    model.add(Dense(10, kernel_initializer = 'he_normal', name = 'fc3'))
    model.add(ELU())
    
    # do not put activation at the end because we want to exact output, not a class identifier
    model.add(Dense(1, name = 'output', kernel_initializer = 'he_normal'))
    
    adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(optimizer = adam, loss = 'mean_squared_error')

    return model