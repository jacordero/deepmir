#!/usr/bin/env python

import os
import json

import numpy as np
import keras
from keras import backend as K

def save_image(output_filename, image):
     image.savefig(output_filename, dpi=(200))

def create_directory(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory {}'.format(directory))
        sys.exit(1)

def load_ids(filename):
     return np.load(filename)['arr_0']


def load_image_data(filename):
     X = np.load(filename)['arr_0']
    
     if K.image_data_format() == 'channels_first':
          X = np.swapaxes(X, 1, 3)

     X = X.astype('float32')
     if np.amax(X) > 1:
          X /= 255
          
     return X

def get_rgb_input_shape(img_rows, img_columns):
    if K.image_data_format() == 'channels_first':
        return (3, img_rows, img_columns)
    return (img_rows, img_columns, 3)

def load_labels(filename, num_classes):
    labels = np.load(filename)['arr_0']
    labels = keras.utils.to_categorical(labels, num_classes)
    return labels


def save_to_file(output_text, output_filename):
     with open(output_filename, "w") as output_file:
          output_file.write(output_text)
