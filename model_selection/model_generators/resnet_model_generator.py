#!/usr/bin/env python

""" 
Generator of ResNet-like models. This script based in the code written by the keras team:
https://github.com/keras-team/keras-applications/blob/master/keras_applications/resnet50.py
"""

from __future__ import print_function

import keras
from keras.layers import Dense, Conv2D, BatchNormalization, Activation
from keras.layers import GlobalAveragePooling2D, Input, Flatten
from keras.regularizers import l2
from keras import backend as K
from keras.models import Model
from keras.optimizers import Adam

import numpy as np
import utils.deepmirna_utils as deep_utils

NUM_CLASSES = 2
IMG_ROWS, IMG_COLUMNS = 25, 100

def resnet_module_full_preactivation(inputs, num_filters):
    x = BatchNormalization()(inputs)
    x = Activation('relu')(x)
    x = Conv2D(num_filters, kernel_size=(3, 3), padding='same')(x)
    x = BatchNormalization()(x)    
    x = Activation('relu')(x)
    x = Conv2D(num_filters, kernel_size=(3, 3), padding='same')(x)
    return x

def create_name(prefix, num_filters):
    name = prefix + "_" + str(num_filters) + "filters"
    return name



def build_model_one_module(num_filters):
    input_shape_img = deep_utils.get_rgb_input_shape(IMG_ROWS, IMG_COLUMNS)
    #num_filters = 32
    
    inputs = Input(shape=input_shape_img)
    x = Conv2D(num_filters, kernel_size=(3, 3), activation='relu')(inputs)
    x_prime = resnet_module_full_preactivation(x, num_filters)
    x = keras.layers.add([x, x_prime])
    
    y = GlobalAveragePooling2D()(x)
    #y = Flatten()(x)
    outputs = Dense(NUM_CLASSES, activation='softmax')(y)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(), metrics=['accuracy'])

    model_name = create_name("resnet_one_module", num_filters)
    return (model, model_name)


def build_model_two_modules(num_filters):
    input_shape_img = deep_utils.get_rgb_input_shape(IMG_ROWS, IMG_COLUMNS)
    #num_filters = 32
    
    inputs = Input(shape=input_shape_img)
    x = Conv2D(num_filters, kernel_size=(3, 3), activation='relu')(inputs)

    x_prime = resnet_module_full_preactivation(x, num_filters)
    x = keras.layers.add([x, x_prime])

    x_prime = resnet_module_full_preactivation(x, num_filters)
    x = keras.layers.add([x, x_prime])
    
    y = GlobalAveragePooling2D()(x)
    #y = Flatten()(x)
    outputs = Dense(NUM_CLASSES, activation='softmax')(y)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(), metrics=['accuracy'])

    model_name = create_name("resnet_two_modules", num_filters)
    return (model, model_name)


def build_model_three_modules(num_filters):
    input_shape_img = deep_utils.get_rgb_input_shape(IMG_ROWS, IMG_COLUMNS)
    #num_filters = 32
    
    inputs = Input(shape=input_shape_img)
    x = Conv2D(num_filters, kernel_size=(3, 3), activation='relu')(inputs)

    x_prime = resnet_module_full_preactivation(x, num_filters)
    x = keras.layers.add([x, x_prime])

    x_prime = resnet_module_full_preactivation(x, num_filters)
    x = keras.layers.add([x, x_prime])
    
    x_prime = resnet_module_full_preactivation(x, num_filters)
    x = keras.layers.add([x, x_prime])
    
    y = GlobalAveragePooling2D()(x)
    #y = Flatten()(x)
    outputs = Dense(NUM_CLASSES, activation='softmax')(y)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(), metrics=['accuracy'])

    model_name = create_name("resnet_three_modules", num_filters)
    return (model, model_name)


def build_model_four_modules(num_filters):
    input_shape_img = deep_utils.get_rgb_input_shape(IMG_ROWS, IMG_COLUMNS)
    #num_filters = 32
    
    inputs = Input(shape=input_shape_img)
    x = Conv2D(num_filters, kernel_size=(3, 3), activation='relu')(inputs)

    x_prime = resnet_module_full_preactivation(x, num_filters)
    x = keras.layers.add([x, x_prime])
    x = Activation('relu')(x)

    x_prime = resnet_module_full_preactivation(x, num_filters)
    x = keras.layers.add([x, x_prime])

    x_prime = resnet_module_full_preactivation(x, num_filters)
    x = keras.layers.add([x, x_prime])

    x_prime = resnet_module_full_preactivation(x, num_filters)
    x = keras.layers.add([x, x_prime])
    
    y = GlobalAveragePooling2D()(x)
    #y = Flatten()(x)
    outputs = Dense(NUM_CLASSES, activation='softmax')(y)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(), metrics=['accuracy'])

    model_name = create_name("resnet_four_modules", num_filters)
    return (model, model_name)


def build_final_model():
    input_shape_img = deep_utils.get_rgb_input_shape(IMG_ROWS, IMG_COLUMNS)
    num_filters = 32
    
    inputs = Input(shape=input_shape_img)
    x = Conv2D(num_filters, kernel_size=(2, 2), activation='relu')(inputs)

    x_prime = resnet_module_v2(x, num_filters)
    x = keras.layers.add([x, x_prime])

    x_prime = resnet_module_v2(x, num_filters)
    x = keras.layers.add([x, x_prime])
    
    x_prime = resnet_module_v2(x, num_filters)
    x = keras.layers.add([x, x_prime])
    
    y = GlobalAveragePooling2D()(x)
    #y = Flatten()(x)
    outputs = Dense(NUM_CLASSES, activation='softmax')(y)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(), metrics=['accuracy'])

    model_name = create_name("resnet_final_model", num_filters)
    return (model, model_name)
