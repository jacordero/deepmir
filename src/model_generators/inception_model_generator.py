#!/usr/bin/env python

from __future__ import print_function

import keras
from keras.layers import Dense, Input, Conv2D, Flatten, concatenate
from keras.layers import MaxPooling2D, Dropout
from keras.models import Model
from keras import backend as K

import numpy as np
import utils.deepmirna_utils as deep_utils

NUM_CLASSES = 2
IMG_ROWS, IMG_COLUMNS = 25, 100
# try with values: 8, 12, 16*, 20
N_FILTERS = 8 #


def inception_module(inputs, n_filters):

    towerOne = MaxPooling2D((2,2), strides=(1,1), padding='same')(inputs)
    towerOne = Conv2D(n_filters, (1,1), activation='relu', padding='same')(towerOne)

    towerTwo = Conv2D(n_filters, (1,1), activation='relu', padding='same')(inputs)
    towerTwo = Conv2D(n_filters, (2,2), activation='relu', padding='same')(towerTwo)

    towerThree = Conv2D(n_filters, (1,1), activation='relu', padding='same')(inputs)
    towerThree = Conv2D(n_filters, (3,3), activation='relu', padding='same')(towerThree)

    towerFour = Conv2D(n_filters, (1, 1), activation='relu', padding='same')(inputs)
    
    x = concatenate([towerOne, towerTwo, towerThree, towerFour], axis=3)
    return x


def build_model_one_module():
    input_shape_img = deep_utils.get_rgb_input_shape(IMG_ROWS, IMG_COLUMNS)
    inputs = Input(input_shape_img)

    x = Conv2D(64, (3,3), activation='relu', padding='same')(inputs)
    x = inception_module(x, n_filters=N_FILTERS)

    x  = Flatten()(x)
    x = Dropout(0.5)(x)
    predictions = Dense(2, activation='softmax')(x)

    model = Model(input=inputs, output=predictions)
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adam(),
                  metrics=['accuracy'])

    model_name = "inception_one_module_" + str(N_FILTERS) + "filters"

    return (model, model_name)



def build_model_two_modules():
    input_shape_img = deep_utils.get_rgb_input_shape(IMG_ROWS, IMG_COLUMNS)
    inputs = Input(input_shape_img)

    x = Conv2D(64, (3,3), activation='relu', padding='same')(inputs)
    x = inception_module(x, n_filters=N_FILTERS)
    x = inception_module(x, n_filters=N_FILTERS)

    x  = Flatten()(x)
    x = Dropout(0.5)(x)
    predictions = Dense(2, activation='softmax')(x)

    model = Model(input=inputs, output=predictions)
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adam(),
                  metrics=['accuracy'])

    model_name = "inception_two_modules_" + str(N_FILTERS) + "filters"

    return (model, model_name)



def build_model_three_modules():
    input_shape_img = deep_utils.get_rgb_input_shape(IMG_ROWS, IMG_COLUMNS)
    inputs = Input(input_shape_img)

    x = Conv2D(64, (3,3), activation='relu', padding='same')(inputs)
    x = inception_module(x, n_filters=N_FILTERS)
    x = inception_module(x, n_filters=N_FILTERS)
    x = inception_module(x, n_filters=N_FILTERS)

    x  = Flatten()(x)
    x = Dropout(0.5)(x)
    predictions = Dense(2, activation='softmax')(x)

    model = Model(input=inputs, output=predictions)
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adam(),
                  metrics=['accuracy'])

    model_name = "inception_three_modules_" + str(N_FILTERS) + "filters"
    
    return (model, model_name)


def build_model_four_modules():
    input_shape_img = deep_utils.get_rgb_input_shape(IMG_ROWS, IMG_COLUMNS)
    inputs = Input(input_shape_img)

    x = Conv2D(64, (3,3), activation='relu', padding='same')(inputs)
    x = inception_module(x, n_filters=N_FILTERS)
    x = inception_module(x, n_filters=N_FILTERS)
    x = inception_module(x, n_filters=N_FILTERS)
    x = inception_module(x, n_filters=N_FILTERS)

    x  = Flatten()(x)
    x = Dropout(0.5)(x)
    predictions = Dense(2, activation='softmax')(x)

    model = Model(input=inputs, output=predictions)
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adam(),
                  metrics=['accuracy'])

    model_name = "inception_four_modules_" + str(N_FILTERS) + "filters"
    return (model, model_name)
