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

def compute_true_positives(predicted_labels, true_labels):
    selected_ids = []
    for counter, predicted_label in enumerate(predicted_labels):
        true_label = true_labels[counter]
        if true_label[1] > true_label[0] and predicted_label[1] > predicted_label[0]:
            selected_ids.append(counter)

    return selected_ids

def compute_true_negatives(predicted_labels, true_labels):
    selected_ids = []
    for counter, predicted_label in enumerate(predicted_labels):
        true_label = true_labels[counter]
        if true_label[0] > true_label[1] and predicted_label[0] > predicted_label[1]:
            selected_ids.append(counter)

    return selected_ids

def compute_false_positives(predicted_labels, true_labels):
    selected_ids = []
    for counter, predicted_label in enumerate(predicted_labels):
        true_label = true_labels[counter]
        # is negative and was computed positive
        if true_label[0] > true_label[1] and predicted_label[1] > predicted_label[0]:
            selected_ids.append(counter)

    return selected_ids

def compute_false_negatives(predicted_labels, true_labels):
    selected_ids = []
    for counter, predicted_label in enumerate(predicted_labels):
        true_label = true_labels[counter]
        # is positive and was computed negative
        if true_label[1] > true_label[0] and predicted_label[0] > predicted_label[1]:
            selected_ids.append(counter)

    return selected_ids

def save_stats(stats, output_filename):
    output_text = "true positives: " + str(stats['tp']) + "\n"
    output_text += "false positives: " + str(stats['fp']) + "\n"
    output_text += "true negatives: " + str(stats['tn']) + "\n"
    output_text += "false negatives: " + str(stats['fn'])
    with open(output_filename, "w") as output_file:
            output_file.write(output_text)

def save_to_file(output_text, output_filename):
     with open(output_filename, "w") as output_file:
          output_file.write(output_text)

            
def save_results_in_json(directory, category, results):
    create_directory(directory)
    with open(directory + "/" + category + ".json", "w") as f:
        json.dump({category: results}, f)

def format_performance_info(performance_values):
     performance_text = ""
     for counter, value in enumerate(performance_values):
          performance_text += str(counter) + ": " + str(np.format_float_positional(value, 2))
          
     return performance_text
