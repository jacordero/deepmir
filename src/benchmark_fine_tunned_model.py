#!/usr/bin/env python

import keras
from keras import backend as K
from keras.models import load_model

import numpy as np

import utlis.deepmirna_utils as deep_utils

MODEL_FILENAME = "../models/fine_tunned_cnn.h5"
DATA_DIR = "../datasets/"

def evaluate(images_filename, labels_filename, names_filename):
    images = deep_utils.load_image_data(images_filename)
    labels = deep_utils.load_labels(labels_filename, 2)
    names = np.load(names_filename)['arr_0']

    model = load_model(MODEL_FILENAME)
    return model.evaluate(images, labels)    


def evaluate_hsa():
    print("** evaluate hsaplus **")
    images_filename = DATA_DIR + "hsa_images.npz"
    labels_filename = DATA_DIR + "hsa_labels.npz"
    names_filename = DATA_DIR + "hsa_names.npz"

    loss, acc = evaluate(images_filename, labels_filename, names_filename)
    return acc


def evaluate_mirbase():
    print("** evaluate allmirbase **")
    images_filename = DATA_DIR + "mirbase_images.npz"
    labels_filename = DATA_DIR + "mirbase_labels.npz"
    names_filename = DATA_DIR + "mirbase_names.npz"
    loss, acc = evaluate(images_filename, labels_filename, names_filename)
    return acc


def evaluate_mirgenedb():
    print("** evaluate mirgenedb **")
    images_filename = DATA_DIR + "mirgenedb_images.npz"
    labels_filename = DATA_DIR + "mirgenedb_labels.npz"
    names_filename = DATA_DIR + "mirgenedb_names.npz"
    loss, acc = evaluate(images_filename, labels_filename, names_filename)
    return acc

    
def run():
    performance = {}

    performance['mirbase'] = evaluate_mirbase()
    performance['mirgenedb'] = evaluate_mirgenedb()
    performance['hsa'] = evaluate_hsa()
    
    for key in sorted(performance.keys()):
        print("{:12s} {:.2f}".format(key, 100*performance[key]))
    
if __name__ == '__main__':
    run()
