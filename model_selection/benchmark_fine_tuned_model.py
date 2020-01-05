#!/usr/bin/env python

import keras
from keras import backend as K
from keras.models import load_model

import numpy as np

import utils.deepmirna_utils as deep_utils

MODEL_FILENAME = "../models/fine_tuned_cnn.h5"
DATA_DIR = "../datasets/"

def evaluate(images_filename, labels_filename, names_filename):
    images = deep_utils.load_image_data(images_filename)
    labels = deep_utils.load_labels(labels_filename, 2)
    names = np.load(names_filename)['arr_0']

    model = load_model(MODEL_FILENAME)
    return model.evaluate(images, labels)    


def evaluate_chen():
    print("** evaluate chen **")
    images_filename = DATA_DIR + "chen_images.npz"
    labels_filename = DATA_DIR + "chen_labels.npz"
    names_filename = DATA_DIR + "chen_names.npz"

    loss, acc = evaluate(images_filename, labels_filename, names_filename)
    return acc


def evaluate_dreplus():
    print("** evaluate dreplus **")
    images_filename = DATA_DIR + "dreplus_images.npz"
    labels_filename = DATA_DIR + "dreplus_labels.npz"
    names_filename = DATA_DIR + "dreplus_names.npz"

    loss, acc = evaluate(images_filename, labels_filename, names_filename)
    return acc


def evaluate_ggaplus():
    print("** evaluate ggaplus **")
    images_filename = DATA_DIR + "ggaplus_images.npz"
    labels_filename = DATA_DIR + "ggaplus_labels.npz"
    names_filename = DATA_DIR + "ggaplus_names.npz"

    loss, acc = evaluate(images_filename, labels_filename, names_filename)
    return acc


def evaluate_hsa():
    print("** evaluate hsa **")
    images_filename = DATA_DIR + "hsa_images.npz"
    labels_filename = DATA_DIR + "hsa_labels.npz"
    names_filename = DATA_DIR + "hsa_names.npz"

    loss, acc = evaluate(images_filename, labels_filename, names_filename)
    return acc


def evaluate_hsaplus():
    print("** evaluate hsaplus **")
    images_filename = DATA_DIR + "hsaplus_images.npz"
    labels_filename = DATA_DIR + "hsaplus_labels.npz"
    names_filename = DATA_DIR + "hsaplus_names.npz"

    loss, acc = evaluate(images_filename, labels_filename, names_filename)
    return acc


def evaluate_mirbase():
    print("** evaluate mirbase **")
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


def evaluate_mmu():
    print("** evaluate mmu **")
    images_filename = DATA_DIR + "mmu_images.npz"
    labels_filename = DATA_DIR + "mmu_labels.npz"
    names_filename = DATA_DIR + "mmu_names.npz"

    loss, acc = evaluate(images_filename, labels_filename, names_filename)
    return acc


def evaluate_mmuplus():
    print("** evaluate mmuplus **")
    images_filename = DATA_DIR + "mmuplus_images.npz"
    labels_filename = DATA_DIR + "mmuplus_labels.npz"
    names_filename = DATA_DIR + "mmuplus_names.npz"

    loss, acc = evaluate(images_filename, labels_filename, names_filename)
    return acc


def evaluate_mmustar():
    print("** evaluate mmustar **")
    images_filename = DATA_DIR + "mmustar_images.npz"
    labels_filename = DATA_DIR + "mmustar_labels.npz"
    names_filename = DATA_DIR + "mmustar_names.npz"

    loss, acc = evaluate(images_filename, labels_filename, names_filename)
    return acc


def evaluate_neghsa():
    print("** evaluate neghsa **")
    images_filename = DATA_DIR + "neghsa_images.npz"
    labels_filename = DATA_DIR + "neghsa_labels.npz"
    names_filename = DATA_DIR + "neghsa_names.npz"

    loss, acc = evaluate(images_filename, labels_filename, names_filename)
    return acc


def evaluate_notbestfold():
    print("** evaluate notbestfold **")
    images_filename = DATA_DIR + "notbestfold_images.npz"
    labels_filename = DATA_DIR + "notbestfold_labels.npz"
    names_filename = DATA_DIR + "notbestfold_names.npz"

    loss, acc = evaluate(images_filename, labels_filename, names_filename)
    return acc


def evaluate_pseudo():
    print("** evaluate pseudo **")
    images_filename = DATA_DIR + "pseudo_images.npz"
    labels_filename = DATA_DIR + "pseudo_labels.npz"
    names_filename = DATA_DIR + "pseudo_names.npz"

    loss, acc = evaluate(images_filename, labels_filename, names_filename)
    return acc


def evaluate_shuffled():
    print("** evaluate shuffled **")
    images_filename = DATA_DIR + "shuffled_images.npz"
    labels_filename = DATA_DIR + "shuffled_labels.npz"
    names_filename = DATA_DIR + "shuffled_names.npz"

    loss, acc = evaluate(images_filename, labels_filename, names_filename)
    return acc


def evaluate_zou():
    print("** evaluate zou **")
    images_filename = DATA_DIR + "zou_images.npz"
    labels_filename = DATA_DIR + "zou_labels.npz"
    names_filename = DATA_DIR + "zou_names.npz"

    loss, acc = evaluate(images_filename, labels_filename, names_filename)
    return acc


# The evaluation methods for neghsa and mirbase require a huge a amount of memory
# That's why they are commented. We recommend to run these methods alone.
def run_evaluations():
    #performance['mirbase'] = evaluate_mirbase()
    #performance['neghsa'] = evaluate_neghsa()
    
    performance = {}
    performance['chen'] = evaluate_chen()
    performance['dreplus'] = evaluate_dreplus()
    performance['ggaplus'] = evaluate_ggaplus()
    performance['hsa'] = evaluate_hsa()
    performance['hsaplus'] = evaluate_hsaplus()    
    performance['mirgenedb'] = evaluate_mirgenedb()
    performance['mmu'] = evaluate_mmu()
    performance['mmuplus'] = evaluate_mmuplus()
    performance['mmustar'] = evaluate_mmustar()
    performance['notbestfold'] = evaluate_notbestfold()
    performance['pseudo'] = evaluate_pseudo()
    performance['shuffled'] = evaluate_shuffled()
    performance['zou'] = evaluate_zou()
    
    for key in sorted(performance.keys()):
        print("{:12s} {:.2f}".format(key, 100 * performance[key]))
    
if __name__ == '__main__':
    run_evaluations()
