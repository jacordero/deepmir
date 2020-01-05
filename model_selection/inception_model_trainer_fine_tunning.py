#!/usr/bin/env python

from __future__ import print_function
from datetime import date
import time

import keras
import numpy as np

import utils.deepmirna_utils as deep_utils
import model_generators.inception_model_generator as model_generator
from utils.data_loader import DataLoader

PRETRAIN_EPOCHS = 40
TRAIN_EPOCHS = 100
BATCH_SIZE = 128
SAVE_MODEL = True

def train(model, model_name):

    loader = DataLoader()
    pretrain_data, pretrain_labels, pretrain_names = loader.load_pretrain_datasets()
    
    # pretrain model
    model.fit(pretrain_data, pretrain_labels,
              batch_size=BATCH_SIZE, epochs=PRETRAIN_EPOCHS)
    
    # fine tune pretrained model
    train_data, train_labels, train_names = loader.load_train_datasets()
    test_data, test_labels, test_names = loader.load_test_datasets()
    model.fit(train_data, train_labels,
              validation_data=(test_data, test_labels),
              batch_size=BATCH_SIZE, epochs=TRAIN_EPOCHS)
    
    if SAVE_MODEL:
        utils.create_directory("../models")
        model_filename = "../models/fine_tunned_" + model_name + ".h5" 
        model.save(model_filename)

    scores = model.evaluate(test_data, test_labels, verbose=1)
    return scores

if __name__ == '__main__':
    start = time.time()
    script_name = "inception_model_trainer_fine_tunning.py"
    print("******************************************************")
    print("Script: {}".format(script_name))
    print("Date: {}".format(str(date.today())))
    print("******************************************************")
    
    model, model_name = model_generator.build_model_one_module()
    scores = train(model, model_name)
    print("Results: {}, hsa, {:.4f}".format(model_name, scores[1]))

    model, model_name = model_generator.build_model_two_modules()
    scores = train(model, model_name)
    print("Results: {}, hsa, {:.4f}".format(model_name, scores[1]))

    model, model_name = model_generator.build_model_three_modules()
    scores = train(model, model_name)
    print("Results: {}, hsa, {:.4f}".format(model_name, scores[1]))

    model, model_name = model_generator.build_model_four_modules()
    scores = train(model, model_name)
    print("Results: {}, hsa, {:.4f}".format(model_name, scores[1]))
    
    end = time.time()
    elapsed_time = (end - start) / 60
    print("\nTask finished in {} minutes!!".format(elapsed_time))
