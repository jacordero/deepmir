#!/usr/bin/env python

from __future__ import print_function
from datetime import date
import time

import keras
import numpy as np

from utils.data_loader import DataLoader
import model_generators.resnet_model_generator as model_generator
import utils.deepmirna_utils as deep_utils

# Training parameters
PRETRAIN_EPOCHS = 40
TRAIN_EPOCHS = 100
BATCH_SIZE = 128
SAVE_MODEL = True

def train(model, model_name):

    loader = DataLoader()
    
    # pretrain model    
    pretrain_data, pretrain_labels, pretrain_names = loader.load_pretrain_datasets()
    model.fit(pretrain_data, pretrain_labels,
              batch_size=BATCH_SIZE,
              epochs=PRETRAIN_EPOCHS,
              shuffle=True)
    
    # fine tune pretrained model
    train_data, train_labels, train_names = loader.load_train_datasets()
    test_data, test_labels, test_names = loader.load_test_datasets()
    model.fit(train_data, train_labels,
              batch_size=BATCH_SIZE,
              epochs=TRAIN_EPOCHS,
              validation_data=(test_data, test_labels),
              shuffle=True)
    
    # save trained model
    if SAVE_MODEL:
        deep_utils.create_directory("../models")
        model_filename = "../models/fine_tunned_" + model_name + ".h5"
        model.save(model_filename)

    scores = model.evaluate(test_data, test_labels, verbose=1)
    return scores


if __name__ == '__main__':
    start = time.time()

    script_name = "resnet_model_trainer_fine_tunning.py"
    print("******************************************************")
    print("Running: {}".format(script_name))
    print("Date: {}".format(str(date.today())))
    print("******************************************************")

    NUM_FILTERS = 28        
    
    model, model_name = model_generator.build_model_one_module(NUM_FILTERS)
    scores = train(model, model_name)
    print("Results: {}, {}, {:.4f}".format(model_name, dataset, scores[1]))

    model, model_name = model_generator.build_model_two_modules(NUM_FILTERS)
    scores = train(model, model_name)
    print("Results: {}, {}, {:.4f}".format(model_name, dataset, scores[1]))

    model, model_name = model_generator.build_model_three_modules(NUM_FILTERS)
    scores = train(model, model_name)
    print("Results: {}, {}, {:.4f}".format(model_name, dataset, scores[1]))

    model, model_name = model_generator.build_model_four_modules(NUM_FILTERS)
    scores = train(model, model_name)
    print("Results: {}, {}, {:.4f}".format(model_name, dataset, scores[1]))
    
    end = time.time()
    elapsed_time = (end - start) / 60
    print("\nTask finished in {} minutes!!".format(elapsed_time))
