#!/usr/bin/env python

from __future__ import print_function
from datetime import date
import time

import keras
import numpy as np

import utils.deepmirna_utils as deep_utils
import model_generators.inception_model_generator as model_generator
from utils.data_loader import DataLoader

TRAIN_EPOCHS = 100
BATCH_SIZE = 128
N_FILTERS = 8 #values used in our experiments: 8, 12, 16, and 20

def train(model, model_name):

    loader = DataLoader()
    train_data, train_labels, train_names = loader.load_train_datasets()
    test_data, test_labels, test_names = loader.load_test_datasets()

    model.fit(train_data, train_labels,
              validation_data=(test_data, test_labels),
              batch_size=BATCH_SIZE, epochs=TRAIN_EPOCHS)
    
    utils.create_directory("../models")
    model_filename = "../models/base_" + model_name + ".h5" 
    model.save(model_filename)

    scores = model.evaluate(test_data, test_labels, verbose=1)
    return scores

if __name__ == '__main__':
    start = time.time()
    script_name = "inception_model_trainer_base.py"
    print("******************************************************")
    print("Script: {}".format(script_name))
    print("Date: {}".format(str(date.today())))
    print("******************************************************")
    
    model, model_name = model_generator.build_model_one_module(N_FILTERS)
    scores = train(model, model_name)
    print("Results: {}, modhsa, {:.4f}".format(model_name, scores[1]))

    model, model_name = model_generator.build_model_two_modules(N_FILTERS)
    scores = train(model, model_name)
    print("Results: {}, modhsa, {:.4f}".format(model_name, scores[1]))

    model, model_name = model_generator.build_model_three_modules(N_FILTERS)
    scores = train(model, model_name)
    print("Results: {}, modhsa, {:.4f}".format(model_name, scores[1]))

    model, model_name = model_generator.build_model_four_modules(N_FILTERS)
    scores = train(model, model_name)
    print("Results: {}, modhsa, {:.4f}".format(model_name, scores[1]))
    
    end = time.time()
    elapsed_time = (end - start) / 60
    print("\nTask finished in {} minutes!!".format(elapsed_time))
