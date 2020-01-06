#!/usr/bin/env python

import os
import sys
import numpy as np

import deepmirna_utils as utils

NUM_CLASSES = 2

class DataLoader:

    def __init__(self):
        self.train_data_filename = "../datasets/modhsa_train_images.npz"
        self.train_labels_filename = "../datasets/modhsa_train_labels.npz"
        self.train_names_filename = "../datasets/modhsa_train_names.npz"
        self.test_data_filename = "../datasets/modhsa_test_images.npz"
        self.test_labels_filename = "../datasets/modhsa_test_labels.npz"
        self.test_names_filename = "../datasets/modhsa_test_names.npz"
        self.pretrain_data_filename = "../datasets/nonhsa_modmirbase_images.npz"
        self.pretrain_labels_filename = "../datasets/nonhsa_modmirbase_labels.npz"
        self.pretrain_names_filename = "../datasets/nonhsa_modmirbase_names.npz"

    def load_pretrain_datasets(self):
        data = utils.load_image_data(self.pretrain_data_filename)
        labels = utils.load_labels(self.pretrain_labels_filename, NUM_CLASSES)
        names = utils.load_names(self.pretrain_names_filename)
        return (data, labels, names)

    def load_train_datasets(self):
        data = utils.load_image_data(self.train_data_filename)
        labels = utils.load_labels(self.train_labels_filename, NUM_CLASSES)
        names = utils.load_names(self.train_names_filename)
        return (data, labels, names)

    def load_test_datasets(self):
        data = utils.load_image_data(self.test_data_filename)
        labels = utils.load_labels(self.test_labels_filename, NUM_CLASSES)
        names = utils.load_names(self.test_names_filename)
        return (data, labels, names)        
