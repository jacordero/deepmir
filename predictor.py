#!/usr/bin/env python

import sys
import os

from subprocess import Popen, PIPE
from pyfaidx import Fasta

import imageio
import numpy as np
import keras
from keras import backend as K
from keras.models import load_model

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
MODEL_FILENAME = CURRENT_DIR + "/models/fine_tuned_cnn.h5"
HAIRPIN_IMAGE_GENERATOR_JAR = './hairpin_image_generator/ImageCalc.jar'

def generate_hairpin_images(fasta_filename, data_directory):
    
    # create directory for new hairpin images
    images_directory = data_directory + "/images"
    try:
        if not os.path.exists(images_directory):
            os.makedirs(images_directory)
    except OSError:
        print ('Error: Creating directory {}'.format(images_directory))
        sys.exit(1)

    # generate new hairpin images
    sequences = Fasta(fasta_filename)
    for key in sequences.keys():
        generate_hairpin_image(str(sequences[key][:].seq), key, images_directory)


def generate_hairpin_image(sequence, sequence_identifier, output_directory):

    hairpin_image_name = output_directory + "/" + sequence_identifier + '.png'
    process = Popen(['java', '-jar', HAIRPIN_IMAGE_GENERATOR_JAR, '-o', 
    	hairpin_image_name, '-s', sequence], stdout=PIPE, stderr=PIPE)
    stdout, stderr = process.communicate()

    #print(stdout)
    # hairpin image was generated with success
    if stderr.decode('utf-8') != '':
        print(stderr.decode('utf-8'))

    # TODO: handle non success
        

def generate_hairpin_array(data_directory):
    
    images_directory = data_directory + "/images"
    # select images that will be converted to numpy arrays
    image_short_filenames = []
    image_long_filenames = []
    for (dirpath, dirnames, filenames) in os.walk(images_directory):
        for filename in filenames:
            if filename.endswith(".png"):
                im = imageio.imread(dirpath + "/" + filename)
                # skip if the shape is wrong
                # throw an exception here instead
                if im.shape != (25, 100, 3): 
                    continue
                    
                image_long_filenames.append(dirpath + "/" + filename)
                image_short_filenames.append(filename.split(".")[0])
            
    num_images = len(image_long_filenames)
    print("Images to process: {}".format(num_images))

    # convert selected images to numpy arrays
    images_tensor = np.zeros(shape=(num_images, 25, 100, 3), dtype=float, )
    names_tensor = np.array(image_short_filenames, dtype=np.string_)
    for image_index in range(num_images):
        image_filename = image_long_filenames[image_index]
        im = imageio.imread(image_filename)
        if im.shape == (25, 100, 3):
            images_tensor[image_index] = im
        if image_index > 0 and (image_index % 1000) == 0:
            print("Number of images processed: {}".format(image_index))
    
    output_images_filename = data_directory + "/images.npz"
    output_names_filename = data_directory + "/names.npz"
    np.savez_compressed(output_images_filename, images_tensor)
    np.savez_compressed(output_names_filename, names_tensor)
    print("Numpy arrays for images were created in: {}".format(data_directory))


def load_image_data(filename):
     X = np.load(filename)['arr_0']
    
     if K.image_data_format() == 'channels_first':
          X = np.swapaxes(X, 1, 3)

     X = X.astype('float32')
     if np.amax(X) > 1:
          X /= 255
          
     return X


def compute_predictions(data_directory):
    images = load_image_data(data_directory + "/images.npz")
    names = np.load(data_directory + "/names.npz")['arr_0']
    
    model = load_model(MODEL_FILENAME)
    predictions = np.argmax(model.predict(images), axis=1)
    
    results_filename = data_directory + "/results.csv"
    print("Writing prediction results to: {}".format(results_filename))
    with open(results_filename, 'w') as results:
        results.write("hairpin,label\n")
        
        for name, prediction in zip(names.tolist(), predictions.tolist()):
            if prediction == 1:
                label = "pre-miRNA"
            else:
                label = "not pre-miRNA"
                
            results.write("{},{}\n".format(name.decode('utf-8'), label))

          
if __name__ == "__main__":
    test_filename = "sequences.fasta"
    data_directory = CURRENT_DIR + '/data/'
    data_directory += os.path.basename(test_filename).rstrip('.fasta')
    
    generate_hairpin_images(test_filename, data_directory)
    generate_hairpin_array(data_directory)
    compute_predictions(data_directory)

    print("Predictions for sequences in {} are finished.".format(data_directory))
