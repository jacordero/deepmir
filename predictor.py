#!/usr/bin/env python

import sys
import os

from subprocess import Popen, PIPE

from pyfaidx import Fasta

APP_DIR = os.path.dirname(os.path.realpath(__file__))
HAIRPIN_GENERATOR_JAR = './hairpin_image_generator/ImageCalc.jar'

def generate_hairpin_images(filename):

	# create directory for new hairpin images

	output_directory = APP_DIR + '/hairpin_images/' + os.path.basename(filename).rstrip('.fasta')
	try:
		if not os.path.exists(output_directory):
			os.makedirs(output_directory)
	except OSError:
		print ('Error: Creating directory {}'.format(output_directory))
		sys.exit(1)

    # generate new hairpin images
	sequences = Fasta(test_filename)
	for key in sequences.keys():
		generate_hairpin_image(str(sequences[key][:].seq), key, output_directory)


def generate_hairpin_image(sequence, sequence_identifier, output_directory):

    hairpin_image_name = output_directory + "/" + sequence_identifier + '.png'
    process = Popen(['java', '-jar', HAIRPIN_GENERATOR_JAR, '-o', 
    	hairpin_image_name, '-s', sequence], stdout=PIPE, stderr=PIPE)
    stdout, stderr = process.communicate()

    print(stdout)
    # hairpin image was generated with success
    if stderr.decode('utf-8') == '':
        stdout = stdout.decode('utf-8')
        fold_info = stdout.split('\n')[1]
        fold = fold_info.split(':')[1].strip()
    # hairpin image generation failed
    else:
        print(stderr.decode('utf-8'))
        image_id = ''
        fold = ''
    return (fold, image_id)


if __name__ == "__main__":
	#test_filename = "sequences.fasta"
	test_filename = "test_seq.fasta"
	generate_hairpin_images(test_filename)
	# load images and convert to numpy files
	# 


