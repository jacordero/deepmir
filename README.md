## DeepMir


This repository contains the source code required to replicate the experiments described in the pre-print submited to bioRxiv [Detection of pre-microRNA with Convolutional Neural Networks](https://www.biorxiv.org/content/10.1101/840579v1). Additionally, it contains a ready to use implementation of the pre-miRNA detection framework proposed in the pre-print.

The content of this project is organized as follows:
* the *model_selection* directory contains the scripts used to train and evaluate several CNNs based on VGG, Inception, and ResNet architectures.
* the *hairpin_image_generator* directory contains the JAR file used for hairpin image generation.
* The *notebooks* directory contains Jupyter notebooks used to analize CNN models and generate images included in the pre-print.
* The *datasets* directory contains the datasets used to train our best CNN model. Additionally, the data inside *benchmark* was used to evaluate the performance of the CNN model.
* The *models* directory contains the best CNN model obtained by using a fine-tuning procedure during training. The base CNN model (without fine-tuning) is also included here.
* The *examples* directory contains examples of fasta files that the script *predictor.py* can process.
* The *predictor.py* script is an implementation of the pre-miRNA detection framework described in the pre-print.

### Dependencies

This project requires:
* python 3.5 or later
* pyfaidx 0.5.7
* imageio 2.5.0
* xldr
* Keras 2.2.4 
* tensorflow 1.14.0
* ipython 7.9.0
* scikit-learn 0.21.3
* matplotlib 3.0.3
* pandas 0.24.2
* jupyter 5.7 or later

### Usage

To use the pre-miRNA detection framework run
```
python predictor [filename.fasta]
```
If the input file is successfully processed, the prediction results will be available in the file *user_data/filename/results.csv*. Additionally, the hairpin images corresponding to the sequences defined in *filename.fasta* will be available in the directory *user_data/filename/images*. Numpy files containing the hairpin images encoded as arrays and their corresponding names encoded will also be available as *user_data/filename/images.npz* and *user_data/filename/names.npz*.


To train new CNN models run the trainer scripts inside the directory *model_selection*. For example,
```
python vgg_model_trainer_fine_tuning.py
```
This script will train several models based on the VGG architecture using fine-tuning. The resulting models will be available in the *models* directory. To modify the resulting CNN models, modify the constants defined in each trainer script: *TRAIN_EPOCHS*, *PRETRAIN_EPOCHS*, *BATCH_SIZE*, *N_FILTERS*, and *DENSE_UNITS*.

The script *model_selection/benchmark_fine_tuned_model.py* computes the performance of *fine-tuning-CNN* using several benchmark datasets.

**Note**: this repository contains some relatively large datasets such as *nonhsa_modmirbase_images.npz*, *modmirbase_train_images.npz*, *mirbase_images.npz*, and *neghsa_images.npz*. Take this into consideration when working with this project. 
