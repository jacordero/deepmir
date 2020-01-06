# deepmir
Detection of pre-microRNA with Convolutional Neural Networks

This repository includes the source code used to train the CNN models described in https://www.biorxiv.org/content/10.1101/840579v1.

Requirements:
* python3.5+
* pyfaidx-0.5.7
* imageio-2.5.0 
* Keras-2.2.4 
* tensorflow-1.14.0
* jupyter-5.7.8+
* ipython-7.9.0
* scikit-learn-0.21.3
* matplotlib-3.0.3
* pandas-0.24.2

The script *src/benchmark_fine_tunned_model.py* evaluates the peformance of the best CNN on different datasets. Additionally, the src directory contains scripts to train several CNN models. The architecture of these CNNs is defined in scripts contained in the *src/model_generators/* directory.

