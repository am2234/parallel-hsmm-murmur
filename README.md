# A recurrent neural network and parallel hidden Markov model algorithm to detect heart murmurs

Andrew McDonald, Mark Gales, Anurag Agarwal

## What's in this repository?

This code uses four main scripts, described below, to train and run the [Cambridge University Engineering Department (CUED) Acoustics Lab](http://acoustics.eng.cam.ac.uk/) entry for the [2022 PhysioNet Challenge](https://moody-challenge.physionet.org/2022/).

The algorithm is a hybrid recurrent neural network and hidden semi-Markov model to both segment and detect heart murmurs.

The `results` folder contains scripts to generate results and figures used in the research article 'A recurrent neural network and parallel hidden Markov model algorithm to segment and detect heart murmurs in phonocardiograms'. To run the scripts in `results` you will need to set the enviroment variable `CIRCOR_FOLDER` to the path to your local copy of the training data. The `final_model` folder contains the outputs of the algorithm on the full training set.


## How do I run these scripts?

The below instructions are adapted from the official challenge instructions.

You can install the dependencies for these scripts by creating a Docker image (see below) and running

    pip install requirements.txt

You can train your model by running

    python train_model.py training_data model

where

- `training_data` (input; required) is a folder with the training data files and
- `model` (output; required) is a folder for saving your model.

You can test your trained model by running

    python run_model.py model test_data test_outputs

where

- `model` (input; required) is a folder for loading your model, and
- `test_data` (input; required) is a folder with the validation or test data files (you can use the training data for debugging and cross-validation), and
- `test_outputs` is a folder for saving your model outputs.

The [2022 Challenge website](https://physionetchallenges.org/2022/#data) provides a training database with a description of the contents and structure of the data files.

You can evaluate your model by running the official [evaluation code](https://github.com/physionetchallenges/evaluation-2022) (included in this repo)

    python evaluate_model.py labels outputs scores.csv class_scores.csv

where `labels` is a folder with labels for the data, such as the training database on the PhysioNet webpage; `outputs` is a folder containing files with your model's outputs for the data; `scores.csv` (optional) is a collection of scores for your model; and `class_scores.csv` (optional) is a collection of per-class scores for your model.

## Docker

requires nvidia-docker2


## Acknowledgements

This work expands upon the PhD research of Dr Edmund Kay at Cambridge University Engineering Department. The results in his thesis inspired many of the improvements in this work.