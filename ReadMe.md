

# Deep Neural Network for Classifying Normal and Abnormal X-ray Images using BYOL Self-Supervised Learning

This repository contains code to train a deep neural network for classifying normal and abnormal X-ray images using the BYOL (Bootstrap Your Own Latent) self-supervised learning technique. The model is fine-tuned using 1%, 10%, and 100% of the data available in the MURA dataset.
The code is based on [lucidrains/byol-pytorch](https://github.com/lucidrains/byol-pytorch), [pyaf/DenseNet-MURA-PyTorch](https://github.com/pyaf/DenseNet-MURA-PyTorch)


## Introduction

The purpose of this project is to train a deep learning model that can classify X-ray studies (multi-view radiographic images ) into normal and abnormal categories. We utilize the BYOL self-supervised learning approach, which enables the model to learn meaningful representations from unlabeled data and then fine-tune the model using labeled data.

## Dataset

The dataset used in this project is the "MURA" dataset, which contains a large collection of X-ray images. It consists of both normal and abnormal X-ray images, making it suitable for binary classification tasks.

Please download the "MURA" dataset from [here](https://stanfordmlgroup.github.io/competitions/mura/) and organize it according to the directory structure specified in the Usage section.

## Installation

1.  Clone this repository to your local machine:

bashCopy code

`git clone https://github.com/assafcaf/MuraBYOL.git
cd mura-byol` 

2.  Create a virtual environment (optional but recommended) and install the required dependencies:

bashCopy code

``conda create -n mura-byol``

``conda activate mura-byol``

``pip install -r requirements.txt`` 

## Usage

Before running the training or evaluation scripts, please ensure that the dataset is correctly prepared and located in the proper location by utilizing the `dataPreprocessing.py` script in the source directory

To utilized the script run `python ./src/dataPreprocessing.py --data-in <path_to_mura_files>`

## Training BYOL

To train the BYOL model on the MURA dataset, use the following command:

bashCopy code

`python .\src\trainBoylModel.py`
 
 For more advanced parameter tuning pls you can run `python .\src\trainBoylModel.py --help`
 
 to modify the command-line arguments as per your requirements, such as the data directory, batch size, and learning rate.

## Finetune
To finetune the model use the following command:
bashCopy code
` python .\src\finetuneBoylModel.py --model-dir <path_to_byol_model>`

For example --model-dir = `"trained_models\boyl\model1\improved-resnet18.pt"`

For more advanced parameter tuning you can run `python .\src\finetuneBoylModel.py --help` to modify the command-line arguments as per your requirements, such as the data directory, batch size, and learning rate.
## Evaluation

To evaluate the pretrained model using  Linear classifaier on the test set, execute the evaluation script:

bashCopy code
`python .\src\linear_representation_evaluation.py --model-pth<path_to_fine_tune_model>`

For example:
 `model-pth = trained_models\boyl\model1\improved-resnet18.pt`

To evaluate the finetuned model on the test set, execute the evaluation script:

bashCopy code
`python .\src\evaluate_BoylMode.py --model-pth<path_to_fine_tune_model>` 

for example: path_to_fine_tune_model = `\trained_models\boyl\model1\model-finetuned-processed_100.pt`


 

for example: path_to_fine_tune_model = `\trained_models\boyl\model1\model-finetuned-processed_100.pt`
## Baseline

To train the baseline model (densnet169) on the MURA dataset, use the following command:
bashCopy code

`python .\src\trainBoylModel.py --data-in MURA-v1.1 `
 
## Results

| Finetune on: | accuracy | recall | precision|  F1| 
|--|--|--|--|--|
| 1% | 0.71816 | 0.753354 | 0.798687 | 0.712891
| 10% | 0.757075 |  0.793587|  0.866521| 0.731978
| 100% | 0.76533 |  0.793782| 0.838074 | 0.753937