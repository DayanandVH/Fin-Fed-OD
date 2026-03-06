import warnings
warnings.filterwarnings("ignore")
import logging
logging.getLogger("requests").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

import os
import numpy as np
import argparse
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.metrics import average_precision_score
from joblib import Parallel, delayed
import json
import pickle
import pandas as pd

import matplotlib.pyplot as plt

import torch

from mnist_train_outlier_types import model_training


# seed = 42
# lim = 100


def read_and_preprocess_data(train, test_sets_data_label, ds_name):

    trainX = train.iloc[:, :-1]
    trainy = train.iloc[:, -1]
    trainy = np.array(trainy, dtype=int)

    test_sets_data, test_sets_label = [], []
    for ind, _ in enumerate(test_sets_data_label):
        i = test_sets_data_label[ind]
        testX = i.iloc[:, :-1]
        testy = i.iloc[:, -1]
        testy = np.array(testy, dtype=int)
        test_sets_data.append(testX)
        test_sets_label.append(testy)
    
    trainX, valX, trainy, valy = train_test_split(trainX, trainy, test_size=0.2, random_state=seed, stratify=trainy)

    return trainX, test_sets_data, valX, trainy, test_sets_label, valy


def read_and_preprocess_input_data(ds_pair_index, ds_name, folder_name):

    train_type1 = pd.read_csv(str(folder_name) + '/Type_1_train.csv')
    train_type2 = pd.read_csv(str(folder_name) + '/Type_2_train.csv')
    train_type3 = pd.read_csv(str(folder_name) + '/Type_3_train.csv')
    train_type4 = pd.read_csv(str(folder_name) + '/Type_4_train.csv')

    test_type1 = pd.read_csv(str(folder_name) + '/Type_1_test.csv')
    test_type2 = pd.read_csv(str(folder_name) + '/Type_2_test.csv')
    test_type3 = pd.read_csv(str(folder_name) + '/Type_3_test.csv')
    test_type4 = pd.read_csv(str(folder_name) + '/Type_4_test.csv')

    train_sets_data_label = [train_type1, train_type2, train_type3, train_type4]
    test_sets_data_label = [test_type1, test_type2, test_type3, test_type4]

    train = train_sets_data_label[ds_pair_index-1]
    trainX, test_sets_data, valX, trainy, test_sets_label, valy = read_and_preprocess_data(train, test_sets_data_label, ds_name)
    
    return trainX, test_sets_data, valX, trainy, test_sets_label, valy, ds_name


def main(ds_outlier_type, counter, seed, ds_name, input_data_path):

    # detect cuda
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu").type
    print('Using device: ' + device)

    trainX, test_sets_data, valX, trainy, test_sets_label, valy, ds_name = read_and_preprocess_input_data(ds_outlier_type, ds_name, input_data_path)
    for exp in range(0, 1):
        print("Exp: ", exp)
        epochs = 1000
        update_epoch_no = 0
        epoch_list = [50, 100, 200, 300, 500, 1000]  ## Epochs at which model output is generated, classification, and OD algorithms are applied

        z_train, z_test_sets, rec_err_train, rec_err_test_sets = model_training(trainX, trainy, test_sets_data, test_sets_label, valX, 128, 0.001, device, epochs, update_epoch_no, epoch_list, ds_outlier_type)


ds_name = 'mnist'
input_dim = int(28*28)
input_data_path = '../data/run5_MNIST_dataset_outliers_3per_inlier_0digit_outliers_3568digits'

# seeds = np.random.randint(1, 1001, size=1)
seeds = [42]
for counter, seed in enumerate(seeds):
    for ds_pair_index in range(1, 5):
        main(ds_pair_index, counter, seed, ds_name, input_data_path)
# main(ds_pair_index, start_i, end_i, counter, seed)
# main()
