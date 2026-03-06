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

from train_outlier_types import model_training


# seed = 42
# lim = 100

def scaler_mixdatatype(trainX, trainy, test_sets_data, test_sets_label, num_features, cat_features):
    # Standardizing and one-hot encoding the input data before passing to the model training
    ohe = OneHotEncoder(sparse=False, handle_unknown="ignore")
    trainX_one_hot = ohe.fit_transform(trainX[cat_features])
    trainX_one_hot = pd.DataFrame(trainX_one_hot, columns=ohe.get_feature_names(cat_features))
    data_with_onehot = pd.concat([trainX_one_hot, trainX[num_features]], axis=1)

    trainX, valX, trainy, valy = train_test_split(data_with_onehot, trainy, test_size=0.1, random_state=seed, stratify=trainy)

    scaler = StandardScaler()
    scaler.fit(trainX[num_features])
    num_trainX_transformed = scaler.transform(trainX[num_features])
    num_trainX_transformed = pd.DataFrame(num_trainX_transformed, columns=num_features)
    num_valX_transformed = scaler.transform(valX[num_features])
    num_valX_transformed = pd.DataFrame(num_valX_transformed, columns=num_features)
    cat_trainX = trainX.drop(num_features, axis=1, inplace=False)
    cat_trainX = cat_trainX.reset_index(drop=True)
    cat_valX = valX.drop(num_features, axis=1, inplace=False)
    cat_valX = cat_valX.reset_index(drop=True)
    trainX = pd.concat([cat_trainX, num_trainX_transformed], axis=1)
    valX = pd.concat([cat_valX, num_valX_transformed], axis=1)
    print("trainX: ", trainX.shape)
    print("valX: ", valX.shape)

    normalized_test_sets_data = []
    for ind, i in enumerate(test_sets_data):
        testX = test_sets_data[ind]
        testX_one_hot = ohe.transform(testX[cat_features])
        testX_one_hot = pd.DataFrame(testX_one_hot, columns=ohe.get_feature_names(cat_features))
        data_with_onehot = pd.concat([testX_one_hot, testX[num_features]], axis=1)
        num_testX_transformed = scaler.transform(data_with_onehot[num_features])
        num_testX_transformed = pd.DataFrame(num_testX_transformed, columns=num_features)
        cat_testX = data_with_onehot.drop(num_features, axis=1, inplace=False)
        cat_testX = cat_testX.reset_index(drop=True)
        testX = pd.concat([cat_testX, num_testX_transformed], axis=1)
        print("testX: ", testX.shape)
        normalized_test_sets_data.append(testX)

    return trainX, normalized_test_sets_data, valX, trainy, test_sets_label, valy, scaler


def read_and_preprocess_data(train, test_sets_data_label, ds_name):
    if ds_name == 'credit_default':
        num_features = ['BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6', 'LIMIT_BAL']
        cat_features = ['SEX', 'EDUCATION', 'MARRIAGE', 'AGE', 'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']
        label_col_name = 'default payment next month'
    # for i in list(ds.columns): ds[i] = pd.to_numeric(ds[i]) #Only needed for Credit Default
    if ds_name == 'adult_data':
        num_features = ['age', 'fnlwgt', 'Education-num', 'Capital-gain', 'Capital-loss', 'Hours-per-week']
        cat_features = ['workclass', 'education', 'Marital-status', 'occupation', 'relationship', 'race', 'sex', 'Native-country']
        label_col_name = 'Class'

    trainX = train[cat_features + num_features]
    trainy = train.iloc[:, -1]
    trainy = np.array(trainy)
    # print("trainX: ", len(trainX))
    # print("trainX: ", trainX.shape)
    # print("trainy: ", len(trainy))
    # print("trainy: ", trainy.shape)
    cat_uniqval_count = [len(set(list(trainX[i].values))) for i in cat_features]

    test_sets_data, test_sets_label = [], []
    for ind, _ in enumerate(test_sets_data_label):
        i = test_sets_data_label[ind]
        testX = i[cat_features + num_features]
        testy = i.iloc[:, -1]
        testy = np.array(testy)
        # print("testX: ", len(testX))
        # print("testX: ", testX.shape)
        # print("testy: ", len(testy))
        # print("testy: ", testy.shape)
        test_sets_data.append(testX)
        test_sets_label.append(testy)

    # sd_trainX_ds, sd_testX_ds, sd_valX_ds, sd_outlier_ds, cat_cols_one_hot_ds, std_scaler_ds, std_sample_ds = standard_scaler_mixdatatype(data, label, num_features_ds, cat_features_ds, sel_ind_ds)
    # mm_trainX_ds, mm_testX_ds, mm_valX_ds, mm_outlier_ds, cat_cols_one_hot_ds, mm_scaler_ds, mm_sample_ds = minmax_scaler_mixdatatype(data, label, num_features_ds, cat_features_ds, sel_ind_ds)

    trainX, test_sets_data, valX, trainy, test_sets_label, valy, std_scaler_ds = scaler_mixdatatype(trainX, trainy, test_sets_data, test_sets_label, num_features, cat_features)

    return trainX, test_sets_data, valX, trainy, test_sets_label, valy, cat_uniqval_count


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
    trainX, test_sets_data, valX, trainy, test_sets_label, valy, cat_uniqval_count = read_and_preprocess_data(train, test_sets_data_label, ds_name)
    
    return trainX, test_sets_data, valX, trainy, test_sets_label, valy, cat_uniqval_count, ds_name


def main(run_no, ds_outlier_type, counter, seed, ds_name, input_data_path):

    # detect cuda
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu").type
    print('Using device: ' + device)

    trainX, test_sets_data, valX, trainy, test_sets_label, valy, cat_uniqval_count, ds_name = read_and_preprocess_input_data(ds_outlier_type, ds_name, input_data_path)
    for exp in range(0, 1):
        print("Exp: ", exp)
        # epochs = 1000
        # update_epoch_no = 0
        epoch_list = [50, 100, 200, 300, 500, 1000]  ## Epochs at which model output is generated, classification, and OD algorithms are applied

        # z_train, z_test_sets, rec_err_train, rec_err_test_sets = model_training(trainX, trainy, test_sets_data, test_sets_label, valX, 128, 0.001, device, epochs, update_epoch_no, epoch_list, ds_outlier_type)
        model_training(trainX, trainy, test_sets_data, test_sets_label, valX, valy, cat_uniqval_count, run_no, ds_name, device, epoch_list, ds_outlier_type)


# Creating an ArgumentParser Object
parser = argparse.ArgumentParser()
# Fetching the arguments
parser.add_argument('ds_ind', help = 'Enter DS number', type = int)
args = parser.parse_args()
# epochs = args.total_epochs
# fl_rounds = args.rounds
ds_index = args.ds_ind

for i in range(1, 2):

    run_no = 1

    if ds_index == 1:
        ds_name = 'credit_default'
        input_data_path = '/netscratch/herurkar/FL_AE_OD/data/run' + str(run_no) + '_synthetic_outlier_CD_dataset_outliers_3per_corr_cat_cols_25_corr_num_cols_25_per_types_1missing_2lessprob_3gaussian_4scaling'
    if ds_index == 2:
        ds_name = 'adult_data'
        input_data_path = '/netscratch/herurkar/FL_AE_OD/data/run' + str(run_no) + '_synthetic_outlier_AD_dataset_outliers_3per_corr_cat_cols_25_corr_num_cols_25_per_types_1missing_2lessprob_3gaussian_4scaling'

    # seeds = np.random.randint(1, 1001, size=1)
    seeds = [42]
    for counter, seed in enumerate(seeds):
        for ds_pair_index in range(1, 5):
            main(run_no, ds_pair_index, counter, seed, ds_name, input_data_path)
    # main(ds_pair_index, start_i, end_i, counter, seed)
    # main()
