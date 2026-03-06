import logging
import warnings
logging.captureWarnings(True)
warnings.filterwarnings("ignore")

import os
import time
import argparse
import numpy as np
import json
import pickle
from collections import OrderedDict
from typing import List, Tuple
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import SVC
# from pyod.models.lof import LOF
# from pyod.models.knn import KNN
# from pyod.models.cblof import CBLOF
# from pyod.models.hbos import HBOS
# from pyod.models.ocsvm import OCSVM
from pyod.models.iforest import IForest
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import auc
from sklearn.metrics import classification_report
from sklearn.metrics import average_precision_score
import pandas as pd

import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.optim as optim
# from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR

import flwr as fl
from flwr.common import Metrics

import matplotlib.pyplot as plt

from mnist_autoencoder_init import AutoEncoder


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu").type
print('Using device: ' + device)
# DEVICE = torch.device("cpu")


# def draw_and_store_plots(train_loss, val_loss, test_loss, epochs, ds_outlier_type, file_path):
def draw_and_store_plots(train_loss, val_loss, epochs, cid, file_path):

    # if not os.path.exists(str(file_path) + '/visualization_plots'): os.mkdir(str(file_path) + '/visualization_plots')
    # vis_plot_path = str(file_path)+'/visualization_plots'

    y_max = np.max(np.hstack((train_loss, val_loss))) + 0.01
    y_min = np.min(np.hstack((train_loss, val_loss))) - 0.01
    plt.plot(range(1, epochs+1), train_loss, label='Train_loss')
    plt.plot(range(1, epochs+1), val_loss, alpha=0.5, label='Val_loss')
    # plt.scatter(epochs, test_loss, label='Test_loss')
    # plt.annotate(test_loss, (epochs, test_loss))
    plt.legend(loc='best')
    plt.ylim(y_min, y_max)
    plt.xlabel('Epochs')
    plt.ylabel('Error Value')
    plt.title('Train_Test_Loss_Curve')
    ind = int(int(cid)+1)
    plt.savefig(str(file_path)+'/Train_test_curve_epoch_'+str(epochs)+'_for_Client'+str(ind)+'.png')
    # plt.show()
    plt.close()


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


def load_datasets(batch_size, ds_name, folder_name):
    

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

    # Split each partition into train/val and create DataLoader
    trainloaders = []
    valloaders = []
    testloaders = []
    full_train_data = []
    trainy_sets, valy_sets, testy_sets = [], [], []
    cat_uniqval_loader = []
    for train_set in train_sets_data_label:
        trainX, test_sets_data, valX, trainy, test_sets_label, valy = read_and_preprocess_data(train_set, test_sets_data_label, ds_name)
        test_sets = []
        test_sets_data1 = []
        for ind, i in enumerate(test_sets_data):
            i = pd.DataFrame(i)
            test_sets_data1.append(i)

        x_torch_train = torch.FloatTensor(trainX.astype(np.float32).values).to(device)
        trainloaders.append(DataLoader(dataset=x_torch_train, batch_size=batch_size, shuffle=False))
        full_train_data.append(torch.FloatTensor(trainX.astype(np.float32).values).to(device))
        valloaders.append(torch.FloatTensor(valX.astype(np.float32).values).to(device))
        for ind, i in enumerate(test_sets_data1):
            test_sets.append(torch.FloatTensor(i.astype(np.float32).values).to(device))
        testloaders.append(test_sets)
        trainy_sets.append(trainy)
        valy_sets.append(valy)
        testy_sets.append(test_sets_label)
    
    return trainloaders, valloaders, testloaders, full_train_data, trainy_sets, valy_sets, testy_sets


def od_model_for_latent_classification(trainX, test_sets, trainy, test_sets_label, cid, epoch, op_path):

    clf = IForest(n_estimators=100, max_samples="auto", contamination=0.3, max_features=1., behaviour='old', random_state=42)
    clf.fit(trainX)

    # result_inliers = pd.DataFrame(columns=['Outlier_type', 'Precision', 'Recall', 'f1-score', 'Average_Precision_Score'])
    # result_outliers = pd.DataFrame(columns=['Outlier_type', 'Precision', 'Recall', 'f1-score', 'Average_Precision_Score'])
    result_outliers = pd.DataFrame(columns=['Client', 'Precision', 'Recall', 'f1-score', 'Pr-auc', 'Roc-auc', 'Average_Precision_Score'])
    for ind, i in enumerate(test_sets):
        testX = test_sets[ind]
        testy = test_sets_label[ind]
        pred_test = clf.predict(testX)
        outlierness_score = clf.predict_proba(testX)
        outlierness_score = outlierness_score[:, 1]
        precision, recall, _ = precision_recall_curve(np.array(testy), np.array(outlierness_score), pos_label=1)
        f1 = f1_score(testy, pred_test)
        pr_auc = auc(recall, precision)
        roc_auc = roc_auc_score(testy, outlierness_score)
        fpr, tpr, _ = roc_curve(testy, outlierness_score)
        prec = precision_score(testy, pred_test)
        rec = recall_score(testy, pred_test)
        report = classification_report(testy, pred_test, output_dict=True)
        ap = average_precision_score(testy, outlierness_score)
        # ap = average_precision_score(testy, rec_test)
        # result_inliers.loc[ind] = ['Type' + str(ind + 1), report["0"]['precision'], report["0"]['recall'], report["0"]['f1-score']] + [ap]
        # result_outliers.loc[ind] = ['Type' + str(ind + 1), report["1"]['precision'], report["1"]['recall'], report["1"]['f1-score']] + [ap]
        result_outliers.loc[ind] = ['Client' + str(ind + 1), prec, rec, f1, pr_auc, roc_auc, ap]
    
    result_other_outliers = result_outliers.drop(labels=int(cid), axis=0, inplace=False)
    # result_outliers.loc[len(result_outliers)] = ['Avg_Score_of_Other_Outliers', np.mean(result_other_outliers['Precision'].values), np.mean(result_other_outliers['Recall'].values), np.mean(result_other_outliers['f1-score'].values), np.mean(result_other_outliers['Average_Precision_Score'].values)]
    result_outliers.loc[len(result_outliers)] = ['Avg_Score_of_Other_Outliers', np.mean(result_other_outliers['Precision'].values), np.mean(result_other_outliers['Recall'].values), np.mean(result_other_outliers['f1-score'].values), np.mean(result_other_outliers['Pr-auc'].values), np.mean(result_other_outliers['Roc-auc'].values), np.mean(result_other_outliers['Average_Precision_Score'].values)]
    ind1 = int(int(cid)+1)
    result_outliers.round(2).to_csv(str(op_path) + '/OD_Model_Result_on_Latentspace_Outliers_Type' + str(ind1) + '_train_data_at_epoch_'+str(epoch) + '.csv')


def classification_model_at_latent(trainX, test_sets, trainy, test_sets_label, cid, epoch, op_path):

    # clf = SVC(gamma='auto')
    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(trainX, trainy)

    # result_inliers = pd.DataFrame(columns=['Outlier_type', 'Precision', 'Recall', 'f1-score', 'Average_Precision_Score'])
    # result_outliers = pd.DataFrame(columns=['Outlier_type', 'Precision', 'Recall', 'f1-score', 'Average_Precision_Score'])
    result_outliers = pd.DataFrame(columns=['Client', 'Precision', 'Recall', 'f1-score', 'Pr-auc', 'Roc-auc', 'Average_Precision_Score'])
    for ind, i in enumerate(test_sets):
        testX = test_sets[ind]
        testy = test_sets_label[ind]
        pred_test = clf.predict(testX)
        outlierness_score = clf.predict_proba(testX)
        outlierness_score = outlierness_score[:, 1]
        precision, recall, _ = precision_recall_curve(np.array(testy), np.array(outlierness_score), pos_label=1)
        f1 = f1_score(testy, pred_test)
        pr_auc = auc(recall, precision)
        roc_auc = roc_auc_score(testy, outlierness_score)
        fpr, tpr, _ = roc_curve(testy, outlierness_score)
        prec = precision_score(testy, pred_test)
        rec = recall_score(testy, pred_test)
        # print("Classification Result: ", classification_report(testy, pred_test, output_dict=True))
        report = classification_report(testy, pred_test, output_dict=True)
        ap = average_precision_score(testy, outlierness_score)
        # result_inliers.loc[ind] = ['Type'+str(ind+1), report["0"]['precision'], report["0"]['recall'], report["0"]['f1-score']] + [ap]
        # result_outliers.loc[ind] = ['Type' + str(ind + 1), report["1"]['precision'], report["1"]['recall'], report["1"]['f1-score']] + [ap]
        result_outliers.loc[ind] = ['Client' + str(ind + 1), prec, rec, f1, pr_auc, roc_auc, ap]
    
    result_other_outliers = result_outliers.drop(labels=int(cid), axis=0, inplace=False)
    # result_outliers.loc[len(result_outliers)] = ['Avg_Score_of_Other_Outliers', np.mean(result_other_outliers['Precision'].values), np.mean(result_other_outliers['Recall'].values), np.mean(result_other_outliers['f1-score'].values), np.mean(result_other_outliers['Average_Precision_Score'].values)]
    result_outliers.loc[len(result_outliers)] = ['Avg_Score_of_Other_Outliers', np.mean(result_other_outliers['Precision'].values), np.mean(result_other_outliers['Recall'].values), np.mean(result_other_outliers['f1-score'].values), np.mean(result_other_outliers['Pr-auc'].values), np.mean(result_other_outliers['Roc-auc'].values), np.mean(result_other_outliers['Average_Precision_Score'].values)]
    ind1 = int(int(cid)+1)
    result_outliers.round(2).to_csv(str(op_path) + '/Supervised_Classification_Model_Result_Outliers_Type' + str(ind1) + '_train_data_at_epoch_'+str(epoch) + '.csv')

    if not os.path.exists(str(op_path) + '/latent_representation'): os.mkdir(str(op_path) + '/latent_representation')
    latent_path = str(op_path) + '/latent_representation'
    with open(str(latent_path)+'/Train_Set_Model_Latent_Representation_Outliers_Type' + str(ind1) + '_train_data_at_epoch_'+str(epoch) + '.npy', 'wb') as f:
        np.save(f, trainX)
    with open(str(latent_path)+'/Train_Set_Labels_Outliers_Type' + str(ind1) + '_train_data_at_epoch_'+str(epoch) + '.npy', 'wb') as f:
        np.save(f, trainy)
    with open(str(latent_path)+'/Test_Set_Model_Latent_Representation_Outliers_Type' + str(ind1) + '_train_data_at_epoch_'+str(epoch) + '.npy', 'wb') as f:
        np.save(f, test_sets)
    with open(str(latent_path)+'/Test_Set_Labels_Outliers_Type' + str(ind1) + '_train_data_at_epoch_'+str(epoch) + '.npy', 'wb') as f:
        np.save(f, test_sets_label)


def OD_using_AE_rec_err_output(rec_train, rec_err_test_sets, trainy, test_sets_label, cid, epoch, op_path):

    # result_inliers = pd.DataFrame(columns=['Client', 'Precision', 'Recall', 'f1-score', 'Average_Precision_Score'])
    # result_outliers = pd.DataFrame(columns=['Client', 'Precision', 'Recall', 'f1-score', 'Average_Precision_Score'])
    result_outliers = pd.DataFrame(columns=['Client', 'Precision', 'Recall', 'f1-score', 'Pr-auc', 'Roc-auc', 'Average_Precision_Score'])
    for ind, i in enumerate(rec_err_test_sets):
        rec_test = rec_err_test_sets[ind]
        # set a threshold
        loss_threshold = np.percentile(rec_test, 97)
        # label the prediction based on threshold
        pred_test = [1 if i > loss_threshold else 0 for i in rec_test]
        testy = test_sets_label[ind]
        precision, recall, _ = precision_recall_curve(np.array(testy), np.array(rec_test), pos_label=1)
        f1 = f1_score(testy, pred_test)
        pr_auc = auc(recall, precision)
        roc_auc = roc_auc_score(testy, rec_test)
        fpr, tpr, _ = roc_curve(testy, rec_test)
        prec = precision_score(testy, pred_test)
        rec = recall_score(testy, pred_test)
        sorted_testy = [i for _, i in sorted(zip(rec_test, testy))]
        sorted_rec_test = [i for i, _ in sorted(zip(rec_test, testy))]
        # print("Classification Result: ", classification_report(testy, pred_test, output_dict=True))
        report = classification_report(testy, pred_test, output_dict=True)
        ap = average_precision_score(sorted_testy, sorted_rec_test)
        # ap = average_precision_score(testy, rec_test)
        # result_inliers.loc[ind] = ['Client' + str(ind + 1), report["0"]['precision'], report["0"]['recall'], report["0"]['f1-score']] + [ap]
        # result_outliers.loc[ind] = ['Client' + str(ind + 1), report["1"]['precision'], report["1"]['recall'], report["1"]['f1-score']] + [ap]
        result_outliers.loc[ind] = ['Client' + str(ind + 1), prec, rec, f1, pr_auc, roc_auc, ap]
    
    result_other_outliers = result_outliers.drop(labels=int(cid), axis=0, inplace=False)
    # result_outliers.loc[len(result_outliers)] = ['Avg_Score_of_Other_Outliers', np.mean(result_other_outliers['Precision'].values), np.mean(result_other_outliers['Recall'].values), np.mean(result_other_outliers['f1-score'].values), np.mean(result_other_outliers['Average_Precision_Score'].values)]
    result_outliers.loc[len(result_outliers)] = ['Avg_Score_of_Other_Outliers', np.mean(result_other_outliers['Precision'].values), np.mean(result_other_outliers['Recall'].values), np.mean(result_other_outliers['f1-score'].values), np.mean(result_other_outliers['Pr-auc'].values), np.mean(result_other_outliers['Roc-auc'].values), np.mean(result_other_outliers['Average_Precision_Score'].values)]
    ind1 = int(int(cid)+1)
    result_outliers.round(2).to_csv(str(op_path) + '/AE_Result_Client' + str(ind1) + '_train_data_at_epoch_'+str(epoch) + '.csv')

    if not os.path.exists(str(op_path) + '/rec_error'): os.mkdir(str(op_path) + '/rec_error')
    rec_err_path = str(op_path) + '/rec_error'
    with open(str(rec_err_path)+'/Train_Set_Model_Reconstruction_Error_Outliers_Type' + str(ind1) + '_train_data_at_epoch_'+str(epoch) + '.npy', 'wb') as f:
        np.save(f, rec_train)
    with open(str(rec_err_path)+'/Train_Set_Labels_Outliers_Type' + str(ind1) + '_train_data_at_epoch_'+str(epoch) + '.npy', 'wb') as f:
        np.save(f, trainy)
    with open(str(rec_err_path)+'/Test_Set_Model_Reconstruction_Error_Outliers_Type' + str(ind1) + '_train_data_at_epoch_'+str(epoch) + '.npy', 'wb') as f:
        np.save(f, rec_err_test_sets)
    with open(str(rec_err_path)+'/Test_Set_Labels_Outliers_Type' + str(ind1) + '_train_data_at_epoch_'+str(epoch) + '.npy', 'wb') as f:
        np.save(f, test_sets_label)


def train(cid, server_round, model, trainloader, valloader, epochs: int, batch_size:int, verbose=False):
    """Train the network on the training set."""
    
    # trainloader.append(DataLoader(dataset=trainX, batch_size=batch_size, shuffle=False))
    
    # Init Optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # scheduler = CosineAnnealingLR(optimizer, T_max=epochs, verbose=True)

    # Init Loss
    # Only MSE loss
    loss = nn.MSELoss(reduction='mean').to(device)

    train_loss, val_loss = [], []
    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()
        total_loss = 0
        # iterate over distinct minibatches
        for batch_id, batch in enumerate(trainloader):
            # set network in training mode
            model.train()
            model.to(device)
            # move to device
            batch = batch.to(device)
            # reset encoder and decoder gradients
            optimizer.zero_grad()
            # conduct forward encoder/decoder pass
            rec_batch, _ = model(batch)            
            
            ####### Use for only MSE loss
            batch_loss = loss(input=rec_batch, target=batch)
            # run error back-propagation
            batch_loss.backward(retain_graph=True)
            # optimize encoder and decoder parameters
            optimizer.step()
            total_loss += batch_loss.item()
        
        # update learning rate according to the scheduler
        # scheduler.step()

        # set model in test/evaluation mode
        model.eval()
        # x_torch_val = torch.FloatTensor(valX.astype(np.float32).values).to(device)
        rec_val, _ = model(valloader)
        
        ###### Only for MSE loss
        valid_loss = loss(input=rec_val, target=valloader)
        
        print("Epoch {} Training loss {} Validation loss {} Time {} ".format(epoch, round(total_loss / len(trainloader), 4), round(valid_loss.item(), 4), round(abs(time.time() - epoch_start_time), 4)))
        train_loss.append(total_loss/len(trainloader))
        val_loss.append(valid_loss.item())
        
    # draw_and_store_plots(train_loss, val_loss, epochs, cid, os.curdir)
    draw_and_store_plots(train_loss, val_loss, epochs, cid, file_path)

    
def test(model, server_round, trainX, testloader, trainy, testy, cid, epoch):
    """Evaluate the network on the entire test set."""
    
    # Only for MSE loss
    # loss = nn.MSELoss(reduction='mean').to(device)
    reconstruction_MSE = nn.MSELoss(reduction='none').to(device)
    model.eval()
    
    rec_train, z_train = model(trainX)
    rec_err_train = torch.mean(torch.sub(trainX, rec_train), dim=1)
    rec_err_train = np.abs(rec_err_train.detach().cpu().numpy())
       
    # For multiple items in testset
    z_test_sets = []
    rec_err_test_sets = []
    # set model in test/evaluation mode
    # model.eval()
    for ind, i in enumerate(testloader):
        testX = testloader[ind]
        # x_torch_test = torch.FloatTensor(testX.astype(np.float32).values).to(device)
        rec_test, z_test = model(testX)
        rec_err_test = reconstruction_MSE(input=rec_test, target=testX).sum(dim=1)
        # z_test_sets.append(z_test.detach().numpy())
        z_test_sets.append(z_test.detach().cpu().numpy())
        # rec_err_test_sets.append(np.abs(rec_err_test.detach().numpy()))
        rec_err_test_sets.append(np.abs(rec_err_test.detach().cpu().numpy()))
    test_loss = rec_err_test_sets[int(cid)].mean().item()

    if server_round in round_list:
        if not os.path.exists(str(file_path)+'/fl_model_'+str(model_architecture)+'_num_clients_'+str(NUM_CLIENTS)+'_fl_rounds_'+str(server_round)+'_client_epochs_'+str(epochs)+'_batch_size_'+str(batch_size)):
            os.mkdir(str(file_path)+'/fl_model_'+str(model_architecture)+'_num_clients_'+str(NUM_CLIENTS)+'_fl_rounds_'+str(server_round)+'_client_epochs_'+str(epochs)+'_batch_size_'+str(batch_size))
        output_path = str(file_path)+'/fl_model_'+str(model_architecture)+'_num_clients_'+str(NUM_CLIENTS)+'_fl_rounds_'+str(server_round)+'_client_epochs_'+str(epochs)+'_batch_size_'+str(batch_size)

        OD_using_AE_rec_err_output(rec_err_train, rec_err_test_sets, trainy, testy, cid, epoch, output_path)
        # od_model_for_latent_classification(z_train.detach().numpy(), z_test_sets, trainy, testy, cid, epoch, output_path)
        od_model_for_latent_classification(z_train.detach().cpu().numpy(), z_test_sets, trainy, testy, cid, epoch, output_path)
        # classification_model_at_latent(z_train.detach().numpy(), z_test_sets, trainy, testy, cid, epoch, output_path)
        classification_model_at_latent(z_train.detach().cpu().numpy(), z_test_sets, trainy, testy, cid, epoch, output_path)
    
    return test_loss, rec_err_train, rec_err_test_sets


def get_parameters(model) -> List[np.ndarray]:
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def set_parameters(model, parameters: List[np.ndarray]):
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)
    

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, cid, model, trainloader, valloader, testloader, train_data, trainy, valy, testy):
        self.cid = cid
        self.model = model
        self.trainloader = trainloader
        self.valloader = valloader
        self.testloader = testloader
        self.train_data = train_data
        self.trainy = trainy
        self.valy = valy
        self.testy = testy

    def get_parameters(self, config):
        print(f"[Client {self.cid}] get_parameters")
        return get_parameters(self.model)

    def fit(self, parameters, config):
        server_round = config["server_round"]
        print(f"[Client {self.cid}, round {server_round}] fit, config: {config}")
        set_parameters(self.model, parameters)
        train(self.cid, server_round, self.model, self.trainloader, self.valloader, epochs=epochs, batch_size=batch_size)
        return get_parameters(self.model), len(self.trainloader), {}

    def evaluate(self, parameters, config):
        server_round = config["server_round"]
        print(f"[Client {self.cid}, round {server_round}] evaluate, config: {config}")
        set_parameters(self.model, parameters)
        test_loss, train_rec_err_per_sample, test_rec_err_per_sample = test(self.model, server_round, self.train_data, self.testloader, self.trainy, self.testy, self.cid, epochs)
        return float(test_loss), len(self.testloader), {"loss": float(test_loss)}


def client_fn(cid) -> FlowerClient:
    """Create a Flower client representing a single organization."""

    ds_name = 'mnist'
    input_dim = int(28*28)
    input_data_path = '../data/run'+str(run_no)+'_MNIST_dataset_outliers_3per_inlier_0digit_outliers_3568digits'
    print('input_data_path: ', input_data_path)

    # Load data
    trainloaders, valloaders, testloaders, full_train_data, trainy_sets, valy_sets, testy_sets = load_datasets(batch_size, ds_name, input_data_path)
    # trainloaders, valloaders, testloader, input_dim = load_datasets()
    # Note: each client gets a different trainloader/valloader, so each client
    # will train and evaluate on their own unique data
    trainloader = trainloaders[int(cid)]
    valloader = valloaders[int(cid)]
    testloader = testloaders[int(cid)]
    train_data = full_train_data[int(cid)]
    trainy = trainy_sets[int(cid)]
    valy = valy_sets[int(cid)]
    testy = testy_sets[int(cid)]
    
    # Load model
    model = AutoEncoder([input_dim]+model_architecture[0], model_architecture[1]+[input_dim]).to(device)

    # Create a  single Flower client representing a single organization
    return FlowerClient(cid, model, trainloader, valloader, testloader, train_data, trainy, valy, testy)


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [m["loss"] for num_examples, m in metrics]
    # print('weighted average accuracies: ', accuracies)
    examples = [num_examples for num_examples, _ in metrics]
    # print('weighted average examples: ', examples)

    # Aggregate and return custom metric (weighted average)
    return {"loss": sum(accuracies) / sum(examples)}


def get_round_info(server_round: int):
    """Return training configuration dict for each round.
    """
    config = {
            "server_round": server_round,  # The current round of federated learning
    }   
    return config



# Creating an ArgumentParser Object
parser = argparse.ArgumentParser()
# Fetching the arguments
# parser.add_argument('total_epochs', help = 'Enter total epochs for training', type = int)
# parser.add_argument('rounds', help = 'Enter total FL rounds', type = int)
parser.add_argument('exp_run', help = 'Enter experiment run number', type = int)
parser.add_argument('fl_agg', help = 'Fed Aggregation number', type = int)
args = parser.parse_args()
# epochs = args.total_epochs
# fl_rounds = args.rounds
run_no = args.exp_run
fl_agg_no = args.fl_agg

seed = 42
batch_size = 128
epochs = 1000
NUM_CLIENTS = 4
if fl_agg_no == 1: strategy = 'FedAvg'
if fl_agg_no == 2: strategy = 'FedProx_1'
if fl_agg_no == 3: strategy = 'FedProx_0.1'
if fl_agg_no == 4: strategy = 'FedProx_0.01'
model_architecture = [[1024,256],[256,1024]]
model_foldername = 'model_1024_256'
round_list = [50, 100, 200]
fl_rounds = 200

for i in range(1,2):
    
        # for fl_rounds in round_list:
        # for epochs in epoch_list:

        # print('epochs: ', epochs)
        print('fl_rounds: ', fl_rounds)
        print('model_architecture: ', model_foldername)
        print('run_no: ', run_no)

        # MNIST
        base_dir = '../output/new_type/run'+str(run_no)+'_MNIST_outliers_3per_inlier_0digit_outliers_3568digits_tanh'
        print('base_dir: ', base_dir)

        if strategy == 'FedAvg':
            base_output_dir = str(base_dir)+'/fl_total_model'
        if strategy == 'FedProx_1':
            base_output_dir = str(base_dir)+'/fl_total_model_FedProx_1'
            mu = 1
        if strategy == 'FedProx_0.1':
            base_output_dir = str(base_dir)+'/fl_total_model_FedProx_0.1'
            mu = 0.1
        if strategy == 'FedProx_0.01':
            base_output_dir = str(base_dir)+'/fl_total_model_FedProx_0.01'
            mu = 0.01

        if not os.path.exists(str(base_dir)): os.mkdir(str(base_dir))
        if not os.path.exists(str(base_output_dir)): os.mkdir(str(base_output_dir))
        if not os.path.exists(str(base_output_dir)+'/'+str(model_foldername)): os.mkdir(str(base_output_dir)+'/'+str(model_foldername))
        file_path = str(base_output_dir)+'/'+str(model_foldername)

        # Create an instance of the model and get the parameters for inilialization of client models
        # params = get_parameters(AutoEncoder([input_dim, 128, 64], [64, 128, input_dim]).to(device))

        # Create FedAvg strategy
        if strategy == 'FedAvg':
            strategy = fl.server.strategy.FedAvg(
                fraction_fit=1.0,
                fraction_evaluate=1.0,
                min_fit_clients=NUM_CLIENTS,
                min_evaluate_clients=NUM_CLIENTS,
                min_available_clients=NUM_CLIENTS,
                # initial_parameters=fl.common.ndarrays_to_parameters(params),
                evaluate_metrics_aggregation_fn=weighted_average,  # <-- pass the metric aggregation function
                on_fit_config_fn=get_round_info,  # Function used to configure training. Pass the fit_config function,
                on_evaluate_config_fn=get_round_info, # Function used to configure validation.
            )

        # Create FedProx strategy
        if strategy == 'FedProx_1' or strategy == 'FedProx_0.1' or strategy == 'FedProx_0.01':
            strategy = fl.server.strategy.FedProx(
                fraction_fit=1.0,
                fraction_evaluate=1.0,
                min_fit_clients=NUM_CLIENTS,
                min_evaluate_clients=NUM_CLIENTS,
                min_available_clients=NUM_CLIENTS,
                # initial_parameters=fl.common.ndarrays_to_parameters(params),
                evaluate_metrics_aggregation_fn=weighted_average,  # <-- pass the metric aggregation function
                on_fit_config_fn=get_round_info,  # Function used to configure training. Pass the fit_config function,
                on_evaluate_config_fn=get_round_info, # Function used to configure validation.
                proximal_mu= mu,
            )

        # Specify client resources if you need GPU (defaults to 1 CPU and 0 GPU)
        client_resources = None
        # if device.type == "cuda":
        if device == "cuda":
            client_resources = {"num_gpus": 1}

        start_time = time.time()

        # Start simulation
        fl.simulation.start_simulation(
            client_fn=client_fn,
            num_clients=NUM_CLIENTS,
            config=fl.server.ServerConfig(num_rounds=fl_rounds),
            strategy=strategy,
            client_resources=client_resources,
            ray_init_args={"num_cpus": 2, "ignore_reinit_error": True, "include_dashboard": False,}
        )

        print('Total time taken: ', round(abs(time.time() - start_time), 4))

        with open(str(file_path)+'/Time_Taken.txt', 'w') as f:
            f.write('FL Total time taken: '+str(round(abs(time.time() - start_time), 4)))
