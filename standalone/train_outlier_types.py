import os
import time
import numpy as np
import json
import pickle
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
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

import matplotlib.pyplot as plt

from autoencoder_init import AutoEncoder


def draw_and_store_plots(train_loss, val_loss, test_loss, epochs, ds_outlier_type, file_path):

    if not os.path.exists(str(file_path) + '/visualization_plots'): os.mkdir(str(file_path) + '/visualization_plots')
    vis_plot_path = str(file_path)+'/visualization_plots'

    # y_max = np.max(np.hstack((train_loss, val_loss, test_loss))) + 0.1
    # y_min = np.min(np.hstack((train_loss, val_loss, test_loss))) - 0.1
    y_max = np.max(np.hstack((train_loss, val_loss))) + 0.1
    y_min = np.min(np.hstack((train_loss, val_loss))) - 0.1
    # plt.figure(figsize=(10, 5))
    # plt.subplot(1,2,1)
    plt.plot(range(1, epochs+1), train_loss, label='Train_loss')
    plt.plot(range(1, epochs+1), val_loss, alpha=0.5, label='Val_loss')
    plt.scatter(epochs, test_loss, label='Test_loss')
    plt.annotate(test_loss, (epochs, test_loss))
    plt.legend(loc='best')
    plt.ylim(y_min, y_max)
    plt.xlabel('Epochs')
    plt.ylabel('Error Value')
    plt.title('Train_Test_Loss_Curve')
    plt.savefig(str(vis_plot_path)+'/Train_test_curve_for_training_outlier_type'+str(ds_outlier_type)+'.png')
    # plt.show()
    plt.close()


def visualize_and_store_latent_weight(encoder_latent_weight_matrix, decoder_latent_weight_matrix, epoch, file_path, ds_outlier_type):

    if not os.path.exists(str(file_path) + '/visualization_plots'): os.mkdir(str(file_path) + '/visualization_plots')
    vis_plot_path = str(file_path)+'/visualization_plots'

    np.savetxt(vis_plot_path + '/model_outliertype'+str(ds_outlier_type)+'_encoder_latent_weight_matrix_epoch'+str(epoch)+'.txt', encoder_latent_weight_matrix)
    np.savetxt(vis_plot_path + '/model_outliertype'+str(ds_outlier_type)+'_decoder_latent_weight_matrix_epoch'+str(epoch)+'.txt', decoder_latent_weight_matrix)


def od_model_for_latent_classification(trainX, test_sets, trainy, test_sets_label, ds_outlier_type, epoch, file_path):

    clf = IForest(n_estimators=100, max_samples="auto", contamination=0.3, max_features=1., behaviour='old', random_state=42)
    clf.fit(trainX)

    # result_inliers = pd.DataFrame(columns=['Outlier_type', 'Precision', 'Recall', 'f1-score', 'Average_Precision_Score'])
    result_outliers = pd.DataFrame(columns=['Outlier_type', 'Precision', 'Recall', 'f1-score', 'Pr-auc', 'Roc-auc', 'Average_Precision_Score'])
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
        print("Classification Result: ", classification_report(testy, pred_test, output_dict=True))
        report = classification_report(testy, pred_test, output_dict=True)
        ap = average_precision_score(testy, outlierness_score)
        # ap = average_precision_score(testy, rec_test)
        # result_inliers.loc[ind] = ['Type' + str(ind + 1), report["0"]['precision'], report["0"]['recall'], report["0"]['f1-score']] + [ap]
        # result_outliers.loc[ind] = ['Type' + str(ind + 1), report["1"]['precision'], report["1"]['recall'], report["1"]['f1-score']] + [ap]
        result_outliers.loc[ind] = ['Client' + str(ind + 1), prec, rec, f1, pr_auc, roc_auc, ap]

    result_other_outliers = result_outliers.drop(labels=int(ds_outlier_type - 1), axis=0, inplace=False)
    result_outliers.loc[len(result_outliers)] = ['Avg_Score_of_Other_Outliers', np.mean(result_other_outliers['Precision'].values), np.mean(result_other_outliers['Recall'].values), np.mean(result_other_outliers['f1-score'].values), np.mean(result_other_outliers['Pr-auc'].values), np.mean(result_other_outliers['Roc-auc'].values), np.mean(result_other_outliers['Average_Precision_Score'].values)]

    if not os.path.exists(str(file_path) + '/visualization_plots'): os.mkdir(str(file_path) + '/visualization_plots')
    vis_plot_path = str(file_path) + '/visualization_plots'
    result_outliers.round(2).to_csv(str(vis_plot_path) + '/OD_Model_Result_on_Latentspace_Outliers_Type' + str(ds_outlier_type) + '_train_data_at_epoch_'+str(epoch) + '.csv')


def classification_model_at_latent(trainX, test_sets, trainy, test_sets_label, ds_outlier_type, epoch, file_path):

    # clf = SVC(gamma='auto')
    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(trainX, trainy)

    # result_inliers = pd.DataFrame(columns=['Outlier_type', 'Precision', 'Recall', 'f1-score', 'Average_Precision_Score'])
    result_outliers = pd.DataFrame(columns=['Outlier_type', 'Precision', 'Recall', 'f1-score', 'Pr-auc', 'Roc-auc', 'Average_Precision_Score'])
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
        print("Classification Result: ", classification_report(testy, pred_test, output_dict=True))
        report = classification_report(testy, pred_test, output_dict=True)
        ap = average_precision_score(testy, outlierness_score)
        # result_inliers.loc[ind] = ['Type'+str(ind+1), report["0"]['precision'], report["0"]['recall'], report["0"]['f1-score']] + [ap]
        # result_outliers.loc[ind] = ['Type' + str(ind + 1), report["1"]['precision'], report["1"]['recall'], report["1"]['f1-score']] + [ap]
        result_outliers.loc[ind] = ['Client' + str(ind + 1), prec, rec, f1, pr_auc, roc_auc, ap]

    result_other_outliers = result_outliers.drop(labels=int(ds_outlier_type-1), axis=0, inplace=False)
    result_outliers.loc[len(result_outliers)] = ['Avg_Score_of_Other_Outliers', np.mean(result_other_outliers['Precision'].values), np.mean(result_other_outliers['Recall'].values), np.mean(result_other_outliers['f1-score'].values), np.mean(result_other_outliers['Pr-auc'].values), np.mean(result_other_outliers['Roc-auc'].values), np.mean(result_other_outliers['Average_Precision_Score'].values)]

    if not os.path.exists(str(file_path) + '/visualization_plots'): os.mkdir(str(file_path) + '/visualization_plots')
    vis_plot_path = str(file_path) + '/visualization_plots'
    result_outliers.round(2).to_csv(str(vis_plot_path) + '/Supervised_Classification_Model_Result_Outliers_Type' + str(ds_outlier_type) + '_train_data_at_epoch_'+str(epoch) + '.csv')

    if not os.path.exists(str(file_path) + '/latent_representation'): os.mkdir(str(file_path) + '/latent_representation')
    latent_path = str(file_path) + '/latent_representation'
    with open(str(latent_path)+'/Train_Set_Model_Latent_Representation_Outliers_Type' + str(ds_outlier_type) + '_train_data_at_epoch_'+str(epoch) + '.npy', 'wb') as f:
        np.save(f, trainX)
    with open(str(latent_path)+'/Train_Set_Labels_Outliers_Type' + str(ds_outlier_type) + '_train_data_at_epoch_'+str(epoch) + '.npy', 'wb') as f:
        np.save(f, trainy)
    with open(str(latent_path)+'/Test_Set_Model_Latent_Representation_Outliers_Type' + str(ds_outlier_type) + '_train_data_at_epoch_'+str(epoch) + '.npy', 'wb') as f:
        np.save(f, test_sets)
    with open(str(latent_path)+'/Test_Set_Labels_Outliers_Type' + str(ds_outlier_type) + '_train_data_at_epoch_'+str(epoch) + '.npy', 'wb') as f:
        np.save(f, test_sets_label)


def OD_using_AE_rec_err_output(rec_train, rec_err_test_sets, trainy, test_sets_label, ds_outlier_type, epoch, file_path):

    # sorting train samples based on rec erros and then getting gt labels for top rec error samples
    # train_pred_labels = [i for _, i in zip(rec_train, trainy)]
    # train_precision_score = precision_score(trainy, train_pred_labels)

    # result_inliers = pd.DataFrame(columns=['Outlier_type', 'Precision', 'Recall', 'f1-score', 'Average_Precision_Score'])
    result_outliers = pd.DataFrame(columns=['Outlier_type', 'Precision', 'Recall', 'f1-score', 'Pr-auc', 'Roc-auc', 'Average_Precision_Score'])
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
        print("Classification Result: ", classification_report(testy, pred_test, output_dict=True))
        report = classification_report(testy, pred_test, output_dict=True)
        ap = average_precision_score(sorted_testy, sorted_rec_test)
        # ap = average_precision_score(testy, rec_test)
        # result_inliers.loc[ind] = ['Type' + str(ind + 1), report["0"]['precision'], report["0"]['recall'], report["0"]['f1-score']] + [ap]
        # result_outliers.loc[ind] = ['Type' + str(ind + 1), report["1"]['precision'], report["1"]['recall'], report["1"]['f1-score']] + [ap]
        result_outliers.loc[ind] = ['Client' + str(ind + 1), prec, rec, f1, pr_auc, roc_auc, ap]

    result_other_outliers = result_outliers.drop(labels=int(ds_outlier_type - 1), axis=0, inplace=False)
    result_outliers.loc[len(result_outliers)] = ['Avg_Score_of_Other_Outliers', np.mean(result_other_outliers['Precision'].values), np.mean(result_other_outliers['Recall'].values), np.mean(result_other_outliers['f1-score'].values), np.mean(result_other_outliers['Pr-auc'].values), np.mean(result_other_outliers['Roc-auc'].values), np.mean(result_other_outliers['Average_Precision_Score'].values)]

    if not os.path.exists(str(file_path) + '/visualization_plots'): os.mkdir(str(file_path) + '/visualization_plots')
    vis_plot_path = str(file_path) + '/visualization_plots'
    # result_inliers.round(2).to_csv(str(vis_plot_path) + '/AE_Result_Inliers_Type' + str(ds_outlier_type) + '_train_data_at_epoch_'+str(epoch) + '.csv')
    result_outliers.round(2).to_csv(str(vis_plot_path) + '/AE_Result_Outliers_Type' + str(ds_outlier_type) + '_train_data_at_epoch_'+str(epoch) + '.csv')

    if not os.path.exists(str(file_path) + '/rec_error'): os.mkdir(str(file_path) + '/rec_error')
    rec_err_path = str(file_path) + '/rec_error'
    with open(str(rec_err_path)+'/Train_Set_Model_Reconstruction_Error_Outliers_Type' + str(ds_outlier_type) + '_train_data_at_epoch_'+str(epoch) + '.npy', 'wb') as f:
        np.save(f, rec_train)
    with open(str(rec_err_path)+'/Train_Set_Labels_Outliers_Type' + str(ds_outlier_type) + '_train_data_at_epoch_'+str(epoch) + '.npy', 'wb') as f:
        np.save(f, trainy)
    with open(str(rec_err_path)+'/Test_Set_Model_Reconstruction_Error_Outliers_Type' + str(ds_outlier_type) + '_train_data_at_epoch_'+str(epoch) + '.npy', 'wb') as f:
        np.save(f, rec_err_test_sets)
    with open(str(rec_err_path)+'/Test_Set_Labels_Outliers_Type' + str(ds_outlier_type) + '_train_data_at_epoch_'+str(epoch) + '.npy', 'wb') as f:
        np.save(f, test_sets_label)


# def model_training(all_trainX_ds1, testX_ds1, valX_ds1, all_trainX_ds2, testX_ds2, valX_ds2, outlier_data1, outlier_data2, scaler_ds1, scaler_ds2, sample_ds1, sample_ds2, batch_size, device, epochs):
def model_training(trainX, trainy, test_sets_data, test_sets_label, valX, valy, cat_uniqval_count, run_no, ds_name, device, epoch_list, ds_outlier_type):

    input_dim = trainX.shape[1]
    batch_size = 128
    epochs = 500

    # writer = SummaryWriter()

    x_torch_train = torch.FloatTensor(trainX.astype(np.float32).values).to(device)
    dataloader = DataLoader(dataset=x_torch_train, batch_size=batch_size, shuffle=False)

    # Build model
    # model = AutoEncoder([input_dim, 128, 64, 32, 16, 8, 4, 2], [2, 4, 8, 16, 32, 64, 128, input_dim])
    # model_dir_name = 'model_128_64_32_16_8_4_2'
    # model = AutoEncoder([input_dim,128,64,32,16,8], [8,16,32,64,128,input_dim])
    # model_dir_name = 'model_128_64_32_16_8'
    # model = AutoEncoder([input_dim, 128, 64, 32, 16], [16, 32, 64, 128, input_dim])
    # model_dir_name = 'model_128_64_32_16'
    # model = AutoEncoder([input_dim, 128, 64, 32], [32, 64, 128, input_dim])
    # model_dir_name = 'model_128_64_32'
    model = AutoEncoder([input_dim, 128, 64], [64, 128, input_dim])
    model_dir_name = 'model_128_64'
    # model = AutoEncoder([input_dim, 8, 4,], [4, 8, input_dim])
    # model = AutoEncoder([input_dim, 12, 4, 1], [1, 4, 12, input_dim])
    # model_dir_name = 'model_12_4_1'

    # for i in list(model.parameters()):
    #     i = np.array(i.detach().numpy())
    #     print(i.shape)
    #     print(i)

    # Init Optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # scheduler = CosineAnnealingLR(optimizer, T_max=epochs, verbose=True)

    # Init Loss
    # Only MSE loss
    # loss = nn.MSELoss(reduction='mean').to(device)

    # BCE and MSE loss
    reconstruction_BCE = nn.CrossEntropyLoss(reduction='none').to(device)
    reconstruction_MSE = nn.MSELoss(reduction='none').to(device)

    if ds_name == 'credit_default':
        base_dir = '../output/new_type/MemAE/run' + str(run_no) + '_CD_outliers_3per_corr_cat_cols_25per_corr_num_cols_25per_types_4_1missing_2lessprob_3gaussian_4scaling'
    if ds_name == 'adult_data':
        base_dir = '../output/new_type/MemAE/run' + str(run_no) + '_AD_outliers_3per_corr_cat_cols_25per_corr_num_cols_25per_types_4_1missing_2lessprob_3gaussian_4scaling'
    # file_path = str(base_dir)
    file_path = str(base_dir) + '/centralized'
    if not os.path.exists(str(base_dir)): os.mkdir(str(base_dir))
    if not os.path.exists(str(file_path)): os.mkdir(str(file_path))
    if not os.path.exists(str(file_path) + '/' + str(model_dir_name)): os.mkdir(str(file_path) + '/' + str(model_dir_name))
    res_folder = str(file_path) + '/' + str(model_dir_name)


    train_loss, val_loss = [], []
    for epoch in range(1, epochs + 1):

        epoch_start_time = time.time()
        total_loss = 0
        # iterate over distinct minibatches
        for batch_id, batch in enumerate(dataloader):
            # set network in training mode
            model.train()
            model.to(device)
            # move to device
            batch = batch.to(device)
            # reset encoder and decoder gradients
            optimizer.zero_grad()
            # conduct forward encoder/decoder pass
            rec_batch, _ = model(batch)

            ######## Use for BCE and MSE loss
            # init rec loss
            rec_error_batch_bce = torch.zeros(batch.shape[0]).to(device)
            # rec_error_batch_mse = torch.zeros(batch.shape[0]).to(device)
            curr_col_ind = 0
            # for cat_attr in cat_features:
            for ind, cat_ind_loc in enumerate(cat_uniqval_count):
                # compute reconstruction error of categorical and numerical parts
                rec_error_batch_bce_single_attr = reconstruction_BCE(input=rec_batch[:, curr_col_ind:cat_ind_loc+curr_col_ind], target=batch[:, curr_col_ind:cat_ind_loc+curr_col_ind].argmax(axis=1))
                rec_error_batch_bce += rec_error_batch_bce_single_attr
                curr_col_ind+=cat_ind_loc
            # for num_attr in numerical_attributes_scaled:
            rec_error_batch_mse = reconstruction_MSE(input=rec_batch[:, int(sum(cat_uniqval_count)):], target=batch[:, int(sum(cat_uniqval_count)):]).sum(dim=1)
            # rec_error_batch_mse += rec_error_batch_mse_single_attr
            # compute total reconstruction error
            rec_error_batch = rec_error_batch_bce + rec_error_batch_mse
            rec_error_batch_mean = rec_error_batch.mean()
            # print('rec_error_batch_mean: ', type(rec_error_batch_mean))
            # print('rec_error_batch_mean: ', rec_error_batch_mean)
            # rec_error_batch_bce_mean = rec_error_batch_bce.mean()
            # rec_error_batch_mse_mean = rec_error_batch_mse.mean()

            # run error back-propagation
            rec_error_batch_mean.backward(retain_graph=True)

            # optimize encoder and decoder parameters
            optimizer.step()
            total_loss += rec_error_batch_mean.item()

            ####### Use for only MSE loss
            # batch_loss = loss(input=rec_batch, target=batch)
            # # run error back-propagation
            # batch_loss.backward(retain_graph=True)
            # # optimize encoder and decoder parameters
            # optimizer.step()
            # total_loss += batch_loss.item()

        # # update learning rate according to the scheduler
        # scheduler.step()

        # Displaying and storing latent layer weights
        # if epoch == epochs:
        #     print('\n Epoch: ', epoch)
        #     for ind, i in enumerate(list(model.parameters())):
        #         if ind == (len(list(model.parameters())) / 2) - 2:
        #             enc_lat = np.array(i.detach().numpy())
        #         if ind == (len(list(model.parameters())) / 2):
        #             dec_lat = np.array(i.detach().numpy())
        #     visualize_and_store_latent_weight(enc_lat, dec_lat, epoch, os.curdir, ds_outlier_type)

        # set model in test/evaluation mode
        model.eval()
        x_torch_val = torch.FloatTensor(valX.astype(np.float32).values).to(device)
        rec_val, _ = model(x_torch_val)
        ###### Only for MSE loss
        # valid_loss = loss(input=rec_val, target=x_torch_val)

        ###### For BCE and MSE loss
        val_rec_error_bce = torch.zeros(x_torch_val.shape[0]).to(device)
        curr_col_ind = 0
        # for cat_attr in cat_features:
        for ind, cat_ind_loc in enumerate(cat_uniqval_count):
            # compute reconstruction error of categorical and numerical parts
            val_rec_error_bce_single_attr = reconstruction_BCE(input=rec_val[:, curr_col_ind:cat_ind_loc + curr_col_ind], target=x_torch_val[:, curr_col_ind:cat_ind_loc + curr_col_ind].argmax(axis=1))
            val_rec_error_bce += val_rec_error_bce_single_attr
            curr_col_ind += cat_ind_loc
        # for num_attr in numerical_attributes_scaled:
        val_rec_error_mse = reconstruction_MSE(input=rec_val[:, int(sum(cat_uniqval_count)):], target=x_torch_val[:, int(sum(cat_uniqval_count)):]).sum(dim=1)
        # compute total reconstruction error
        val_rec_error = val_rec_error_bce + val_rec_error_mse
        valid_loss = val_rec_error.mean()

        # print("Epoch {} Training loss {} Validation loss {} Time {} ".format(epoch, round(total_loss / len(dataloader), 4), round(valid_loss.item(), 4), round(abs(time.time() - epoch_start_time), 4)))
        print("Epoch {} Training loss {} Validation loss {} Time {} ".format(epoch, total_loss / len(dataloader), valid_loss.item(), round(abs(time.time() - epoch_start_time), 4)))
        train_loss.append(total_loss/len(dataloader))
        val_loss.append(valid_loss.item())

        if epoch in epoch_list:
            z_test_sets = []
            rec_err_test_sets = []
            # set model in test/evaluation mode
            # model.eval()
            for ind, i in enumerate(test_sets_data):
                testX = test_sets_data[ind]
                x_torch_test = torch.FloatTensor(testX.astype(np.float32).values).to(device)
                rec_test, z_test = model(x_torch_test)
                test_rec_error_bce = torch.zeros(x_torch_test.shape[0]).to(device)
                curr_col_ind = 0
                # for cat_attr in cat_features:
                for ind, cat_ind_loc in enumerate(cat_uniqval_count):
                    # compute reconstruction error of categorical and numerical parts
                    test_rec_error_bce_single_attr = reconstruction_BCE(input=rec_test[:, curr_col_ind:cat_ind_loc + curr_col_ind], target=x_torch_test[:, curr_col_ind:cat_ind_loc + curr_col_ind].argmax(axis=1))
                    test_rec_error_bce += test_rec_error_bce_single_attr
                    curr_col_ind += cat_ind_loc
                # for num_attr in numerical_attributes_scaled:
                test_rec_error_mse = reconstruction_MSE(input=rec_test[:, int(sum(cat_uniqval_count)):], target=x_torch_test[:, int(sum(cat_uniqval_count)):]).sum(dim=1)
                # compute total reconstruction error
                rec_err_test = test_rec_error_bce + test_rec_error_mse
                z_test_sets.append(z_test.detach().numpy())
                rec_err_test_sets.append(np.abs(rec_err_test.detach().numpy()))

            rec_train, z_train = model(x_torch_train)
            train_rec_error_bce = torch.zeros(x_torch_train.shape[0]).to(device)
            curr_col_ind = 0
            # for cat_attr in cat_features:
            for ind, cat_ind_loc in enumerate(cat_uniqval_count):
                # compute reconstruction error of categorical and numerical parts
                train_rec_error_bce_single_attr = reconstruction_BCE(input=rec_train[:, curr_col_ind:cat_ind_loc + curr_col_ind], target=x_torch_train[:, curr_col_ind:cat_ind_loc + curr_col_ind].argmax(axis=1))
                train_rec_error_bce += train_rec_error_bce_single_attr
                curr_col_ind += cat_ind_loc
            # for num_attr in numerical_attributes_scaled:
            train_rec_error_mse = reconstruction_MSE(input=rec_train[:, int(sum(cat_uniqval_count)):], target=x_torch_train[:, int(sum(cat_uniqval_count)):]).sum(dim=1)
            # compute total reconstruction error
            rec_err_train = train_rec_error_bce + train_rec_error_mse

            # # classification_model_at_latent(z_train.detach().numpy(), z_test_sets, trainy, test_sets_label, ds_outlier_type, epoch, res_folder)
            OD_using_AE_rec_err_output(rec_err_train.detach().numpy(), rec_err_test_sets, trainy, test_sets_label, ds_outlier_type, epoch, res_folder)
            od_model_for_latent_classification(z_train.detach().numpy(), z_test_sets, trainy, test_sets_label, ds_outlier_type, epoch, res_folder)
            classification_model_at_latent(z_train.detach().numpy(), z_test_sets, trainy, test_sets_label, ds_outlier_type, epoch, res_folder)

    z_test_sets = []
    rec_err_test_sets = []
    test_loss = 0
    # set model in test/evaluation mode
    model.eval()
    for ind, i in enumerate(test_sets_data):
        testX = test_sets_data[ind]
        x_torch_test = torch.FloatTensor(testX.astype(np.float32).values).to(device)
        rec_test, z_test = model(x_torch_test)
        test_rec_error_bce = torch.zeros(x_torch_test.shape[0]).to(device)
        curr_col_ind = 0
        # for cat_attr in cat_features:
        for ind, cat_ind_loc in enumerate(cat_uniqval_count):
            # compute reconstruction error of categorical and numerical parts
            test_rec_error_bce_single_attr = reconstruction_BCE(input=rec_test[:, curr_col_ind:cat_ind_loc + curr_col_ind], target=x_torch_test[:, curr_col_ind:cat_ind_loc + curr_col_ind].argmax(axis=1))
            test_rec_error_bce += test_rec_error_bce_single_attr
            curr_col_ind += cat_ind_loc
        # for num_attr in numerical_attributes_scaled:
        test_rec_error_mse = reconstruction_MSE(input=rec_test[:, int(sum(cat_uniqval_count)):], target=x_torch_test[:, int(sum(cat_uniqval_count)):]).sum(dim=1)
        # compute total reconstruction error
        rec_err_test = test_rec_error_bce + test_rec_error_mse
        test_loss = rec_err_test.mean().item()
        # z_test_sets.append(z_test.detach().numpy())
        # rec_err_test_sets.append(np.abs(rec_err_test.detach().numpy()))
        z_test_sets.append(z_test.detach().numpy())
        rec_err_test_sets.append(np.abs(rec_err_test.detach().numpy()))

    rec_train, z_train = model(x_torch_train)
    rec_err_per_sample_train = torch.mean(torch.sub(x_torch_train, rec_train), dim=1)

    train_loss = np.array(train_loss)
    val_loss = np.array(val_loss)
    test_loss = np.array(test_loss)

    draw_and_store_plots(train_loss, val_loss, test_loss, epochs, ds_outlier_type, res_folder)

    # writer.add_graph(model, [[x_torch_test_ds1], [1], [0]])

    # # print("model: ", model)
    # # print("model parameters: ", model.parameters())
    print('model.state_dict().items(): ', model.state_dict().items())
    print('model.state_dict().keys(): ', model.state_dict().keys())
    # print('model.state_dict().keys(): ', type(model.state_dict().keys()))
    # # print('model.state_dict().keys()[0]: ', model.state_dict().keys()[0])
    # # print('model.state_dict().keys()[0]: ', type(model.state_dict().keys()[0]))
    # print('model.state_dict().values(): ', model.state_dict().values())
    # print('model.state_dict().values(): ', type(model.state_dict().values()))
    # print('model.state_dict().values(): ', list(model.state_dict().values()))
    # print('model.state_dict().values(): ', type(list(model.state_dict().values())))

    # for i in list(model.parameters()):
    #     i = np.array(i.detach().numpy())
    #     print(i.shape)
    #     print(i)

    return z_train.detach().numpy(), z_test_sets, np.abs(rec_err_per_sample_train.detach().numpy()), rec_err_test_sets





