# Fin-Fed-OD: Federated Outlier Detection on Financial Tabular Data
This repository executes experiments explained in Fin-Fed-OD and collects results of known and unknown outliers.
## Notice
This is developed for research purposes.

## Implemented models
We have implemented the following models.
- [x] [Fin-Fed-OD]()
- [x] [DAGMM](https://sites.cs.ucsb.edu/~bzong/doc/iclr18-dagmm.pdf)
- [x] [MemAE](https://openaccess.thecvf.com/content_ICCV_2019/papers/Gong_Memorizing_Normality_to_Detect_Anomaly_Memory-Augmented_Deep_Autoencoder_for_Unsupervised_ICCV_2019_paper.pdf)

## Dependencies
The most important ones:
- flwr[simulation]==1.3.0
- pyod
- torch with CUDA
- numpy
- python
- pandas
- scikit-learn
- seaborn

## Usage
For training the model start from the root of the project.
```
$ cd code/Model_Training_Code

$ cd AE or cd DAGMM or cd MemAE
AE for Deep Autoencoder, DAGMM to run DAGMM model, and MemAE to run MemAE model

To train Federated Learning Models for Tabular Data (Credit Default or Adult Data)
$ python3 fl_clients_all_type_outlier_prediction_bce_and_mse_loss.py
--exp_run [experiment number (1...5)]
--ds_ind [dataset 1.Credit Default 2.Adult Data]
--fl_agg [Fedeaed Aggregation Algorithm 1.FedAvg 2.FedProx with mu=1 3.FedProx with mu=0.1 4.FedProx with mu=0.01]

To train Federated Learning Models for MNIST Data
$ python3 fl_clients_mnist_all_type_outlier_prediction_mse_loss.py
--exp_run [experiment number (1...5)]
--fl_agg [Fedeaed Aggregation Algorithm 1.FedAvg 2.FedProx with mu=1 3.FedProx with mu=0.1 4.FedProx with mu=0.01]

To train Standalone Models
$ cd standalone

To train Standalone Models for Tabular Data
$ python3 main_outlier_types.py
--ds_ind [dataset 1.Credit Default 2.Adult Data]

To train Standalone Models for MNIST Data
$ python3 mnist_main_outlier_types.py

```
