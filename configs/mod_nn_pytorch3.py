import numpy as np
import os

GENERAL_PATH = os.getcwd()

model_params = {
    'configs_file_name' : 'mod_nn_pytorch3',
    'model_name' : 'nn_pytorch',
    'hidden_dims': [256, 128, 64, 32],
    'epochs': 50,
    'batch_size': 64,
    'lr': 0.001
}

dataset_params = {
    'dataset_name' : 'SeasonsModel',
    'type_features' : 'alexNet_compact',
    'data_train_path' : os.path.join(GENERAL_PATH, 'data/processed/train_SeasonsModel_alexNet_compact.csv'),
    'data_test_path' : os.path.join(GENERAL_PATH, 'data/processed/test_SeasonsModel_alexNet_compact.csv'),
} 
