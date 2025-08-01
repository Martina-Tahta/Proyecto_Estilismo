import numpy as np
import os

GENERAL_PATH = os.getcwd()

current_filename = os.path.splitext(os.path.basename(__file__))[0]

model_params = {
    'configs_file_name' : current_filename,
    'model_name' : 'nn_pytorch',
    'hidden_dims': [128, 64, 32],
    'epochs': 500,
    'batch_size': 64,
    'lr': 1e-4,
    'weight_decay': 1e-3,
    'dropout': 0.3
}

DATASET = 'SeasonsModel'
TYPE_FEATURES = 'alexNet_face_only_compact'

dataset_params = {
    'dataset_name' : DATASET,
    'type_features' : TYPE_FEATURES,
    
    'data_train_path' : os.path.join(GENERAL_PATH, f'data/processed/{DATASET}/{TYPE_FEATURES}/train_{DATASET}_{TYPE_FEATURES}.csv'),
    'data_val_path' : os.path.join(GENERAL_PATH, f'data/processed/{DATASET}/{TYPE_FEATURES}/val_{DATASET}_{TYPE_FEATURES}.csv'),
    'data_test_path' : os.path.join(GENERAL_PATH, f'data/processed/{DATASET}/{TYPE_FEATURES}/test_{DATASET}_{TYPE_FEATURES}.csv'),
} 
