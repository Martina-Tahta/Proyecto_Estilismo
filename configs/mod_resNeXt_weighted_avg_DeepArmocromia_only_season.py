import numpy as np
import os

GENERAL_PATH = os.getcwd()

current_filename = os.path.splitext(os.path.basename(__file__))[0]

model_params = {
    'configs_file_name' : current_filename,
    'model_name' : 'resNeXt_weighted_avg',
    'num_classes' : 4,
    'variant': "resnext101_32x8d",  # "resnext50_32x4d" or "resnext101_32x8d", ...
    'epochs': 50,
    'batch_size': 16,
    'lr': 1e-4,
    'weight_decay': 1e-4,
    'early_stopping_patience': 5,
    'dropout': 0
}

DATASET = 'DeepArmocromia_season_only'
TYPE_FEATURES = 'resNeXt_weighted_avg'

dataset_params = {
    'dataset_name' : DATASET,
    'type_features' : TYPE_FEATURES,
    
    'data_train_path' : os.path.join(GENERAL_PATH, f'data/split_dataset/{DATASET}/train_{DATASET}.csv'),
    'data_val_path' : os.path.join(GENERAL_PATH, f'data/split_dataset/{DATASET}/val_{DATASET}.csv'),
    'data_test_path' : os.path.join(GENERAL_PATH, f'data/split_dataset/{DATASET}/test_{DATASET}.csv'),
} 
