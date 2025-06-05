import numpy as np
import os

GENERAL_PATH = os.getcwd()

current_filename = os.path.splitext(os.path.basename(__file__))[0]
CLASSES_CSV = os.path.join(GENERAL_PATH, 'data/classes_csv/DeepArmocromia_classes.csv')

model_params = {
    'configs_file_name' : current_filename,
    'model_name' : 'resNeXt_weighted_avg',
    'variant': "resnext101_32x8d",  # "resnext50_32x4d" or "resnext101_32x8d", ...
    'epochs': 50,
    'batch_size': 128,
    'lr': 1e-4,
    'weight_decay': 1e-2,
    'early_stopping_patience': 20,
    'dropout': 0.5,
    'classes_csv': CLASSES_CSV
}

DATASET = 'DeepArmocromia'
TYPE_FEATURES = 'resNeXt_weighted_avg'

dataset_params = {
    'dataset_name' : DATASET,
    'type_features' : TYPE_FEATURES,
    
    'data_train_path' : os.path.join(GENERAL_PATH, f'data/split_dataset/{DATASET}/train_{DATASET}.csv'),
    'data_val_path' : os.path.join(GENERAL_PATH, f'data/split_dataset/{DATASET}/val_{DATASET}.csv'),
    'data_test_path' : os.path.join(GENERAL_PATH, f'data/split_dataset/{DATASET}/test_{DATASET}.csv'),
}
