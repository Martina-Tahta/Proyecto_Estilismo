import numpy as np
import os

GENERAL_PATH = os.getcwd()

current_filename = os.path.splitext(os.path.basename(__file__))[0]
CLASSES_CSV = os.path.join(GENERAL_PATH, 'data/classes_csv/real_12_classes.csv')

model_params = {
    'configs_file_name' : current_filename,
    'model_name' : 'resNeXt_ft',
    'variant': "resnext50_32x4d", 
    'train_blocks': [],
    'epochs': 50,
    'batch_size': 64,
    'lr_backbone': 1e-3,
    'lr_fc': 1e-3,
    'weight_decay': 1e-5,
    'early_stopping_patience': 50,
    'dropout': 0.5,
    'verbose': True,
    'classes_csv': CLASSES_CSV,
    'classifier': 0
}

DATASET = 'SuperDataset'
TYPE_FEATURES = 'resNeXt'

dataset_params = {
    'dataset_name' : DATASET,
    'type_features' : TYPE_FEATURES,
    
    'data_train_path' : os.path.join(GENERAL_PATH, f'data/split_dataset/{DATASET}/train_balanced_{DATASET}.csv'),
    'data_val_path' : os.path.join(GENERAL_PATH, f'data/split_dataset/{DATASET}/val_{DATASET}.csv'),
    'data_test_path' : os.path.join(GENERAL_PATH, f'data/split_dataset/{DATASET}/test_{DATASET}.csv'),
}
