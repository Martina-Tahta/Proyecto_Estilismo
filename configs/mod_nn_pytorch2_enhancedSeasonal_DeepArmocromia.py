import numpy as np
import os

GENERAL_PATH = os.getcwd()

current_filename = os.path.splitext(os.path.basename(__file__))[0]
CLASSES_CSV = os.path.join(GENERAL_PATH, 'data/classes_csv/DeepArmocromia_classes.csv')

model_params = {
    'configs_file_name' : current_filename,
    'model_name' : 'nn_pytorch',
    'hidden_dims': [128, 64],
    'epochs': 300,
    'batch_size': 128,
    'lr': 1e-3,
    'weight_decay': 1e-6,
    'dropout': 0.5,
    'patience': 30,
    'classes_csv': CLASSES_CSV
}

DATASET = 'DeepArmocromia'
TYPE_FEATURES = 'enhancedSeasonal'

dataset_params = {
    'dataset_name' : DATASET,
    'type_features' : TYPE_FEATURES,
    
    'data_train_path' : os.path.join(GENERAL_PATH, f'data/processed/{DATASET}/{TYPE_FEATURES}/train_{DATASET}_{TYPE_FEATURES}.csv'),
    'data_val_path' : os.path.join(GENERAL_PATH, f'data/processed/{DATASET}/{TYPE_FEATURES}/val_{DATASET}_{TYPE_FEATURES}.csv'),
    'data_test_path' : os.path.join(GENERAL_PATH, f'data/processed/{DATASET}/{TYPE_FEATURES}/test_{DATASET}_{TYPE_FEATURES}.csv'),
} 
