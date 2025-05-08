import numpy as np
import os

GENERAL_PATH = os.getcwd()

model_params = {
    'configs_file_name' : 'mod_ensamble1',
    'model_name' : 'ensamble',
    'rf': {'n_estimators':10,
            'max_depth': 5,
            'min_samples_split': 4,
            'min_samples_leaf': 2},
    'gb': {'n_estimators': 10,
            'learning_rate': 0.05,
            'max_depth': 5,
            'min_samples_split': 4,
            'min_samples_leaf': 2},
    'xgb': {'n_estimators': 10,
            'learning_rate': 0.05,
            'max_depth': 5,
            'min_child_weight': 2,
            'subsample': 0.8,
            'colsample_bytree': 0.8},
    'ada': {'n_estimators': 10,
            'learning_rate': 0.05},
    'model_weights': [2, 1, 2, 1]
}

dataset_params = {
    'dataset_name' : 'SeasonsModel',
    'type_features' : 'enhancedSeasonal',
    'data_train_path' : os.path.join(GENERAL_PATH, 'data/processed/train_SeasonsModel_enhancedSeasonal.csv'),
    'data_test_path' : os.path.join(GENERAL_PATH, 'data/processed/test_SeasonsModel_enhancedSeasonal.csv'),
} 

