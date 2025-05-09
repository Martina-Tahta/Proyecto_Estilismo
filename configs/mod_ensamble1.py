import os

GENERAL_PATH = os.getcwd()
current_filename = os.path.splitext(os.path.basename(__file__))[0]

model_params = {
    'configs_file_name' : current_filename,
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

DATASET = 'SeasonsModel'
TYPE_FEATURES = 'enhancedSeasonal'

dataset_params = {
    'dataset_name' : DATASET,
    'type_features' : TYPE_FEATURES,
    
    'data_train_path' : os.path.join(GENERAL_PATH, f'data/processed/{DATASET}/{TYPE_FEATURES}/train_{DATASET}_{TYPE_FEATURES}.csv'),
    'data_val_path' : os.path.join(GENERAL_PATH, f'data/processed/{DATASET}/{TYPE_FEATURES}/val_{DATASET}_{TYPE_FEATURES}.csv'),
    'data_test_path' : os.path.join(GENERAL_PATH, f'data/processed/{DATASET}/{TYPE_FEATURES}/test_{DATASET}_{TYPE_FEATURES}.csv'),
} 

