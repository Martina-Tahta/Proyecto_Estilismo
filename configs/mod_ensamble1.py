import os

GENERAL_PATH = os.getcwd()
current_filename = os.path.splitext(os.path.basename(__file__))[0]
CLASSES_CSV = os.path.join(GENERAL_PATH, 'data/classes_csv/DeepArmocromia_classes.csv')

model_params = {
    'configs_file_name' : current_filename,
    'model_name' : 'ensamble',
    'rf': {'n_estimators':200,
            'max_depth': None,
            'min_samples_split': 5,
            'min_samples_leaf': 2},
#     'gb': {'n_estimators': 10,
#             'learning_rate': 0.05,
#             'max_depth': 5,
#             'min_samples_split': 4,
#             'min_samples_leaf': 2},
    'xgb': {'n_estimators': 150,
            'learning_rate': 0.1,
            'max_depth': 6,
            'min_child_weight': 1,
            'subsample': 0.85,
            'colsample_bytree': 0.8},
    'ada': {'n_estimators': 10,
            'learning_rate': 0.5},
    'model_weights': [1, 2, 1],
    'classes_csv': CLASSES_CSV
}

DATASET = 'DeepArmocromia'
TYPE_FEATURES = 'enhancedSeasonal'

dataset_params = {
    'dataset_name' : DATASET,
    'type_features' : TYPE_FEATURES,
    
    'data_train_path' : os.path.join(GENERAL_PATH, f'data/split_dataset/{DATASET}/train_{DATASET}.csv'),
    'data_val_path' : os.path.join(GENERAL_PATH, f'data/split_dataset/{DATASET}/val_{DATASET}.csv'),
    'data_test_path' : os.path.join(GENERAL_PATH, f'data/split_dataset/{DATASET}/test_{DATASET}.csv'),
}

