import os
from enhancedSeasonal_feature_extractor import EnhancedSeasonalColorDatabase
from alexNet_feature_extractor import AlexNetFeatureExtractor
import pandas as pd

def process_split_dataset(raw_datasets_paths, processed_dataset_save_path, name_dataset, type_features=0):
    train_path, val_path, test_path = raw_datasets_paths
    features_name = ''
    
    if type_features == 0:
        extractor = EnhancedSeasonalColorDatabase()
        features_name = 'enhancedSeasonal'
        extractor_kwargs = {} 

    elif type_features == 1:
        extractor = AlexNetFeatureExtractor()
        features_name = 'alexNet'
        extractor_kwargs = {'compact': False, 'face_seg': False}

    elif type_features == 2:
        extractor = AlexNetFeatureExtractor()
        features_name = 'alexNet_compact'
        extractor_kwargs = {'compact': True, 'face_seg': False}

    elif type_features == 3:
        extractor = AlexNetFeatureExtractor()
        features_name = 'alexNet_face_only_compact'
        extractor_kwargs = {'compact': True, 'face_seg': True}
    
    train_features_df = extractor.extract_imgs_features(train_path, **extractor_kwargs)
    val_features_df = extractor.extract_imgs_features(val_path, **extractor_kwargs)
    test_features_df = extractor.extract_imgs_features(test_path, **extractor_kwargs)

    processed_dataset_save_path = processed_dataset_save_path + f"/{features_name}/"
    os.makedirs(processed_dataset_save_path, exist_ok=True)  
    train_features_df.to_csv(os.path.join(processed_dataset_save_path, f"train_{name_dataset}_{features_name}.csv"), index=False)
    val_features_df.to_csv(os.path.join(processed_dataset_save_path, f"val_{name_dataset}_{features_name}.csv"), index=False)
    test_features_df.to_csv(os.path.join(processed_dataset_save_path, f"test_{name_dataset}_{features_name}.csv"), index=False)


def combine_features(path_datasets, dataset, type_feature1, type_feature2):
    path_datasets_new = path_datasets + f"/combine_{type_feature1}_{type_feature2}/"
    os.makedirs(path_datasets_new, exist_ok=True)

    for t in ['train', 'val', 'test']:
        df1 = pd.read_csv(path_datasets+f'/{type_feature1}/{t}_{dataset}_{type_feature1}.csv')
        df1 = df1.drop(['image_file', 'season'], axis=1)
        df2 = pd.read_csv(path_datasets+f'/{type_feature2}/{t}_{dataset}_{type_feature2}.csv')

        df_combined = pd.concat([df1, df2], axis=1)
        df_combined.to_csv(os.path.join(path_datasets_new, f"{t}_{dataset}_combine_{type_feature1}_{type_feature2}.csv"), index=False)



def main():
    GENERAL_PATH = os.getcwd()
    dataset = 'SeasonsModel'
    raw_dataset_path = os.path.join(GENERAL_PATH, f'data/split_dataset/{dataset}')
    processed_dataset_save_path = os.path.join(GENERAL_PATH, f'data/processed/{dataset}')
    type_features = 3
    process_split_dataset([raw_dataset_path+f'/train_{dataset}.csv', raw_dataset_path+f'/val_{dataset}.csv', raw_dataset_path+f'/test_{dataset}.csv'],
                          processed_dataset_save_path, dataset, type_features=type_features)
    
    # combine_features(processed_dataset_save_path, dataset, 'enhancedSeasonal', 'alexNet_compact')


    # dataset = 'Ours'
    # raw_dataset_path = os.path.join(GENERAL_PATH, f'data/raw/{dataset}')
    # processed_dataset_save_path = os.path.join(GENERAL_PATH, f'data/processed')
    # process_raw_dataset(raw_dataset_path,processed_dataset_save_path, dataset, type_features=1, split=False)


if __name__ == "__main__":
    main()