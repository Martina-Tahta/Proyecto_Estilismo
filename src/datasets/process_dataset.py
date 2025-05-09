import os
from enhancedSeasonal_feature_extractor import EnhancedSeasonalColorDatabase
from alexNet_feature_extractor import AlexNetFeatureExtractor

def process_split_dataset(raw_datasets_paths, processed_dataset_save_path, name_dataset, type_features=0):
    train_path, val_path, test_path = raw_datasets_paths
    features_name = ''
    
    if type_features==0:
        extractor = EnhancedSeasonalColorDatabase()
        features_name = 'enhancedSeasonal'
        
    elif type_features==1:
        extractor = AlexNetFeatureExtractor()
        features_name = 'alexNet'
        
    elif type_features==2:
        extractor = AlexNetFeatureExtractor()
        features_name = 'alexNet_compact'

    elif type_features==3:
        extractor = AlexNetFeatureExtractor()
        features_name = 'alexNet_face_only_compact'
    
    train_features_df = extractor.extract_imgs_features(train_path)
    val_features_df = extractor.extract_imgs_features(val_path)
    test_features_df = extractor.extract_imgs_features(test_path)


    os.makedirs(processed_dataset_save_path, exist_ok=True)  
    train_features_df.to_csv(os.path.join(processed_dataset_save_path, f"train_{name_dataset}_{features_name}.csv"), index=False)
    val_features_df.to_csv(os.path.join(processed_dataset_save_path, f"val_{name_dataset}_{features_name}.csv"), index=False)
    test_features_df.to_csv(os.path.join(processed_dataset_save_path, f"test_{name_dataset}_{features_name}.csv"), index=False)


def main():
    GENERAL_PATH = os.getcwd()
    dataset = 'SeasonsModel'
    raw_dataset_path = os.path.join(GENERAL_PATH, f'data/split_dataset/{dataset}')
    processed_dataset_save_path = os.path.join(GENERAL_PATH, f'data/processed/{dataset}')
    process_split_dataset([raw_dataset_path+f'/train_{dataset}.csv', raw_dataset_path+f'/val_{dataset}.csv', raw_dataset_path+f'/test_{dataset}.csv'],
                          processed_dataset_save_path, dataset, type_features=0)

    # dataset = 'Ours'
    # raw_dataset_path = os.path.join(GENERAL_PATH, f'data/raw/{dataset}')
    # processed_dataset_save_path = os.path.join(GENERAL_PATH, f'data/processed')
    # process_raw_dataset(raw_dataset_path,processed_dataset_save_path, dataset, type_features=1, split=False)


if __name__ == "__main__":
    main()