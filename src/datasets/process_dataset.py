import os

from extract_features import extract_imgs_features_enhancedSeasonal, extract_imgs_features_AlexNet
from split_dataset import split_and_save_dataset


def process_raw_dataset(raw_dataset_path, processed_dataset_save_path, name_dataset, type_features=0, test_size=0.2, random_state=42, split=True):
    features_name = ''
    if type_features==0:
        features_df = extract_imgs_features_enhancedSeasonal(raw_dataset_path)
        features_name = 'enhancedSeasonal'
    elif type_features==1:
        features_df = extract_imgs_features_AlexNet(raw_dataset_path)
        features_name = 'alexNet'
    elif type_features==2:
        features_df = extract_imgs_features_AlexNet(raw_dataset_path, compact=True)
        features_name = 'alexNet_compact'

    elif type_features==3:
        features_df = extract_imgs_features_AlexNet(raw_dataset_path, compact=True, face_seg=True)
        features_name = 'alexNet_face_only_compact'
    
    if split:
        split_and_save_dataset(features_df, processed_dataset_save_path, 
                            name_dataset, features_name=features_name, test_size=test_size, random_state=random_state)
    else:
        features_df.to_csv(os.path.join(processed_dataset_save_path, f"test_{name_dataset}_{features_name}.csv"), index=False)

def main():
    GENERAL_PATH = os.getcwd()
    dataset = 'SeasonsModel'
    raw_dataset_path = os.path.join(GENERAL_PATH, f'data/raw/{dataset}')
    processed_dataset_save_path = os.path.join(GENERAL_PATH, f'data/processed')
    process_raw_dataset(raw_dataset_path,processed_dataset_save_path, dataset, type_features=3)

    # dataset = 'Ours'
    # raw_dataset_path = os.path.join(GENERAL_PATH, f'data/raw/{dataset}')
    # processed_dataset_save_path = os.path.join(GENERAL_PATH, f'data/processed')
    # process_raw_dataset(raw_dataset_path,processed_dataset_save_path, dataset, type_features=1, split=False)


if __name__ == "__main__":
    main()