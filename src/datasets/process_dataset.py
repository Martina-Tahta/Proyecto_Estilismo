import os

from extract_features import extract_imgs_features
from split_dataset import split_and_save_dataset


def process_raw_dataset(raw_dataset_path, processed_dataset_save_path, name_dataset, test_size=0.2, random_state=42, split=True):
    features_df = extract_imgs_features(raw_dataset_path)
    if split:
        split_and_save_dataset(features_df, processed_dataset_save_path, 
                            name_dataset, test_size=test_size, random_state=random_state)
    else:
        features_df.to_csv(os.path.join(processed_dataset_save_path, f"test_{name_dataset}.csv"), index=False)

def main():
    GENERAL_PATH = os.getcwd()
    # dataset = 'SeasonsModel'
    # raw_dataset_path = os.path.join(GENERAL_PATH, f'data/raw/{dataset}')
    # processed_dataset_save_path = os.path.join(GENERAL_PATH, f'data/processed')
    # process_raw_dataset(raw_dataset_path,processed_dataset_save_path, dataset)

    dataset = 'Ours'
    raw_dataset_path = os.path.join(GENERAL_PATH, f'data/raw/{dataset}')
    processed_dataset_save_path = os.path.join(GENERAL_PATH, f'data/processed')
    process_raw_dataset(raw_dataset_path,processed_dataset_save_path, dataset, split=False)


if __name__ == "__main__":
    main()