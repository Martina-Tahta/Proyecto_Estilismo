import os
from enhancedSeasonal_feature_extractor import EnhancedSeasonalColorDatabase
from alexNet_feature_extractor import AlexNetFeatureExtractor
from resNeXt_feature_extractor import ResNeXtFeatureExtractor
import pandas as pd
from segmentation import segment_images_in_directory, segment_face_hair

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

    elif type_features == 4:
        extractor = ResNeXtFeatureExtractor()
        features_name = 'resNeXt'
        extractor_kwargs = {'compact': False}
    
    elif type_features == 5:
        extractor = ResNeXtFeatureExtractor()
        features_name = 'resNeXt_compact'
        extractor_kwargs = {'compact': True}
    
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
        
def segment_images_from_csv(csv_path, output_dir, device="cpu"):
    """
    Reads a CSV file with an 'image_path' column, segments each image using
    segment_face_hair from segmentation.py, and saves the segmented images
    in the specified output directory.
    
    Parameters:
        csv_path : str
            Path to the CSV file containing image paths.
        output_dir : str
            Directory to store segmented images.
        device : str
            Device on which segmentation should run ("cpu" or "cuda").
    """

    df = pd.read_csv(csv_path)
    os.makedirs(output_dir, exist_ok=True)
    
    for _, row in df.iterrows():
        image_path = row['image_path']
        filename = os.path.basename(image_path)
        output_path = os.path.join(output_dir, filename)
        try:
            segmented_image = segment_face_hair(image_path, device)
            segmented_image.save(output_path)
            print(f"Segmented and saved: {output_path}")
        except Exception as e:
            print(f"Error processing {image_path}: {e}")


def update_image_paths(csv_path, new_directory, output_csv_path):
    """
    Reads a CSV file with an 'image_path' column, updates each image path to reflect
    the new directory using relative paths, and saves the updated CSV to the specified output path.

    Parameters:
        csv_path : str
            Path to the original CSV file.
        new_directory : str
            Directory where the updated image files are stored.
        output_csv_path : str
            Path to save the updated CSV file.
    """
    df = pd.read_csv(csv_path)
    df['image_path'] = df['image_path'].apply(lambda x: os.path.relpath(os.path.join(new_directory, os.path.basename(x)), start=os.getcwd()))
    df.to_csv(output_csv_path, index=False)
    print(f"Updated CSV saved to: {output_csv_path}")


def main():
    GENERAL_PATH = os.getcwd()
    dataset = 'DeepArmocromia'
    raw_dataset_path = os.path.join(GENERAL_PATH, f'data/split_dataset/{dataset}')
    processed_dataset_save_path = os.path.join(GENERAL_PATH, f'data/processed/{dataset}')
    type_features = 2
    
    # segmented_dataset_path = os.path.join(GENERAL_PATH, f'data/segmented_dataset/{dataset}')
    # train_csv_path = os.path.join(GENERAL_PATH, f'data/split_dataset/{dataset}_segmented/train_{dataset}.csv')
    # val_csv_path = os.path.join(GENERAL_PATH, f'data/split_dataset/{dataset}_segmented/val_{dataset}.csv')
    # test_csv_path = os.path.join(GENERAL_PATH, f'data/split_dataset/{dataset}_segmented/test_{dataset}.csv')
    
    # segment_images_from_csv(train_csv_path, segmented_dataset_path, device="cuda")
    # segment_images_from_csv(val_csv_path, segmented_dataset_path, device="cuda")
    # segment_images_from_csv(test_csv_path, segmented_dataset_path, device="cuda")
    
    # update_image_paths(train_csv_path, segmented_dataset_path, train_csv_path)
    # update_image_paths(test_csv_path, segmented_dataset_path, test_csv_path)
    # update_image_paths(val_csv_path, segmented_dataset_path, val_csv_path)
        
    process_split_dataset([raw_dataset_path+f'/train_{dataset}.csv', raw_dataset_path+f'/val_{dataset}.csv', raw_dataset_path+f'/test_{dataset}.csv'],
                          processed_dataset_save_path, dataset, type_features=type_features)

    
    # combine_features(processed_dataset_save_path, dataset, 'enhancedSeasonal', 'alexNet_compact')
    # dataset = 'Ours'
    # raw_dataset_path = os.path.join(GENERAL_PATH, f'data/raw/{dataset}')
    # processed_dataset_save_path = os.path.join(GENERAL_PATH, f'data/processed')
    # process_raw_dataset(raw_dataset_path,processed_dataset_save_path, dataset, type_features=1, split=False)


if __name__ == "__main__":
    main()