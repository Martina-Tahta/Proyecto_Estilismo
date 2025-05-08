import pandas as pd
import os
import re

from enhancedSeasonal_feature_extractor import EnhancedSeasonalColorDatabase
from alexNet_feature_extractor import AlexNetFeatureExtractor

def extract_imgs_features_enhancedSeasonal(raw_dataset_path):
    # Create database imgs existing model
    print("Creating enhanced database...")
    db_creator = EnhancedSeasonalColorDatabase(raw_dataset_path)
    # Process images and create features
    data = []
    print(f"Looking for images in: {db_creator.image_directory}")

    counter_img = 0
    counter_processed_img = 0
    for image_file in os.listdir(db_creator.image_directory):
        counter_img += 1
        if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            season_match = re.match(r'([a-zA-Z_]+)_\d+', image_file)
            if season_match:
                season = season_match.group(1)
                image_path = os.path.join(db_creator.image_directory, image_file)
                features = db_creator.extract_enhanced_features(image_path)
                
                if features:
                    features['image_file'] = image_file
                    features['season'] = season
                    data.append(features)
                    counter_processed_img += 1
                    #print(f"Processed {image_file}")

    if counter_img == counter_processed_img:
        print("All images were processed.")
    else:
        print("Some images were not processed.")
    df = pd.DataFrame(data)
    return df



def extract_imgs_features_AlexNet(raw_dataset_path, compact=False, face_seg=False):
    # Create database imgs existing model
    print("Creating alexNet database...")
    db_creator = AlexNetFeatureExtractor(raw_dataset_path)
    # Process images and create features
    data = []
    print(f"Looking for images in: {db_creator.image_directory}")

    counter_img = 0
    counter_processed_img = 0
    for image_file in os.listdir(db_creator.image_directory):
        counter_img += 1
        if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            season_match = re.match(r'([a-zA-Z_]+)_\d+', image_file)
            if season_match:
                season = season_match.group(1)
                image_path = os.path.join(db_creator.image_directory, image_file)
                
                if not face_seg:
                    if compact:
                        features = db_creator.extract_compact_features(image_path)
                    else:
                        features = db_creator.extract_features(image_path)
                else:
                    if compact:
                        features = db_creator.extract_only_face_compact_features(image_path)
                    # else:
                    #     features = db_creator.extract_features(image_path)
                
                features['image_file'] = image_file
                features['season'] = season
                data.append(features)
                counter_processed_img += 1
                #print(f"Processed {image_file}")

    if counter_img == counter_processed_img:
        print("All images were processed.")
    else:
        print("Some images were not processed.")
    df = pd.concat(data, ignore_index=True)
    return df
