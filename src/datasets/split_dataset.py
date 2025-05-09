import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
import os
import re

def raw_dataset_split(raw_dataset_path, output_dir, name_dataset, test_size=0.2, random_state=42):
    #split train, test, val
    image_files = []
    seasons = []
    for image_file in os.listdir(raw_dataset_path):
        if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            season_match = re.match(r'([a-zA-Z_]+)_\d+', image_file)
            if season_match:
                season = season_match.group(1)
                image_file = os.path.relpath(os.path.join(raw_dataset_path, image_file))
                image_files.append(image_file)
                seasons.append(season)

    X_train, X_test, y_train, y_test = train_test_split(
        image_files, seasons, test_size=test_size, stratify=seasons, random_state=random_state
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=test_size, stratify=y_train, random_state=random_state
    )
    
    train_df = pd.DataFrame({
        'image_path': X_train,
        'season':     y_train
    })
    test_df = pd.DataFrame({
        'image_path': X_test,
        'season':     y_test
    })
    val_df = pd.DataFrame({
        'image_path': X_val,
        'season':     y_val
    })

    os.makedirs(output_dir, exist_ok=True)
    train_df.to_csv(os.path.join(output_dir, f"train_{name_dataset}.csv"), index=False)
    test_df.to_csv(os.path.join(output_dir, f"test_{name_dataset}.csv"), index=False)
    val_df.to_csv(os.path.join(output_dir, f"val_{name_dataset}.csv"), index=False)
    

def rebalanace_train_set(train_set_path):
    # Separate majority and minority classes with SMOTE-like approach
    df = pd.read_csv(train_set_path)
    classes = df['season'].value_counts()
    majority_class = classes.index[0]
    n_samples = int(classes[majority_class] * 1.5)  # Increase samples
    
    balanced_dfs = []
    for season in classes.index:
        season_df = df[df['season'] == season]
        if len(season_df) < n_samples:
            upsampled_df = resample(
                season_df,
                replace=True,
                n_samples=n_samples,
                random_state=42
            )
            balanced_dfs.append(upsampled_df)
        else:
            balanced_dfs.append(season_df)
    
    df = pd.concat(balanced_dfs)
    df.to_csv(train_set_path, index=False)
    
    
def main():
    GENERAL_PATH = os.getcwd()
    dataset = 'SeasonsModel'
    raw_dataset_path = os.path.join(GENERAL_PATH, f'data/raw/{dataset}')
    processed_dataset_save_path = os.path.join(GENERAL_PATH, f'data/split_dataset/{dataset}')
    raw_dataset_split(raw_dataset_path, processed_dataset_save_path, dataset)
    rebalanace_train_set(processed_dataset_save_path+f'/train_{dataset}.csv')


if __name__ == "__main__":
    main()



