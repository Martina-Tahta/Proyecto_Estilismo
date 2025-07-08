import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
import os
import re
import random
import shutil

def data_from_folder(folder_path, seasonsModel=True):
    image_files = []
    seasons = []
    for image_file in os.listdir(folder_path):
        if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            if seasonsModel:
                season_match = re.match(r'([a-zA-Z_]+)_\d+', image_file)
                if season_match:
                    season = season_match.group(1)
                    image_file = os.path.relpath(os.path.join(folder_path, image_file))
                    image_files.append(image_file)
                    seasons.append(season)

            else: 
                season = os.path.basename(folder_path)
                image_file = os.path.relpath(os.path.join(folder_path, image_file))
                image_files.append(image_file)
                seasons.append(season)

    return image_files, seasons


def dataset_split(output_dir, name_dataset, raw_dataset_path=None, csv=None, test_size=0.2, random_state=42):
    if raw_dataset_path is not None:
        image_files, seasons = data_from_folder(raw_dataset_path)
    elif csv is not None:
        image_files = csv['image_path']
        seasons = csv['season'] 


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
    

def count_images_per_category(train_path):
    """
    Recorre cada season dentro de 'train_path' y cuenta las imágenes PNG en cada categoría.
    
    Args:
        train_path (str): Ruta al directorio 'train' de tu dataset.
    """
    for season in os.listdir(train_path):
        season_path = os.path.join(train_path, season)
        if not os.path.isdir(season_path):
            continue
        
        print(f"Season: {season}")
        for category in os.listdir(season_path):
            category_path = os.path.join(season_path, category)
            if not os.path.isdir(category_path):
                continue
            
            png_count = sum(
                1 for fname in os.listdir(category_path)
                if fname.lower().endswith('.png')
            )
            print(f"  Category '{category}': {png_count} samples")
        print()


def DeepArmocromia_train_val_split(dataset_path, val_ratio=0.2, seed=42):
    random.seed(seed)

    train_src = os.path.join(dataset_path, 'their_train')
    # Nombres finales: dataset_path/train y dataset_path/val
    final_train = os.path.join(dataset_path, 'train')
    final_val   = os.path.join(dataset_path, 'val')

    # Crea directorios finales si no existen (vacíos para limpiar)
    os.makedirs(final_train, exist_ok=True)
    os.makedirs(final_val, exist_ok=True)

    for season in os.listdir(train_src):
        season_path = os.path.join(train_src, season)
        if not os.path.isdir(season_path):
            continue

        for category in os.listdir(season_path):
            cat_path = os.path.join(season_path, category)
            if not os.path.isdir(cat_path):
                continue

            files = [f for f in os.listdir(cat_path) if f.lower().endswith('.png')]
            random.shuffle(files)

            val_count = int(len(files) * val_ratio)
            val_files, train_files = files[:val_count], files[val_count:]

            # Crear en final_train/season/category y final_val/season/category
            train_cat_dst = os.path.join(final_train, season, category)
            val_cat_dst   = os.path.join(final_val,   season, category)
            os.makedirs(train_cat_dst, exist_ok=True)
            os.makedirs(val_cat_dst,   exist_ok=True)

            for f in train_files:
                shutil.copy2(os.path.join(cat_path, f), os.path.join(train_cat_dst, f))
            for f in val_files:
                shutil.copy2(os.path.join(cat_path, f), os.path.join(val_cat_dst, f))

            print(f"{season}/{category}: train={len(train_files)}, val={len(val_files)}")
        

def DeepArmocromia_generate_image_csv(dataset_root, output_csv_path, name_dataset):
    """
    Recorre las carpetas train, val y test y genera un CSV para cada una con columnas:
    - image_path: ruta relativa comenzando con 'data/raw/DeepArmocromia/...'
    - season: combinación 'categoria_estacion' (por ejemplo, 'true_winter')

    Args:
        dataset_root (str): Ruta al directorio que contiene las carpetas train, val y test.
        output_dir (str): Ruta donde se guardarán los archivos CSV.
    """
    subsets = ['train', 'val', 'test']
    os.makedirs(output_csv_path, exist_ok=True)

    for subset in subsets:
        records = []
        subset_dir = os.path.join(dataset_root, subset)
        if not os.path.isdir(subset_dir):
            print(f"Aviso: '{subset}' no existe en {dataset_root}, se omite.")
            continue

        for season in os.listdir(subset_dir):
            season_dir = os.path.join(subset_dir, season)
            if not os.path.isdir(season_dir):
                continue

            for category in os.listdir(season_dir):
                category_dir = os.path.join(season_dir, category)
                if not os.path.isdir(category_dir):
                    continue

                for fname in os.listdir(category_dir):
                    if fname.lower().endswith('.png'):
                        rel_path = os.path.join(
                            'data/raw/DeepArmocromia', 
                            subset, season, category, fname
                        )
                        season_label = f"{category}_{season}"
                        records.append({
                            'image_path': rel_path,
                            'season': season_label
                        })

        df = pd.DataFrame(records)
        csv_path = os.path.join(output_csv_path, f"{subset}_{name_dataset}.csv")
        df.to_csv(csv_path, index=False)
        print(f"CSV para '{subset}' generado en: {csv_path} (registros: {len(df)})")


def get_generated_images_dataframe(general_path):
    all_dfs = []
    for s in ['bright_spring', 'bright_winter', 'dark_autumn', 'dark_winter', 'light_spring', 'light_summer', 'soft_autumn', 'soft_summer', 'true_autumn', 'true_spring', 'true_summer', 'true_winter']:
        image_files, seasons = data_from_folder(os.path.join(general_path, f'data/raw/Generated/{s}'), seasonsModel=False)
        df = pd.DataFrame({
            'image_path': image_files,
            'season': seasons
        })

        all_dfs.append(df)

    return pd.concat(all_dfs, ignore_index=True)


def combine_datasets(general_path):
    season_mapping = {
        'cool_summer': 'true_summer',
        'cool_winter': 'true_winter',
        'deep_autumn': 'dark_autumn',
        'deep_winter': 'dark_winter',
        'warm_autumn': 'true_autumn',
        'warm_spring': 'true_spring',
        'bright_spring': 'bright_spring',
        'bright_winter': 'bright_winter',
        'light_spring': 'light_spring',
        'light_summer': 'light_summer',
        'soft_autumn': 'soft_autumn',
        'soft_summer': 'soft_summer'
    }

    image_files, seasons = data_from_folder(os.path.join(general_path, 'data/segmented_dataset/SeasonsModel'))
    seasonsModel = pd.DataFrame({
        'image_path': image_files,
        'season': seasons
    })

    deepArm_train = pd.read_csv(os.path.join(general_path, 'data/split_dataset/DeepArmocromia_og/train_DeepArmocromia.csv'))
    deepArm_test = pd.read_csv(os.path.join(general_path, 'data/split_dataset/DeepArmocromia_og/test_DeepArmocromia.csv'))
    deepArm_train['season'] = deepArm_train['season'].map(season_mapping)
    deepArm_test['season'] = deepArm_test['season'].map(season_mapping)
    deepArm_all = pd.concat([deepArm_train, deepArm_test], ignore_index=True)

    superDataset = pd.concat([seasonsModel, deepArm_all], ignore_index=True)

    save_dir = os.path.join(general_path, f'data/split_dataset/SuperDataset')
    dataset_split(save_dir, 'SuperDataset', csv=superDataset)

    aiGenerated = get_generated_images_dataframe(general_path)
    train_set = pd.read_csv(os.path.join(save_dir, f"train_SuperDataset.csv"))
    train_set = pd.concat([train_set, aiGenerated], ignore_index=True)
    train_set.to_csv(os.path.join(save_dir, f"train_SuperDataset.csv"), index=False)

    print("\n🔢 Cantidad de muestras por clase en el train:")
    print(train_set['season'].value_counts().sort_index())

    balanced_path = os.path.join(save_dir, "train_balanced_SuperDataset.csv")
    shutil.copy2(os.path.join(save_dir, "train_SuperDataset.csv"), balanced_path)
    rebalanace_train_set(balanced_path, increase_rate=1)

    balanced = pd.read_csv(balanced_path)
    print("\n🔢 Cantidad de muestras por clase en el train:")
    print(balanced['season'].value_counts().sort_index())


def map_to_season(output_dir, name_dataset):
    train_df = pd.read_csv(os.path.join(output_dir, f"train_{name_dataset}.csv"))
    train_balanced_df = pd.read_csv(os.path.join(output_dir, f"train_balanced_{name_dataset}.csv"))
    test_df = pd.read_csv(os.path.join(output_dir, f"test_{name_dataset}.csv"))
    val_df = pd.read_csv(os.path.join(output_dir, f"val_{name_dataset}.csv"))

    season_mapping = {
        'true_summer': 'summer',
        'true_winter': 'winter',
        'dark_autumn': 'autumn',
        'dark_winter': 'winter',
        'true_autumn': 'autumn',
        'true_spring': 'spring',
        'bright_spring': 'spring',
        'bright_winter': 'winter',
        'light_spring': 'spring',
        'light_summer': 'summer',
        'soft_autumn': 'autumn',
        'soft_summer': 'summer'
    }
    train_df['season'] = train_df['season'].map(season_mapping)
    train_balanced_df['season'] = train_balanced_df['season'].map(season_mapping)
    test_df['season'] = test_df['season'].map(season_mapping)
    val_df['season'] = val_df['season'].map(season_mapping)

    output_dir += 'SeasonsOnly'
    name_dataset += 'SeasonsOnly'
    os.makedirs(output_dir, exist_ok=True)
    train_df.to_csv(os.path.join(output_dir, f"train_{name_dataset}.csv"), index=False)
    train_balanced_df.to_csv(os.path.join(output_dir, f"train_balanced_{name_dataset}.csv"), index=False)
    test_df.to_csv(os.path.join(output_dir, f"test_{name_dataset}.csv"), index=False)
    val_df.to_csv(os.path.join(output_dir, f"val_{name_dataset}.csv"), index=False)


def rebalanace_train_set(train_set_path, increase_rate=1.5):
    # Separate majority and minority classes with SMOTE-like approach
    df = pd.read_csv(train_set_path)
    classes = df['season'].value_counts()
    majority_class = classes.index[0]
    n_samples = int(classes[majority_class] * increase_rate)  # Increase samples
    
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

    # Split individual datasets
    # dataset = 'SeasonsModel'
    # raw_dataset_path = os.path.join(GENERAL_PATH, f'data/raw/{dataset}')
    # processed_dataset_save_path = os.path.join(GENERAL_PATH, f'data/split_dataset/{dataset}')
    # dataset_split(raw_dataset_path, processed_dataset_save_path, dataset)
    # rebalanace_train_set(processed_dataset_save_path+f'/train_{dataset}.csv')

    # Split DeepArmocromia (only train, validation)
    # dataset = 'DeepArmocromia'
    # dataset_path = os.path.join(GENERAL_PATH, f'data/raw/{dataset}')
    # DeepArmocromia_train_val_split(dataset_path)
    # output_csv_path=os.path.join(GENERAL_PATH, f'data/split_dataset/{dataset}')
    # DeepArmocromia_generate_image_csv(dataset_path, output_csv_path, dataset)
    # rebalanace_train_set(os.path.join(output_csv_path, f'train_{dataset}.csv'))

    # Split for super dataset: combines SeasonsModel, DeepArmocromia and AI generated images (only for train)
    #combine_datasets(GENERAL_PATH)
    output_dir = os.path.join(GENERAL_PATH, 'data/split_dataset/SuperDataset')
    map_to_season(output_dir, 'SuperDataset')

if __name__ == "__main__":
    main()



