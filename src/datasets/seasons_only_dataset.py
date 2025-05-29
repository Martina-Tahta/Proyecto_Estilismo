import pandas as pd
import os

def create_seasons_only_csv(path_csv,output_csv_path):
    data = pd.read_csv(path_csv)
    data['subcategory'] = data['season']
    data['season'] = data['season'].str.split('_').str[1]
    
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    data.to_csv(output_csv_path, index=False)
    

def main():
    GENERAL_PATH = os.getcwd()
    dataset = 'DeepArmocromia'
    path_csv = os.path.join(GENERAL_PATH, f'data/split_dataset/{dataset}')
    output_csv = os.path.join(GENERAL_PATH, f'data/split_dataset/{dataset}_season_only')

    create_seasons_only_csv(os.path.join(path_csv,f'train_{dataset}.csv'),os.path.join(output_csv,f'train_{dataset}_season_only.csv'))
    create_seasons_only_csv(os.path.join(path_csv,f'val_{dataset}.csv'),os.path.join(output_csv,f'val_{dataset}_season_only.csv'))
    create_seasons_only_csv(os.path.join(path_csv,f'test_{dataset}.csv'),os.path.join(output_csv,f'test_{dataset}_season_only.csv'))

if __name__ == "__main__":
    main()