import pandas as pd
from sklearn.model_selection import train_test_split
import os

def split_and_save_dataset(df, output_dir, name_dataset, features_name, test_size=0.2, random_state=42):
    """
    Divide un CSV original en train y test, y guarda los archivos resultantes.
    """
    X = df.drop(columns=['season']) 
    y = df['season']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)

    os.makedirs(output_dir, exist_ok=True)
    train_df.to_csv(os.path.join(output_dir, f"train_{name_dataset}_{features_name}.csv"), index=False)
    test_df.to_csv(os.path.join(output_dir, f"test_{name_dataset}_{features_name}.csv"), index=False)

