import os
import torch
import torch.nn as nn
import torch.optim as optim
import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import Dataset, DataLoader
from sklearn.utils import resample


# Dataset para PyTorch
class TabularDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y.values, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# Red neuronal flexible
class FlexibleNN(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, dropout=0):
        super().__init__()
        layers = []
        dims = [input_dim] + hidden_dims
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout))
        layers.append(nn.Linear(dims[-1], output_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


# Clase principal
class NNSeasonalColorModel:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.classes = None
        self.hidden_dims = None
        self.lr = None
        self.batch_size = None
        self.feature_cols = None

    # def preprocess_data(self, csv_path):
    #     """Enhanced preprocessing with feature selection and robust scaling"""
    #     # Separate majority and minority classes with SMOTE-like approach
    #     df = pd.read_csv(csv_path)
    #     classes = df['season'].value_counts()
    #     majority_class = classes.index[0]
    #     n_samples = int(classes[majority_class] * 1.5)  # Increase samples
        
    #     balanced_dfs = []
    #     for season in classes.index:
    #         season_df = df[df['season'] == season]
    #         if len(season_df) < n_samples:
    #             upsampled_df = resample(
    #                 season_df,
    #                 replace=True,
    #                 n_samples=n_samples,
    #                 random_state=42
    #             )
    #             balanced_dfs.append(upsampled_df)
    #         else:
    #             balanced_dfs.append(season_df)
        
    #     df = pd.concat(balanced_dfs)
        
    #     # Remove highly correlated features
    #     feature_cols = [col for col in df.columns 
    #                    if col not in ['image_file', 'season']]
    #     correlation_matrix = df[feature_cols].corr().abs()
    #     upper = correlation_matrix.where(
    #         np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
    #     )
    #     to_drop = [column for column in upper.columns 
    #                if any(upper[column] > 0.95)]
    #     df = df.drop(to_drop, axis=1)
    #     self.feature_cols = [col for col in df.columns 
    #                    if col not in ['image_file', 'season']]
    #     return df


    def train_model(self, train_path, val_path, model_params=None, hidden_dims=[128, 64, 32], 
                    epochs=50, batch_size=32, lr=1e-3, weight_decay=1e-5, dropout=0, save_name=None):

        if model_params:
            hidden_dims = model_params.get("hidden_dims", hidden_dims)
            epochs = model_params.get("epochs", epochs)
            batch_size = model_params.get("batch_size", batch_size)
            lr = model_params.get("lr", lr)
            weight_decay = model_params.get("weight_decay", weight_decay)
            dropout = model_params.get("dropout", dropout)

        # Preprocess data
        #df = self.preprocess_data(train_path)
        df_train = pd.read_csv(train_path)
        #print(df_train.columns.tolist())
        self.feature_cols = df_train.drop(['image_file', 'season'], axis=1).columns.tolist()
        X_train = df_train[self.feature_cols]
        y_train = df_train['season']

        y_encoded, classes = pd.factorize(y_train)
        self.classes = classes

        self.scaler = RobustScaler() #StandardScaler()
        X_scaled = self.scaler.fit_transform(X_train)
        train_ds = TabularDataset(X_scaled, pd.Series(y_encoded))
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)


        df_val = pd.read_csv(val_path)
        X_val = df_val[self.feature_cols]
        X_val_scaled = self.scaler.transform(X_val)
        y_val = df_val['season']
        y_val_encoded = self.classes.get_indexer(y_val)
        val_ds = TabularDataset(X_val_scaled, pd.Series(y_val_encoded))
        val_loader = DataLoader(val_ds, batch_size=batch_size)


        input_dim = X_train.shape[1]
        output_dim = len(self.classes)
        self.hidden_dims = hidden_dims
        self.lr = lr
        self.batch_size = batch_size

        self.model = FlexibleNN(input_dim, hidden_dims, output_dim, dropout=dropout)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)

        best_val_loss = float('inf')
        patience = 20
        counter = 0
        best_model_state = None

        for epoch in range(epochs):
            # ----- ENTRENAMIENTO -----
            self.model.train()
            total_loss = 0
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            avg_train_loss = total_loss / len(train_loader)

            # ----- VALIDACIÓN -----
            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    outputs = self.model(batch_X)
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()
            avg_val_loss = val_loss / len(val_loader)

            # ----- LOG -----
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f}")
            # ----- EARLY STOPPING -----
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model_state = self.model.state_dict()
                counter = 0
            else:
                counter += 1
                if counter >= patience:
                    print(f"\n Early stopping at epoch {epoch+1}. Best val loss: {best_val_loss:.4f}")
                    break

        if best_model_state:
            self.model.load_state_dict(best_model_state)

        if save_name is not None:
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'scaler': self.scaler,
                'classes': self.classes,
                'feature_cols': self.feature_cols,
                'hidden_dims': self.hidden_dims
            }, os.path.join(f'runs/model_NNpytorch_{save_name}.pt'))
        
        return best_val_loss
    

    def eval_model(self, test_path, results_folder=None):
        df = pd.read_csv(test_path)
        X_test = df[self.feature_cols]
        y_test = df['season']

        X_test_scaled = self.scaler.transform(X_test)

        #y_encoded = [self.classes.tolist().index(label) for label in y_test]
        y_encoded = self.classes.get_indexer(y_test)

        test_ds = TabularDataset(X_test_scaled, pd.Series(y_encoded))
        test_loader = DataLoader(test_ds, batch_size=self.batch_size)

        # Evaluación
        self.model.eval()
        all_preds, all_targets = [], []
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                outputs = self.model(batch_X)
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.tolist())
                all_targets.extend(batch_y.tolist())

        report = classification_report(all_targets, all_preds, target_names=self.classes, zero_division=0)
        print("\nClassification Report:\n", report)
    
        if results_folder is not None:
            # Plot confusion matrix
            plt.figure(figsize=(15, 12))
            cm = confusion_matrix(all_targets, all_preds)
            sns.heatmap(
                cm, 
                annot=True, 
                fmt='d', 
                cmap='Blues',
                xticklabels=self.classes,
                yticklabels=self.classes
            )
            plt.title('Confusion Matrix')
            plt.ylabel('True Season')
            plt.xlabel('Predicted Season')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(f'{results_folder}/confusion_matrix.png')

            report = classification_report(all_targets, all_preds, target_names=self.classes)
            report_file_path = f'{results_folder}/classification_report.txt'
            with open(report_file_path, 'w') as f:
                f.write(report)

    def load_params_model(self, model_path): 
        bundle = torch.load(model_path, weights_only=False)

        # Asignar atributos cargados
        self.scaler = bundle['scaler']
        self.classes = bundle['classes']
        self.feature_cols = bundle['feature_cols']
        self.hidden_dims = bundle['hidden_dims']

        input_dim = len(self.feature_cols)
        output_dim = len(self.classes)

        self.model = FlexibleNN(input_dim, self.hidden_dims, output_dim)
        self.model.load_state_dict(bundle['model_state_dict'])
        self.model.eval()

    def predict_season(self, img_features):
        if self.model is None or self.scaler is None or self.classes is None:
            raise ValueError("El modelo no está entrenado o no se cargo.")

        try:
            if isinstance(img_features, dict):
                img_features = pd.DataFrame([img_features]) 
            else:
                img_features = pd.DataFrame(img_features)
            img_features = img_features[self.feature_cols]

            X_scaled = self.scaler.transform(img_features)
            X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

            # Predicción
            with torch.no_grad():
                outputs = self.model(X_tensor)
                probs = torch.softmax(outputs, dim=1).numpy()[0]
                pred_idx = np.argmax(probs)

            return {
                'predicted_season': self.classes[pred_idx],
                'confidence_scores': {
                    season: float(prob) for season, prob in zip(self.classes, probs)
                }
            }

        except Exception as e:
            return {'error': str(e)}
    
    
