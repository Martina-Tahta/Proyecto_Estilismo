import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import pandas as pd
import torch.nn.functional as F
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import os
from tqdm import tqdm
import gc


class CSVDataset(Dataset):
        def __init__(self, df, transform, class2idx):
            self.df = df.reset_index(drop=True)
            self.transform = transform
            self.class2idx = class2idx

        def __len__(self):
            return len(self.df)

        def __getitem__(self, idx):
            row = self.df.iloc[idx]
            img = Image.open(row['image_path']).convert('RGB')
            img = self.transform(img)
            label = self.class2idx[row['season']]
            return img, label

class SoftmaxWeightedPool2d(nn.Module):
    def __init__(self, channels: int, height: int, width: int):
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(1, channels, height, width))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        B, C, H, W = x.shape
        attn = F.softmax(self.logits.view(1, C, -1), dim=-1)
        attn = attn.view(1, C, H, W)
        return (x * attn).sum(dim=(2,3))  # [B, C]

class ResNeXtWeightedClassifier(nn.Module):
    def __init__(self, variant: str = "resnext50_32x4d", num_classes=12, device=None):
        super().__init__()
        self.num_classes = num_classes
        # 1) Backbone ResNeXt sin avgpool ni fc
        backbone = getattr(models, variant)(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-2])

        # 2) Weighted pooling entrenable
        # Asumimos mapas de [B, 2048, 7, 7]
        self.pool = SoftmaxWeightedPool2d(channels=2048, height=7, width=7)
        
        # 3) Clasificador final
        self.classifier = nn.Sequential(
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

        mean = [0.485, 0.456, 0.406]
        std  = [0.229, 0.224, 0.225]
        self.train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

        self.test_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature_extractor(x)       # [B, 2048, 7, 7]
        pooled   = self.pool(features)             # [B, 2048]
        dropped  = self.dropout(pooled)     
        out      = self.classifier(dropped)         # [B, num_classes]
        return out
    
    def create_dataloader(self, csv_path: str, batch_size: int = 32,
                          shuffle: bool = False, num_workers: int = 4, train=True):
        df = pd.read_csv(csv_path)
        if train:
            dataset = CSVDataset(df, self.train_transform, self.class2idx)
        else:
            dataset = CSVDataset(df, self.test_transform, self.class2idx)
        return DataLoader(dataset, batch_size=batch_size,
                          shuffle=shuffle, num_workers=num_workers)

 
    def train_model(self, train_csv, val_csv, name_classes, model_params=None, save_name=None):
        # --- Extract config ---
        if model_params:
            epochs       = model_params.get("epochs", 50)
            batch_size   = model_params.get("batch_size", 16)  # reduced default batch size
            lr           = model_params.get("lr", 1e-4)
            weight_decay = model_params.get("weight_decay", 1e-5)
            patience     = model_params.get("early_stopping_patience", 10)
            dropout      = model_params.get("dropout", 0.2)
        else:
            epochs, batch_size, lr, weight_decay, patience = 50, 16, 1e-4, 1e-5, 10

        # --- Load data and map classes ---
        self.classes = name_classes
        self.class2idx = {c: i for i, c in enumerate(self.classes)}

        train_loader = self.create_dataloader(train_csv, batch_size=batch_size, shuffle=True, train=True)
        val_loader   = self.create_dataloader(val_csv,   batch_size=batch_size, shuffle=False, train=False)

        self.dropout  = nn.Dropout(p=dropout)

        # --- Setup optimizer, loss, scaler ---
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)
        scaler    = torch.cuda.amp.GradScaler()

        best_val_acc = 0.0
        epochs_no_improve = 0

        for epoch in range(1, epochs + 1):
            # --- Training loop with tqdm ---
            self.train()
            total_loss = 0.0
            train_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [Train]", unit="batch")
            for images, labels in train_bar:
                images = images.to(self.device)
                labels = labels.to(self.device)
                optimizer.zero_grad()

                with torch.cuda.amp.autocast():
                    logits = self(images)
                    loss   = criterion(logits, labels)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                loss_val = loss.item()
                total_loss += loss_val * images.size(0)
                train_bar.set_postfix(loss=loss_val)

                # Clear refs to free memory
                del images, labels, logits, loss
            train_bar.close()

            avg_loss = total_loss / len(train_loader.dataset)

            # --- Validation loop with tqdm ---
            self.eval()
            correct = 0
            val_bar = tqdm(val_loader, desc=f"Epoch {epoch}/{epochs} [Val]  ", unit="batch")
            with torch.no_grad():
                for images, labels in val_bar:
                    images = images.to(self.device)
                    labels = labels.to(self.device)

                    with torch.cuda.amp.autocast():
                        preds = self(images).argmax(dim=1)
                    correct += (preds == labels).sum().item()
                    val_bar.set_postfix(acc=correct / ((val_bar.n+1)*batch_size))

                    del images, labels, preds
            val_bar.close()

            val_acc = correct / len(val_loader.dataset)
            print(f"\nEpoch {epoch}/{epochs}  Train Loss: {avg_loss:.4f}  Val Acc: {val_acc:.4f}")

            # Early Stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                epochs_no_improve = 0
                if save_name:
                    torch.save(self.state_dict(), os.path.join('runs', f'model_{save_name}.pt'))
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f"Early stopping: no improvement in {patience} epochs.")
                    break

            # Force garbage collection and clear CUDA cache
            del train_bar, val_bar
            torch.cuda.empty_cache()
            gc.collect()
            

    def eval_model(self, test_csv, results_folder=None, batch_size=32, num_workers=4):
        test_loader = self.create_dataloader(test_csv, batch_size=batch_size,
                                            num_workers=num_workers, shuffle=False, train=False)

        self.eval()
        all_preds, all_targets = [], []
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                preds = self(images).argmax(dim=1)
                all_preds.extend(preds.cpu().tolist())
                all_targets.extend(labels.cpu().tolist())

        report = classification_report(all_targets, all_preds, target_names=self.classes)
        print("\nClassification Report:\n", report)

        if results_folder:
            cm = confusion_matrix(all_targets, all_preds)
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=self.classes, yticklabels=self.classes)
            plt.title('Confusion Matrix')
            plt.ylabel('True Season')
            plt.xlabel('Predicted Season')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(f'{results_folder}/confusion_matrix.png')
            with open(f'{results_folder}/classification_report.txt', 'w') as f:
                f.write(report)

    def load_params_model(self, weights_path: str, class_names: list):
            """
            Carga pesos guardados y define las clases asociadas.

            Args:
                weights_path (str): Ruta al archivo .pt o .pth del modelo.
                class_names (list): Lista de clases (str) en orden correcto.
            """
            self.classes = class_names
            self.class2idx = {c: i for i, c in enumerate(self.classes)}
            self.dropout = nn.Dropout(p=0.2)  # Reasegura que exista para .forward()
            self.load_state_dict(torch.load(weights_path, map_location=self.device))
            self.eval()
            print(f"Modelo cargado desde {weights_path}.")

    def test_model(self, test_csv: str, batch_size: int = 32, num_workers: int = 4):
        """
        Evalúa el modelo sobre un CSV de test (debe incluir columnas: image_path, season).

        Args:
            test_csv (str): Ruta al CSV.
            batch_size (int): Batch size.
            num_workers (int): Cantidad de workers del DataLoader.
        """
        df = pd.read_csv(test_csv)
        if not hasattr(self, 'class2idx'):
            self.classes = sorted(df['season'].unique())
            self.class2idx = {c: i for i, c in enumerate(self.classes)}
        test_loader = self.create_dataloader(test_csv, batch_size=batch_size, shuffle=False, num_workers=num_workers, train=False)

        self.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for images, labels in tqdm(test_loader, desc="Testing", unit="batch"):
                images = images.to(self.device)
                labels = labels.to(self.device)
                preds = self(images).argmax(dim=1)
                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(labels.cpu().tolist())

        report = classification_report(all_labels, all_preds, target_names=self.classes)
        print("\n--- TEST REPORT ---\n")
        print(report)

        cm = confusion_matrix(all_labels, all_preds)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Purples',
                    xticklabels=self.classes, yticklabels=self.classes)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()