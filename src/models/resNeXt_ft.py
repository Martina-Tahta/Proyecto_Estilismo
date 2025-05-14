from typing import Optional, Tuple, List
import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm

from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


class ImageCSVDataset(Dataset):
    """Dataset que lee un .csv con las columnas `image_path` y `season`."""
    label_map = None

    def __init__(self, csv_path: str, img_dir=None, transform=None):
        self.df = pd.read_csv(csv_path)
        if "image_path" not in self.df.columns or "season" not in self.df.columns:
            raise ValueError("El CSV debe tener columnas 'image_' y 'season'.")
        self.img_dir = img_dir or ""
        self.transform = transform

        if ImageCSVDataset.label_map is None:
            classes = sorted(self.df["season"].unique())
            ImageCSVDataset.label_map = {cls: idx for idx, cls in enumerate(classes)}

        self.label_map = ImageCSVDataset.label_map

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        path = (
            os.path.join(self.img_dir, row["image_path"])
            if self.img_dir and not os.path.isabs(row["image_path"])
            else row["image_path"]
        )
        image = Image.open(path).convert("RGB")
        label = self.label_map[row["season"]]

        image = self.transform(image)
        return image, label



class ResNeXt_FT:
    def __init__(self, variant="resnext50_32x4d", device=None):
        self.num_classes = 12
        self.classes = None
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")

        if variant == "resnext50_32x4d":
            weights = models.ResNeXt50_32X4D_Weights.DEFAULT
        elif variant == "resnext101_32x8d":
            weights = models.ResNeXt101_32X8D_Weights.DEFAULT

        self.model = getattr(models, variant)(weights=weights)
        self.model.to(self.device)

        self.train_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.test_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])


    def create_dataloader(self, csv_path, img_dir=None, batch_size=32, num_workers=4, test=False):
        """Crea DataLoader a partir de CSV ."""
        if test:
            ds = ImageCSVDataset(csv_path, img_dir, self.test_transform)
        else:
            ds = ImageCSVDataset(csv_path, img_dir, self.train_transform)

        if self.classes is None:
            inv_map = {v: k for k, v in ds.label_map.items()}
            self.classes = tuple(inv_map[i] for i in range(len(inv_map)))

        loader = DataLoader(ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers, pin_memory=True)
        return loader

    
    def train_model(self, train_path, val_path, model_params=None, verbose=True, save_name=None):
        # --- Extracción de parámetros de configuración ---
        if model_params:
            train_blocks = model_params.get("train_blocks", ['layer4'])
            blocks       = model_params.get("blocks", {'layer4':[0,1,2]})   # índices de los sub-bloques
            epochs       = model_params.get("epochs", 50)
            batch_size   = model_params.get("batch_size", 32)
            lr_backbone  = model_params.get("lr_backbone", 1e-4)
            lr_fc        = model_params.get("lr_fc", 1e-3)
            weight_decay = model_params.get("weight_decay", 1e-5)
            early_stopping_patience = model_params.get("early_stopping_patience", 10)
            dropout      = model_params.get("dropout", 0.2)

        # --- 1. Congelar todo el modelo de golpe ---
        self.model.requires_grad_(False)

        # --- 2. Descongelar solo los sub-bloques deseados de cada layer ---
        # Por cada layer en train_blocks, sacamos sus children (los bottleneck blocks)
        # y descongelamos solo aquellos cuyos índices aparecen en `blocks`.
        unfrozen_params = []  # para el optimizador
        for layer_name in train_blocks:
            layer_module = getattr(self.model, layer_name)
            children = list(layer_module.children())
            for idx in blocks[layer_name]:
                # permite índices negativos
                real_idx = idx if idx >= 0 else len(children) + idx
                block = children[real_idx]
                # marcar todos sus parámetros como entrenables
                for p in block.parameters():
                    p.requires_grad = True
                unfrozen_params.append(block.parameters())

        # --- 3. Reemplazar y habilitar la capa fc (con dropout) ---
        self.model.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.model.fc.in_features, self.num_classes)
        )
        nn.init.xavier_uniform_(self.model.fc[1].weight)
        self.model.fc[1].bias.data.zero_()
        # aseguramos que gradúe
        for p in self.model.fc.parameters():
            p.requires_grad = True
        unfrozen_params.append(self.model.fc.parameters())
        self.model.fc.to(self.device)

        # --- 4. Construir optimizador con grupos de LR discriminativos ---
        opt_groups = []
        # lr_backbone para cada bloque
        for params in unfrozen_params[:-1]:
            opt_groups.append({"params": params, "lr": lr_backbone})
        # lr_fc para la cabeza
        opt_groups.append({"params": self.model.fc.parameters(), "lr": lr_fc})

        self.optimizer = torch.optim.AdamW(opt_groups, weight_decay=weight_decay)
        self.criterion  = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=epochs)

        train_loader = self.create_dataloader(train_path, batch_size=batch_size, test=False)
        val_loader = self.create_dataloader(val_path, batch_size=batch_size, test=True)

        best_val_acc = -float("inf")
        patience_cnt = 0

        scaler = torch.cuda.amp.GradScaler()   # opcional (AMP)

        for epoch in range(1, epochs + 1):
            self.model.train()
            running_loss, correct, total = 0.0, 0, 0

            for images, labels in tqdm(train_loader, disable=not verbose):
                images, labels = images.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()
                with torch.cuda.amp.autocast():             # AMP
                    logits = self.model(images)
                    loss   = self.criterion(logits, labels)

                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()

                running_loss += loss.item() * images.size(0)
                preds = logits.argmax(1)
                total += labels.size(0)
                correct += (preds == labels).sum().item()

            self.scheduler.step()

            train_loss = running_loss / len(train_loader.dataset)
            train_acc  = 100 * correct / total
            val_acc    = self._evaluate(val_loader)

            if verbose:
                print(f"Epoch {epoch:02d}/{epochs} | "
                    f"train loss: {train_loss:.4f} | "
                    f"train acc: {train_acc:.2f}% | "
                    f"val acc: {val_acc:.2f}%")

            # ----- Early stopping + checkpoint -----
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_cnt = 0
                torch.save(self.model.state_dict(), os.path.join(f'runs/model_resNeXt_ft_{save_name}.pt'))   # guarda el mejor
            else:
                patience_cnt += 1
                if early_stopping_patience and patience_cnt >= early_stopping_patience:
                    if verbose:
                        print("Early stopping: no improvement.")
                    break

    def _evaluate(self, loader: DataLoader) -> float:
        self.model.eval()
        correct = total = 0
        with torch.no_grad():
            for images, labels in loader:
                images, labels = images.to(self.device), labels.to(self.device)
                preds = self.model(images).argmax(1)
                total += labels.size(0)
                correct += (preds == labels).sum().item()
        return 100.0 * correct / total

    # ---------------------------------------------------------------------
    def  eval_model(self, test_path, results_folder=None):
        test_loader = self.create_dataloader(test_path, test=True)
        
        self.model.eval()
        #correct = total = 0
        all_preds, all_targets = [], []
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                preds = self.model(images).argmax(1)
                all_preds.extend(preds.tolist())
                all_targets.extend(labels.tolist())
                # total += labels.size(0)
                # correct += (preds == labels).sum().item()



        report = classification_report(all_targets, all_preds, target_names=self.classes)
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


    # ---------------------------------------------------------------------
    def save(self, path: str) -> None:
        torch.save(self.model.state_dict(), path)

    def load(self, path: str) -> None:
        state = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state)
        self.model.to(self.device)