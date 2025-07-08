from typing import Optional, Tuple, List
import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

from huggingface_hub import snapshot_download
import clip  # OpenAI CLIP repository

from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


class ImageCSVDataset(Dataset):
    """Dataset que lee un .csv con las columnas `image_path` y `season`."""
    label_map = None

    def __init__(self, classes, csv_path, img_dir=None, transform=None):
        self.df = pd.read_csv(csv_path)
        if "image_path" not in self.df.columns or "season" not in self.df.columns:
            raise ValueError("El CSV debe tener columnas 'image_path' y 'season'.")
        self.img_dir = img_dir or ""
        self.transform = transform

        if ImageCSVDataset.label_map is None:
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
        if self.transform:
            image = self.transform(image)
        return image, label


class Farl16_FT(nn.Module):
    def __init__(
        self,
        num_classes: int = 12,
        classifier: int = 0,
        device: Optional[str] = None
    ):
        super().__init__()
        self.num_classes = num_classes
        self.classes = None
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")

        # 1) Descargar e inicializar FaRL-16 desde Hugging Face
        repo_id = "Green-Sky/FaRL-Base-Patch16-LAIONFace20M-ep64"
        hf_dir = snapshot_download(repo_id)
        # Cargar modelo CLIP ViT-B/16
        clip_model, _ = clip.load("ViT-B/16", device=self.device)
        # Cargar pesos FaRL
        ckpt = torch.load(os.path.join(hf_dir, "pytorch_model.bin"), map_location=self.device)
        clip_model.load_state_dict(ckpt, strict=False)
        self.backbone = clip_model

        # Determinar dimensión de embedding
        # CLIP ViT-B/16 visual.proj: out_dim x embed_dim
        embed_dim = self.backbone.visual.proj.shape[1]

        # 2) Construir cabeza de clasificación
        if classifier == 0:
            red_dim = embed_dim // 2
            self.classifier = nn.Sequential(
                nn.Linear(embed_dim, red_dim),
                nn.ReLU(),
                nn.Dropout(p=0.5),
                nn.Linear(red_dim, self.num_classes)
            )
            nn.init.xavier_uniform_(self.classifier[0].weight)
            self.classifier[0].bias.data.zero_()
        else:
            self.classifier = nn.Sequential(
                nn.Dropout(p=0.2),
                nn.Linear(embed_dim, self.num_classes)
            )
            nn.init.xavier_uniform_(self.classifier[1].weight)
            self.classifier[1].bias.data.zero_()

        # 3) Transformaciones
        mean = [0.481, 0.457, 0.407]
        std  = [0.268, 0.261, 0.275]
        self.train_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.4, contrast=0.2, saturation=0.2),
            transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        self.test_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

        # 4) Mover a device
        self.to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Extraer embedding con autocast (half precision)
        embed = self.backbone.encode_image(x)
        # Asegurar mismo dtype entre embedding y classifier (float32)
        embed = embed.float()
        return self.classifier(embed)

    def create_dataloader(self, name_classes, csv_path, img_dir=None,
                          batch_size=32, num_workers=4, test=False, shuffle=True):
        transform = self.test_transform if test else self.train_transform
        ds = ImageCSVDataset(name_classes, csv_path, img_dir, transform)
        if self.classes is None:
            inv = {v: k for k, v in ds.label_map.items()}
            self.classes = tuple(inv[i] for i in range(len(inv)))
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                          num_workers=num_workers, pin_memory=True)

    def train_model(self, train_path, val_path, name_classes,
                    model_params=None, verbose=True, save_name=None):
        # Configuraciones
        mp = model_params or {}
        train_blocks  = mp.get("train_blocks", [])  # no aplica a FaRL
        epochs        = mp.get("epochs", 50)
        batch_size    = mp.get("batch_size", 32)
        lr_backbone   = mp.get("lr_backbone", 1e-4)
        lr_fc         = mp.get("lr_fc", 1e-3)
        weight_decay  = mp.get("weight_decay", 1e-5)
        early_stop    = mp.get("early_stopping_patience", 10)

        self.classes = name_classes
        self.class2idx = {c: i for i, c in enumerate(name_classes)}

         # 4) Congelar todo el backbone
        for p in self.backbone.parameters():
            p.requires_grad = False

        # 5) Descongelar solo los bloques especificados
        unfrozen_params = []
        for idx in train_blocks:
            block = self.backbone.visual.transformer.resblocks[idx]
            for p in block.parameters():
                p.requires_grad = True
            unfrozen_params.append(block.parameters())

        # 6) Asegurar que la cabeza clasificadora se entrene
        for p in self.classifier.parameters():
            p.requires_grad = True
        unfrozen_params.append(self.classifier.parameters())

        # DataLoaders
        train_loader = self.create_dataloader(name_classes, train_path,
                                              batch_size=batch_size, test=False)
        val_loader   = self.create_dataloader(name_classes, val_path,
                                              batch_size=batch_size, test=True)

        # Optimizador y scheduler
        opt_groups = []
        for params in unfrozen_params[:-1]:
            opt_groups.append({"params": params, "lr": lr_backbone})
        opt_groups.append({"params": self.classifier.parameters(), "lr": lr_fc})
        optimizer = torch.optim.AdamW(opt_groups, weight_decay=weight_decay)
        criterion = nn.CrossEntropyLoss()
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=1, eta_min=1e-5
        )
        scaler = torch.cuda.amp.GradScaler()

        best_val = -float("inf")
        patience = 0

        for epoch in range(1, epochs+1):
            self.train()
            running_loss = correct = total = 0
            for imgs, labels in tqdm(train_loader, disable=not verbose,
                                     desc=f"Epoch {epoch}/{epochs} [Train]"):
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                with torch.cuda.amp.autocast():
                    logits = self(imgs)
                    loss   = criterion(logits, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

                running_loss += loss.item() * imgs.size(0)
                preds = logits.argmax(dim=1)
                total += labels.size(0)
                correct += (preds == labels).sum().item()
            train_loss = running_loss / len(train_loader.dataset)
            train_acc  = 100 * correct / total
            val_acc    = self._evaluate(val_loader)

            if verbose:
                print(f"Epoch {epoch}/{epochs} | train loss: {train_loss:.4f} | "
                      f"train acc: {train_acc:.2f}% | val acc: {val_acc:.2f}%")

            if val_acc > best_val:
                best_val = val_acc
                patience = 0
                if save_name:
                    os.makedirs("runs", exist_ok=True)
                    torch.save(self.state_dict(), os.path.join(
                        "runs", f"model_farl16_ft_{save_name}.pth"))
            else:
                patience += 1
                if early_stop and patience >= early_stop:
                    if verbose: print("Early stopping: no improvement.")
                    break
        torch.cuda.empty_cache()

    def _evaluate(self, loader: DataLoader) -> float:
        self.eval()
        correct = total = 0
        with torch.no_grad():
            for imgs, labels in loader:
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                preds = self(imgs).argmax(dim=1)
                total += labels.size(0)
                correct += (preds == labels).sum().item()
        return 100.0 * correct / total

    def eval_model(self, test_path, results_folder=None):
        test_loader = self.create_dataloader(self.classes, test_path, test=True)
        self.eval()
        all_preds, all_targets = [], []
        with torch.no_grad():
            for imgs, labels in test_loader:
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                preds = self(imgs).argmax(1)
                all_preds.extend(preds.cpu().tolist())
                all_targets.extend(labels.cpu().tolist())

        report = classification_report(all_targets, all_preds, target_names=self.classes)
        print("\nClassification Report:\n", report)
        if results_folder:
            os.makedirs(results_folder, exist_ok=True)
            cm = confusion_matrix(all_targets, all_preds)
            plt.figure(figsize=(15,12))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=self.classes, yticklabels=self.classes)
            plt.title('Confusion Matrix')
            plt.ylabel('True Season')
            plt.xlabel('Predicted Season')
            plt.tight_layout()
            plt.savefig(f"{results_folder}/confusion_matrix.png")
            with open(f"{results_folder}/classification_report.txt", 'w') as f:
                f.write(report)

    def load_params_model(self, weights_path: str, class_names: list):
        self.classes = class_names
        self.class2idx = {c:i for i,c in enumerate(class_names)}
        self.load_state_dict(torch.load(weights_path, map_location=self.device))
        self.eval()
        print(f"Modelo cargado desde {weights_path}.")

    def test_model(self, test_csv, seasons_only=None, top3=None, batch_size = 32, num_workers = 4):
        df = pd.read_csv(test_csv)
        if not hasattr(self, 'class2idx'):
            self.classes = sorted(df['season'].unique())
            self.class2idx = {c:i for i,c in enumerate(self.classes)}
        test_loader = self.create_dataloader(
            self.classes, test_csv, batch_size=batch_size,
            num_workers=num_workers, shuffle=False, test=True)
        self.eval()
        all_preds, all_labels = [], []
        for imgs, labels in tqdm(test_loader, desc="Testing", unit="batch"):
            imgs = imgs.to(self.device)
            labels = labels.to(self.device)
            preds = self(imgs).argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
        report = classification_report(all_labels, all_preds, target_names=self.classes)
        print("\n--- TEST REPORT ---\n", report)
        cm = confusion_matrix(all_labels, all_preds)
        plt.figure(figsize=(10,8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Purples',
                    xticklabels=self.classes, yticklabels=self.classes)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()