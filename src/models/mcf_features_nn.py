import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd

from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from MCF.models_mcf import mae_vit_base_patch16_dec512d2b as modelMCF


class ColorimetryDataset(Dataset):
    def __init__(self, csv_file, transform=None, class_map=None):
        self.df = pd.read_csv(csv_file)
        self.transform = transform
        # Build mapping from season names to indices if not provided
        if class_map is None:
            seasons = sorted(self.df['season'].unique())
            self.class_map = {s: i for i, s in enumerate(seasons)}
        else:
            self.class_map = class_map
        # Convert season names to indices
        self.labels = self.df['season'].map(self.class_map).values
        self.paths = self.df['image_path'].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.paths[idx]
        label = int(self.labels[idx])
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label


class ColorimetryClassifier(nn.Module):
    """
    Modelo que utiliza un encoder MCF preentrenado para extraer features
    y un clasificador feed-forward de 12 clases al final.
    """
    def __init__(self,
                 mcf_variant: str = 'resnext101_32x8d',
                 feature_dim: int = 2048,
                 hidden_dim: int = 512,
                 num_classes: int = 12,
                 dropout_p: float = 0.3,
                 freeze_encoder: bool = True):
        super(ColorimetryClassifier, self).__init__()
        # Cargar encoder MCF preentrenado
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.encoder = modelMCF(pretrained_ckpt='MCF/pretrain_mae_online4_diffmask_laion_01_t02_cos_b64_single_dec512d2b_checkpoint-15.pth')
        self.encoder.to(self.device)
        self.encoder.eval()  # Modo evaluación para el encoder


        # Congelar encoder si no queremos entrenar todos sus pesos
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        # Clasificador feed-forward
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p),
            nn.Linear(hidden_dim, num_classes)
        )

        self.train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
        self.test_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
        self.to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Extraer features con MCF
        features = self.encoder(x)
        logits = self.classifier(features)
        return logits

    
    def train_model(self, train_csv, val_csv, model_params=None, verbose=True, save_name=None):
        # --- Extracción de parámetros de configuración ---
        if model_params:#
            epochs       = model_params.get("epochs", 50)
            batch_size   = model_params.get("batch_size", 32)
            lr        = model_params.get("lr", 1e-3)
            weight_decay = model_params.get("weight_decay", 1e-5)
            early_stopping_patience = model_params.get("early_stopping_patience", 10)
   
        """
        Entrena el modelo usando datasets definidos en CSVs.
        CSVs deben tener columnas 'image_path' y 'season'.
        El DataLoader de entrenamiento usa shuffle=True.
        """
        # Datasets
        train_dataset = ColorimetryDataset(train_csv, transform=self.train_transform)
        val_dataset = ColorimetryDataset(val_csv,
                                        transform=self.test_transform,
                                        class_map=train_dataset.class_map)

        # DataLoaders
        train_loader = DataLoader(train_dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=4,
                                  pin_memory=True)
        val_loader = DataLoader(val_dataset,
                                batch_size=batch_size,
                                shuffle=False,
                                num_workers=4,
                                pin_memory=True)

        # Optimizer y loss
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.parameters()),
                               lr=lr,
                               weight_decay=weight_decay)
        criterion = nn.CrossEntropyLoss()

        self.to(self.device)
        best_val_acc = 0.0

        for epoch in range(1, epochs + 1):
            # Modo entrenamiento
            self.train()
            train_loss = 0.0
            correct = total = 0
            for images, labels in train_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                optimizer.zero_grad()
                outputs = self(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
            train_loss /= total
            train_acc = correct / total

            # Validación
            self.eval()
            val_loss = 0.0
            correct = total = 0
            with torch.no_grad():
                for images, labels in val_loader:
                    images = images.to(self.device)
                    labels = labels.to(self.device)
                    outputs = self(images)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item() * images.size(0)
                    _, preds = torch.max(outputs, 1)
                    correct += (preds == labels).sum().item()
                    total += labels.size(0)
            val_loss /= total
            val_acc = correct / total

            print(f"Epoch {epoch}/{epochs} | "
                  f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")

            # Guardar mejor modelo
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(self.state_dict(), 'best_colorimetry_model.pth')
        print(f"Mejor Val Acc: {best_val_acc:.4f}")


    def eval_model(self,
                   test_csv: str,
                   results_folder: str = None,
                   batch_size: int = 32,
                   num_workers: int = 4):

        test_dataset = ColorimetryDataset(test_csv, transform=self.test_transform,
                                          class_map=self.class_map)
        test_loader = DataLoader(test_dataset,
                                 batch_size=batch_size,
                                 shuffle=False,
                                 num_workers=num_workers,
                                 pin_memory=True)

        self.eval()
        all_preds, all_targets = [], []
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                preds = self(images).argmax(dim=1)
                all_preds.extend(preds.cpu().tolist())
                all_targets.extend(labels.cpu().tolist())

        report = classification_report(all_targets, all_preds,
                                       target_names=self.classes)
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