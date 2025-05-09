import torch
import torchvision
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import pandas as pd
import numpy as np
import os
import re

#NO ESTA DETECTANDO CARASSS

class AlexNetFeatureExtractor:
    def __init__(self):
        alexnet = models.alexnet(pretrained=True)
        self.feature_extractor = alexnet.features
        self.feature_extractor.eval()

        self.transform_img = transforms.Compose([
            transforms.Resize((224, 224)),  # Tamaño estándar para AlexNet
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Imagenes RGB normalizadas
                                std=[0.229, 0.224, 0.225])
        ])

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.deeplab = models.segmentation.deeplabv3_resnet101(pretrained=True)
        self.deeplab = self.deeplab.to(self.device)
        self.deeplab.eval()
        self.transform_deeplab = transforms.Compose([
            transforms.Resize((520, 520)),  
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def extract_imgs_features(self, images_directory, compact=False, face_seg=False):
        # Create database imgs existing model
        print("Creating alexNet database...")
        # Process images and create features
        data = []
        print(f"Looking for images in: {images_directory}")

        df = pd.read_csv(images_directory)

        for _, row in df.iterrows():
            image_path = row['image_path']
            img_season = row['season']
            if not face_seg:
                if compact:
                    features = self.extract_compact_features(image_path)
                else:
                    features = self.extract_features(image_path)
            else:
                if compact:
                    features = self.extract_only_face_compact_features(image_path)
                # else:
                #     features = self.extract_only_face_features(image_path)
            
            features['image_file'] = image_path
            features['season'] = img_season
            data.append(features)
            print(f"Processed {image_path}")

        df = pd.concat(data, ignore_index=True)
        return df
    
    def extract_features(self, image_path):
        img = Image.open(image_path).convert('RGB')
        input_tensor = self.transform_img(img).unsqueeze(0) 
        with torch.no_grad():
            features = self.feature_extractor(input_tensor)

        flattened_features = features.view(features.size(0), -1)
        features_np = flattened_features.numpy()
        df = pd.DataFrame(features_np)
        return df
    

    def extract_compact_features(self, image_path):
        img = Image.open(image_path).convert('RGB')
        input_tensor = self.transform_img(img).unsqueeze(0) 
        with torch.no_grad():
            features = self.feature_extractor(input_tensor)
            pooled = features.mean(dim=(2, 3))  

        feature_vector = pooled.numpy().flatten()
        df = pd.DataFrame([feature_vector], columns=[f'feat_{i}' for i in range(len(feature_vector))])
        return df
    
    def extract_only_face_compact_features(self, image_path):
        img = Image.open(image_path).convert('RGB')
        input_dl = self.transform_deeplab(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.deeplab(input_dl)['out'][0]
            seg = output.argmax(0).cpu().numpy()  # shape: (520, 520)

        # 3. Crear máscara de cara y pelo
        face_hair_mask = np.logical_or(seg == 1, seg == 13).astype(np.uint8)  # 1 = face, 13 = hair
        if face_hair_mask.sum() == 0:
            print(f"Advertencia: no se detectó cara ni pelo en {image_path}")
            return None

        # 4. Aplicar máscara sobre imagen original (sin normalizar)
        img_resized = img.resize((520, 520))
        img_np = np.array(img_resized)
        masked_img_np = img_np * face_hair_mask[:, :, None]  # aplicar en RGB

        # 5. Convertir a PIL y transformarlo para AlexNet
        masked_img = Image.fromarray(masked_img_np)
        input_tensor = self.transform_img(masked_img).unsqueeze(0)

        # 6. Pasar por AlexNet y hacer global average pooling
        with torch.no_grad():
            features = self.feature_extractor(input_tensor)
            pooled = features.mean(dim=(2, 3))  

        feature_vector = pooled.numpy().flatten()
        df = pd.DataFrame([feature_vector], columns=[f'feat_{i}' for i in range(len(feature_vector))])
        return df

