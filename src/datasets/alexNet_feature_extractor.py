import torch
import torchvision
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import pandas as pd
import numpy as np
import os
import re
import matplotlib.pyplot as plt
from insightface.app import FaceAnalysis


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
        
        self.face_app = FaceAnalysis(
            name="antelopev2",                 # ← different model pack
            allowed_modules=['detection', 'parsing']
        )
        ctx_id = 0 if self.device.type == 'cuda' else -1
        self.face_app.prepare(ctx_id=ctx_id, det_size=(320, 320))

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
        # 1) load & numpy
        pil = Image.open(image_path).convert('RGB')
        img = np.array(pil)

        # 2) detect & parsing
        faces = self.face_app.get(img)
        if not faces:
            print(f"No face detected in {image_path}")
            return None

        parsing = faces[0].parsing
        if parsing is None:
            print(f"No parsing map available for {image_path}")
            return None

        # 3) build mask for face (1) or hair (13)
        parsing = np.asarray(parsing)
        mask = np.logical_or(parsing == 1, parsing == 13).astype(np.uint8)

        # 3) apply mask & (optional) plot
        masked = img * mask[:, :, None]
        plt.imshow(masked); plt.axis('off'); plt.title("Face+Hair Mask"); plt.show()

        plt.figure(figsize=(6, 6))
        plt.imshow(masked)
        plt.axis('off')
        plt.title(f"Face+Hair mask for {os.path.basename(image_path)}")
        plt.show()

        # 4) back to PIL → AlexNet
        masked_pil = Image.fromarray(masked).resize((224, 224))
        inp        = self.transform_img(masked_pil).unsqueeze(0)
        with torch.no_grad():
            feats  = self.feature_extractor(inp)
            pooled = feats.mean(dim=(2, 3)).cpu().numpy().flatten()

        cols = [f'feat_{i}' for i in range(len(pooled))]
        return pd.DataFrame([pooled], columns=cols)

