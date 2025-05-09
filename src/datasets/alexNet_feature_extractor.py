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
import facer


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
        
        # — FaRL detector + parser from pyfacer —
        #    'retinaface/mobilenet' is a fast face detector  
        #    'farl/celebm/448' is the BiSeNet-V1 model on CelebAMask-HQ
        self.detector = facer.face_detector("retinaface/mobilenet",
                                           device=self.device)
        self.parser   = facer.face_parser("farl/celebm/448",
                                         device=self.device)

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
        # 1) load & to B×3×H×W tensor
        img_pil = Image.open(image_path).convert("RGB")
        img_np  = np.array(img_pil)
        # facer expects CHW, scale [0–1]
        img_t   = facer.hwc2bchw(img_np).to(self.device)

        # 2) detect faces
        with torch.inference_mode():
            dets = self.detector(img_t)

            # 3) parse the first face
            #    returns a dict with 'seg'→{'logits': Tensor[N×19×h×w], ...}
            faces = self.parser(img_t, dets)
            seg_logits = faces["seg"]["logits"]  # N×19×H×W

        if seg_logits.numel() == 0:
            print(f"No face parsed in {image_path}")
            return None

        # 4) get H×W mask of (skin=1) ∪ (hair=17)
        seg_probs = seg_logits.softmax(dim=1)
        seg_map   = seg_probs.argmax(dim=1)[0].cpu().numpy()
        mask      = np.logical_or(seg_map == 1, seg_map == 17).astype(np.uint8)

        # 5) apply mask & (optional) plot
        masked = img_np * mask[:, :, None]
        
        plt.figure(figsize=(5,5))
        plt.imshow(masked)
        plt.axis("off")
        plt.title(os.path.basename(image_path))
        plt.show()

        # 6) back to PIL → AlexNet pooling
        masked_pil = Image.fromarray(masked).resize((224, 224))
        inp        = self.transform_img(masked_pil).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feats  = self.feature_extractor(inp)
            pooled = feats.mean(dim=(2,3)).cpu().numpy().flatten()

        cols = [f"feat_{i}" for i in range(len(pooled))]
        return pd.DataFrame([pooled], columns=cols)

