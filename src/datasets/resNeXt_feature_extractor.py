import torch
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import pandas as pd
import numpy as np
import facer
from segmentation import load_parser, segment_face_hair

class ResNeXtFeatureExtractor:
    def __init__(self, variant="resnext101_32x8d"):
        # 1) Cargar ResNeXt pretrained
        resnext = getattr(models, variant)(pretrained=True)
        # 2) Quitamos avgpool y fc: nos quedamos con conv+blocks
        modules = list(resnext.children())[:-2]
        self.feature_extractor = torch.nn.Sequential(*modules)
        self.feature_extractor.eval()

        # 3) Transformaciones estándar
        self.transform_img = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.feature_extractor.to(self.device)

        # 4) Detector y parser igual
        self.detector = facer.face_detector("retinaface/mobilenet",
                                            device=self.device)
        self.parser   = facer.face_parser("farl/celebm/448",
                                          device=self.device)

    def extract_imgs_features(self, csv_path, compact=False):
        data = []
        df_input = pd.read_csv(csv_path)
        for _, row in df_input.iterrows():
            img_path, season = row['image_path'], row['season']
            feats = (self.extract_compact_features(img_path) 
                        if compact else self.extract_features(img_path))
            
            if feats is not None:
                feats['image_file'] = img_path
                feats['season'] = season
                data.append(feats)
            print(f"Processed {img_path}")
        return pd.concat(data, ignore_index=True)

    def extract_features(self, image_path):
        img = Image.open(image_path).convert('RGB')
        inp = self.transform_img(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feats = self.feature_extractor(inp)
        # feats: [1, 2048, 7, 7]
        flat = feats.view(feats.size(0), -1).cpu().numpy()
        return pd.DataFrame(flat)

    def extract_compact_features(self, image_path):
        img = Image.open(image_path).convert('RGB')
        inp = self.transform_img(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feats = self.feature_extractor(inp)
            # pool spatial dims → [1,2048]
            pooled = feats.mean(dim=(2, 3)).cpu().numpy().flatten()
        cols = [f'feat_{i}' for i in range(pooled.shape[0])]
        return pd.DataFrame([pooled], columns=cols)

 