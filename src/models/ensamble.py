import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, AdaBoostClassifier
from sklearn.metrics import classification_report, confusion_matrix
from xgboost import XGBClassifier
from sklearn.utils import resample
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import warnings

import numpy as np
import cv2
import face_recognition
import pandas as pd


class EnhancedSeasonalColorFeatureExtractor:
    def __init__(self):
        return
        
    def extract_imgs_features(self, images_directory, full_path=''):  
        # Create database imgs existing model
        print("Creating enhanced database...")
        # Process images and create features
        data = []
        print(f"Looking for images in: {images_directory}")

        df = pd.read_csv(images_directory)
        
        for _, row in df.iterrows():
            image_path = row['image_path']
            img_season = row['season']

            features = self.extract_enhanced_features(full_path + image_path)
            
            if features:
                # features['image_file'] = image_path
                # features['season'] = img_season
                # data.append(features)
                # #print(f"Processed {image_path}")
                vals = np.array(list(features.values()), dtype=float)
                if not np.isnan(vals).any():
                    features['image_path'] = image_path
                    features['season']     =img_season
                    data.append(features)
                else:
                    print(f"Skipping {row['image_path']}: contiene NaNs")
        print(f"All images have been processed")
        #df = pd.DataFrame(data)
        df = pd.DataFrame(data).dropna(how='any', axis=0)
        
        return df


    def extract_enhanced_features(self, image_path):
        """
        Extract enhanced color features from an image
        """
        try:
            # Load image
            image = face_recognition.load_image_file(image_path)
            face_locations = face_recognition.face_locations(image)
            
            if not face_locations:
                return None
                
            # Convert to different color spaces
            image_cv = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            image_lab = cv2.cvtColor(image_cv, cv2.COLOR_BGR2LAB)
            image_hsv = cv2.cvtColor(image_cv, cv2.COLOR_BGR2HSV)

            features = {}
            
            # Get face landmarks
            face_landmarks = face_recognition.face_landmarks(image, face_locations)
            if not face_landmarks:
                return None
                
            landmarks = face_landmarks[0]
            top, right, bottom, left = face_locations[0]
            face = image_cv[top:bottom, left:right]
            face_lab = image_lab[top:bottom, left:right]
            face_hsv = image_hsv[top:bottom, left:right]
            
            # Extract eye features
            eye_features = self._extract_eye_features(
                image_cv, image_lab, image_hsv, landmarks)
            features.update(eye_features)
            
            # Extract skin features
            skin_features = self._extract_skin_features(
                face, face_lab, face_hsv, landmarks, image_path)
            features.update(skin_features)
            
            # Extract hair features
            hair_features = self._extract_hair_features(
                image_cv, image_lab, image_hsv, top, left, right)
            features.update(hair_features)
            
            # Extract contrast features
            contrast_features = self._extract_contrast_features(
                face, face_lab, face_hsv)
            features.update(contrast_features)
            
            return features
            
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
            return None
            
    def _extract_eye_features(self, image_cv, image_lab, image_hsv, landmarks):
        """Extract detailed eye color features"""
        features = {}
        
        for eye_name, eye_points in [('left_eye', landmarks['left_eye']), 
                                   ('right_eye', landmarks['right_eye'])]:
            eye_region = self._get_eye_region(image_cv, eye_points)
            eye_region_lab = self._get_eye_region(image_lab, eye_points)
            eye_region_hsv = self._get_eye_region(image_hsv, eye_points)
            
            # BGR values
            bgr_mean = cv2.mean(eye_region)[:3]
            features.update({
                f'{eye_name}_b': bgr_mean[0],
                f'{eye_name}_g': bgr_mean[1],
                f'{eye_name}_r': bgr_mean[2],
            })
            
            # LAB values
            lab_mean = cv2.mean(eye_region_lab)[:3]
            features.update({
                f'{eye_name}_l': lab_mean[0],
                f'{eye_name}_a': lab_mean[1],
                f'{eye_name}_b_lab': lab_mean[2],
            })
            
            # HSV values
            hsv_mean = cv2.mean(eye_region_hsv)[:3]
            features.update({
                f'{eye_name}_h': hsv_mean[0],
                f'{eye_name}_s': hsv_mean[1],
                f'{eye_name}_v': hsv_mean[2],
            })
        
        return features
    
    def _extract_skin_features(self, face, face_lab, face_hsv, landmarks, image_path):
        """Extract detailed skin color features"""
        features = {}
        
        # # Get skin tone features first (only once per image)
        # skin_tone_features = skin_tone_detection(image_path)
        # features.update({
        #     'skin_dominant_color': skin_tone_features[0],
        #     'skin_tone': skin_tone_features[1]
        # })
        
        # Define multiple sampling regions for more robust skin color detection
        regions = {
            'forehead': (0.2, 0.3, 0.8, 0.4),
            'left_cheek': (0.5, 0.2, 0.7, 0.35),
            'right_cheek': (0.5, 0.65, 0.7, 0.8),
            'chin': (0.8, 0.4, 0.9, 0.6)
        }
        
        for region_name, (y1, x1, y2, x2) in regions.items():
            h, w = face.shape[:2]
            region = face[int(h*y1):int(h*y2), int(w*x1):int(w*x2)]
            region_lab = face_lab[int(h*y1):int(h*y2), int(w*x1):int(w*x2)]
            region_hsv = face_hsv[int(h*y1):int(h*y2), int(w*x1):int(w*x2)]
            
            if region.size > 0:
                # BGR values
                bgr_mean = cv2.mean(region)[:3]
                features.update({
                    f'skin_{region_name}_b': bgr_mean[0],
                    f'skin_{region_name}_g': bgr_mean[1],
                    f'skin_{region_name}_r': bgr_mean[2],
                })
                
                # LAB values
                lab_mean = cv2.mean(region_lab)[:3]
                features.update({
                    f'skin_{region_name}_l': lab_mean[0],
                    f'skin_{region_name}_a': lab_mean[1],
                    f'skin_{region_name}_b_lab': lab_mean[2],
                })
                
                # HSV values
                hsv_mean = cv2.mean(region_hsv)[:3]
                features.update({
                    f'skin_{region_name}_h': hsv_mean[0],
                    f'skin_{region_name}_s': hsv_mean[1],
                    f'skin_{region_name}_v': hsv_mean[2],
                })
        
        return features
    
    def _extract_hair_features(self, image_cv, image_lab, image_hsv, top, left, right):
        """Extract detailed hair color features"""
        features = {}
        
        # Sample multiple regions of hair
        hair_regions = [
            image_cv[max(0, top-30):top, left:right],  # Top
            image_cv[max(0, top-30):top, left:left+30],  # Left side
            image_cv[max(0, top-30):top, right-30:right]  # Right side
        ]
        
        hair_regions_lab = [
            image_lab[max(0, top-30):top, left:right],
            image_lab[max(0, top-30):top, left:left+30],
            image_lab[max(0, top-30):top, right-30:right]
        ]
        
        hair_regions_hsv = [
            image_hsv[max(0, top-30):top, left:right],
            image_hsv[max(0, top-30):top, left:left+30],
            image_hsv[max(0, top-30):top, right-30:right]
        ]
        
        for i, (region, region_lab, region_hsv) in enumerate(zip(
            hair_regions, hair_regions_lab, hair_regions_hsv)):
            if region.size > 0:
                # BGR values
                bgr_mean = cv2.mean(region)[:3]
                features.update({
                    f'hair_region{i}_b': bgr_mean[0],
                    f'hair_region{i}_g': bgr_mean[1],
                    f'hair_region{i}_r': bgr_mean[2],
                })
                
                # LAB values
                lab_mean = cv2.mean(region_lab)[:3]
                features.update({
                    f'hair_region{i}_l': lab_mean[0],
                    f'hair_region{i}_a': lab_mean[1],
                    f'hair_region{i}_b_lab': lab_mean[2],
                })
                
                # HSV values
                hsv_mean = cv2.mean(region_hsv)[:3]
                features.update({
                    f'hair_region{i}_h': hsv_mean[0],
                    f'hair_region{i}_s': hsv_mean[1],
                    f'hair_region{i}_v': hsv_mean[2],
                })
        
        return features
    
    def _extract_contrast_features(self, face, face_lab, face_hsv):
        """Extract contrast and relationship features"""
        features = {}

        # Calculate overall contrast in grayscale
        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        features['contrast'] = np.std(gray)
        
        # Calculate color variance in BGR
        for i, channel in enumerate(['b', 'g', 'r']):
            features[f'{channel}_variance'] = np.std(face[:, :, i])
        
        # Calculate color variance in LAB
        for i, channel in enumerate(['l', 'a', 'b']):
            features[f'{channel}_variance'] = np.std(face_lab[:, :, i])
        
        # Calculate color variance in HSV
        for i, channel in enumerate(['h', 's', 'v']):
            features[f'{channel}_variance'] = np.std(face_hsv[:, :, i])
        
        # --- New section: YCrCb-based features ---
        face_ycrcb = cv2.cvtColor(face, cv2.COLOR_BGR2YCrCb)
        # Calculate means
        features['y_mean']  = np.mean(face_ycrcb[:, :, 0])
        features['cr_mean'] = np.mean(face_ycrcb[:, :, 1])
        features['cb_mean'] = np.mean(face_ycrcb[:, :, 2])
        # Calculate standard deviations
        features['y_variance']  = np.std(face_ycrcb[:, :, 0])
        features['cr_variance'] = np.std(face_ycrcb[:, :, 1])
        features['cb_variance'] = np.std(face_ycrcb[:, :, 2])
        
        return features
    
    def _get_eye_region(self, image, eye_points):
        """Helper function to get eye region from landmarks"""
        eye_points = np.array(eye_points)
        x, y, w, h = cv2.boundingRect(eye_points)
        return image[y:y+h, x:x+w]
    



class EnhancedSeasonalColorModel:
    def __init__(self):
        self.feature_cols = None
        self.pipeline = None
        self.extractor = EnhancedSeasonalColorFeatureExtractor()
        
    def create_ensemble(self, model_params=None):
        """Create a voting ensemble of multiple models"""
        
        model_params = model_params or {}

        rf_params = model_params.get('rf', {})
        #gb_params = model_params.get('gb', {})
        xgb_params = model_params.get('xgb', {})
        ada_params = model_params.get('ada', {})
        weights = model_params.get('model_weights', [1, 1, 1])#, 1])


        models = [
            ('rf', RandomForestClassifier(
                class_weight='balanced',
                **rf_params
            )),
            # ('gb', GradientBoostingClassifier(
            #     **gb_params
            # )),
            ('xgb', XGBClassifier(
                eval_metric='mlogloss',
                use_label_encoder=False,
                **xgb_params
            )),
            ('ada', AdaBoostClassifier(
                **ada_params
            ))
        ]
            
        return VotingClassifier(
            estimators=models,
            voting='soft',
            weights=weights,
            verbose=True,
        )
    
    def train_model(self, train_path, val_path=None, names=None, model_params=None, save_name=""):
        """Train an improved ensemble model with advanced parameter tuning"""
        # Preprocess data
        df = self.extractor.extract_imgs_features(train_path)
        
        # Prepare features
        X_train = df.drop(['image_path','season'], axis=1)
        y_train = df['season']
        
        
        # Create pipeline with robust scaler
        self.pipeline = Pipeline([
            ('scaler', RobustScaler()),
            ('classifier', self.create_ensemble(model_params=model_params))
        ])
        
        # Fit the model
        self.pipeline.fit(X_train, y_train)
        
        # Save model
        joblib.dump(self.pipeline, f'runs/model_ensamble_{save_name}.joblib')
        
        
    def eval_model(self, test_path, results_folder=None):
        df = self.extractor.extract_imgs_features(test_path)
        X_test = df.drop(['image_path','season'], axis=1)
        y_test = df['season']

        # Evaluate
        y_pred = self.pipeline.predict(X_test)                

        if results_folder is not None:
            # Plot confusion matrix
            plt.figure(figsize=(15, 12))
            cm = confusion_matrix(y_test, y_pred)
            sns.heatmap(
                cm, 
                annot=True, 
                fmt='d', 
                cmap='Blues',
                xticklabels=np.unique(y_test),
                yticklabels=np.unique(y_test)
            )
            plt.title('Confusion Matrix')
            plt.ylabel('True Season')
            plt.xlabel('Predicted Season')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(f'{results_folder}/confusion_matrix.png')

            report = classification_report(y_test, y_pred)
            report_file_path = f'{results_folder}/classification_report.txt'
            with open(report_file_path, 'w') as f:
                f.write(report)

    def load_params_model(self, model_path, names=None):
        """Initialize the tester with the trained model"""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            self.pipeline = joblib.load(model_path)
            
        # Get feature names from the model pipeline
        if hasattr(self.pipeline, 'feature_names_in_'):
            self.feature_cols = self.pipeline.feature_names_in_
        else:
            self.feature_cols = self.pipeline.steps[0][1].feature_names_in_

        self.extractor = EnhancedSeasonalColorFeatureExtractor()
        
  

    def test_model(self, test_dataset_path, seasons_only=False, top3=None):
        df = self.extractor.extract_imgs_features(test_dataset_path)
        X_test = df.drop(['image_path','season'], axis=1)
        y_test = df['season']

        # Evaluate
        y_pred = self.pipeline.predict(X_test)                


        # Plot confusion matrix
        plt.figure(figsize=(15, 12))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=np.unique(y_test),
            yticklabels=np.unique(y_test)
        )
        plt.title('Confusion Matrix')
        plt.ylabel('True Season')
        plt.xlabel('Predicted Season')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

        report = classification_report(y_test, y_pred)
        print("\nClassification Report:\n", report)


    # def predict_season(self, img_features):
    #     if self.pipeline is None:
    #         return {'error': 'Pipeline no cargado o existente. Asegúrese de inicializar el modelo correctamente.'}
        
    #     try:
    #         if isinstance(img_features, dict):
    #             img_features = pd.DataFrame([img_features]) 
    #         else:
    #             img_features = pd.DataFrame(img_features)
    #         img_features = img_features[self.feature_cols]
    #         prediction = self.pipeline.predict(img_features)
    #         probabilities = self.pipeline.predict_proba(img_features)

    #         season_probs = {
    #             season: prob
    #             for season, prob in zip(self.pipeline.classes_, probabilities[0])
    #         }

    #         return {
    #             'predicted_season': prediction[0],
    #             'confidence_scores': season_probs
    #         }

    #     except Exception as e:
    #         return {'error': f"Error al realizar la predicción: {str(e)}"}