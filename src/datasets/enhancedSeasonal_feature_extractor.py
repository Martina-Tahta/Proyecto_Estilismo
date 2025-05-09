import numpy as np
import cv2
import face_recognition
from sklearn.preprocessing import StandardScaler
import os
import re
import pandas as pd

from skin_undertone import skin_tone_detection

class EnhancedSeasonalColorDatabase:
    def __init__(self):
        return
        
    def extract_imgs_features(self, images_directory):  
        # Create database imgs existing model
        print("Creating enhanced database...")
        # Process images and create features
        data = []
        print(f"Looking for images in: {images_directory}")

        df = pd.read_csv(images_directory)
        
        for _, row in df.iterrows():
            image_path = row['image_path']
            img_season = row['season']

            features = self.extract_enhanced_features(image_path)
            
            if features:
                features['image_file'] = image_path
                features['season'] = img_season
                data.append(features)
                print(f"Processed {image_path}")

        df = pd.DataFrame(data)
        
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
        
        # Get skin tone features first (only once per image)
        skin_tone_features = skin_tone_detection(image_path)
        features.update({
            'skin_dominant_color': skin_tone_features[0],
            'skin_tone': skin_tone_features[1]
        })
        
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