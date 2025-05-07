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


class EnhancedSeasonalColorModel:
    def __init__(self):
        self.feature_cols = None
        self.pipeline = None
        
    def preprocess_data(self, csv_path):
        """Enhanced preprocessing with feature selection and robust scaling"""
        # Separate majority and minority classes with SMOTE-like approach
        df = pd.read_csv(csv_path)
        classes = df['season'].value_counts()
        majority_class = classes.index[0]
        n_samples = int(classes[majority_class] * 1.5)  # Increase samples
        
        balanced_dfs = []
        for season in classes.index:
            season_df = df[df['season'] == season]
            if len(season_df) < n_samples:
                upsampled_df = resample(
                    season_df,
                    replace=True,
                    n_samples=n_samples,
                    random_state=42
                )
                balanced_dfs.append(upsampled_df)
            else:
                balanced_dfs.append(season_df)
        
        df = pd.concat(balanced_dfs)
        
        # Remove highly correlated features
        feature_cols = [col for col in df.columns 
                       if col not in ['image_file', 'season']]
        correlation_matrix = df[feature_cols].corr().abs()
        upper = correlation_matrix.where(
            np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
        )
        to_drop = [column for column in upper.columns 
                   if any(upper[column] > 0.95)]
        df = df.drop(to_drop, axis=1)
        self.feature_cols = [col for col in df.columns if col not in ['image_file', 'season']]
        return df
        
    def create_ensemble(self, model_params=None):
        """Create a voting ensemble of multiple models"""
        
        
        model_params = model_params or {}

        rf_params = model_params.get('rf', {})
        gb_params = model_params.get('gb', {})
        xgb_params = model_params.get('xgb', {})
        ada_params = model_params.get('ada', {})
        weights = model_params.get('model_weights', [1, 1, 1, 1])


        models = [
            ('rf', RandomForestClassifier(
                class_weight='balanced',
                **rf_params
            )),
            ('gb', GradientBoostingClassifier(
                **gb_params
            )),
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
    
    def train_model(self, train_path, model_params=None, save_name=""):
        """Train an improved ensemble model with advanced parameter tuning"""
        # Preprocess data
        df = self.preprocess_data(train_path)
        
        # Prepare features
        X_train = df[self.feature_cols]
        y_train = df['season']
        
        
        # Create pipeline with robust scaler
        self.pipeline = Pipeline([
            ('scaler', RobustScaler()),
            ('classifier', self.create_ensemble())
        ])
        
        # Cross-validation
        cv_scores = cross_val_score(
            self.pipeline, X_train, y_train, 
            cv=5, scoring='accuracy'
        )
        print("\nCross-validation scores:", cv_scores)
        print("Mean CV score:", cv_scores.mean())
        print("CV score std:", cv_scores.std())
        
        # Fit the model
        self.pipeline.fit(X_train, y_train)
        
        # Save model
        joblib.dump(self.pipeline, f'runs/model_ensamble_{save_name}.joblib')

        return cv_scores.mean() #DEVOLVER ESTO SIRVE PARA COMPARAR ENTRE DISTINTOS HIPERPARAMS
        
    def eval_model(self, test_path, results_folder=None):
        df = pd.read_csv(test_path)
        X_test = df[self.feature_cols]
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

    def load_params_model(self, model_path):
        """Initialize the tester with the trained model"""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            self.pipeline = joblib.load(model_path)
            
        # Get feature names from the model pipeline
        if hasattr(self.pipeline, 'feature_names_in_'):
            self.feature_cols = self.pipeline.feature_names_in_
        else:
            self.feature_cols = self.pipeline.steps[0][1].feature_names_in_
        

    def predict_season(self, img_features):
        if self.pipeline is None:
            return {'error': 'Pipeline no cargado o existente. Asegúrese de inicializar el modelo correctamente.'}
        
        try:
            if isinstance(img_features, dict):
                img_features = pd.DataFrame([img_features]) 
            else:
                img_features = pd.DataFrame(img_features)
            img_features = img_features[self.feature_cols]
            prediction = self.pipeline.predict(img_features)
            probabilities = self.pipeline.predict_proba(img_features)

            season_probs = {
                season: prob
                for season, prob in zip(self.pipeline.classes_, probabilities[0])
            }

            return {
                'predicted_season': prediction[0],
                'confidence_scores': season_probs
            }

        except Exception as e:
            return {'error': f"Error al realizar la predicción: {str(e)}"}