# depression_model.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib
import json

class DepressionDetectionModel:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.threshold = 0.5
        
    def prepare_data(self, feature_df):
        """Prepare data for training with robust cleaning"""
        # Separate features and labels
        X = feature_df.drop(['participant_id', 'phq8_score', 'depression'], axis=1)
        y = feature_df['depression']
        
        # Handle infinity values
        X = X.replace([np.inf, -np.inf], np.nan)
        
        # Handle missing values by filling with column mean
        X = X.fillna(X.mean())
        
        # If any columns are still all NaN, fill with 0
        X = X.fillna(0)
        
        # Verify no infinite values remain
        if not np.isfinite(X.values).all():
            print("⚠️  Warning: Some features still contain non-finite values")
            print("Replacing remaining non-finite values with 0...")
            X = X.replace([np.inf, -np.inf, np.nan], 0)
        
        # Store feature names
        self.feature_names = X.columns.tolist()
        
        print(f"\n✓ Prepared {len(X)} samples with {len(self.feature_names)} features")
        
        return X, y
    
    def train(self, feature_df, test_size=0.2, random_state=42):
        """Train the depression detection model"""
        X, y = self.prepare_data(feature_df)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train Random Forest
        print("Training Random Forest model...")
        self.model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced',
            random_state=random_state,
            n_jobs=-1
        )
        
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test_scaled)
        y_pred_proba = self.model.predict_proba(X_test_scaled)[:, 1]
        
        print("\n=== Model Performance ===")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['No Depression', 'Depression']))
        
        print("\nConfusion Matrix:")
        cm = confusion_matrix(y_test, y_pred)
        print(cm)
        print(f"\nTrue Negatives: {cm[0,0]} | False Positives: {cm[0,1]}")
        print(f"False Negatives: {cm[1,0]} | True Positives: {cm[1,1]}")
        
        print(f"\nROC-AUC Score: {roc_auc_score(y_test, y_pred_proba):.4f}")
        
        # Cross-validation
        print("\nCross-validation scores:")
        cv_scores = cross_val_score(
            self.model, X_train_scaled, y_train, cv=5, scoring='roc_auc'
        )
        print(f"CV ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        
        # Feature importance
        self.print_feature_importance()
        
        return {
            'train_score': self.model.score(X_train_scaled, y_train),
            'test_score': self.model.score(X_test_scaled, y_test),
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'cv_scores': cv_scores.tolist()
        }
    
    def print_feature_importance(self, top_n=20):
        """Print top important features"""
        if self.model is None:
            print("Model not trained yet")
            return
        
        importances = self.model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        print(f"\nTop {top_n} Most Important Features:")
        for i in range(min(top_n, len(indices))):
            idx = indices[i]
            print(f"{i+1}. {self.feature_names[idx]}: {importances[idx]:.4f}")
    
    def predict(self, features):
        """Predict depression from audio features"""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        # Prepare features
        feature_dict = {name: 0.0 for name in self.feature_names}
        feature_dict.update(features)
        
        X = pd.DataFrame([feature_dict])
        X = X[self.feature_names]  # Ensure correct order
        
        # Handle missing values
        X = X.fillna(0)
        
        # Scale
        X_scaled = self.scaler.transform(X)
        
        # Predict
        prediction = self.model.predict(X_scaled)[0]
        probability = self.model.predict_proba(X_scaled)[0]
        
        return {
            'depression_detected': bool(prediction),
            'probability': float(probability[1]),
            'confidence': float(max(probability)),
            'risk_level': self.get_risk_level(probability[1])
        }
    
    def get_risk_level(self, probability):
        """Convert probability to risk level"""
        if probability < 0.3:
            return "Low"
        elif probability < 0.6:
            return "Moderate"
        else:
            return "High"
    
    def save_model(self, model_path='models/depression_model.pkl'):
        """Save trained model"""
        import os
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'threshold': self.threshold
        }
        
        joblib.dump(model_data, model_path)
        print(f"Model saved to {model_path}")
    
    def load_model(self, model_path='models/depression_model.pkl'):
        """Load trained model"""
        model_data = joblib.load(model_path)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        self.threshold = model_data.get('threshold', 0.5)
        
        print(f"Model loaded from {model_path}")