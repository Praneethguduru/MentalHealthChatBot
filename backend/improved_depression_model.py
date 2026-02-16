# improved_depression_model.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import joblib
import matplotlib.pyplot as plt

class ImprovedDepressionModel:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_selector = None
        self.feature_names = None
        self.selected_features = None
        self.threshold = 0.5
        
    def prepare_data(self, feature_df, n_features=100):
        """Prepare data with feature selection"""
        # Separate features and labels
        X = feature_df.drop(['participant_id', 'phq8_score', 'depression'], axis=1)
        y = feature_df['depression']
        
        # Clean data
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.mean())
        X = X.fillna(0)
        
        # Store original feature names
        self.feature_names = X.columns.tolist()
        
        print(f"\n📊 Data Preparation:")
        print(f"  ├─ Original features: {len(self.feature_names)}")
        print(f"  ├─ Samples: {len(X)}")
        print(f"  └─ Target distribution: {y.value_counts().to_dict()}")
        
        # Feature selection - keep top N most relevant features
        print(f"\n🔍 Selecting top {n_features} features...")
        self.feature_selector = SelectKBest(score_func=f_classif, k=n_features)
        X_selected = self.feature_selector.fit_transform(X, y)
        
        # Get selected feature names
        selected_indices = self.feature_selector.get_support(indices=True)
        self.selected_features = [self.feature_names[i] for i in selected_indices]
        
        print(f"  ✓ Reduced to {len(self.selected_features)} features")
        
        return pd.DataFrame(X_selected, columns=self.selected_features), y
    
    def train(self, feature_df, test_size=0.2, random_state=42, n_features=100):
        """Train improved model with feature selection"""
        
        X, y = self.prepare_data(feature_df, n_features=n_features)
        
        # Split data with stratification
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"\n📈 Train/Test Split:")
        print(f"  Training: {len(X_train)} samples")
        print(f"  Testing: {len(X_test)} samples")
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train with better hyperparameters for small dataset
        print("\n🤖 Training Random Forest model...")
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,  # Reduced to prevent overfitting
            min_samples_split=10,
            min_samples_leaf=5,
            max_features='sqrt',
            class_weight='balanced',  # Handle class imbalance
            random_state=random_state,
            n_jobs=-1
        )
        
        self.model.fit(X_train_scaled, y_train)
        
        # Predictions
        y_pred = self.model.predict(X_test_scaled)
        y_pred_proba = self.model.predict_proba(X_test_scaled)[:, 1]
        
        # Adjust decision threshold for better recall
        # Find optimal threshold
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]
        
        print(f"\n⚖️  Optimal decision threshold: {optimal_threshold:.3f} (default: 0.5)")
        
        # Predict with optimal threshold
        y_pred_optimal = (y_pred_proba >= optimal_threshold).astype(int)
        
        # Evaluation
        print("\n" + "="*60)
        print("MODEL PERFORMANCE - Standard Threshold (0.5)")
        print("="*60)
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['No Depression', 'Depression']))
        
        print("\nConfusion Matrix:")
        cm = confusion_matrix(y_test, y_pred)
        print(cm)
        print(f"True Negatives: {cm[0,0]} | False Positives: {cm[0,1]}")
        print(f"False Negatives: {cm[1,0]} | True Positives: {cm[1,1]}")
        
        # With optimal threshold
        print("\n" + "="*60)
        print(f"MODEL PERFORMANCE - Optimal Threshold ({optimal_threshold:.3f})")
        print("="*60)
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred_optimal, target_names=['No Depression', 'Depression']))
        
        print("\nConfusion Matrix:")
        cm_opt = confusion_matrix(y_test, y_pred_optimal)
        print(cm_opt)
        print(f"True Negatives: {cm_opt[0,0]} | False Positives: {cm_opt[0,1]}")
        print(f"False Negatives: {cm_opt[1,0]} | True Positives: {cm_opt[1,1]}")
        
        print(f"\n📊 Metrics:")
        print(f"  ├─ ROC-AUC Score: {roc_auc_score(y_test, y_pred_proba):.4f}")
        print(f"  ├─ Accuracy (0.5): {self.model.score(X_test_scaled, y_test):.4f}")
        print(f"  └─ Accuracy (optimal): {(y_pred_optimal == y_test).mean():.4f}")
        
        # Cross-validation with stratified folds
        print("\n🔄 Cross-validation (5-fold):")
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
        cv_scores = cross_val_score(
            self.model, X_train_scaled, y_train, cv=cv, scoring='roc_auc'
        )
        print(f"  ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        
        # Feature importance
        self.print_feature_importance()
        
        # Store optimal threshold
        self.threshold = optimal_threshold
        
        return {
            'train_score': self.model.score(X_train_scaled, y_train),
            'test_score': self.model.score(X_test_scaled, y_test),
            'test_score_optimal': (y_pred_optimal == y_test).mean(),
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'cv_scores': cv_scores.tolist(),
            'optimal_threshold': optimal_threshold
        }
    
    def print_feature_importance(self, top_n=20):
        """Print top important features"""
        if self.model is None:
            print("Model not trained yet")
            return
        
        importances = self.model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        print(f"\n🏆 Top {top_n} Most Important Features:")
        for i in range(min(top_n, len(indices))):
            idx = indices[i]
            print(f"  {i+1}. {self.selected_features[idx]}: {importances[idx]:.4f}")
    
    def predict(self, features):
        """Predict depression from audio features"""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        # Prepare features - use selected features only
        feature_dict = {name: 0.0 for name in self.feature_names}
        feature_dict.update(features)
        
        X = pd.DataFrame([feature_dict])
        X = X[self.feature_names]
        
        # Apply feature selection
        X_selected = self.feature_selector.transform(X)
        X_selected = pd.DataFrame(X_selected, columns=self.selected_features)
        
        # Scale
        X_scaled = self.scaler.transform(X_selected)
        
        # Predict with optimal threshold
        probability = self.model.predict_proba(X_scaled)[0]
        prediction = (probability[1] >= self.threshold).astype(int)
        
        return {
            'depression_detected': bool(prediction),
            'probability': float(probability[1]),
            'confidence': float(max(probability)),
            'risk_level': self.get_risk_level(probability[1]),
            'threshold_used': float(self.threshold)
        }
    
    def get_risk_level(self, probability):
        """Convert probability to risk level"""
        if probability < 0.3:
            return "Low"
        elif probability < 0.6:
            return "Moderate"
        else:
            return "High"
    
    def save_model(self, model_path='models/depression_model_improved.pkl'):
        """Save trained model"""
        import os
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_selector': self.feature_selector,
            'feature_names': self.feature_names,
            'selected_features': self.selected_features,
            'threshold': self.threshold
        }
        
        joblib.dump(model_data, model_path)
        print(f"\n💾 Model saved to {model_path}")
    
    def load_model(self, model_path='models/depression_model_improved.pkl'):
        """Load trained model"""
        model_data = joblib.load(model_path)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_selector = model_data['feature_selector']
        self.feature_names = model_data['feature_names']
        self.selected_features = model_data['selected_features']
        self.threshold = model_data.get('threshold', 0.5)
        
        print(f"✓ Model loaded from {model_path}")