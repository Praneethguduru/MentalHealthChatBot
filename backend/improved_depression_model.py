# improved_depression_model.py — OPTIMIZED
# Changes:
#   - predict() now builds DataFrame only from selected_features (skips full feature rebuild)
#   - Added input validation in predict() to avoid crashes on empty feature dicts
#   - Removed matplotlib import (was unused at runtime — only used during training)
#   - Training unchanged (offline step, not a bottleneck)

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import joblib


class ImprovedDepressionModel:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_selector = None
        self.feature_names = None       # all original features
        self.selected_features = None   # subset kept after selection
        self.threshold = 0.5

    # ── Data preparation (training only) ────────────────────────────────────────
    def prepare_data(self, feature_df, n_features: int = 100):
        X = feature_df.drop(["participant_id", "phq8_score", "depression"], axis=1)
        y = feature_df["depression"]

        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.mean()).fillna(0)

        self.feature_names = X.columns.tolist()
        print(f"\n  Data: {len(X)} samples, {len(self.feature_names)} original features")
        print(f"  Target distribution: {y.value_counts().to_dict()}")

        print(f"  Selecting top {n_features} features …")
        self.feature_selector = SelectKBest(score_func=f_classif, k=n_features)
        X_selected = self.feature_selector.fit_transform(X, y)
        selected_idx = self.feature_selector.get_support(indices=True)
        self.selected_features = [self.feature_names[i] for i in selected_idx]
        print(f"  Reduced to {len(self.selected_features)} features")

        return pd.DataFrame(X_selected, columns=self.selected_features), y

    # ── Training (offline step) ─────────────────────────────────────────────────
    def train(self, feature_df, test_size: float = 0.2, random_state: int = 42, n_features: int = 100):
        X, y = self.prepare_data(feature_df, n_features=n_features)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        print(f"\n  Train: {len(X_train)} | Test: {len(X_test)}")

        X_train_s = self.scaler.fit_transform(X_train)
        X_test_s = self.scaler.transform(X_test)

        print("  Training Random Forest …")
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            min_samples_split=10,
            min_samples_leaf=5,
            max_features="sqrt",
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        )
        self.model.fit(X_train_s, y_train)

        y_pred = self.model.predict(X_test_s)
        y_proba = self.model.predict_proba(X_test_s)[:, 1]

        fpr, tpr, thresholds = roc_curve(y_test, y_proba)
        optimal_idx = np.argmax(tpr - fpr)
        self.threshold = float(thresholds[optimal_idx])

        y_pred_opt = (y_proba >= self.threshold).astype(int)

        print("\n" + "=" * 60)
        print("MODEL PERFORMANCE — Standard Threshold (0.5)")
        print(classification_report(y_test, y_pred, target_names=["No Depression", "Depression"]))
        cm = confusion_matrix(y_test, y_pred)
        print(f"TN:{cm[0,0]}  FP:{cm[0,1]}  FN:{cm[1,0]}  TP:{cm[1,1]}")

        roc = roc_auc_score(y_test, y_proba)
        print(f"\nROC-AUC: {roc:.4f}")
        print(f"Optimal threshold: {self.threshold:.3f}")

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
        cv_scores = cross_val_score(self.model, X_train_s, y_train, cv=cv, scoring="roc_auc")
        print(f"CV ROC-AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

        self.print_feature_importance()

        return {
            "train_score": self.model.score(X_train_s, y_train),
            "test_score": self.model.score(X_test_s, y_test),
            "test_score_optimal": float((y_pred_opt == y_test).mean()),
            "roc_auc": roc,
            "cv_scores": cv_scores.tolist(),
            "optimal_threshold": self.threshold,
        }

    def print_feature_importance(self, top_n: int = 20):
        if not self.model:
            return
        importances = self.model.feature_importances_
        indices = np.argsort(importances)[::-1]
        print(f"\n  Top {top_n} features:")
        for i in range(min(top_n, len(indices))):
            idx = indices[i]
            print(f"    {i+1}. {self.selected_features[idx]}: {importances[idx]:.4f}")

    # ── Prediction (runtime — OPTIMIZED) ────────────────────────────────────────
    def predict(self, features: dict) -> dict:
        """
        Predict depression from a feature dict.

        OPTIMIZED: builds a DataFrame directly from selected_features
        instead of the full feature_names set, cutting DataFrame construction
        time significantly when there are thousands of original features.
        """
        if not self.model:
            raise ValueError("Model not trained/loaded yet")
        if not features:
            raise ValueError("Empty feature dict")

        # Build row using only the features the model was trained on
        row = {f: features.get(f, 0.0) for f in self.selected_features}
        X = pd.DataFrame([row], columns=self.selected_features)
        X = X.fillna(0.0)

        X_scaled = self.scaler.transform(X)
        proba = self.model.predict_proba(X_scaled)[0]
        prediction = int(proba[1] >= self.threshold)

        return {
            "depression_detected": bool(prediction),
            "probability": float(proba[1]),
            "confidence": float(max(proba)),
            "risk_level": self._risk_level(proba[1]),
            "threshold_used": float(self.threshold),
        }

    @staticmethod
    def _risk_level(probability: float) -> str:
        if probability < 0.3:
            return "Low"
        elif probability < 0.6:
            return "Moderate"
        return "High"

    # ── Serialisation ────────────────────────────────────────────────────────────
    def save_model(self, model_path: str = "models/depression_model_improved.pkl"):
        import os
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump({
            "model": self.model,
            "scaler": self.scaler,
            "feature_selector": self.feature_selector,
            "feature_names": self.feature_names,
            "selected_features": self.selected_features,
            "threshold": self.threshold,
        }, model_path)
        print(f"\n  ✅ Model saved → {model_path}")

    def load_model(self, model_path: str = "models/depression_model_improved.pkl"):
        data = joblib.load(model_path)
        self.model = data["model"]
        self.scaler = data["scaler"]
        self.feature_selector = data["feature_selector"]
        self.feature_names = data["feature_names"]
        self.selected_features = data["selected_features"]
        self.threshold = data.get("threshold", 0.5)
        print(f"  ✅ Model loaded ← {model_path}")