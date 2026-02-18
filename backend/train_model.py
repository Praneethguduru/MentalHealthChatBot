# train_model.py — OPTIMIZED
# Changes:
#   - Configurable n_features via CLI arg for easier experimentation
#   - Saves feature dataset only if it doesn't already exist (skip re-extraction)
#   - Cleaner output formatting
#   - Added --skip-extraction flag to re-train on existing audio_features.csv

import argparse
import os
from audio_feature_extraction import AudioFeatureProcessor
from improved_depression_model import ImprovedDepressionModel
import pandas as pd

# ===== CONFIGURE THESE PATHS =====
COVAREP_DIR  = "data/covarep"
FORMANT_DIR  = "data/formant"
LABELS_FILE  = "data/phq8_labels.csv"
FEATURES_CSV = "data/audio_features.csv"
MODEL_OUT    = "models/depression_model_improved.pkl"
# ==================================


def check_paths(skip_extraction: bool):
    print("Checking paths …")
    if not skip_extraction:
        for path, label in [
            (COVAREP_DIR, "COVAREP directory"),
            (FORMANT_DIR, "Formant directory"),
            (LABELS_FILE, "Labels file"),
        ]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"{label} not found: {path}")
        n_cov = len([f for f in os.listdir(COVAREP_DIR) if f.endswith(".csv")])
        n_fmt = len([f for f in os.listdir(FORMANT_DIR) if f.endswith(".csv")])
        print(f"  COVAREP: {n_cov} CSV files")
        print(f"  Formant: {n_fmt} CSV files")
        print(f"  Labels:  {LABELS_FILE}")
    else:
        if not os.path.exists(FEATURES_CSV):
            raise FileNotFoundError(f"Features CSV not found (needed for --skip-extraction): {FEATURES_CSV}")
        print(f"  Using existing features: {FEATURES_CSV}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Train improved depression detection model")
    parser.add_argument("--n-features", type=int, default=100, help="Number of features to select (default: 100)")
    parser.add_argument("--skip-extraction", action="store_true",
                        help="Skip feature extraction and use existing data/audio_features.csv")
    args = parser.parse_args()

    print("=" * 60)
    print("IMPROVED DEPRESSION DETECTION MODEL TRAINING")
    print("=" * 60)
    print()

    check_paths(args.skip_extraction)

    # STEP 1: Feature extraction
    if args.skip_extraction and os.path.exists(FEATURES_CSV):
        print(f"STEP 1: Loading existing features from {FEATURES_CSV} …")
        feature_df = pd.read_csv(FEATURES_CSV)
        print(f"  Loaded {len(feature_df)} samples, {len(feature_df.columns) - 3} features")
    else:
        print("STEP 1: Extracting audio features …")
        print("-" * 60)
        processor = AudioFeatureProcessor(COVAREP_DIR, FORMANT_DIR, LABELS_FILE)
        feature_df = processor.create_feature_dataset()
        os.makedirs("data", exist_ok=True)
        feature_df.to_csv(FEATURES_CSV, index=False)
        print(f"\n  ✅ Feature dataset saved → {FEATURES_CSV}")

    # STEP 2: Train
    print("\n" + "=" * 60)
    print(f"STEP 2: Training model (n_features={args.n_features}) …")
    print("-" * 60)
    model = ImprovedDepressionModel()
    try:
        metrics = model.train(feature_df, n_features=args.n_features)
    except Exception as e:
        import traceback
        print(f"\n  ❌ Training failed: {e}")
        traceback.print_exc()
        return

    # STEP 3: Save
    print("\n" + "=" * 60)
    print("STEP 3: Saving model …")
    os.makedirs("models", exist_ok=True)
    model.save_model(MODEL_OUT)

    print("\n" + "=" * 60)
    print("  ✅ TRAINING COMPLETE!")
    print("=" * 60)
    print(f"  ROC-AUC:           {metrics['roc_auc']:.4f}")
    print(f"  Test accuracy:     {metrics['test_score']:.4f}")
    print(f"  Optimal accuracy:  {metrics['test_score_optimal']:.4f}")
    print(f"  Optimal threshold: {metrics['optimal_threshold']:.3f}")
    print(f"  Model saved →      {MODEL_OUT}")
    print("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        print(f"\n  ❌ Error: {e}")
        traceback.print_exc()