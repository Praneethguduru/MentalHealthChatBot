# train_model.py
from audio_feature_extraction import AudioFeatureProcessor
from improved_depression_model import ImprovedDepressionModel  # Changed
import pandas as pd
import os

# ===== CONFIGURE THESE PATHS =====
COVAREP_DIR = "data/covarep"
FORMANT_DIR = "data/formant"
LABELS_FILE = "data/phq8_labels.csv"
# ==================================

def check_paths():
    """Verify that all paths exist"""
    print("Checking paths...")
    
    if not os.path.exists(COVAREP_DIR):
        raise FileNotFoundError(f"COVAREP directory not found: {COVAREP_DIR}")
    
    if not os.path.exists(FORMANT_DIR):
        raise FileNotFoundError(f"Formant directory not found: {FORMANT_DIR}")
    
    if not os.path.exists(LABELS_FILE):
        raise FileNotFoundError(f"Labels file not found: {LABELS_FILE}")
    
    # Count files
    covarep_count = len([f for f in os.listdir(COVAREP_DIR) if f.endswith('.csv')])
    formant_count = len([f for f in os.listdir(FORMANT_DIR) if f.endswith('.csv')])
    
    print(f"✓ COVAREP directory found: {covarep_count} CSV files")
    print(f"✓ Formant directory found: {formant_count} CSV files")
    print(f"✓ Labels file found")
    print()

def main():
    print("="*60)
    print("IMPROVED DEPRESSION DETECTION MODEL TRAINING")
    print("="*60)
    print()
    
    # Check paths
    check_paths()
    
    # Step 1: Extract and combine features
    print("STEP 1: Loading and processing audio features...")
    print("-"*60)
    processor = AudioFeatureProcessor(COVAREP_DIR, FORMANT_DIR, LABELS_FILE)
    feature_df = processor.create_feature_dataset()
    
    # Save feature dataset
    os.makedirs('data', exist_ok=True)
    feature_df.to_csv('data/audio_features.csv', index=False)
    print(f"\n✓ Feature dataset saved to data/audio_features.csv")
    
    # Step 2: Train improved model
    print("\n" + "="*60)
    print("STEP 2: Training improved model with feature selection...")
    print("-"*60)
    
    model = ImprovedDepressionModel()
    
    try:
        # Train with 100 best features (reduced from 27,975)
        metrics = model.train(feature_df, n_features=100)
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Step 3: Save model
    print("\n" + "="*60)
    print("STEP 3: Saving trained model...")
    print("-"*60)
    os.makedirs('models', exist_ok=True)
    model.save_model('models/depression_model_improved.pkl')
    
    print("\n" + "="*60)
    print("✅ TRAINING COMPLETE!")
    print("="*60)
    print(f"  ROC-AUC Score: {metrics['roc_auc']:.4f}")
    print(f"  Test Accuracy (standard): {metrics['test_score']:.4f}")
    print(f"  Test Accuracy (optimized): {metrics['test_score_optimal']:.4f}")
    print(f"  Optimal Threshold: {metrics['optimal_threshold']:.3f}")
    print(f"  Model: models/depression_model_improved.pkl")
    print("="*60)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()