# test_model.py — OPTIMIZED
# Changes:
#   - Added argparse so you can pass participant IDs on the command line
#   - Handles missing participants gracefully without crashing
#   - Cleaner output formatting with colour indicators

import argparse
from improved_depression_model import ImprovedDepressionModel
from audio_feature_extraction import AudioFeatureProcessor
import pandas as pd
import os

MODEL_PATH   = "models/depression_model_improved.pkl"
COVAREP_DIR  = "data/covarep"
FORMANT_DIR  = "data/formant"
LABELS_FILE  = "data/phq8_labels.csv"


def load_model() -> ImprovedDepressionModel:
    print("Loading trained model …")
    model = ImprovedDepressionModel()
    model.load_model(MODEL_PATH)
    print(f"  Selected features : {len(model.selected_features)}")
    print(f"  Decision threshold: {model.threshold:.3f}\n")
    return model


def test_participant(participant_id: str, model: ImprovedDepressionModel,
                     processor: AudioFeatureProcessor, labels_df: pd.DataFrame):
    print("=" * 60)
    print(f"PARTICIPANT {participant_id}")
    print("=" * 60)

    cov = processor.load_covarep_features(participant_id)
    fmt = processor.load_formant_features(participant_id)

    if cov is None or fmt is None:
        print(f"  ⚠️  No audio data found for participant {participant_id}\n")
        return

    features = {**cov, **fmt}

    try:
        result = model.predict(features)
    except Exception as e:
        print(f"  ❌ Prediction failed: {e}\n")
        return

    actual_row = labels_df[labels_df["Participant_ID"] == int(participant_id)]
    if not actual_row.empty:
        actual_score = actual_row["PHQ_8Total"].values[0]
        actual_dep = actual_score >= 10
        print(f"  Actual PHQ-8 score : {actual_score}/24")
        print(f"  Actual depression  : {'YES' if actual_dep else 'NO'}")
        print()

    print(f"  Predicted depression : {'YES' if result['depression_detected'] else 'NO'}")
    print(f"  Probability          : {result['probability']:.2%}")
    print(f"  Risk level           : {result['risk_level']}")
    print(f"  Confidence           : {result['confidence']:.2%}")
    print(f"  Threshold used       : {result['threshold_used']:.3f}")

    if not actual_row.empty:
        correct = result["depression_detected"] == actual_dep
        print(f"\n  Result: {'✅ CORRECT' if correct else '❌ INCORRECT'}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Test depression model on specific participants")
    parser.add_argument(
        "participants",
        nargs="*",
        default=["300", "301", "302", "303", "304"],
        help="Participant IDs to test (default: 300 301 302 303 304)",
    )
    args = parser.parse_args()

    model = load_model()

    processor = AudioFeatureProcessor(COVAREP_DIR, FORMANT_DIR, LABELS_FILE)

    if not os.path.exists(LABELS_FILE):
        print(f"  ⚠️  Labels file not found: {LABELS_FILE}")
        labels_df = pd.DataFrame(columns=["Participant_ID", "PHQ_8Total"])
    else:
        labels_df = pd.read_csv(LABELS_FILE)

    print(f"\nTesting {len(args.participants)} participant(s): {args.participants}\n")
    for pid in args.participants:
        test_participant(pid, model, processor, labels_df)


if __name__ == "__main__":
    main()