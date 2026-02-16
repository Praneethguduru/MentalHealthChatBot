# test_model.py
from improved_depression_model import ImprovedDepressionModel
from audio_feature_extraction import AudioFeatureProcessor
import pandas as pd

# Load the trained model
print("Loading trained model...")
model = ImprovedDepressionModel()
model.load_model('models/depression_model_improved.pkl')

print("\n" + "="*60)
print("MODEL LOADED SUCCESSFULLY")
print("="*60)
print(f"Selected features: {len(model.selected_features)}")
print(f"Decision threshold: {model.threshold:.3f}")

# Test on a specific participant
def test_participant(participant_id):
    """Test depression detection on a specific participant"""
    
    processor = AudioFeatureProcessor(
        "data/covarep",
        "data/formant",
        "data/PHQ8_Labels.csv"
    )
    
    # Load features for this participant
    covarep_features = processor.load_covarep_features(str(participant_id))
    formant_features = processor.load_formant_features(str(participant_id))
    
    if covarep_features is None or formant_features is None:
        print(f"❌ No audio data found for participant {participant_id}")
        return
    
    # Combine features
    all_features = {}
    all_features.update(covarep_features)
    all_features.update(formant_features)
    
    # Predict
    result = model.predict(all_features)
    
    # Load actual PHQ-8 score
    labels_df = pd.read_csv("data/phq8_labels.csv")
    actual_row = labels_df[labels_df['Participant_ID'] == int(participant_id)]
    
    print("\n" + "="*60)
    print(f"PREDICTION FOR PARTICIPANT {participant_id}")
    print("="*60)
    
    if len(actual_row) > 0:
        actual_score = actual_row['PHQ_8Total'].values[0]
        actual_depression = actual_score >= 10
        print(f"\n📊 Actual PHQ-8:")
        print(f"  Score: {actual_score}/24")
        print(f"  Depression: {'YES' if actual_depression else 'NO'}")
    
    print(f"\n🤖 Model Prediction:")
    print(f"  Depression Detected: {'YES' if result['depression_detected'] else 'NO'}")
    print(f"  Probability: {result['probability']:.2%}")
    print(f"  Risk Level: {result['risk_level']}")
    print(f"  Confidence: {result['confidence']:.2%}")
    
    if len(actual_row) > 0:
        correct = (result['depression_detected'] == actual_depression)
        print(f"\n✓ Prediction: {'CORRECT' if correct else 'INCORRECT'}")
    
    print("="*60)

# Example: Test a few participants
print("\n" + "="*60)
print("TESTING MODEL ON SAMPLE PARTICIPANTS")
print("="*60)

# Test 5 random participants
test_participants = ['300', '301', '302', '303', '304']

for pid in test_participants:
    test_participant(pid)
    print()