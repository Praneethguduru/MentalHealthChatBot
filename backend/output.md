(.venv) (base) praneeth@Praneeths-MacBook-Air backend % python train_model.py  
============================================================
IMPROVED DEPRESSION DETECTION MODEL TRAINING
============================================================

Checking paths...
✓ COVAREP directory found: 189 CSV files
✓ Formant directory found: 189 CSV files
✓ Labels file found

STEP 1: Loading and processing audio features...
------------------------------------------------------------
======================================================================
STEP 1: Loading PHQ-8 labels...
======================================================================
Available columns: ['Participant_ID', 'PHQ_8NoInterest', 'PHQ_8Depressed', 'PHQ_8Sleep', 'PHQ_8Tired', 'PHQ_8Appetite', 'PHQ_8Failure', 'PHQ_8Concentrating', 'PHQ_8Moving', 'PHQ_8Total']

Loaded 219 participants with PHQ-8 labels
Depression cases (PHQ-8 >= 10): 65
Non-depression cases (PHQ-8 < 10): 154

======================================================================
STEP 2: Finding participants with audio features...
======================================================================

Audio file summary:
  Total COVAREP files: 189
  Total Formant files: 189
  Participants with BOTH: 189

Matching results:
  Participants with labels AND audio: 189
  Participants excluded (no audio): 30

======================================================================
STEP 3: Extracting acoustic features...
======================================================================
Processing participant 1/189: 300
Processing participant 10/189: 309
Processing participant 20/189: 319
Processing participant 30/189: 329
Processing participant 40/189: 339
Processing participant 50/189: 350
Processing participant 60/189: 360
Processing participant 70/189: 370
Processing participant 80/189: 380
Processing participant 90/189: 390
Processing participant 100/189: 402
Processing participant 110/189: 412
Processing participant 120/189: 422
Processing participant 130/189: 432
Processing participant 140/189: 442
Processing participant 150/189: 452
Processing participant 160/189: 463
Processing participant 170/189: 473
Processing participant 180/189: 483

Cleaning features...

======================================================================
FEATURE EXTRACTION COMPLETE - SUMMARY
======================================================================

📊 Data Pipeline Results:
  ├─ Total PHQ-8 labels: 219
  ├─ Participants with audio files: 189
  ├─ Successfully processed: 189
  └─ Skipped (errors/missing): 0

📈 Feature Statistics:
  ├─ Total features per participant: 27975
  ├─ COVAREP features: 23580
  └─ Formant features: 4395

🎯 Depression Distribution:
  ├─ Depression (PHQ-8 >= 10): 57 (30.2%)
  └─ No Depression (PHQ-8 < 10): 132 (69.8%)

📉 PHQ-8 Score Statistics:
  ├─ Mean: 6.75
  ├─ Median: 5.00
  ├─ Std Dev: 5.92
  └─ Range: 0 - 23
======================================================================


✓ Feature dataset saved to data/audio_features.csv

============================================================
STEP 2: Training improved model with feature selection...
------------------------------------------------------------

📊 Data Preparation:
  ├─ Original features: 27975
  ├─ Samples: 189
  └─ Target distribution: {0: 132, 1: 57}

🔍 Selecting top 100 features...
/Users/praneeth/Desktop/MentalHealthChatBot/.venv/lib/python3.10/site-packages/sklearn/feature_selection/_univariate_selection.py:110: UserWarning: Features [    2     7     8 ... 27962 27968 27973] are constant.
  warnings.warn("Features %s are constant." % constant_features_idx, UserWarning)
/Users/praneeth/Desktop/MentalHealthChatBot/.venv/lib/python3.10/site-packages/sklearn/feature_selection/_univariate_selection.py:111: RuntimeWarning: divide by zero encountered in divide
  f = msb / msw
/Users/praneeth/Desktop/MentalHealthChatBot/.venv/lib/python3.10/site-packages/sklearn/feature_selection/_univariate_selection.py:111: RuntimeWarning: invalid value encountered in divide
  f = msb / msw
  ✓ Reduced to 100 features

📈 Train/Test Split:
  Training: 151 samples
  Testing: 38 samples

🤖 Training Random Forest model...

⚖️  Optimal decision threshold: inf (default: 0.5)

============================================================
MODEL PERFORMANCE - Standard Threshold (0.5)
============================================================

Classification Report:
/Users/praneeth/Desktop/MentalHealthChatBot/.venv/lib/python3.10/site-packages/sklearn/metrics/_classification.py:1731: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", result.shape[0])
/Users/praneeth/Desktop/MentalHealthChatBot/.venv/lib/python3.10/site-packages/sklearn/metrics/_classification.py:1731: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", result.shape[0])
/Users/praneeth/Desktop/MentalHealthChatBot/.venv/lib/python3.10/site-packages/sklearn/metrics/_classification.py:1731: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", result.shape[0])
               precision    recall  f1-score   support

No Depression       0.71      1.00      0.83        27
   Depression       0.00      0.00      0.00        11

     accuracy                           0.71        38
    macro avg       0.36      0.50      0.42        38
 weighted avg       0.50      0.71      0.59        38


Confusion Matrix:
[[27  0]
 [11  0]]
True Negatives: 27 | False Positives: 0
False Negatives: 11 | True Positives: 0

============================================================
MODEL PERFORMANCE - Optimal Threshold (inf)
============================================================

Classification Report:
/Users/praneeth/Desktop/MentalHealthChatBot/.venv/lib/python3.10/site-packages/sklearn/metrics/_classification.py:1731: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", result.shape[0])
/Users/praneeth/Desktop/MentalHealthChatBot/.venv/lib/python3.10/site-packages/sklearn/metrics/_classification.py:1731: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", result.shape[0])
/Users/praneeth/Desktop/MentalHealthChatBot/.venv/lib/python3.10/site-packages/sklearn/metrics/_classification.py:1731: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", result.shape[0])
               precision    recall  f1-score   support

No Depression       0.71      1.00      0.83        27
   Depression       0.00      0.00      0.00        11

     accuracy                           0.71        38
    macro avg       0.36      0.50      0.42        38
 weighted avg       0.50      0.71      0.59        38


Confusion Matrix:
[[27  0]
 [11  0]]
True Negatives: 27 | False Positives: 0
False Negatives: 11 | True Positives: 0

📊 Metrics:
  ├─ ROC-AUC Score: 0.5000
  ├─ Accuracy (0.5): 0.7105
  └─ Accuracy (optimal): 0.7105

🔄 Cross-validation (5-fold):
  ROC-AUC: 0.5000 (+/- 0.0000)

🏆 Top 20 Most Important Features:
  1. covarep_0.080026_max: 0.0000
  2. covarep_0.45563_max: 0.0000
  3. covarep_-0.045552_max: 0.0000
  4. formant_4448.7_min: 0.0000
  5. covarep_0.01957_min: 0.0000
  6. covarep_0.61289_std: 0.0000
  7. covarep_0.12288_std: 0.0000
  8. covarep_-0.073821_mean: 0.0000
  9. covarep_-11.485_mean: 0.0000
  10. formant_471.05_mean: 0.0000
  11. covarep_-0.006243_max: 0.0000
  12. covarep_1.4869_std: 0.0000
  13. covarep_0.0082242_mean: 0.0000
  14. covarep_-0.049219_max: 0.0000
  15. covarep_0.027732_std: 0.0000
  16. covarep_-0.013219_min: 0.0000
  17. covarep_-0.10971_max: 0.0000
  18. covarep_-0.10971_median: 0.0000
  19. formant_1192.3_std: 0.0000
  20. covarep_-0.05797_mean: 0.0000

============================================================
STEP 3: Saving trained model...
------------------------------------------------------------

💾 Model saved to models/depression_model_improved.pkl

============================================================
✅ TRAINING COMPLETE!
============================================================
  ROC-AUC Score: 0.5000
  Test Accuracy (standard): 0.7105
  Test Accuracy (optimized): 0.7105
  Optimal Threshold: inf
  Model: models/depression_model_improved.pkl
============================================================




(.venv) (base) praneeth@Praneeths-MacBook-Air backend % python test_model.py
Loading trained model...
✓ Model loaded from models/depression_model_improved.pkl

============================================================
MODEL LOADED SUCCESSFULLY
============================================================
Selected features: 100
Decision threshold: inf

============================================================
TESTING MODEL ON SAMPLE PARTICIPANTS
============================================================

============================================================
PREDICTION FOR PARTICIPANT 300
============================================================

📊 Actual PHQ-8:
  Score: 2/24
  Depression: NO

🤖 Model Prediction:
  Depression Detected: NO
  Probability: 49.53%
  Risk Level: Moderate
  Confidence: 50.47%

✓ Prediction: CORRECT
============================================================


============================================================
PREDICTION FOR PARTICIPANT 301
============================================================

📊 Actual PHQ-8:
  Score: 3/24
  Depression: NO

🤖 Model Prediction:
  Depression Detected: NO
  Probability: 49.53%
  Risk Level: Moderate
  Confidence: 50.47%

✓ Prediction: CORRECT
============================================================


============================================================
PREDICTION FOR PARTICIPANT 302
============================================================

📊 Actual PHQ-8:
  Score: 4/24
  Depression: NO

🤖 Model Prediction:
  Depression Detected: NO
  Probability: 49.53%
  Risk Level: Moderate
  Confidence: 50.47%

✓ Prediction: CORRECT
============================================================


============================================================
PREDICTION FOR PARTICIPANT 303
============================================================

📊 Actual PHQ-8:
  Score: 0/24
  Depression: NO

🤖 Model Prediction:
  Depression Detected: NO
  Probability: 49.53%
  Risk Level: Moderate
  Confidence: 50.47%

✓ Prediction: CORRECT
============================================================


============================================================
PREDICTION FOR PARTICIPANT 304
============================================================

📊 Actual PHQ-8:
  Score: 6/24
  Depression: NO

🤖 Model Prediction:
  Depression Detected: NO
  Probability: 49.53%
  Risk Level: Moderate
  Confidence: 50.47%

✓ Prediction: CORRECT
============================================================