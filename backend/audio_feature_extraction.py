# audio_feature_extraction.py
import pandas as pd
import numpy as np
from pathlib import Path
import os

class AudioFeatureProcessor:
    def __init__(self, covarep_dir, formant_dir, labels_file):
        """
        Initialize the audio feature processor.
        
        Args:
            covarep_dir: Directory containing COVAREP CSV files
            formant_dir: Directory containing Formant CSV files
            labels_file: Path to PHQ-8 labels CSV
        """
        self.covarep_dir = Path(covarep_dir)
        self.formant_dir = Path(formant_dir)
        self.labels_file = labels_file
        
    def load_labels(self):
        """Load PHQ-8 labels - Updated for your exact column format"""
        df = pd.read_csv(self.labels_file)
        
        print(f"Available columns: {df.columns.tolist()}")
        
        # Your column is PHQ_8Total, not PHQ8_Score
        if 'PHQ_8Total' not in df.columns:
            raise ValueError(f"PHQ_8Total column not found. Available: {df.columns.tolist()}")
        
        # Create depression label (PHQ-8 >= 10)
        df['depression'] = (df['PHQ_8Total'] >= 10).astype(int)
        
        # Rename for consistency in the rest of the code
        result = pd.DataFrame({
            'Participant_ID': df['Participant_ID'],
            'PHQ8_Score': df['PHQ_8Total'],  # Map PHQ_8Total to PHQ8_Score
            'depression': df['depression']
        })
        
        print(f"\nLoaded {len(result)} participants with PHQ-8 labels")
        print(f"Depression cases (PHQ-8 >= 10): {result['depression'].sum()}")
        print(f"Non-depression cases (PHQ-8 < 10): {(~result['depression'].astype(bool)).sum()}")
        
        return result
    
    def find_available_participants(self):
        """Find participants who have BOTH COVAREP and Formant data"""
        # Get all COVAREP files
        covarep_files = list(self.covarep_dir.glob("*.csv"))
        covarep_ids = set()
        
        for file in covarep_files:
            # Extract participant ID from filename
            # Try different patterns: XXX_COVAREP.csv or XXX.csv
            filename = file.stem
            if '_COVAREP' in filename:
                participant_id = filename.replace("_COVAREP", "")
            elif '_covarep' in filename:
                participant_id = filename.replace("_covarep", "")
            else:
                participant_id = filename
            
            covarep_ids.add(str(participant_id))
        
        # Get all Formant files
        formant_files = list(self.formant_dir.glob("*.csv"))
        formant_ids = set()
        
        for file in formant_files:
            # Extract participant ID from filename
            filename = file.stem
            if '_FORMANT' in filename:
                participant_id = filename.replace("_FORMANT", "")
            elif '_formant' in filename:
                participant_id = filename.replace("_formant", "")
            else:
                participant_id = filename
            
            formant_ids.add(str(participant_id))
        
        # Find intersection (participants with BOTH)
        available_ids = covarep_ids.intersection(formant_ids)
        
        print(f"\nAudio file summary:")
        print(f"  Total COVAREP files: {len(covarep_ids)}")
        print(f"  Total Formant files: {len(formant_ids)}")
        print(f"  Participants with BOTH: {len(available_ids)}")
        
        return available_ids
    
    def load_covarep_features(self, participant_id):
        """Load COVAREP features for a participant with safety checks"""
        # Try different possible file naming patterns
        possible_files = [
            self.covarep_dir / f"{participant_id}_COVAREP.csv",
            self.covarep_dir / f"{participant_id}.csv",
            self.covarep_dir / f"{participant_id}_covarep.csv",
        ]
        
        covarep_file = None
        for file_path in possible_files:
            if file_path.exists():
                covarep_file = file_path
                break
        
        if covarep_file is None:
            return None
        
        try:
            df = pd.read_csv(covarep_file)
            
            # Remove non-numeric columns and compute statistics
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            if len(numeric_cols) == 0:
                return None
            
            # Compute statistical features with safety checks
            features = {}
            for col in numeric_cols:
                # Replace infinity with NaN and drop NaN values
                col_data = df[col].replace([np.inf, -np.inf], np.nan).dropna()
                
                if len(col_data) > 0:
                    mean_val = col_data.mean()
                    std_val = col_data.std()
                    min_val = col_data.min()
                    max_val = col_data.max()
                    median_val = col_data.median()
                    
                    # Only add if values are finite
                    if np.isfinite(mean_val):
                        features[f'covarep_{col}_mean'] = mean_val
                    if np.isfinite(std_val):
                        features[f'covarep_{col}_std'] = std_val
                    if np.isfinite(min_val):
                        features[f'covarep_{col}_min'] = min_val
                    if np.isfinite(max_val):
                        features[f'covarep_{col}_max'] = max_val
                    if np.isfinite(median_val):
                        features[f'covarep_{col}_median'] = median_val
            
            return features
        except Exception as e:
            print(f"  Error loading COVAREP for {participant_id}: {e}")
            return None
    
    def load_formant_features(self, participant_id):
        """Load Formant features for a participant with safety checks"""
        # Try different possible file naming patterns
        possible_files = [
            self.formant_dir / f"{participant_id}_FORMANT.csv",
            self.formant_dir / f"{participant_id}.csv",
            self.formant_dir / f"{participant_id}_formant.csv",
        ]
        
        formant_file = None
        for file_path in possible_files:
            if file_path.exists():
                formant_file = file_path
                break
        
        if formant_file is None:
            return None
        
        try:
            df = pd.read_csv(formant_file)
            
            # Remove non-numeric columns and compute statistics
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            if len(numeric_cols) == 0:
                return None
            
            # Compute statistical features with safety checks
            features = {}
            for col in numeric_cols:
                # Replace infinity with NaN and drop NaN values
                col_data = df[col].replace([np.inf, -np.inf], np.nan).dropna()
                
                if len(col_data) > 0:
                    mean_val = col_data.mean()
                    std_val = col_data.std()
                    min_val = col_data.min()
                    max_val = col_data.max()
                    median_val = col_data.median()
                    
                    # Only add if values are finite
                    if np.isfinite(mean_val):
                        features[f'formant_{col}_mean'] = mean_val
                    if np.isfinite(std_val):
                        features[f'formant_{col}_std'] = std_val
                    if np.isfinite(min_val):
                        features[f'formant_{col}_min'] = min_val
                    if np.isfinite(max_val):
                        features[f'formant_{col}_max'] = max_val
                    if np.isfinite(median_val):
                        features[f'formant_{col}_median'] = median_val
            
            return features
        except Exception as e:
            print(f"  Error loading Formant for {participant_id}: {e}")
            return None
    
    def create_feature_dataset(self):
        """Create combined feature dataset"""
        print("="*70)
        print("STEP 1: Loading PHQ-8 labels...")
        print("="*70)
        labels_df = self.load_labels()
        total_labels = len(labels_df)
        
        print("\n" + "="*70)
        print("STEP 2: Finding participants with audio features...")
        print("="*70)
        available_audio_ids = self.find_available_participants()
        
        # Filter labels to only include participants with audio
        labels_df['Participant_ID'] = labels_df['Participant_ID'].astype(str)
        available_labels = labels_df[labels_df['Participant_ID'].isin(available_audio_ids)]
        
        print(f"\nMatching results:")
        print(f"  Participants with labels AND audio: {len(available_labels)}")
        print(f"  Participants excluded (no audio): {total_labels - len(available_labels)}")
        
        if len(available_labels) == 0:
            raise ValueError("❌ No participants found with both PHQ-8 labels and audio features!")
        
        print("\n" + "="*70)
        print("STEP 3: Extracting acoustic features...")
        print("="*70)
        
        all_features = []
        skipped = []
        processed_count = 0
        
        for idx, row in available_labels.iterrows():
            participant_id = str(row['Participant_ID'])
            
            # Progress indicator
            processed_count += 1
            if processed_count % 10 == 0 or processed_count == 1:
                print(f"Processing participant {processed_count}/{len(available_labels)}: {participant_id}")
            
            # Load features
            covarep_features = self.load_covarep_features(participant_id)
            formant_features = self.load_formant_features(participant_id)
            
            # Skip if features are missing or invalid
            if covarep_features is None or formant_features is None:
                skipped.append(participant_id)
                continue
            
            if len(covarep_features) == 0 or len(formant_features) == 0:
                skipped.append(participant_id)
                continue
            
            # Combine all features
            combined_features = {
                'participant_id': participant_id,
                'phq8_score': row['PHQ8_Score'],
                'depression': row['depression']
            }
            combined_features.update(covarep_features)
            combined_features.update(formant_features)
            
            all_features.append(combined_features)
        
        # Create DataFrame
        feature_df = pd.DataFrame(all_features)
        
        # Clean infinity and NaN values
        print("\nCleaning features...")
        feature_cols = [col for col in feature_df.columns if col not in ['participant_id', 'phq8_score', 'depression']]
        
        # Replace infinity with NaN
        feature_df[feature_cols] = feature_df[feature_cols].replace([np.inf, -np.inf], np.nan)
        
        # Fill NaN with column mean
        feature_df[feature_cols] = feature_df[feature_cols].fillna(feature_df[feature_cols].mean())
        
        # Fill any remaining NaN with 0
        feature_df[feature_cols] = feature_df[feature_cols].fillna(0)
        
        # Print comprehensive summary
        print("\n" + "="*70)
        print("FEATURE EXTRACTION COMPLETE - SUMMARY")
        print("="*70)
        print(f"\n📊 Data Pipeline Results:")
        print(f"  ├─ Total PHQ-8 labels: {total_labels}")
        print(f"  ├─ Participants with audio files: {len(available_audio_ids)}")
        print(f"  ├─ Successfully processed: {len(feature_df)}")
        print(f"  └─ Skipped (errors/missing): {len(skipped)}")
        
        print(f"\n📈 Feature Statistics:")
        print(f"  ├─ Total features per participant: {len(feature_df.columns) - 3}")
        print(f"  ├─ COVAREP features: {sum(1 for col in feature_df.columns if 'covarep' in col)}")
        print(f"  └─ Formant features: {sum(1 for col in feature_df.columns if 'formant' in col)}")
        
        print(f"\n🎯 Depression Distribution:")
        depression_count = feature_df['depression'].sum()
        no_depression_count = len(feature_df) - depression_count
        depression_pct = (depression_count / len(feature_df) * 100) if len(feature_df) > 0 else 0
        
        print(f"  ├─ Depression (PHQ-8 >= 10): {depression_count} ({depression_pct:.1f}%)")
        print(f"  └─ No Depression (PHQ-8 < 10): {no_depression_count} ({100-depression_pct:.1f}%)")
        
        print(f"\n📉 PHQ-8 Score Statistics:")
        print(f"  ├─ Mean: {feature_df['phq8_score'].mean():.2f}")
        print(f"  ├─ Median: {feature_df['phq8_score'].median():.2f}")
        print(f"  ├─ Std Dev: {feature_df['phq8_score'].std():.2f}")
        print(f"  └─ Range: {feature_df['phq8_score'].min():.0f} - {feature_df['phq8_score'].max():.0f}")
        
        print("="*70)
        
        # Warnings
        if len(skipped) > 0:
            print(f"\n⚠️  Skipped {len(skipped)} participants due to errors:")
            print(f"   {', '.join(skipped[:10])}")
            if len(skipped) > 10:
                print(f"   ... and {len(skipped) - 10} more")
        
        if len(feature_df) < 20:
            print("\n⚠️  WARNING: Very few samples! Model may not train well.")
            print("   Recommended minimum: 50+ samples")
            print(f"   Current samples: {len(feature_df)}")
        
        if depression_pct < 20 or depression_pct > 80:
            print(f"\n⚠️  WARNING: Class imbalance detected!")
            print(f"   Depression: {depression_pct:.1f}% | No Depression: {100-depression_pct:.1f}%")
            print("   Model may be biased toward majority class")
        
        print()
        
        return feature_df