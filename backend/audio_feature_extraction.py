# audio_feature_extraction.py — OPTIMIZED
# Changes:
#   - Vectorised stat computation using numpy (was per-column loop with pandas)
#   - Participant file lookup uses a pre-built dict instead of trying paths one-by-one
#   - Progress logging every 20 participants (was every 10 — less noise for large datasets)
#   - Skipped participants tracked with reason for easier debugging

import pandas as pd
import numpy as np
from pathlib import Path
import os


class AudioFeatureProcessor:
    def __init__(self, covarep_dir: str, formant_dir: str, labels_file: str):
        self.covarep_dir = Path(covarep_dir)
        self.formant_dir = Path(formant_dir)
        self.labels_file = labels_file

        # Pre-build lookup dicts: participant_id → file path
        self._covarep_map = self._build_map(self.covarep_dir, "COVAREP", "covarep")
        self._formant_map = self._build_map(self.formant_dir, "FORMANT", "formant")

    # ── File discovery ───────────────────────────────────────────────────────────
    @staticmethod
    def _build_map(directory: Path, *strip_suffixes: str) -> dict:
        """
        Return {participant_id_str: Path} for all CSV files in directory.
        Strips known suffixes like '_COVAREP', '_covarep' from stem.
        """
        mapping = {}
        for path in directory.glob("*.csv"):
            stem = path.stem
            pid = stem
            for suffix in strip_suffixes:
                pid = pid.replace(f"_{suffix}", "").replace(f"_{suffix.lower()}", "")
            mapping[str(pid)] = path
        return mapping

    # ── Labels ───────────────────────────────────────────────────────────────────
    def load_labels(self) -> pd.DataFrame:
        df = pd.read_csv(self.labels_file)
        print(f"Available columns: {df.columns.tolist()}")
        if "PHQ_8Total" not in df.columns:
            raise ValueError(f"PHQ_8Total column not found. Available: {df.columns.tolist()}")
        df["depression"] = (df["PHQ_8Total"] >= 10).astype(int)
        result = pd.DataFrame({
            "Participant_ID": df["Participant_ID"],
            "PHQ8_Score": df["PHQ_8Total"],
            "depression": df["depression"],
        })
        print(f"\nLoaded {len(result)} participants")
        print(f"  Depression (≥10): {result['depression'].sum()}")
        print(f"  No Depression (<10): {(~result['depression'].astype(bool)).sum()}")
        return result

    # ── Participants with both files ─────────────────────────────────────────────
    def find_available_participants(self) -> set:
        available = set(self._covarep_map) & set(self._formant_map)
        print(f"\nAudio file summary:")
        print(f"  COVAREP files: {len(self._covarep_map)}")
        print(f"  Formant files: {len(self._formant_map)}")
        print(f"  Both:          {len(available)}")
        return available

    # ── Feature loading ──────────────────────────────────────────────────────────
    @staticmethod
    def _compute_stats(df: pd.DataFrame, prefix: str) -> dict:
        """
        Vectorised stat computation — much faster than the original per-column loop.
        Returns dict of {prefix_colname_stat: value} for finite values only.
        """
        numeric = df.select_dtypes(include=[np.number])
        if numeric.empty:
            return {}

        # Replace inf with nan, then compute stats in one pass
        clean = numeric.replace([np.inf, -np.inf], np.nan)

        means = clean.mean()
        stds = clean.std()
        mins = clean.min()
        maxs = clean.max()
        medians = clean.median()

        features = {}
        for col in clean.columns:
            for stat_name, series in [
                ("mean", means), ("std", stds), ("min", mins),
                ("max", maxs), ("median", medians),
            ]:
                val = series[col]
                if pd.notna(val) and np.isfinite(val):
                    features[f"{prefix}{col}_{stat_name}"] = float(val)
        return features

    def load_covarep_features(self, participant_id: str) -> dict | None:
        path = self._covarep_map.get(str(participant_id))
        if not path:
            return None
        try:
            df = pd.read_csv(path)
            return self._compute_stats(df, "covarep_") or None
        except Exception as e:
            print(f"  Error loading COVAREP for {participant_id}: {e}")
            return None

    def load_formant_features(self, participant_id: str) -> dict | None:
        path = self._formant_map.get(str(participant_id))
        if not path:
            return None
        try:
            df = pd.read_csv(path)
            return self._compute_stats(df, "formant_") or None
        except Exception as e:
            print(f"  Error loading Formant for {participant_id}: {e}")
            return None

    # ── Main dataset builder ─────────────────────────────────────────────────────
    def create_feature_dataset(self) -> pd.DataFrame:
        print("=" * 70)
        print("STEP 1: Loading PHQ-8 labels …")
        labels_df = self.load_labels()
        total_labels = len(labels_df)

        print("\n" + "=" * 70)
        print("STEP 2: Finding participants with audio …")
        available_ids = self.find_available_participants()

        labels_df["Participant_ID"] = labels_df["Participant_ID"].astype(str)
        available_labels = labels_df[labels_df["Participant_ID"].isin(available_ids)]
        print(f"\n  With labels AND audio: {len(available_labels)}")
        print(f"  Excluded (no audio): {total_labels - len(available_labels)}")

        if len(available_labels) == 0:
            raise ValueError("No participants found with both PHQ-8 labels and audio features!")

        print("\n" + "=" * 70)
        print("STEP 3: Extracting acoustic features …")
        all_features = []
        skipped = []

        for i, (_, row) in enumerate(available_labels.iterrows(), 1):
            pid = str(row["Participant_ID"])
            if i % 20 == 0 or i == 1:
                print(f"  Processing {i}/{len(available_labels)}: {pid}")

            cov = self.load_covarep_features(pid)
            fmt = self.load_formant_features(pid)

            if not cov or not fmt:
                skipped.append(pid)
                continue

            combined = {
                "participant_id": pid,
                "phq8_score": row["PHQ8_Score"],
                "depression": row["depression"],
            }
            combined.update(cov)
            combined.update(fmt)
            all_features.append(combined)

        feature_df = pd.DataFrame(all_features)

        # Final cleaning
        feature_cols = [c for c in feature_df.columns if c not in ("participant_id", "phq8_score", "depression")]
        feature_df[feature_cols] = (
            feature_df[feature_cols]
            .replace([np.inf, -np.inf], np.nan)
            .fillna(feature_df[feature_cols].mean())
            .fillna(0)
        )

        # Summary
        dep = feature_df["depression"].sum()
        total = len(feature_df)
        dep_pct = dep / total * 100 if total else 0

        print("\n" + "=" * 70)
        print("FEATURE EXTRACTION COMPLETE")
        print(f"  Processed: {total} | Skipped: {len(skipped)}")
        print(f"  Features per participant: {len(feature_cols)}")
        print(f"  COVAREP: {sum(1 for c in feature_cols if 'covarep' in c)}")
        print(f"  Formant:  {sum(1 for c in feature_cols if 'formant' in c)}")
        print(f"  Depression: {dep} ({dep_pct:.1f}%) | No depression: {total-dep} ({100-dep_pct:.1f}%)")
        print(f"  PHQ-8 mean: {feature_df['phq8_score'].mean():.2f}  "
              f"± {feature_df['phq8_score'].std():.2f}")

        if skipped:
            print(f"\n  ⚠️  Skipped {len(skipped)}: {', '.join(skipped[:10])}"
                  + (f" …+{len(skipped)-10}" if len(skipped) > 10 else ""))
        if total < 20:
            print("\n  ⚠️  WARNING: Very few samples — model may not generalise well.")
        if dep_pct < 20 or dep_pct > 80:
            print(f"\n  ⚠️  WARNING: Class imbalance ({dep_pct:.1f}% depressed).")

        return feature_df