# check_labels.py
import pandas as pd

LABELS_FILE = "data/PHQ8_Labels.csv"

# Load and inspect the file
df = pd.read_csv(LABELS_FILE)

print("Column names in your CSV:")
print(df.columns.tolist())
print("\nFirst few rows:")
print(df.head())
print(f"\nTotal rows: {len(df)}")