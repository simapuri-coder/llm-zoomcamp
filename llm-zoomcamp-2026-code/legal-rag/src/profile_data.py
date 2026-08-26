import json
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CSV_PATH = PROJECT_ROOT / "data" / "raw" / "legal_text_classification_sample.csv"
QA_PATH = PROJECT_ROOT / "data" / "raw" / "qa_sample.jsonl"


# ---------------------------------------------------------
# Profile legal case dataset
# ---------------------------------------------------------

def profile_cases():
    print("=" * 60)
    print("LEGAL CASE DATASET")
    print("=" * 60)

    df = pd.read_csv(CSV_PATH)

    print(f"\nNumber of records: {len(df):,}")

    print("\nColumns:")
    for column in df.columns:
        print(f"  - {column}")

    print("\nData types:")
    print(df.dtypes)

    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nDuplicate rows:")
    print(df.duplicated().sum())

    if "case_id" in df.columns:
        print(f"\nUnique case IDs: {df['case_id'].nunique():,}")

    if "case_outcome" in df.columns:
        print("\nCase outcome distribution:")
        print(df["case_outcome"].value_counts())

    if "case_text" in df.columns:
        text_lengths = df["case_text"].fillna("").str.len()

        print("\nCase text length statistics:")
        print(f"  Minimum: {text_lengths.min():,} characters")
        print(f"  Maximum: {text_lengths.max():,} characters")
        print(f"  Average: {text_lengths.mean():,.0f} characters")
        print(f"  Median:  {text_lengths.median():,.0f} characters")

    print("\nFirst case:")
    print(df.iloc[0].to_string())


# ---------------------------------------------------------
# Profile QA dataset
# ---------------------------------------------------------

def profile_qa():
    print("\n" + "=" * 60)
    print("QA DATASET")
    print("=" * 60)

    records = []

    with open(QA_PATH, "r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                records.append(json.loads(line))

    print(f"\nNumber of QA records: {len(records):,}")

    if records:
        print("\nFields in first QA record:")
        for key in records[0].keys():
            print(f"  - {key}")

        print("\nFirst QA question:")
        print(records[0].get("question", "N/A"))

        print("\nFirst QA answer:")
        print(records[0].get("answer", "N/A"))


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

if __name__ == "__main__":
    profile_cases()
    profile_qa()