import re
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "legal_text_classification_sample.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "legal_documents.csv"
)

TARGET_CHARS = 1500
OVERLAP_CHARS = 200


def clean_text(text):
    """Basic cleaning while preserving legal meaning and citations."""

    if pd.isna(text):
        return ""

    text = str(text)

    # Normalize repeated whitespace.
    text = re.sub(r"\s+", " ", text).strip()

    return text


def create_chunks(text, target_chars=TARGET_CHARS):
    """
    Split legal text into approximately target-sized chunks.

    We prefer paragraph/sentence boundaries, but enforce a hard
    maximum so that extremely long legal passages are never
    stored as one embedding.
    """

    text = clean_text(text)

    if not text:
        return []

    # First try legal paragraph boundaries.
    paragraphs = re.split(r"(?<=\.)\s+(?=\d+\s)", text)

    # If paragraph detection is ineffective, use sentence boundaries.
    if len(paragraphs) == 1:
        paragraphs = re.split(r"(?<=[.!?])\s+", text)

    chunks = []
    current = ""

    for paragraph in paragraphs:
        paragraph = paragraph.strip()

        if not paragraph:
            continue

        # If one paragraph itself is too large, split it.
        if len(paragraph) > target_chars:

            if current:
                chunks.append(current)
                current = ""

            for start in range(
                0,
                len(paragraph),
                target_chars
            ):
                piece = paragraph[start:start + target_chars]
                chunks.append(piece.strip())

            continue

        if len(current) + len(paragraph) + 1 <= target_chars:
            current = f"{current} {paragraph}".strip()

        else:
            if current:
                chunks.append(current)

            current = paragraph

    if current:
        chunks.append(current)

    return chunks

def prepare_documents():
    print("Loading legal case data...")

    df = pd.read_csv(INPUT_PATH)

    print(f"Original records: {len(df):,}")

    # Remove records without searchable legal text.
    df = df.dropna(subset=["case_text"]).copy()

    df["case_text"] = df["case_text"].apply(clean_text)

    # Remove empty text after cleaning.
    df = df[df["case_text"].str.len() > 0].copy()

    print(f"Records with usable text: {len(df):,}")

    records = []

    for _, row in df.iterrows():

        chunks = create_chunks(row["case_text"])

        for chunk_number, chunk in enumerate(chunks):

            records.append(
                {
                    "case_id": row["case_id"],
                    "case_title": row["case_title"],
                    "case_outcome": row["case_outcome"],
                    "chunk_id": f"{row['case_id']}_{chunk_number}",
                    "chunk_number": chunk_number,
                    "text": chunk,
                    "text_length": len(chunk),
                }
            )

    processed_df = pd.DataFrame(records)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    processed_df.to_csv(OUTPUT_PATH, index=False)

    print(f"\nCreated {len(processed_df):,} searchable chunks.")
    print(f"Saved to: {OUTPUT_PATH}")

    print("\nChunk length statistics:")
    print(processed_df["text_length"].describe())


if __name__ == "__main__":
    prepare_documents()