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


def create_chunks(
    text,
    target_chars=TARGET_CHARS,
    min_chars=100,
):
    """
    Split legal text into retrieval-friendly chunks.

    Rules:
    - Prefer paragraph/sentence boundaries.
    - Target approximately 1,500 characters.
    - Never exceed 1,500 characters.
    - Merge very short fragments into the previous chunk.
    """

    text = clean_text(text)

    if not text:
        return []

    # Try to identify numbered legal paragraphs.
    paragraphs = re.split(
        r"(?<=\.)\s+(?=\d+\s)",
        text
    )

    # If that doesn't provide useful boundaries,
    # fall back to sentence boundaries.
    if len(paragraphs) == 1:
        paragraphs = re.split(
            r"(?<=[.!?])\s+",
            text
        )

    pieces = []

    for paragraph in paragraphs:
        paragraph = paragraph.strip()

        if not paragraph:
            continue

        # Break very long paragraphs into safe pieces.
        if len(paragraph) > target_chars:

            for start in range(
                0,
                len(paragraph),
                target_chars
            ):
                piece = paragraph[
                    start:start + target_chars
                ].strip()

                if piece:
                    pieces.append(piece)

        else:
            pieces.append(paragraph)

    # Build final chunks.
    chunks = []
    current = ""

    for piece in pieces:

        if not current:
            current = piece
            continue

        candidate = f"{current} {piece}"

        if len(candidate) <= target_chars:
            current = candidate

        else:
            chunks.append(current)
            current = piece

    if current:
        chunks.append(current)

       # Merge very short fragments into the previous chunk.
    # We allow the chunk to go slightly above the target size
    # because preserving meaningful legal context is more
    # important than enforcing an exact character limit.

    merged_chunks = []

    for chunk in chunks:

        if len(chunk) < min_chars and merged_chunks:
            merged_chunks[-1] = (
                f"{merged_chunks[-1]} {chunk}"
            )
        else:
            merged_chunks.append(chunk)

    return merged_chunks
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