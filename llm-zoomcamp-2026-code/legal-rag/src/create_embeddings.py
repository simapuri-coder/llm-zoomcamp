import os
import time

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI


# Load environment variables from .env
load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY")
EMBEDDING_MODEL = os.getenv(
    "OPENAI_EMBEDDING_MODEL",
    "text-embedding-3-small",
)

INPUT_FILE = "data/processed/legal_documents.csv"
OUTPUT_FILE = "data/processed/legal_documents_embeddings.csv"

BATCH_SIZE = 100


def create_embeddings():
    """Create embeddings for all legal document chunks."""

    if not API_KEY:
        raise ValueError(
            "OPENAI_API_KEY is not configured. "
            "Check your .env file."
        )

    print("Loading legal document chunks...")

    df = pd.read_csv(INPUT_FILE)

    print(f"Chunks loaded: {len(df):,}")
    print(f"Embedding model: {EMBEDDING_MODEL}")

    client = OpenAI(api_key=API_KEY)

    all_embeddings = []

    total_batches = (len(df) + BATCH_SIZE - 1) // BATCH_SIZE

    for batch_number, start in enumerate(
        range(0, len(df), BATCH_SIZE),
        start=1,
    ):
        end = min(start + BATCH_SIZE, len(df))

        batch_texts = (
            df.iloc[start:end]["text"]
            .fillna("")
            .tolist()
        )

        print(
            f"Processing batch {batch_number}/{total_batches} "
            f"({start + 1}-{end})..."
        )

        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=batch_texts,
        )

        batch_embeddings = [
            item.embedding
            for item in response.data
        ]

        all_embeddings.extend(batch_embeddings)

        print(
            f"  ✓ Generated {len(batch_embeddings)} embeddings"
        )

        # Small pause between batches
        time.sleep(0.2)

    df["embedding"] = all_embeddings

    df.to_csv(OUTPUT_FILE, index=False)

    print()
    print("Embedding generation complete!")
    print(f"Total embeddings: {len(all_embeddings):,}")
    print(f"Embedding dimensions: {len(all_embeddings[0])}")
    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    create_embeddings()