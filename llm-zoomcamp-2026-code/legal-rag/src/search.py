import ast
import os

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
load_dotenv()

EMBEDDINGS_FILE = "data/processed/legal_documents_embeddings.csv"
EMBEDDING_MODEL = os.getenv(
    "OPENAI_EMBEDDING_MODEL",
    "text-embedding-3-small",
)

client = OpenAI()
def load_embeddings():
    df = pd.read_csv(EMBEDDINGS_FILE)

    df["embedding"] = df["embedding"].apply(ast.literal_eval)

    embeddings = np.array(df["embedding"].tolist(), dtype=np.float32)

    return df, embeddings

def embed_query(query):
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=query,
    )

    return np.array(response.data[0].embedding, dtype=np.float32)

def search(query, df, embeddings, top_k=5):
    query_embedding = embed_query(query)

    # Normalize vectors
    query_norm = query_embedding / np.linalg.norm(query_embedding)
    embedding_norms = embeddings / np.linalg.norm(
        embeddings, axis=1, keepdims=True
    )

    # Calculate cosine similarity
    similarities = embedding_norms @ query_norm

    # Get the indices of the top results
    top_indices = np.argsort(similarities)[-top_k:][::-1]

    results = df.iloc[top_indices].copy()
    results["similarity"] = similarities[top_indices]

    return results
if __name__ == "__main__":
    df, embeddings = load_embeddings()

    query = "What are the legal principles concerning contractual obligations?"

    results = search(
        query,
        df,
        embeddings,
        top_k=5,
    )

    print("\nQuery:", query)
    print("\nTop 5 Results:\n")

    for _, row in results.iterrows():
        print("=" * 80)
        print("Similarity:", round(row["similarity"], 4))
        print("Case:", row["case_title"])
        print("Outcome:", row["case_outcome"])
        print("Chunk:", row["chunk_number"])
        print("Text:", row["text"][:500])