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

def generate_answer(query, results):
    context = "\n\n".join(
        [
            f"Case: {row['case_title']}\n"
            f"Outcome: {row['case_outcome']}\n"
            f"Chunk: {row['chunk_number']}\n"
            f"Text: {row['text']}"
            for _, row in results.iterrows()
        ]
    )

    prompt = f"""
You are a legal research assistant.

Answer the user's question using ONLY the legal passages provided below.

Rules:
- Do not invent facts, cases, citations, or legal principles.
- If the provided passages do not contain enough information to answer,
  clearly say that the retrieved context is insufficient.
- Explain the answer clearly and concisely.
- Identify the relevant cases when supported by the context.
- Do not provide personalized legal advice.

Retrieved legal passages:
{context}

User question:
{query}
"""

    response = client.chat.completions.create(
        model=os.getenv("OPENAI_LLM_MODEL"),
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
    )

    return response.choices[0].message.content

if __name__ == "__main__":
    df, embeddings = load_embeddings()

    query = "What are the legal principles concerning contractual obligations?"

    results = search(
        query,
        df,
        embeddings,
        top_k=5,
    )

    answer = generate_answer(query, results)

    print("\n" + "=" * 80)
    print("QUESTION")
    print("=" * 80)
    print(query)

    print("\n" + "=" * 80)
    print("ANSWER")
    print("=" * 80)
    print(answer)

    print("\n" + "=" * 80)
    print("SOURCES")
    print("=" * 80)

    for _, row in results.iterrows():
        print(
            f"- {row['case_title']} "
            f"(chunk {row['chunk_number']}, "
            f"similarity {row['similarity']:.4f})"
        )