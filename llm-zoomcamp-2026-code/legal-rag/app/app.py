import sys
from pathlib import Path

from flask import Flask, request, jsonify

# Allow imports from the project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.search import load_embeddings, search, generate_answer

app = Flask(__name__)

# Load embeddings once when the application starts
df, embeddings = load_embeddings()


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "documents": len(df)
    })


@app.route("/query", methods=["POST"])
def query():
    data = request.get_json(silent=True) or {}

    question = data.get("question")
    top_k = int(data.get("top_k", 5))

    if not question:
        return jsonify({
            "error": "question is required"
        }), 400

    results = search(
        question,
        df,
        embeddings,
        top_k=top_k
    )

    answer = generate_answer(question, results)

    sources = []
    for _, row in results.iterrows():
        sources.append({
            "case_title": row["case_title"],
            "case_id": row["case_id"],
            "case_outcome": row["case_outcome"],
            "chunk_number": int(row["chunk_number"]),
            "similarity": float(row["similarity"])
        })

    return jsonify({
        "question": question,
        "answer": answer,
        "sources": sources
    })


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )
