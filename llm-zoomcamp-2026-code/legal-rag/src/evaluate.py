import json

from search import load_embeddings, search
def load_qa_data():
    qa_file = "data/raw/qa_sample.jsonl"

    with open(qa_file, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]
    
if __name__ == "__main__":
    qa_data = load_qa_data()

    df, embeddings = load_embeddings()

    question = qa_data[0]["question"]

    results = search(
        question,
        df,
        embeddings,
        top_k=5,
    )

    print("\nQUESTION:")
    print(question)

    print("\nEXPECTED CASE:")
    print(qa_data[0]["source"]["citation"])

    print("\nTOP 5 RETRIEVED CASES:")

    for rank, (_, row) in enumerate(results.iterrows(), start=1):
        print(
            f"{rank}. {row['case_title']} "
            f"| similarity={row['similarity']:.4f} "
            f"| chunk={row['chunk_number']}"
        )