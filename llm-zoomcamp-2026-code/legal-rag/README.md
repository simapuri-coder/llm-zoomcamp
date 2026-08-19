# Legal RAG Application

A Retrieval-Augmented Generation (RAG) application for answering questions about Australian legal cases using a legal case corpus and QA evaluation data.

## Project goals

- Build a searchable legal-document knowledge base.
- Retrieve relevant legal cases for a user question.
- Generate grounded answers with an OpenAI LLM.
- Evaluate retrieval and generated answers using the supplied QA dataset.
- Expose the application through a Flask API.
- Store application metrics in PostgreSQL.
- Visualize monitoring metrics with Grafana.
- Containerize the application with Docker Compose.

## Data

The initial project data supplied for this assignment are:

- `legal_text_classification.csv` — legal case records with `case_id`, `case_outcome`, `case_title`, and `case_text`.
- `qa.jsonl` — question/answer examples with source legal-document information.

Raw datasets should not be committed to GitHub. Keep them in a local `data/raw/` directory or another approved storage location.

## Initial architecture

```text
User Question
     |
     v
 Flask API (/ask)
     |
     v
 Query Embedding
     |
     v
 Legal Search / Retrieval
     |
     v
 Relevant Cases
     |
     v
 OpenAI LLM
     |
     v
 Grounded Answer + Sources
     |
     +------> PostgreSQL -> Grafana
     |
     +------> Evaluation pipeline
```

## Planned project structure

```text
legal-rag/
├── app/
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
├── src/
├── tests/
├── monitoring/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env.example
└── README.md
```

## Development milestones

1. Profile and validate the supplied legal datasets.
2. Design legal-document chunking and metadata handling.
3. Build document embeddings and the search index.
4. Implement retrieval and evaluate Recall@K / MRR.
5. Implement RAG answer generation.
6. Evaluate answer quality and groundedness.
7. Build the Flask API.
8. Add PostgreSQL request/latency logging.
9. Add Grafana dashboards.
10. Containerize and document the complete application.

## Configuration

Use environment variables for API credentials. Never commit an actual API key.

Example variables will be documented in `.env.example`.
