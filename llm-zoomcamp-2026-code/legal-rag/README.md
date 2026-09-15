# Legal RAG Application

A Retrieval-Augmented Generation (RAG) application for legal document question answering.

## Project Overview

- Legal document retrieval using semantic similarity search
- OpenAI embeddings using text-embedding-3-small
- LLM-generated answers using retrieved legal context
- Flask REST API
- PostgreSQL database
- Docker and Docker Compose deployment

## Architecture

User Question -> Query Embedding -> Semantic Search -> Top-K Legal Chunks -> LLM -> Answer + Sources

## Current Corpus

The running application reports 2,312 searchable legal document records.

- Embedding model: text-embedding-3-small
- Embedding dimension: 1,536
- Retrieval: Top-K semantic similarity search

## API

### Health Check

    curl http://127.0.0.1:5000/health

Expected response:

    {"documents": 2312, "status": "healthy"}

### RAG Query

    curl -X POST http://127.0.0.1:5000/query -H "Content-Type: application/json" -d '{"question":"What are the legal principles concerning contractual obligations?"}'

The response provides the generated answer and retrieved legal sources with similarity scores.

## Run with Docker

Build the application:

    docker build -t legal-rag .

Start the application and PostgreSQL:

    docker compose up -d

Check the services:

    docker compose ps

Stop the services:

    docker compose down

## Evaluation

A 200-question QA sample was used during development evaluation. The QA sample contains source cases that are not all represented in the current corpus, so retrieval behavior was also checked against cases available in the corpus.

## Technology Stack

- Python 3.12
- Flask
- OpenAI API
- Pandas
- NumPy
- scikit-learn
- PostgreSQL
- Docker
- Docker Compose
