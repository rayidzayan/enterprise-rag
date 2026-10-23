# Enterprise GraphRAG — Company Knowledge Assistant

An AI-powered company knowledge assistant that combines vector search and graph-based retrieval to answer questions from internal documents with source-grounded responses.

## Features

- **Document ingestion:** Process PDF, DOCX, and TXT files.
- **Vector search:** Retrieve semantically relevant document chunks using PostgreSQL and pgvector.
- **Knowledge graph:** Extract entities, facts, and relationships into Neo4j.
- **Hybrid retrieval:** Combine vector and graph search to gather evidence.
- **Local LLM inference:** Generate answers using Ollama.
- **Evaluation dashboard:** Run repeatable test questions and inspect keyword matches and response times.
- **Graph Explorer:** Visualize extracted entities and relationships.
- **Interactive interface:** Use the Streamlit application to upload documents and ask questions.

## Architecture

Documents → Text extraction and chunking → Embeddings and knowledge graph → Hybrid retrieval → Ollama answer generation → Source evidence

### Technology Stack

| Component | Technology |
|---|---|
| Frontend | Streamlit |
| Vector database | PostgreSQL + pgvector |
| Graph database | Neo4j |
| Language model | Ollama |
| Embeddings | Sentence Transformers (`all-MiniLM-L6-v2`) |
| Document processing | PyMuPDF, python-docx |
| Evaluation | Custom test-question dashboard |

## Example Use Cases

- Search employee policies and company procedures.
- Retrieve relevant facts from multiple documents.
- Explore relationships between entities and policies.
- Inspect answers and the evidence used to support them.
- Evaluate retrieval behavior on a repeatable set of questions.

## Local Setup

### Prerequisites

- Python 3.12
- PostgreSQL with pgvector
- Neo4j
- Ollama with the configured model downloaded

### Installation

1. Clone or copy the project.
2. Create and activate a virtual environment.
3. Install the dependencies listed in `requirements.txt`.
4. Create a `.env` file with your local database credentials and Ollama configuration.
5. Create the required PostgreSQL database and tables.
6. Start PostgreSQL, Neo4j, and Ollama.
7. Index your documents and launch Streamlit.

### Run the application

```powershell
python -m src.ingest
python -m src.graph_ingest
python -m streamlit run app.py
```

The database schema must be initialized before ingestion. Configure `.env` using the variable names expected by the source code. Never commit passwords or other secrets.

## Evaluation

The dashboard runs a small, repeatable set of company-policy questions and reports keyword matches and response times.

Keyword matching is a lightweight diagnostic, not a comprehensive measure of answer correctness. Evaluation results depend on the indexed documents, graph extraction, retrieval behavior, and language model.

## Project Status

This is a local portfolio demonstration. Production deployment would require additional work on authentication, authorization, data isolation, monitoring, document lifecycle management, and security.

## Author

Developed as an applied AI engineering portfolio project.
