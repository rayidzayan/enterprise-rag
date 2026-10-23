import os
import re

import streamlit as st
from groq import Groq
from dotenv import load_dotenv
from neo4j import GraphDatabase
from pgvector.psycopg import register_vector
from sentence_transformers import SentenceTransformer

from src.db import get_pg_connection

load_dotenv()

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:latest")
OLLAMA_HOST = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

_model = None


def get_embedding_model():
    """Load the embedding model once per Python process."""
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def vector_search(question: str, top_k: int = 4) -> list[dict]:
    """Find semantically similar chunks in PostgreSQL/pgvector."""
    model = get_embedding_model()
    embedding = model.encode(
        question,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    with get_pg_connection() as conn:
        register_vector(conn)

        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT c.id, d.filename, c.chunk_index, c.content,
                       1 - (c.embedding <=> %s) AS similarity
                FROM document_chunks c
                JOIN documents d ON d.id = c.document_id
                ORDER BY c.embedding <=> %s
                LIMIT %s
                """,
                (embedding, embedding, top_k),
            )

            rows = cur.fetchall()

    return [
        {
            "chunk_id": row[0],
            "filename": row[1],
            "chunk_index": row[2],
            "content": row[3],
            "score": float(row[4]),
            "source": "vector",
        }
        for row in rows
    ]


def graph_search(question: str, top_k: int = 4) -> list[dict]:
    """Find graph-related entities and their supporting text chunks."""
    if not NEO4J_PASSWORD:
        raise RuntimeError("NEO4J_PASSWORD is missing from .env")

    # Use meaningful question terms to locate graph entities.
    stopwords = {
        "what", "when", "where", "which", "who", "whom", "whose",
        "does", "did", "have", "has", "had", "will", "would", "could",
        "should", "can", "the", "and", "for", "with", "from", "that",
        "this", "are", "was", "were", "how", "about", "tell", "me",
        "please", "explain", "find", "show", "give", "you", "your",
        "company", "policy", "policies",
    }

    terms = list(dict.fromkeys(
        word.lower()
        for word in re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}", question)
        if word.lower() not in stopwords
    ))[:12]

    if not terms:
        return []

    query = """
    MATCH (seed:Entity)
    WHERE any(term IN $terms
              WHERE toLower(seed.name) CONTAINS term)
    WITH collect(DISTINCT seed) AS seeds
    UNWIND seeds AS seed
    OPTIONAL MATCH (seed)-[:SUBJECT_OF|OBJECT_OF]-(f:Fact)
                -[:SUBJECT_OF|OBJECT_OF]-(neighbor:Entity)
    WITH collect(DISTINCT seed) + collect(DISTINCT neighbor) AS found
    UNWIND found AS e
    WITH DISTINCT e
    WHERE e IS NOT NULL
    MATCH (c:Chunk)-[:MENTIONS]->(e)
    OPTIONAL MATCH (d:Document)-[:HAS_CHUNK]->(c)
    RETURN c.chunk_id AS chunk_id,
           coalesce(d.filename, 'Unknown document') AS filename,
           c.chunk_index AS chunk_index,
           c.content AS content,
           collect(DISTINCT e.name) AS entities
    ORDER BY filename, chunk_index
    LIMIT $top_k
    """

    with GraphDatabase.driver(
        NEO4J_URI,
        auth=(NEO4J_USER, NEO4J_PASSWORD),
    ) as driver:
        driver.verify_connectivity()
        result = driver.execute_query(
            query,
            terms=terms,
            top_k=top_k,
            database_="neo4j",
        )

    return [
        {
            "chunk_id": record["chunk_id"],
            "filename": record["filename"],
            "chunk_index": record["chunk_index"],
            "content": record["content"],
            "entities": record["entities"],
            "source": "graph",
        }
        for record in result.records
    ]


def hybrid_search(question: str, top_k: int = 4) -> dict:
    """Combine vector and graph results, prioritizing shared evidence."""
    vector_results = vector_search(question, top_k=top_k)
    graph_results = graph_search(question, top_k=top_k)

    fused = {}

    for item in vector_results:
        key = ("chunk", item["chunk_id"])
        fused[key] = {
            **item,
            "retrieval_methods": ["vector"],
            "graph_entities": [],
        }

    for item in graph_results:
        key = ("chunk", item["chunk_id"])

        if key in fused:
            fused[key]["retrieval_methods"].append("graph")
            fused[key]["graph_entities"] = item.get("entities", [])
        else:
            fused[key] = {
                **item,
                "retrieval_methods": ["graph"],
                "graph_entities": item.get("entities", []),
                "score": None,
            }

    results = list(fused.values())

    # Evidence found by both methods is ranked first.
    results.sort(
        key=lambda item: (
            len(item["retrieval_methods"]),
            item["score"] if item["score"] is not None else -1,
        ),
        reverse=True,
    )

    return {
        "question": question,
        "vector_results": vector_results,
        "graph_results": graph_results,
        "evidence": results[:top_k],
    }



def generate_answer(question: str, evidence: list[dict]) -> str:
    """Generate a grounded answer using the hosted Groq API."""
    if not evidence:
        return (
            "I could not find relevant evidence in the indexed documents. "
            "Try a more specific question or ingest additional documents."
        )

    # Read the API key from local environment variables or Streamlit Secrets.
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        try:
            api_key = st.secrets.get("GROQ_API_KEY")
        except Exception:
            api_key = None

    if not api_key:
        return (
            "Groq API key is not configured. Set GROQ_API_KEY in your "
            "local .env file or in Streamlit Cloud Secrets."
        )

    model_name = os.getenv(
        "GROQ_MODEL",
        "openai/gpt-oss-20b",
    )

    context = "\n\n".join(
        (
            f"[Source: {item['filename']}, "
            f"chunk {item['chunk_index']}] "
            f"[Retrieved by: {', '.join(item['retrieval_methods'])}]\n"
            f"{item['content']}"
        )
        for item in evidence
    )

    client = Groq(api_key=api_key)

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an enterprise document question-answering "
                    "assistant. Answer using only the supplied evidence. "
                    "Do not invent facts or rely on outside knowledge. "
                    "If the evidence does not answer the question, say so. "
                    "Cite supporting documents using their filename and "
                    "chunk number, as shown in the evidence."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"QUESTION:\n{question}\n\n"
                    f"RETRIEVED EVIDENCE:\n{context}\n\n"
                    "Give a concise, accurate answer with source citations."
                ),
            },
        ],
        temperature=0,
        max_completion_tokens=1024,
    )

    answer = response.choices[0].message.content

    if not answer:
        return "The hosted model returned an empty answer. Please try again."

    return answer.strip()



def ask(question: str, top_k: int = 4) -> dict:
    """Retrieve evidence using both methods and generate a grounded answer."""
    results = hybrid_search(question, top_k=top_k)
    results["answer"] = generate_answer(question, results["evidence"])
    return results
