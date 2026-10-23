import hashlib
import re
from pathlib import Path

import fitz
import numpy as np
from docx import Document
from pgvector.psycopg import register_vector
from sentence_transformers import SentenceTransformer

from src.db import get_pg_connection

DOCUMENTS_DIR = Path("data/documents")
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt"}


def extract_text(path: Path) -> str:
    """Extract text from a PDF, Word document, or text file."""
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        with fitz.open(path) as pdf:
            text = "\n".join(page.get_text() for page in pdf.pages)

    elif suffix == ".docx":
        doc = Document(path)
        text = "\n".join(paragraph.text for paragraph in doc.paragraphs)

    elif suffix == ".txt":
        text = path.read_text(encoding="utf-8", errors="replace")

    else:
        raise ValueError(f"Unsupported file type: {suffix}")

    return re.sub(r"\s+", " ", text).strip()


def split_into_chunks(text: str) -> list[str]:
    """Split text into overlapping character-based chunks."""
    text = text.strip()
    if not text:
        return []

    chunks = []
    start = 0

    while start < len(text):
        end = min(start + CHUNK_SIZE, len(text))
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end == len(text):
            break

        start = end - CHUNK_OVERLAP

    return chunks


def initialize_database() -> None:
    """Create the document and vector tables."""
    with get_pg_connection() as conn:
        register_vector(conn)

        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id BIGSERIAL PRIMARY KEY,
                    filename TEXT NOT NULL,
                    content_hash TEXT UNIQUE NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
            """)

            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS document_chunks (
                    id BIGSERIAL PRIMARY KEY,
                    document_id BIGINT NOT NULL
                        REFERENCES documents(id) ON DELETE CASCADE,
                    chunk_index INTEGER NOT NULL,
                    content TEXT NOT NULL,
                    embedding vector({EMBEDDING_DIMENSION}) NOT NULL,
                    UNIQUE (document_id, chunk_index)
                )
            """)

            cur.execute("""
                CREATE INDEX IF NOT EXISTS
                document_chunks_document_id_idx
                ON document_chunks (document_id)
            """)


def ingest_file(
    path: Path,
    model: SentenceTransformer,
) -> int:
    """Extract, chunk, embed, and store one document."""
    text = extract_text(path)

    if not text:
        print(f"Skipped (no extractable text): {path.name}")
        return 0

    content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
    chunks = split_into_chunks(text)

    if not chunks:
        print(f"Skipped (no chunks): {path.name}")
        return 0

    embeddings = model.encode(
        chunks,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    with get_pg_connection() as conn:
        register_vector(conn)

        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO documents (filename, content_hash)
                VALUES (%s, %s)
                ON CONFLICT (content_hash) DO NOTHING
                RETURNING id
                """,
                (path.name, content_hash),
            )

            row = cur.fetchone()

            if row is None:
                print(f"Skipped (already ingested): {path.name}")
                return 0

            document_id = row[0]

            cur.executemany(
                """
                INSERT INTO document_chunks
                    (document_id, chunk_index, content, embedding)
                VALUES (%s, %s, %s, %s)
                """,
                [
                    (document_id, i, chunk, np.asarray(embedding, dtype=np.float32))
                    for i, (chunk, embedding) in enumerate(
                        zip(chunks, embeddings)
                    )
                ],
            )

    print(f"Ingested: {path.name} ({len(chunks)} chunks)")
    return len(chunks)


def main() -> None:
    DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
    initialize_database()

    files = sorted(
        path
        for path in DOCUMENTS_DIR.iterdir()
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    if not files:
        print(f"No supported documents found in {DOCUMENTS_DIR.resolve()}")
        print("Add PDF, DOCX, or TXT files and run this script again.")
        return

    print(f"Loading embedding model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    total_chunks = 0

    for path in files:
        try:
            total_chunks += ingest_file(path, model)
        except Exception as exc:
            print(f"Failed to ingest {path.name}: {exc}")

    print(f"\nIngestion finished. New chunks stored: {total_chunks}")


if __name__ == "__main__":
    main()
