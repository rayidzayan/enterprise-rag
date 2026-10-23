import json
import os
import re

import ollama
from dotenv import load_dotenv
from neo4j import GraphDatabase

from src.db import get_pg_connection

load_dotenv()

MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:latest")
OLLAMA_HOST = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")


def extract_graph_data(text: str) -> dict:
    """Extract entities and relationships from one text chunk."""
    prompt = f"""
Extract factual entities and relationships from the company document below.

Return ONLY a JSON object in this exact structure:
{{
  "entities": [
    {{"name": "Annual Leave", "type": "POLICY"}}
  ],
  "relationships": [
    {{"source": "Full-time employees", "relation": "RECEIVE",
      "target": "Paid annual leave"}}
  ]
}}

Rules:
- Extract only facts explicitly supported by the text.
- Entity types should be short categories such as PERSON, TEAM,
  POLICY, DEPARTMENT, SYSTEM, ORGANIZATION, or BENEFIT.
- Relationship names should be short uppercase verbs or phrases.
- Use consistent entity names within this chunk.
- Do not invent people, policies, or facts.
- Keep entity names concise.
- Return empty arrays when nothing useful can be extracted.

DOCUMENT CHUNK:
{text}
"""

    response = ollama.chat(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        format="json",
        options={"temperature": 0},
    )

    data = json.loads(response.message.content)

    if not isinstance(data.get("entities"), list):
        raise ValueError("Model response has no valid entities list.")
    if not isinstance(data.get("relationships"), list):
        raise ValueError("Model response has no valid relationships list.")

    entities = []
    seen = set()

    for item in data["entities"]:
        if not isinstance(item, dict):
            continue

        name = str(item.get("name", "")).strip()
        entity_type = str(item.get("type", "ENTITY")).strip().upper()

        if not name:
            continue

        key = name.casefold()
        if key not in seen:
            entities.append({"name": name, "type": entity_type[:50]})
            seen.add(key)

    names = {item["name"].casefold(): item["name"] for item in entities}
    relationships = []

    for item in data["relationships"]:
        if not isinstance(item, dict):
            continue

        source = str(item.get("source", "")).strip()
        target = str(item.get("target", "")).strip()
        relation = str(item.get("relation", "RELATED_TO")).strip().upper()

        if not source or not target or not relation:
            continue

        source = names.get(source.casefold())
        target = names.get(target.casefold())

        if source and target:
            relationships.append({
                "source": source,
                "target": target,
                "relation": re.sub(r"[^A-Z0-9_ ]", "", relation)[:80],
            })

    return {"entities": entities, "relationships": relationships}


def get_chunks():
    """Load existing document chunks from PostgreSQL."""
    with get_pg_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT c.id, d.filename, d.content_hash,
                       c.chunk_index, c.content
                FROM document_chunks c
                JOIN documents d ON d.id = c.document_id
                ORDER BY c.id
            """)
            return cur.fetchall()



def store_graph_chunk(tx, chunk_id, filename, content_hash,
                      chunk_index, content, entities, relationships):
    """Store documents, chunks, entities, and fact nodes in Neo4j."""

    # Store the document and its chunk.
    tx.run("""
        MERGE (d:Document {content_hash: $content_hash})
        SET d.filename = $filename
        MERGE (c:Chunk {chunk_id: $chunk_id})
        SET c.chunk_index = $chunk_index,
            c.content = $content
        MERGE (d)-[:HAS_CHUNK]->(c)
    """,
        content_hash=content_hash,
        filename=filename,
        chunk_id=chunk_id,
        chunk_index=chunk_index,
        content=content,
    )

    # Store entities mentioned in the chunk.
    for entity in entities:
        tx.run("""
            MERGE (e:Entity {name_key: $name_key})
            SET e.name = $name,
                e.entity_type = $entity_type
            WITH e
            MATCH (c:Chunk {chunk_id: $chunk_id})
            MERGE (c)-[:MENTIONS]->(e)
        """,
            name_key=entity["name"].casefold(),
            name=entity["name"],
            entity_type=entity["type"],
            chunk_id=chunk_id,
        )

    # Store each extracted relationship as a Fact node.
    for relationship in relationships:
        fact_key = (
            f"{chunk_id}:"
            f"{relationship['source'].casefold()}:"
            f"{relationship['relation']}:"
            f"{relationship['target'].casefold()}"
        )

        tx.run("""
            MATCH (s:Entity {name_key: $source_key})
            MATCH (t:Entity {name_key: $target_key})
            MATCH (c:Chunk {chunk_id: $chunk_id})

            MERGE (f:Fact {fact_key: $fact_key})
            SET f.relation = $relation,
                f.source_name = $source_name,
                f.target_name = $target_name

            MERGE (s)-[:SUBJECT_OF]->(f)
            MERGE (f)-[:OBJECT_OF]->(t)
            MERGE (c)-[:SUPPORTS]->(f)
        """,
            fact_key=fact_key,
            source_key=relationship["source"].casefold(),
            target_key=relationship["target"].casefold(),
            source_name=relationship["source"],
            target_name=relationship["target"],
            relation=relationship["relation"],
            chunk_id=chunk_id,
        )



def main():
    if not NEO4J_PASSWORD:
        raise RuntimeError("NEO4J_PASSWORD is missing from .env")

    chunks = get_chunks()

    if not chunks:
        print("No document chunks found in PostgreSQL.")
        print("Run python -m src.ingest first.")
        return

    client = ollama.Client(host=OLLAMA_HOST)

    # Confirm Ollama is reachable before processing documents.
    client.list()

    total_entities = 0
    total_relationships = 0
    failures = 0

    with GraphDatabase.driver(
        NEO4J_URI,
        auth=(NEO4J_USER, NEO4J_PASSWORD),
    ) as driver:
        driver.verify_connectivity()

        with driver.session(database="neo4j") as session:
            for row in chunks:
                chunk_id, filename, content_hash, chunk_index, content = row
                print(f"Extracting: {filename}, chunk {chunk_index + 1}")

                try:
                    result = extract_graph_data(content)
                    session.execute_write(
                        store_graph_chunk,
                        chunk_id,
                        filename,
                        content_hash,
                        chunk_index,
                        content,
                        result["entities"],
                        result["relationships"],
                    )

                    total_entities += len(result["entities"])
                    total_relationships += len(result["relationships"])

                    print(
                        f"  Entities: {len(result['entities'])}; "
                        f"relationships: {len(result['relationships'])}"
                    )

                except Exception as exc:
                    failures += 1
                    print(f"  Failed on this chunk: {exc}")

    print("\nGraph extraction finished.")
    print(f"Chunks processed: {len(chunks)}")
    print(f"Entities extracted: {total_entities}")
    print(f"Relationships extracted: {total_relationships}")
    print(f"Failed chunks: {failures}")


if __name__ == "__main__":
    main()
