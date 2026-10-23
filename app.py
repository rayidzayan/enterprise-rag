import time
import os
import subprocess
import sys
from pathlib import Path

import ollama
import streamlit as st
from dotenv import load_dotenv

from src.db import check_postgres
from src.graph import check_neo4j
from src.retrieval import ask

ROOT = Path(__file__).resolve().parent
DOCUMENTS_DIR = ROOT / "data" / "documents"

load_dotenv(ROOT / ".env")

st.set_page_config(
    page_title="Enterprise GraphRAG",
    page_icon="🔎",
    layout="wide",
)

st.title("Enterprise GraphRAG")
st.caption(
    "Knowledge retrieval powered by PostgreSQL, pgvector, "
    "Neo4j, and Ollama"
)

# ---------- Sidebar: system status ----------

with st.sidebar:
    st.header("System Status")

    if st.button("Check connections", use_container_width=True):
        st.session_state["check_status"] = True

    if st.session_state.get("check_status", False):
        try:
            st.success(f"PostgreSQL: {check_postgres()}")
        except Exception as exc:
            st.error(f"PostgreSQL unavailable: {exc}")

        try:
            st.success(f"Neo4j: {check_neo4j()}")
        except Exception as exc:
            st.error(f"Neo4j unavailable: {exc}")

        try:
            client = ollama.Client(
                host=os.getenv(
                    "OLLAMA_BASE_URL",
                    "http://localhost:11434",
                )
            )
            client.list()
            st.success("Ollama: Connected")
        except Exception as exc:
            st.error(f"Ollama unavailable: {exc}")

    st.divider()
    st.markdown("**Architecture**")
    st.markdown(
        "- Document ingestion\n"
        "- Vector retrieval\n"
        "- Graph retrieval\n"
        "- LLM answer generation"
    )

# ---------- Main interface ----------

ask_tab, upload_tab, evaluation_tab, graph_tab, about_tab = st.tabs(
    ["Ask Documents", "Upload Documents", "Evaluation", "Graph Explorer", "About"]
)

with ask_tab:
    st.subheader("Ask your company knowledge base")

    question = st.text_area(
        "Your question",
        placeholder=(
            "Example: How many days of annual leave do "
            "full-time employees receive?"
        ),
        height=100,
    )

    if st.button("Generate Answer", type="primary"):
        if not question.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner(
                "Searching vector and graph knowledge, then generating an answer..."
            ):
                try:
                    result = ask(question.strip())

                    st.subheader("Answer")

                    if isinstance(result, dict):
                        answer = (
                            result.get("answer")
                            or result.get("response")
                            or result.get("output")
                        )

                        if answer is not None:
                            st.markdown(str(answer))
                        else:
                            st.write(result)

                        with st.expander("Inspect retrieval details"):
                            st.json(result)
                    else:
                        st.write(result)

                    st.session_state["last_question"] = question.strip()
                    st.session_state["last_result"] = result

                except Exception as exc:
                    st.error(f"Could not answer the question: {exc}")
                    st.info(
                        "Check that PostgreSQL, Neo4j, and Ollama are "
                        "running and that the knowledge base is indexed."
                    )

    if "last_result" in st.session_state:
        with st.expander("Previous answer"):
            st.write(st.session_state["last_result"])

with upload_tab:
    st.subheader("Add company documents")

    st.write(
        "Upload documents to index their contents for future questions."
    )

    uploaded_files = st.file_uploader(
        "Choose documents",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
    )

    st.caption(
        "Supported formats: PDF, Word documents, and plain text."
    )

    if st.button("Index uploaded documents", type="primary"):
        if not uploaded_files:
            st.warning("Choose at least one document first.")
        else:
            DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)

            saved_count = 0

            for uploaded_file in uploaded_files:
                destination = DOCUMENTS_DIR / Path(
                    uploaded_file.name
                ).name

                destination.write_bytes(uploaded_file.getvalue())
                saved_count += 1

            st.success(f"Saved {saved_count} document(s).")

            commands = [
                (
                    "Vector indexing",
                    [sys.executable, "-m", "src.ingest"],
                ),
                (
                    "Graph extraction",
                    [sys.executable, "-m", "src.graph_ingest"],
                ),
            ]

            for label, command in commands:
                st.write(f"**{label}**")

                try:
                    completed = subprocess.run(
                        command,
                        cwd=str(ROOT),
                        capture_output=True,
                        text=True,
                        timeout=1800,
                        check=False,
                    )

                    if completed.stdout:
                        st.code(completed.stdout)

                    if completed.returncode != 0:
                        st.error(
                            f"{label} failed. Check the output below."
                        )
                        if completed.stderr:
                            st.code(completed.stderr)
                        break

                    if completed.stderr:
                        with st.expander(
                            f"{label}: additional messages"
                        ):
                            st.code(completed.stderr)

                    st.success(f"{label} completed.")

                except subprocess.TimeoutExpired:
                    st.error(
                        f"{label} exceeded the 30-minute time limit."
                    )
                    break
                except Exception as exc:
                    st.error(f"{label} failed: {exc}")
                    break

with evaluation_tab:
    st.subheader("GraphRAG Evaluation Dashboard")

    st.write(
        "Run a repeatable set of questions against the indexed "
        "company policy and inspect the answers."
    )

    test_cases = [
        {
            "question": "How many days of annual leave do full-time employees receive?",
            "expected": ["20 days", "annual leave"],
            "source": "sample_company_policy.txt",
        },
        {
            "question": "How far in advance should employees submit leave requests?",
            "expected": ["five working days", "manager"],
            "source": "sample_company_policy.txt",
        },
        {
            "question": "How many days per week can employees work remotely?",
            "expected": ["three days", "remote"],
            "source": "sample_company_policy.txt",
        },
        {
            "question": "What should employees do if a company laptop is stolen?",
            "expected": ["report", "stolen"],
            "source": "sample_company_policy.txt",
        },
        {
            "question": "Can employees share their passwords?",
            "expected": ["no", "password"],
            "source": "sample_company_policy.txt",
        },
    ]

    st.caption(
        "Keyword checks are approximate indicators, not a complete "
        "measure of semantic correctness."
    )

    if st.button("Run Evaluation", type="primary"):
        rows = []

        progress = st.progress(0)
        status = st.empty()

        for index, case in enumerate(test_cases):
            status.write(
                f"Evaluating question {index + 1} of {len(test_cases)}..."
            )

            start = time.perf_counter()

            try:
                result = ask(case["question"])
                elapsed = time.perf_counter() - start

                if isinstance(result, dict):
                    answer = (
                        result.get("answer")
                        or result.get("response")
                        or result.get("output")
                        or str(result)
                    )
                else:
                    answer = str(result)

                answer_lower = answer.casefold()

                matched = [
                    keyword
                    for keyword in case["expected"]
                    if keyword.casefold() in answer_lower
                ]

                keyword_score = (
                    len(matched) / len(case["expected"]) * 100
                )

                rows.append({
                    "Question": case["question"],
                    "Answer": answer,
                    "Expected keywords": ", ".join(case["expected"]),
                    "Matched keywords": ", ".join(matched) or "None",
                    "Keyword score (%)": round(keyword_score, 1),
                    "Response time (s)": round(elapsed, 2),
                    "Status": "Completed",
                    "Source expected": case["source"],
                })

            except Exception as exc:
                elapsed = time.perf_counter() - start

                rows.append({
                    "Question": case["question"],
                    "Answer": str(exc),
                    "Expected keywords": ", ".join(case["expected"]),
                    "Matched keywords": "None",
                    "Keyword score (%)": 0.0,
                    "Response time (s)": round(elapsed, 2),
                    "Status": "Error",
                    "Source expected": case["source"],
                })

            progress.progress((index + 1) / len(test_cases))

        status.write("Evaluation finished.")

        st.session_state["evaluation_rows"] = rows

    if "evaluation_rows" in st.session_state:
        import pandas as pd

        df = pd.DataFrame(st.session_state["evaluation_rows"])

        total = len(df)
        completed = int((df["Status"] == "Completed").sum())
        average_score = float(df["Keyword score (%)"].mean())
        average_time = float(df["Response time (s)"].mean())

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Test questions", total)
        col2.metric("Completed", f"{completed}/{total}")
        col3.metric("Average keyword score", f"{average_score:.1f}%")
        col4.metric("Average response time", f"{average_time:.2f}s")

        st.subheader("Evaluation results")

        st.dataframe(
            df[
                [
                    "Question",
                    "Keyword score (%)",
                    "Response time (s)",
                    "Status",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )

        st.subheader("Inspect individual answers")

        for index, row in df.iterrows():
            with st.expander(f"Test {index + 1}: {row['Question']}"):
                st.write("**Answer**")
                st.write(row["Answer"])

                st.write("**Expected keywords**")
                st.write(row["Expected keywords"])

                st.write("**Matched keywords**")
                st.write(row["Matched keywords"])

                st.write("**Expected source**")
                st.write(row["Source expected"])

        st.download_button(
            "Download evaluation results (CSV)",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="graphrag_evaluation.csv",
            mime="text/csv",
        )

with graph_tab:
    from neo4j import GraphDatabase

    st.subheader("Knowledge Graph Explorer")
    st.write(
        "Explore entities and factual relationships extracted "
        "from your company documents."
    )

    if st.button("Load Knowledge Graph", type="primary"):
        uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        user = os.getenv("NEO4J_USER", "neo4j")
        password = os.getenv("NEO4J_PASSWORD")

        if not password:
            st.error("NEO4J_PASSWORD is missing from your .env file.")
        else:
            driver = None

            try:
                driver = GraphDatabase.driver(
                    uri,
                    auth=(user, password),
                )

                with driver.session(database="neo4j") as session:
                    records = list(session.run("""
                        MATCH (s:Entity)-[:SUBJECT_OF]->(f:Fact)
                              -[:OBJECT_OF]->(t:Entity)
                        RETURN s.name AS source,
                               f.relation AS relation,
                               t.name AS target
                        ORDER BY source, relation, target
                        LIMIT 100
                    """))

                if not records:
                    st.warning(
                        "No connected facts were found. "
                        "Index documents and run graph extraction first."
                    )
                else:
                    import json

                    lines = [
                        "digraph KnowledgeGraph {",
                        '  rankdir="LR";',
                        '  graph [bgcolor="transparent", pad="0.3"];',
                        '  node [shape="box", style="rounded,filled", '
                        'fillcolor="#E8F0FE", color="#6685B8", '
                        'fontname="Arial"];',
                        '  edge [fontname="Arial", color="#64748B"];',
                    ]

                    nodes = set()

                    for record in records:
                        source = record["source"]
                        relation = record["relation"]
                        target = record["target"]

                        source_id = "entity_" + source
                        target_id = "entity_" + target

                        if source_id not in nodes:
                            lines.append(
                                f"  {json.dumps(source_id)} "
                                f"[label={json.dumps(source)}];"
                            )
                            nodes.add(source_id)

                        if target_id not in nodes:
                            lines.append(
                                f"  {json.dumps(target_id)} "
                                f"[label={json.dumps(target)}];"
                            )
                            nodes.add(target_id)

                        lines.append(
                            f"  {json.dumps(source_id)} -> "
                            f"{json.dumps(target_id)} "
                            f"[label={json.dumps(relation)}];"
                        )

                    lines.append("}")
                    dot = "\n".join(lines)

                    st.graphviz_chart(dot)

                    st.metric("Extracted facts displayed", len(records))

                    st.subheader("Relationship details")
                    st.dataframe(
                        [
                            {
                                "Subject": r["source"],
                                "Relationship": r["relation"],
                                "Target": r["target"],
                            }
                            for r in records
                        ],
                        use_container_width=True,
                        hide_index=True,
                    )

            except Exception as exc:
                st.error(f"Could not load the graph: {exc}")

            finally:
                if driver is not None:
                    driver.close()

with about_tab:
    st.subheader("About this project")

    st.markdown(
        """
        This project demonstrates **GraphRAG**, combining semantic
        vector search with graph-based retrieval.

        **Technology stack**
        - PostgreSQL and pgvector for document embeddings
        - Neo4j for entities, facts, and relationships
        - Ollama for local language-model inference
        - Sentence Transformers for text embeddings
        - Streamlit for the application interface

        **Workflow**

        Documents → Chunking and embeddings → Knowledge graph →
        Hybrid retrieval → Evidence-grounded answers
        """
    )
