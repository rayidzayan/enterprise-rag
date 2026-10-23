import os

from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv()


def check_neo4j():
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    username = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD")

    if not password:
        raise RuntimeError("NEO4J_PASSWORD is missing from .env")

    with GraphDatabase.driver(
        uri, auth=(username, password)
    ) as driver:
        driver.verify_connectivity()

        result = driver.execute_query(
            "RETURN 1 AS result",
            database_="neo4j",
        )

        if result.records[0]["result"] != 1:
            raise RuntimeError("Neo4j verification query failed.")

    return {"uri": uri, "status": "Connected"}
