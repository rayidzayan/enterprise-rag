from src.db import check_postgres
from src.graph import check_neo4j

if __name__ == "__main__":
    print("Checking PostgreSQL...")
    print(check_postgres())

    print("\nChecking Neo4j...")
    print(check_neo4j())

    print("\nBoth database connections succeeded!")
