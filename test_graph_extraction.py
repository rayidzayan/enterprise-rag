from src.graph_ingest import extract_graph_data

sample = """
Full-time employees receive 20 days of paid annual leave per calendar year.
Employees must submit leave requests to their manager at least five working days in advance.
"""

try:
    result = extract_graph_data(sample)
    print("ENTITIES:")
    for item in result["entities"]:
        print(item)

    print("\nRELATIONSHIPS:")
    for item in result["relationships"]:
        print(item)

except Exception as exc:
    print("EXTRACTION ERROR:")
    print(type(exc).__name__, str(exc))
