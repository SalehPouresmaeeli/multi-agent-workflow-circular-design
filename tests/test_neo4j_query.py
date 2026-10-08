### Tests for the read-only safety check in src/tools/neo4j_query.py.
### The RAG agent writes its own Cypher queries, so the tool must refuse anything that could change the graph.
### No database is needed: blocked queries are refused before connecting, and for allowed queries the
### connection is replaced by a stand-in that only records that the check let the query through.

import pytest
from src.tools import neo4j_query
from src.tools.neo4j_query import Neo4jGraphQueryTool


class ReachedDatabase(Exception):
    """Raised by the stand-in connection: the query passed the safety check."""


@pytest.fixture
def tool(monkeypatch):
    def fake_driver(*args, **kwargs):
        raise ReachedDatabase()
    monkeypatch.setattr(neo4j_query.GraphDatabase, "driver", fake_driver)
    return Neo4jGraphQueryTool(password="not-used")


@pytest.mark.parametrize("query", [
    "CREATE (n:Material {id: 'test'})",
    "MATCH (n) DETACH DELETE n",
    "MATCH (n {id: 'casing'}) SET n.weight = 2",
    "match (n) set n.weight = 2",                      # lower case
    "MERGE (n:Tool {id: 'screwdriver'})",
    "MATCH (n) REMOVE n.synonyms",
    "DROP INDEX my_index",
    "MATCH (n) WHERE n.id = 'casing' DELETE n",        # write keyword after a quoted value
])
def test_write_queries_are_blocked(tool, query):
    result = tool._run(cypher_query=query)

    assert result.startswith("Error:")
    assert "READ-ONLY" in result


@pytest.mark.parametrize("query", [
    "MATCH (n) RETURN n LIMIT 5",
    "MATCH (n) WHERE n.id = 'design_to_create_value' RETURN n",      # keyword inside a name
    "MATCH (n) WHERE 'remove adhesive' IN n.synonyms RETURN n",      # keyword inside quoted text
    'MATCH (n) WHERE n.id = "set screw" RETURN n',                   # double quotes
    "MATCH (n)-[:`SET_BY`]->(m) RETURN m",                           # backticked name
    "MATCH (n:Material) RETURN n.id, n.synonyms ORDER BY n.id",
])
def test_read_queries_are_allowed(tool, query):
    with pytest.raises(ReachedDatabase):
        tool._run(cypher_query=query)
