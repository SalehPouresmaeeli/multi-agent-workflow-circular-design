### Tests for src/tools/neo4j_ingestion.py — writing a validated graph to Neo4j (plus a CSV copy).
### A pretend Neo4j connection (tests/fakes.py) records the Cypher queries instead of running them,
### and the CSV copy is written to a temporary folder instead of the real output_data/.

import pandas as pd
import pytest
from src.tools import neo4j_ingestion
from src.tools.neo4j_ingestion import Neo4jGraphIngestionTool, INGESTION_ERROR_PREFIX
from src.tools.pydantic_schemas import GraphNode, GraphEdge, KnowledgeGraphSchema
from fakes import use_fake_neo4j


@pytest.fixture
def graph():
    return KnowledgeGraphSchema(
        nodes=[
            GraphNode(id="non_permanent_joints", label="StructuralFeature", synonyms=["detachable joints"]),
            GraphNode(id="remanufacturing", label="CircularStrategy", properties={"weight": 1.5}),
        ],
        edges=[
            GraphEdge(source="non_permanent_joints", target="remanufacturing", relation="ENABLES_STRATEGY",
                      reference_id="[4]", source_type="Academic paper", relevance="High"),
        ],
    )


@pytest.fixture
def tool():
    return Neo4jGraphIngestionTool(password="not-used", database_name="TestGraph")


@pytest.fixture(autouse=True)
def csv_copy_in_temp_folder(tmp_path, monkeypatch):
    # The tool saves its CSV copy next to the project root, worked out from this module's file path.
    # Pointing that path into tmp_path keeps the real output_data/ folder untouched.
    monkeypatch.setattr(neo4j_ingestion, "__file__", str(tmp_path / "src" / "tools" / "neo4j_ingestion.py"))
    return tmp_path / "output_data"


def test_successful_ingestion_reports_counts(monkeypatch, tool, graph):
    driver = use_fake_neo4j(monkeypatch, neo4j_ingestion)

    result = tool._run(graph_data=graph)

    assert result == "Successfully ingested 2 nodes and 1 edges into TestGraph database."
    assert driver.closed


def test_database_is_created_and_waited_for(monkeypatch, tool, graph):
    driver = use_fake_neo4j(monkeypatch, neo4j_ingestion)

    tool._run(graph_data=graph)

    create = driver.queries[0]
    assert create["database"] == "system"
    assert "CREATE DATABASE TestGraph IF NOT EXISTS WAIT" in create["query"]


def test_nodes_are_merged_with_label_and_synonyms(monkeypatch, tool, graph):
    driver = use_fake_neo4j(monkeypatch, neo4j_ingestion)

    tool._run(graph_data=graph)

    node_queries = driver.queries_containing("MERGE (n:")
    assert len(node_queries) == 2
    assert "MERGE (n:`StructuralFeature` {id: $id})" in node_queries[0]["query"]
    assert node_queries[0]["database"] == "TestGraph"
    assert node_queries[0]["params"] == {"id": "non_permanent_joints", "properties": {}, "synonyms": ["detachable joints"]}
    assert node_queries[1]["params"]["properties"] == {"weight": 1.5}


def test_edges_are_merged_with_their_source_record(monkeypatch, tool, graph):
    driver = use_fake_neo4j(monkeypatch, neo4j_ingestion)

    tool._run(graph_data=graph)

    [edge_query] = driver.queries_containing("MERGE (source)-[r:")
    assert "`ENABLES_STRATEGY`" in edge_query["query"]
    assert edge_query["params"] == {
        "source_id": "non_permanent_joints",
        "target_id": "remanufacturing",
        "reference_id": "[4]",
        "source_type": "Academic paper",
        "relevance": "High",
        "properties": {},
    }


def test_graph_given_as_dict_is_accepted(monkeypatch, tool, graph):
    use_fake_neo4j(monkeypatch, neo4j_ingestion)

    result = tool._run(graph_data=graph.model_dump())

    assert result.startswith("Successfully ingested 2 nodes")


def test_csv_copy_is_saved(monkeypatch, tool, graph, csv_copy_in_temp_folder):
    use_fake_neo4j(monkeypatch, neo4j_ingestion)

    tool._run(graph_data=graph)

    [export_dir] = list(csv_copy_in_temp_folder.glob("graph_*"))
    nodes = pd.read_csv(export_dir / "ingested_nodes.csv", keep_default_na=False)
    edges = pd.read_csv(export_dir / "ingested_edges.csv", keep_default_na=False)
    assert list(nodes["id"]) == ["non_permanent_joints", "remanufacturing"]
    assert list(nodes["synonyms"]) == ["detachable joints", ""]
    assert list(edges.columns) == ["source", "target", "relation", "reference_id", "source_type", "relevance", "properties"]
    assert edges.loc[0, "reference_id"] == "[4]"


def test_database_not_online_writes_nothing(monkeypatch, tool, graph):
    driver = use_fake_neo4j(monkeypatch, neo4j_ingestion, responses={
        "CREATE DATABASE": [{"success": False, "message": "still starting"}],
    })

    result = tool._run(graph_data=graph)

    assert result.startswith(INGESTION_ERROR_PREFIX)
    assert "still starting" in result
    assert driver.queries_containing("MERGE") == []
    assert driver.closed


def test_failed_write_is_reported_not_raised(monkeypatch, tool, graph):
    driver = use_fake_neo4j(monkeypatch, neo4j_ingestion, fail_on="MERGE (source)")

    result = tool._run(graph_data=graph)

    assert result.startswith(f"{INGESTION_ERROR_PREFIX} Database write failed")
    assert driver.closed
