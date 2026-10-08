### Tests for src/agents/ingestion_pipeline.py — extraction -> canonicalisation -> Neo4j, step by step.
### The CrewAI crew, the Neo4j connection and the ingestion tool are all replaced by stand-ins, so the
### tests check how the pipeline connects the steps without running any agent or touching a database.

import json
from types import SimpleNamespace
import pytest
from src.agents import ingestion_pipeline
from src.tools.neo4j_ingestion import INGESTION_ERROR_PREFIX
from src.tools.pydantic_schemas import GraphNode, KnowledgeGraphSchema
from fakes import use_fake_neo4j, fake_crew_class


# --- Reading the nodes already in the graph (given to the canonicalisation agent) ---

def test_existing_nodes_are_listed_with_synonyms(monkeypatch):
    use_fake_neo4j(monkeypatch, ingestion_pipeline, responses={"MATCH (n)": [
        {"id": "remanufacturing", "label": "CircularStrategy", "synonyms": []},
        {"id": "fasteners", "label": "ProductComponent", "synonyms": ["screws", "bolts"]},
    ]})

    text = ingestion_pipeline._fetch_existing_nodes_text("uri", "user", "password", "TestGraph")

    assert text.startswith("Total existing nodes: 2")
    assert 'id: remanufacturing | label: CircularStrategy | synonyms: []' in text
    assert 'id: fasteners | label: ProductComponent | synonyms: ["screws", "bolts"]' in text


def test_empty_graph_is_reported_as_first_run(monkeypatch):
    use_fake_neo4j(monkeypatch, ingestion_pipeline)

    text = ingestion_pipeline._fetch_existing_nodes_text("uri", "user", "password", "TestGraph")

    assert "currently empty" in text


def test_unreachable_database_does_not_stop_the_pipeline(monkeypatch):
    use_fake_neo4j(monkeypatch, ingestion_pipeline, fail_on="MATCH (n)")

    text = ingestion_pipeline._fetch_existing_nodes_text("uri", "user", "password", "TestGraph")

    assert text.startswith("Could not reach Neo4j")


# --- The full pipeline ---

@pytest.fixture
def pipeline(monkeypatch):
    """Replaces every outside dependency of run_ingestion_pipeline and records how they were used."""
    validated_graph = KnowledgeGraphSchema(nodes=[GraphNode(id="repair", label="CircularStrategy")], edges=[])
    record = SimpleNamespace(tool_kwargs=None, tool_input=None, tool_result="Successfully ingested 1 nodes and 0 edges.",
                             cache_cleared=False, events=[], graph=validated_graph)

    FakeCrew, crews = fake_crew_class()
    record.crews = crews
    monkeypatch.setattr(ingestion_pipeline, "Crew", FakeCrew)

    # The pipeline reads the canonicalisation task's validated output after the crew finishes
    monkeypatch.setattr(ingestion_pipeline, "canonicalise_task", SimpleNamespace(output=SimpleNamespace(pydantic=validated_graph)))
    monkeypatch.setattr(ingestion_pipeline, "_fetch_existing_nodes_text", lambda *args: "EXISTING NODES TEXT")

    class FakeIngestionTool:
        def __init__(self, **kwargs):
            record.tool_kwargs = kwargs

        def _run(self, graph_data):
            record.tool_input = graph_data
            return record.tool_result
    monkeypatch.setattr(ingestion_pipeline, "Neo4jGraphIngestionTool", FakeIngestionTool)

    def fake_invalidate():
        record.cache_cleared = True
    monkeypatch.setattr(ingestion_pipeline, "invalidate_schema_cache", fake_invalidate)

    return record


RECORDS = [{"DfReX": "Repair", "Ref": "[8]"}]
HEADERS = ["DfReX", "Ref"]


def test_crew_gets_uploaded_rows_and_existing_nodes(pipeline):
    ingestion_pipeline.run_ingestion_pipeline(RECORDS, HEADERS)

    [crew] = pipeline.crews
    assert json.loads(crew.inputs["raw_extraction"]) == {"headers": HEADERS, "rows": RECORDS}
    assert crew.inputs["existing_nodes"] == "EXISTING NODES TEXT"
    assert len(crew.agents) == 2 and len(crew.tasks) == 2     # extraction, then canonicalisation


def test_validated_graph_is_written_to_configured_database(pipeline):
    result = ingestion_pipeline.run_ingestion_pipeline(RECORDS, HEADERS)

    assert result == "Successfully ingested 1 nodes and 0 edges."
    assert pipeline.tool_input is pipeline.graph
    assert pipeline.tool_kwargs["database_name"] == "KnowledgeGraphTEST"      # from the test config in conftest.py
    assert pipeline.cache_cleared                                             # next RAG question sees the new data


def test_progress_events_are_reported(pipeline):
    events = []

    ingestion_pipeline.run_ingestion_pipeline(RECORDS, HEADERS, event_callback=events.append)

    assert events[0]["type"] == "pipeline_start"
    assert "1 rows loaded" in events[0]["message"]
    assert {"type": "task_done", "message": "[1/3] Ingestion: Task complete."} in events
    assert events[-1]["type"] == "done"


def test_failed_write_raises_and_keeps_cache(pipeline):
    pipeline.tool_result = f"{INGESTION_ERROR_PREFIX} Database write failed: boom"

    with pytest.raises(RuntimeError, match="graph was not updated"):
        ingestion_pipeline.run_ingestion_pipeline(RECORDS, HEADERS)

    assert not pipeline.cache_cleared


def test_broken_progress_callback_does_not_stop_the_pipeline(pipeline):
    def broken_callback(event):
        raise ValueError("browser disconnected")

    result = ingestion_pipeline.run_ingestion_pipeline(RECORDS, HEADERS, event_callback=broken_callback)

    assert result.startswith("Successfully ingested")
