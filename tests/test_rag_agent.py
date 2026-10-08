### Tests for src/agents/rag_agent.py — reading the graph's schema, caching it, and setting up the GraphRAG agent.
### A pretend Neo4j connection supplies the schema, and the CrewAI crew is replaced, so no LLM or database is used.

import pytest
from src.agents import rag_agent
from fakes import use_fake_neo4j, fake_crew_class


SCHEMA_RESPONSES = {
    "nodeTypeProperties": [
        {"label": "Material", "props": ["id", "synonyms"]},
        {"label": "CircularStrategy", "props": []},
    ],
    "RETURN DISTINCT labels": [
        {"source": "CircularStrategy", "rel": "PART_OF_ACTIVITY", "target": "LifecycleActivity"},
    ],
    "relTypeProperties": [
        {"relType": ":`PART_OF_ACTIVITY`", "props": ["reference_id", "source_type", "relevance"]},
    ],
    "reference_id IS NOT NULL": [
        {"source_types": ["Academic paper"], "relevance_levels": ["High", "Medium"],
         "sample_references": ["[4]", "[8]"], "reference_count": 2},
    ],
}


# --- Reading the schema from the database ---

def test_schema_lists_labels_relationships_and_source_values(monkeypatch):
    driver = use_fake_neo4j(monkeypatch, rag_agent, responses=SCHEMA_RESPONSES)

    schema = rag_agent.extract_dynamic_schema()

    assert "- Material (properties: id, synonyms)" in schema
    assert "- CircularStrategy (properties: none)" in schema
    assert "- (CircularStrategy)-[PART_OF_ACTIVITY]->(LifecycleActivity)" in schema
    assert "- PART_OF_ACTIVITY (properties: reference_id, source_type, relevance)" in schema   # :`...` stripped
    assert "- relevance values: ['High', 'Medium']" in schema
    assert "- reference_id: 2 distinct values" in schema
    assert driver.closed


def test_schema_without_source_records_has_no_lineage_section(monkeypatch):
    responses = {k: v for k, v in SCHEMA_RESPONSES.items() if k != "reference_id IS NOT NULL"}
    use_fake_neo4j(monkeypatch, rag_agent, responses=responses)

    schema = rag_agent.extract_dynamic_schema()

    assert "Literature Lineage Values" not in schema


# --- Schema cache ---

@pytest.fixture
def schema_reads(monkeypatch):
    """Starts each test with an empty cache and counts how often the database would be read."""
    reads = []
    monkeypatch.setattr(rag_agent, "_CACHED_SCHEMA", None)
    monkeypatch.setattr(rag_agent, "extract_dynamic_schema", lambda: reads.append(1) or f"schema #{len(reads)}")
    return reads


def test_schema_is_read_once_then_cached(schema_reads):
    first = rag_agent.get_current_schema()
    second = rag_agent.get_current_schema()

    assert first == second == "schema #1"
    assert len(schema_reads) == 1


def test_forced_refresh_reads_again(schema_reads):
    rag_agent.get_current_schema()

    assert rag_agent.get_current_schema(force_refresh=True) == "schema #2"


def test_invalidated_cache_is_read_again(schema_reads):
    rag_agent.get_current_schema()
    rag_agent.invalidate_schema_cache()

    assert rag_agent.get_current_schema() == "schema #2"


# --- Running the GraphRAG agent ---

def test_question_and_schema_reach_the_agent(monkeypatch):
    FakeCrew, crews = fake_crew_class(raw_result="Use screws — [4] (Academic paper, High relevance)")
    monkeypatch.setattr(rag_agent, "Crew", FakeCrew)
    monkeypatch.setattr(rag_agent, "get_current_schema", lambda: "SCHEMA FOR TEST")
    events = []

    answer = rag_agent.run_rag_pipeline("How should the casing be joined?", event_callback=events.append)

    [crew] = crews
    assert answer == "Use screws — [4] (Academic paper, High relevance)"
    assert crew.inputs == {"user_question": "How should the casing be joined?"}
    assert "SCHEMA FOR TEST" in crew.agents[0].backstory
    assert crew.agents[0].tools[0].database_name == "KnowledgeGraphTEST"      # from the test config in conftest.py
    assert events[0]["type"] == "agent_start"
    assert events[-1] == {"type": "result", "answer": answer}
