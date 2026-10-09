### Integration tests: these use a REAL LLM and your REAL Neo4j database.
###
### Every agent uses one Gemini model through Google AI Studio (GEMINI_API_KEY in .env), set up in
### tests/conftest.py. The default is gemini/gemini-3.5-flash-lite; choose another with the environment
### variable INTEGRATION_TEST_MODEL. Your llm_config.json is not used or changed.
###
### Skipped by default, and never run on GitHub Actions. Run them on purpose, with Neo4j running,
### from the project root:
###     python -m pytest -m integration -v
###
### Safety:
###   - Everything is written to a separate database, INTEGRATION_DB, which is emptied before and after
###     each test. Your own knowledge graph (database_name in db_config.json) is never touched.
###   - The CSV copy that ingestion normally saves to output_data/ goes to a temporary folder instead.
###   - Tests are skipped if there is no GEMINI_API_KEY in .env or no db_config.json.

import os
import pytest
from neo4j import GraphDatabase

from src.tools import db_config, neo4j_ingestion
from src.tools.llm_config import get_llm, get_model   # importing llm_config also loads .env

pytestmark = pytest.mark.integration

INTEGRATION_DB = "integrationtest"

requires_llm = pytest.mark.skipif(
    not os.getenv("GEMINI_API_KEY"),
    reason="GEMINI_API_KEY is not set in .env",
)
requires_neo4j = pytest.mark.skipif(
    not db_config._CONFIG_PATH.exists(),
    reason="db_config.json not found (copy db_config.example.json and fill it in)",
)

SAMPLE_GUIDELINE = {
    "Lifecycle activity": "Design to recover value",
    "DfReX": "Remanufacturing",
    "Product requirement": "Select non-permanent joints",
    "Design criteria": "Use non-permanent joint types rather than adhesives",
    "Ref": "[4]",
    "Source Type": "Academic paper",
    "Relevance": "High",
}


# --- Fixtures ---

@pytest.fixture
def neo4j_test_db():
    """
    Connects with the details in db_config.json, makes sure INTEGRATION_DB exists, and empties it
    before and after the test. Yields the connection details with database_name set to INTEGRATION_DB.
    """
    config = db_config.load_db_config()
    assert config["database_name"].lower() != INTEGRATION_DB, \
        f"db_config.json points at '{INTEGRATION_DB}', the database these tests empty — choose another name"

    driver = GraphDatabase.driver(config["uri"], auth=(config["username"], config["password"]))
    try:
        driver.verify_connectivity()
    except Exception as e:
        driver.close()
        pytest.fail(f"Could not connect to Neo4j at {config['uri']} — is it running? ({e})")

    with driver.session(database="system") as system:
        system.run(f"CREATE DATABASE {INTEGRATION_DB} IF NOT EXISTS WAIT 30 SECONDS")

    def empty_test_db():
        with driver.session(database=INTEGRATION_DB) as session:
            session.run("MATCH (n) DETACH DELETE n")

    empty_test_db()
    yield {**config, "database_name": INTEGRATION_DB, "driver": driver}
    empty_test_db()
    driver.close()


@pytest.fixture
def csv_copy_in_temp_folder(tmp_path, monkeypatch):
    # Ingestion saves a CSV copy next to the project root; point that into tmp_path instead of output_data/
    monkeypatch.setattr(neo4j_ingestion, "__file__", str(tmp_path / "src" / "tools" / "neo4j_ingestion.py"))


def count(driver, query):
    with driver.session(database=INTEGRATION_DB) as session:
        return session.run(query).single()[0]


# --- Connection checks ---

def test_every_agent_uses_the_integration_model():
    # conftest.py replaces llm_config.json for integration runs, so no agent falls back to a paid model
    model = get_model("rag_agent")

    assert model.startswith("gemini/")
    assert get_model("extraction_agent") == get_model("canonicalisation_agent") == get_model("mentor_agent") == model


@requires_llm
def test_gemini_answers():
    llm = get_llm("rag_agent", temperature=0)

    reply = llm.call("Reply with exactly: OK")

    assert "OK" in reply.upper()


@requires_neo4j
def test_neo4j_is_reachable(neo4j_test_db):
    with neo4j_test_db["driver"].session(database=INTEGRATION_DB) as session:
        assert session.run("RETURN 1 AS ok").single()["ok"] == 1


# --- Neo4j tools against a real database (no LLM) ---

@requires_neo4j
def test_ingested_graph_can_be_queried(neo4j_test_db, csv_copy_in_temp_folder):
    from src.tools.neo4j_ingestion import Neo4jGraphIngestionTool
    from src.tools.neo4j_query import Neo4jGraphQueryTool
    from src.tools.pydantic_schemas import GraphNode, GraphEdge, KnowledgeGraphSchema

    connection = {k: neo4j_test_db[k] for k in ("uri", "username", "password", "database_name")}
    graph = KnowledgeGraphSchema(
        nodes=[
            GraphNode(id="non_permanent_joints", label="StructuralFeature", synonyms=["detachable joints"]),
            GraphNode(id="remanufacturing", label="CircularStrategy"),
        ],
        edges=[GraphEdge(source="non_permanent_joints", target="remanufacturing", relation="ENABLES_STRATEGY",
                         reference_id="[4]", source_type="Academic paper", relevance="High")],
    )

    result = Neo4jGraphIngestionTool(**connection)._run(graph_data=graph)
    answer = Neo4jGraphQueryTool(**connection)._run(
        cypher_query="MATCH (a)-[r:ENABLES_STRATEGY]->(b) RETURN a.id AS source, b.id AS target, r.reference_id AS ref"
    )

    assert result.startswith("Successfully ingested 2 nodes and 1 edges")
    assert '"source": "non_permanent_joints"' in answer
    assert '"ref": "[4]"' in answer


# --- Agents with the real LLM ---

@requires_llm
def test_mentor_explains_a_real_result():
    from src.agents.mentor_agent import generate_mentor_message
    from src.tools.math_engine import calculate_winning_rule

    results = calculate_winning_rule(user_durability=2.5, user_disassembly=4.0)    # screws win, 25.0 vs 16.5

    message = generate_mentor_message(results, [2.5, 4.0])

    assert len(message) > 20
    assert "screw" in message.lower()


@requires_llm
@requires_neo4j
def test_guideline_to_cited_answer_end_to_end(neo4j_test_db, csv_copy_in_temp_folder, monkeypatch):
    """One guideline goes through extraction, canonicalisation and Neo4j; then the RAG agent answers from it."""
    from src.agents import ingestion_pipeline, rag_agent

    # Point the pipeline and the RAG agent at the integration database instead of your real graph
    monkeypatch.setattr(ingestion_pipeline, "_database_name", INTEGRATION_DB)
    monkeypatch.setattr(rag_agent, "_database_name", INTEGRATION_DB)
    rag_agent.invalidate_schema_cache()

    result = ingestion_pipeline.run_ingestion_pipeline([SAMPLE_GUIDELINE], list(SAMPLE_GUIDELINE))

    driver = neo4j_test_db["driver"]
    assert result.startswith("Successfully ingested")
    assert count(driver, "MATCH (n) RETURN count(n)") > 0
    assert count(driver, "MATCH ()-[r]->() WHERE r.reference_id = '[4]' RETURN count(r)") > 0

    answer = rag_agent.run_rag_pipeline("Which design guidelines come from reference [4]?")

    assert "[4]" in answer
    rag_agent.invalidate_schema_cache()     # don't leave the integration database's schema in the cache
