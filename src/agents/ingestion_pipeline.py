import sys
from pathlib import Path
# Fix for Windows console UnicodeEncodeError
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
####################################

import json
from neo4j import GraphDatabase
from crewai import Crew
from src.agents.knowledge_extraction_agent import extraction_agent, guidelines_extraction_task
from src.agents.canonicalisation_agent import canonicalisation_agent, canonicalise_task
from src.tools.neo4j_ingestion import Neo4jGraphIngestionTool, INGESTION_ERROR_PREFIX
from src.agents.rag_agent import invalidate_schema_cache
from src.tools.db_config import load_db_config

### Database Connection details (from db_config.json — see src/tools/db_config.py)
db_config = load_db_config()
_uri: str = db_config["uri"]
_username: str = db_config["username"]
_password: str = db_config["password"]
_database_name: str = db_config["database_name"]

def _fetch_existing_nodes_text(uri: str, username: str, password: str, database_name: str) -> str:
    """
    Queries Neo4j for all existing nodes (id, label, synonyms) and returns a compact
    text block suitable for injection into the canonicalisation prompt. Returns a
    placeholder string when the graph is empty or the database is unreachable.
    """
    try:
        driver = GraphDatabase.driver(uri, auth=(username, password))
        try:
            with driver.session(database=database_name) as session:
                records = session.execute_read(
                    lambda tx: tx.run(
                        "MATCH (n) RETURN n.id AS id, labels(n)[0] AS label, "
                        "coalesce(n.synonyms, []) AS synonyms ORDER BY label, id"
                    ).data()
                )
        finally:
            driver.close()

        if not records:
            return "The knowledge graph is currently empty — this is the first ingestion run. No cross-batch resolution is needed."

        lines = [f"Total existing nodes: {len(records)}\n"]
        for r in records:
            syn_text = ", ".join(f'"{s}"' for s in (r["synonyms"] or []))
            lines.append(f"  id: {r['id']} | label: {r['label']} | synonyms: [{syn_text}]")
        return "\n".join(lines)

    except Exception as e:
        return f"Could not reach Neo4j to fetch existing nodes ({e}). Proceed with intra-batch resolution only."


_AGENT_ROLE_MAP = {
    "Lead Systems Engineering NLP Architect": "Extraction Agent",
    "Graph Entity Resolution Specialist": "Canonicalisation Agent",
}

#################################
# Function to run the ingestion pipeline
def run_ingestion_pipeline(excel_records: list, column_headers: list, event_callback=None) -> str:
    """
    Orchestrates the full GraphRAG ingestion pipeline:
    1. Extraction agent       — parses raw records into a structured knowledge graph.
    2. Canonicalisation agent — resolves duplicates, standardises IDs and labels.
    3. Ingestion              — writes the validated schema to Neo4j directly (a typed payload
       hand-off with no decision to make, so it skips the LLM rather than risking a transcription
       error through another tool call).
    """
    def push(event):
        if event_callback:
            try:
                event_callback(event)
            except Exception:
                pass

    push({"type": "pipeline_start", "message": f"Pipeline started: {len(excel_records)} rows loaded. Running 3 steps..."})

    completed_tasks = [0]

    def _step_cb(step_output):
        try:
            if hasattr(step_output, 'tool'):
                push({"type": "step", "message": f"Using tool: {step_output.tool}"})
            elif hasattr(step_output, 'return_values'):
                push({"type": "step", "message": "Evaluating result..."})
        except Exception:
            pass

    def _task_cb(task_output):
        try:
            completed_tasks[0] += 1
            role = getattr(task_output, 'agent', '')
            name = _AGENT_ROLE_MAP.get(role, role or 'Agent')
            push({"type": "task_done", "message": f"[{completed_tasks[0]}/3] {name}: Task complete."})
        except Exception:
            completed_tasks[0] += 1
            push({"type": "task_done", "message": f"[{completed_tasks[0]}/3] Task complete."})

    # 1. Instantiate the database write tool
    ingestion_tool = Neo4jGraphIngestionTool(uri=_uri, username=_username, password=_password, database_name=_database_name)

    # 2. Format the raw input data for the first agent
    raw_data_payload = json.dumps({
        "headers": column_headers,
        "rows": excel_records
    }, indent=2)

    # 3. Snapshot the existing graph so the canonicalisation agent can resolve across batches
    push({"type": "step", "message": "Fetching existing graph nodes for cross-batch resolution..."})
    existing_nodes_text = _fetch_existing_nodes_text(_uri, _username, _password, _database_name)

    # 4. Assemble the extraction -> canonicalisation crew (the only two steps that need an LLM)
    pipeline_crew = Crew(
        agents=[extraction_agent, canonicalisation_agent],
        tasks=[guidelines_extraction_task, canonicalise_task],
        verbose=True,
        step_callback=_step_cb,
        task_callback=_task_cb
    )

    print("Executing GraphRAG Ingestion Pipeline (extraction → canonicalisation → ingestion)...\n")

    # 5. Kickoff the LLM crew — existing_nodes_text is interpolated into canonicalise_task's prompt
    pipeline_crew.kickoff(inputs={"raw_extraction": raw_data_payload, "existing_nodes": existing_nodes_text})
    validated_graph = canonicalise_task.output.pydantic

    # 6. Ingest the validated graph directly — no LLM involved, it's a straight typed hand-off
    ingestion_result = ingestion_tool._run(graph_data=validated_graph)

    # The tool reports database failures as text rather than raising, so a failed write would otherwise be
    # announced as "Pipeline complete". Surface it as an error instead — nothing reached the graph.
    # Raising (rather than pushing an error here) lets the caller decide: the web app turns it into a red
    # error line for the user, and a script gets a real exception instead of a success-looking string.
    if ingestion_result.startswith(INGESTION_ERROR_PREFIX):
        raise RuntimeError(f"Ingestion failed, so the graph was not updated. {ingestion_result}")

    completed_tasks[0] += 1
    push({"type": "task_done", "message": f"[{completed_tasks[0]}/3] Ingestion: Task complete."})

    # Invalidate the Graph RAG schema cache so the next chat question sees the newly uploaded data
    invalidate_schema_cache()

    push({"type": "done", "message": f"Pipeline complete. {ingestion_result[:120] if ingestion_result else 'Ingestion finished.'}"})
    return ingestion_result