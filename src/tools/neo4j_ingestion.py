import sys
from pathlib import Path
# Fix for Windows console UnicodeEncodeError
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
####################################

from crewai.tools import BaseTool
from neo4j import GraphDatabase
from pydantic import Field, BaseModel
from typing import Type
from src.tools.pydantic_schemas import KnowledgeGraphSchema    # Import the schema to validate inputs

# Failures are returned as text (an agent-facing tool must not raise), so callers detect them by this prefix.
INGESTION_ERROR_PREFIX = "ERROR:"
# Seconds to wait for a newly created database to come online before giving up (Neo4j's own default is 300).
DATABASE_WAIT_SECONDS = 30

class Neo4jToolInput(BaseModel):
    """Input schema for the Neo4j Tool."""
    graph_data: KnowledgeGraphSchema = Field(
        ..., 
        description="The canonicalised knowledge graph data to be ingested."
    )

class Neo4jGraphIngestionTool(BaseTool):
    name: str = "Neo4j Graph Ingestion Tool"
    description: str = "Ingests canonicalised nodes and edges into a local Neo4j database using Cypher MERGE queries."
    args_schema: Type[BaseModel] = Neo4jToolInput   # defining what the AI must hand to the tool (a KnowledgeGraphSchema object)
    
    # Connection details (Default local Neo4j Desktop credentials)
    uri: str = "neo4j://127.0.0.1:7687"
    username: str = "neo4j"
    password: str = "password"            # UPDATE THIS to local database password
    database_name: str = "KnowledgeGraph"     # Update with the exact name from SHOW DATABASES. It is the default name

    def _run(self, graph_data: KnowledgeGraphSchema) -> str:
        """Executes the database transaction."""
        
        # If CrewAI passes a dict instead of a Pydantic object, parse it.
        if isinstance(graph_data, dict):
            graph_data = KnowledgeGraphSchema(**graph_data)
            
        # --- EXPORT GRAPH DATA TO CSV BEFORE WRITING TO NEO4J ---
        try:
            import pandas as pd
            from pathlib import Path
            import datetime
            
            # Create exported_data directory with timestamp in the root
            root_dir = Path(__file__).resolve().parent.parent.parent
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            export_dir = root_dir / "output_data" / f"graph_{timestamp}"
            export_dir.mkdir(parents=True, exist_ok=True)
            
            # Prepare data for CSVs
            nodes_data = [
                {
                    "id": n.id, 
                    "label": n.label, 
                    "synonyms": ", ".join(n.synonyms), 
                    "properties": str(n.properties) if n.properties else ""
                } 
                for n in graph_data.nodes
            ]
            
            edges_data = [
                {
                    "source": e.source, 
                    "target": e.target, 
                    "relation": e.relation,
                    "reference_id": e.reference_id,
                    "source_type": e.source_type,
                    "relevance": e.relevance,
                    "properties": str(e.properties) if e.properties else ""
                } 
                for e in graph_data.edges
            ]
            
            if nodes_data:
                pd.DataFrame(nodes_data).to_csv(export_dir / "ingested_nodes.csv", index=False)
            if edges_data:
                pd.DataFrame(edges_data).to_csv(export_dir / "ingested_edges.csv", index=False)
        except Exception as csv_err:
            print(f"Warning: Failed to save CSV exports: {csv_err}")
        # ---------------------
        
        # 1. Establish the connection to the database
        driver = GraphDatabase.driver(self.uri, auth=(self.username, self.password))
        
        try:
            # Tell the server to create the target database if it is missing
            # Open a session strictly with the 'system' database to run server-level commands
            with driver.session(database="system") as admin_session:
                # WAIT holds the command until the database is actually online. Without it Neo4j creates the
                # database in the background and returns immediately, so the writes below hit a database that
                # is not ready yet and the whole ingestion is lost. The wait is capped at DATABASE_WAIT_SECONDS;
                # on timeout Neo4j returns success=false (it does not raise), so the rows are checked below.
                creation = admin_session.run(
                    f"CREATE DATABASE {self.database_name} IF NOT EXISTS WAIT {DATABASE_WAIT_SECONDS} SECONDS"
                ).data()   # warning: only for Neo4j desktop
                not_ready = [row for row in creation if not row.get("success", True)]
                if not_ready:
                    return (f"{INGESTION_ERROR_PREFIX} Database '{self.database_name}' was not online within "
                            f"{DATABASE_WAIT_SECONDS} seconds ({not_ready[0].get('message', 'no message given')}). "
                            f"Nothing was written — the database is still being created, so re-run once it is online.")

            # 2. Open a session to run queries
            with driver.session(database=self.database_name) as session:
                
                # --- PROCESS NODES ---
                for node in graph_data.nodes:
                    # the label is wrapped in backquotes to safely handle standardisation.
                    # explicitly save the synonyms array as a node property here! It adds new synonyms without overwritten previous list of synonyms. 
                    query = f"""
                    MERGE (n:`{node.label}` {{id: $id}})
                    SET n += $properties
                    WITH n, $synonyms AS new_synonyms
                    SET n.synonyms = [x IN new_synonyms WHERE NOT x IN coalesce(n.synonyms, [])] + coalesce(n.synonyms, [])
                    """
                    session.run(query, id=node.id, properties=node.properties, synonyms=node.synonyms)
                
                # --- PROCESS EDGES ---
                for edge in graph_data.edges:
                    # MATCH the existing nodes first, then MERGE the relationship between them.
                    # The lineage (reference_id, source_type, relevance) is part of the MERGE key, so each distinct source
                    # gets its own relationship instead of overwriting the lineage of another source for the same triple.
                    query = f"""
                    MATCH (source {{id: $source_id}})
                    MATCH (target {{id: $target_id}})
                    MERGE (source)-[r:`{edge.relation}` {{reference_id: $reference_id, source_type: $source_type, relevance: $relevance}}]->(target)
                    SET r += $properties
                    """
                    session.run(query,
                                source_id=edge.source,
                                target_id=edge.target,
                                reference_id=edge.reference_id,
                                source_type=edge.source_type,
                                relevance=edge.relevance,
                                properties=edge.properties)
            
            return f"Successfully ingested {len(graph_data.nodes)} nodes and {len(graph_data.edges)} edges into {self.database_name} database."
            
        except Exception as e:
            # Some nodes or edges may already have been written before the failure.
            return f"{INGESTION_ERROR_PREFIX} Database write failed: {str(e)}"
            
        finally:
            # Always close the driver connection to prevent memory leaks
            driver.close()

# Local test block of Graph Ingestion Tool to ensure the tool connects and writes
if __name__ == "__main__":
    # Use the absolute path to match the top-level import
    from src.tools.pydantic_schemas import GraphNode, GraphEdge, KnowledgeGraphSchema
    
    # Example1: a dummy Pydantic object manually, including our new synonyms field
    mock_data1 = KnowledgeGraphSchema(
        nodes=[
            GraphNode(id="test_sensor", label="Component", synonyms=["test sensor", "sensor module"], properties={"weight": 1.5}),
            GraphNode(id="waterproof_casing", label="Component", synonyms=["casing"], properties={})
        ],
        edges=[
            GraphEdge(source="waterproof_casing", target="test_sensor", relation="PROTECTS", properties={"confidence": 0.9})
        ]
    )

    # Example2: a dummy Pydantic object manually, including our new synonyms field
    mock_data2 = KnowledgeGraphSchema(
        nodes=[
            GraphNode(id="sealing_component", label="Component", synonyms=['Rubber Seals', 'silicone gasket', 'O-ring seal'], properties={"weight": 1.5}),
            GraphNode(id="ip_67_certification", label="Certification", synonyms=['ip 67 waterproof rating', 'IP-67'], properties={}),
            GraphNode(id="environmental_condition", label="Condition", synonyms=['High Humidity', 'outdoor rain exposure'], properties={"weight": 1.0}),
            GraphNode(id="moisture_ingress", label="Risk", synonyms=['moisture ingress'], properties={}),
            GraphNode(id="printed_circuit_board", label="Electronic", synonyms=['main circuit board', 'electrical PCB'], properties={"weight": 0.5})
        ],
        edges=[
            GraphEdge(source="sealing_component", target="moisture_ingress", relation="PREVENTS", properties={"confidence": 0.9}),
            GraphEdge(source="sealing_component", target="ip_67_certification", relation="REQUIRES", properties={"confidence": 0.8}),
            GraphEdge(source="ip_67_certification", target="printed_circuit_board", relation="PROTECTS", properties={"confidence": 0.9}),
            GraphEdge(source="sealing_component", target="environmental_condition", relation="SEALS_AGAINST", properties={"confidence": 0.8}),
            GraphEdge(source="printed_circuit_board", target="environmental_condition", relation="MUST_SURVIVE", properties={"confidence": 1.0})
        ]
    )

    from src.tools.db_config import load_db_config
    db_config = load_db_config()
    tool = Neo4jGraphIngestionTool(uri=db_config["uri"], username=db_config["username"], password=db_config["password"],
                                   database_name="KnowledgeGraphTEST")    # writes to a separate test database
    mock_data = mock_data2                                # choose the mock_data for the test
    result = tool._run(graph_data=mock_data)
    print(result)

    """ To view graph in Neo4j desktop  use following query
    MATCH (n)-[r]->(m) 
    RETURN n, r, m      
    """ 