# To find correct path of files
import sys
from pathlib import Path
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
####################################

from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from neo4j import GraphDatabase
import json
import re

# Write keywords are matched as whole words only, so names that merely contain them (e.g. 'design_to_create_value') are not blocked.
_FORBIDDEN_KEYWORDS = re.compile(r"\b(CREATE|MERGE|SET|DELETE|REMOVE|DROP)\b", re.IGNORECASE)
# Quoted strings and backticked names are ignored by the check, so search terms like 'remove adhesive' are not blocked.
_QUOTED_TEXT = re.compile(r"'(?:[^'\\]|\\.)*'|\"(?:[^\"\\]|\\.)*\"|`[^`]*`")

class Neo4jQueryInput(BaseModel):
    """Input schema for the Neo4j Query Tool."""
    cypher_query: str = Field(
        ..., 
        description="A valid, read-only Cypher query to execute against a Neo4j database."
    )

class Neo4jGraphQueryTool(BaseTool):
    name: str = "Neo4j Graph Query Tool"
    description: str = "Executes READ-ONLY Cypher queries against the knowledge graph to retrieve information."
    args_schema: type[BaseModel] = Neo4jQueryInput
    
    # Connection details
    uri: str = "neo4j://127.0.0.1:7687"  
    username: str = "neo4j"
    password: str = "password"                 # Update this
    database_name: str = "KnowledgeGraph"      # Update to the target database

    def _run(self, cypher_query: str) -> str:
        """Executes the read-only database transaction."""
        
        # 1st SAFETY CHECK: Block destructive commands (whole Cypher keywords outside quoted text)
        if _FORBIDDEN_KEYWORDS.search(_QUOTED_TEXT.sub("''", cypher_query)):
            return "Error: This tool is strictly for READ-ONLY queries. Write operations are blocked."

        driver = GraphDatabase.driver(self.uri, auth=(self.username, self.password))
        
        try:
            with driver.session(database=self.database_name) as session:
                
                # EXECUTE READ TRANSACTION : used execute_read which tells the server we are only fetching data
                def fetch_data(tx):
                    result = tx.run(cypher_query)
                    return [record.data() for record in result]
                
                # 2nd SAFETY CHECK: explicitly declare a transaction as "read-only". Even if a destructive command bypassed the 1st SAFETY CHECK, the database engine itself would reject the write attempt.
                records = session.execute_read(fetch_data)  

                # Handle empty results cleanly
                if not records:
                    return "No data found in the knowledge graph for this query."
                    
                # Return the results as a clean JSON string for the AI to read
                return json.dumps(records, indent=2)
                
        except Exception as e:
            return f"Cypher execution error: {str(e)}"
            
        finally:
            driver.close()

# Local test block of Graph Query Tool 
if __name__ == "__main__":
    from src.tools.db_config import load_db_config
    db_config = load_db_config()
    tool = Neo4jGraphQueryTool(uri=db_config["uri"], username=db_config["username"], password=db_config["password"],
                               database_name="KnowledgeGraphTEST")
    
    # A simple test query to fetch all nodes and their labels
    test_query = "MATCH (n) RETURN labels(n) AS label, n.id AS id LIMIT 10"
    print(tool._run(cypher_query=test_query)) 