# To find correct path of files
import sys
from pathlib import Path
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
####################################

# to run use "python .\src\rag_agent.py" in the root folder 
##############################################################
from neo4j import GraphDatabase
from src.tools.neo4j_query import Neo4jGraphQueryTool
from src.tools.llm_config import get_llm
from src.tools.db_config import load_db_config
from crewai import Agent, Task, Crew

my_llm = get_llm("rag_agent", temperature=0)

### Database Connection details (from db_config.json — see src/tools/db_config.py)
db_config = load_db_config()
_uri: str = db_config["uri"]
_username: str = db_config["username"]
_password: str = db_config["password"]
_database_name: str = db_config["database_name"]

#################################
_CACHED_SCHEMA = None   # Add a global cache variable for graph schema

# Function to extract the graph schema from the database
def extract_dynamic_schema() -> str:
    """Dynamically reads the Neo4j database and builds a text schema for the AI agent."""
    
    driver = GraphDatabase.driver(_uri, auth=(_username, _password))
    schema_text = "Here is the strict schema of the Neo4j database you are querying:\n\n"
    
    try:
        with driver.session(database=_database_name) as session:
            
            # 1. Fetch all Node Labels and their Properties
            schema_text += "Node Labels and Properties:\n"
            node_query = """
            CALL db.schema.nodeTypeProperties() 
            YIELD nodeLabels, propertyName 
            RETURN nodeLabels[0] AS label, collect(propertyName) AS props
            """
            node_records = session.run(node_query)
            for record in node_records:
                # E.g., "- Component (properties: id, weight, synonyms)"
                props_str = ", ".join(record["props"]) if record["props"] else "none"
                schema_text += f"- {record['label']} (properties: {props_str})\n"
                
            # 2. Fetch the Permitted Relationships
            schema_text += "\nPermitted Relationships:\n"
            rel_query = """
            MATCH (n)-[r]->(m) 
            RETURN DISTINCT labels(n)[0] AS source, type(r) AS rel, labels(m)[0] AS target
            """
            rel_records = session.run(rel_query)
            for record in rel_records:
                # E.g., "- (Component)-[PROTECTS]->(Component)"
                schema_text += f"- ({record['source']})-[{record['rel']}]->({record['target']})\n"

            # 3. Fetch the Relationship Properties (this is where the literature lineage of each edge is stored)
            schema_text += "\nRelationship Types and Properties:\n"
            rel_prop_query = """
            CALL db.schema.relTypeProperties()
            YIELD relType, propertyName
            RETURN relType, collect(propertyName) AS props
            """
            rel_prop_records = session.run(rel_prop_query)
            for record in rel_prop_records:
                # relType comes back as ":`PART_OF_ACTIVITY`". E.g., "- PART_OF_ACTIVITY (properties: reference_id, source_type, relevance)"
                rel_name = record["relType"].lstrip(":").strip("`")
                props_str = ", ".join(record["props"]) if record["props"] else "none"
                schema_text += f"- {rel_name} (properties: {props_str})\n"

            # 4. Fetch the lineage values actually stored, so the agent filters with the exact spelling (e.g., 'High' not 'high')
            lineage_query = """
            MATCH ()-[r]->()
            WHERE r.reference_id IS NOT NULL
            RETURN collect(DISTINCT r.source_type) AS source_types,
                   collect(DISTINCT r.relevance) AS relevance_levels,
                   collect(DISTINCT r.reference_id)[..5] AS sample_references,
                   count(DISTINCT r.reference_id) AS reference_count
            """
            lineage = session.run(lineage_query).single()
            if lineage and lineage["reference_count"]:
                schema_text += "\nLiterature Lineage Values (stored on relationships):\n"
                schema_text += f"- source_type values: {lineage['source_types']}\n"
                schema_text += f"- relevance values: {lineage['relevance_levels']}\n"
                schema_text += f"- reference_id: {lineage['reference_count']} distinct values, e.g. {lineage['sample_references']}\n"

        return schema_text
        
    finally:
        driver.close()

# Cache and retrieve the schema
def get_current_schema(force_refresh: bool = False) -> str:
    """
    Returns the cached schema. Only hits the database if the cache is empty 
    or if a refresh is explicitly forced.
    """
    global _CACHED_SCHEMA
    
    if _CACHED_SCHEMA is None or force_refresh:
        print("Fetching fresh schema from Neo4j...")
        _CACHED_SCHEMA = extract_dynamic_schema()
    else:
        print("Using cached Neo4j schema...")    
    return _CACHED_SCHEMA

# Force update the schema when needed
def invalidate_schema_cache():
    """Wipes the cache so the next query is forced to fetch fresh data."""
    global _CACHED_SCHEMA
    _CACHED_SCHEMA = None

# Run the RAG pipeline
def run_rag_pipeline(user_question: str, event_callback=None) -> str:
    """
    The main bridge function. Extracts the freshest schema, builds the agent,
    and answers the user's question.
    """
    def push(event):
        if event_callback:
            try:
                event_callback(event)
            except Exception:
                pass

    current_schema = get_current_schema()
    push({"type": "agent_start", "message": "RAG Agent: Graph schema loaded. Translating question to Cypher..."})

    # Instantiate the query tool
    query_tool = Neo4jGraphQueryTool(uri=_uri, username=_username, password=_password, database_name=_database_name)

    def _step_cb(step_output):
        try:
            if hasattr(step_output, 'tool'):
                push({"type": "step", "message": f"RAG Agent: Using tool — {step_output.tool}"})
            elif hasattr(step_output, 'return_values'):
                push({"type": "step", "message": "RAG Agent: Evaluating result..."})
        except Exception:
            pass

    def _task_cb(task_output):
        push({"type": "task_done", "message": "RAG Agent: Answer formulated."})

    # CREATE THE CYPHER EXPERT AGENT
    cypher_agent = Agent(
        role="Principal Knowledge Graph Engineer",
        goal="Translate human questions into precise Cypher queries, execute them using your tool, and provide a clear answer.",
        backstory=(
            "You are an expert in Neo4j and the Cypher query language. "
            "You only write READ-ONLY queries using MATCH and RETURN. "
            "You never invent data. You strictly follow this database schema:\n"
            f"{current_schema}\n\n"
            "CRITICAL SEARCH STRATEGY:\n"
            "When a user asks about a specific entity (e.g., a component name), you must NEVER assume it is exactly the 'id'. "
            "You must always write your MATCH clause to check if the user's term matches EITHER the 'id' property OR exists inside the 'synonyms' list. "
            "Example of correct behaviour: MATCH (n:Component) WHERE n.id = 'term' OR 'term' IN n.synonyms\n\n"
            "LITERATURE LINEAGE:\n"
            "Every relationship records where its knowledge came from in three properties: 'reference_id' (the literature reference), "
            "'source_type' (the type of literature) and 'relevance' (the importance of the guideline). "
            "A link backed by several sources is stored as several parallel relationships, one per source. Therefore:\n"
            "- Always return the lineage of the relationships you retrieve, so every fact in your answer can be traced to its source.\n"
            "- Group parallel relationships so the same link is not listed repeatedly, keeping each source's lineage together, e.g. "
            "MATCH (n)-[r]->(m) WHERE n.id = 'term' RETURN n.id AS source, type(r) AS relation, m.id AS target, "
            "collect(DISTINCT r.reference_id + ' (' + r.source_type + ', ' + r.relevance + ')') AS sources\n"
            "- For questions about references, source types or relevance, filter on the relationship properties, e.g. "
            "WHERE r.reference_id = '[4]' or WHERE r.relevance = 'High'. Always use the exact values listed in the schema."
        ),
        tools=[query_tool],
        llm=my_llm,
        verbose=True,
        allow_delegation=False,
        max_iter=10
    )

    # 3. DEFINE THE GRAPH RAG TASK
    query_task = Task(
        description=(
            "Answer the following user question: '{user_question}'\n\n"
            "Instructions:\n"
            "1. Translate the question into a Cypher query using the provided schema.\n"
            "2. Use the Neo4j Graph Query Tool to execute it. You must keep a strict mental tally of exactly how many times you execute this tool.\n"
            "3. RETHINKING: If the tool returns 'No data found' or a syntax error or 'JUDGMENT' step failed, increment your tally, rethink your query, and try again.\n"
            "4. MAXIMUM RETRY LIMIT: You have a strict limit of 10 attempts. If your 9th attempt fails, your 10th attempt must be your final try. If the 10th fails, STOP QUERYING. Inform the user professionally that the specific data could not be found in the current graph.\n"
            "5. JUDGMENT: Once you have the JSON data, read the JSON data and judge critically whether it is logically the answer of the question:'{user_question}'. If not, and return to the 'RETHINKING' step.\n"
            "6. If the JUDGMENT step passed successfully, translate the JSON data into a technical, human-readable answer with a professional tone. Use bullet points if listing multiple components. "
            "Cite the literature lineage of every fact (reference, source type and relevance), e.g. 'Use non-permanent joints — [4] (Academic paper, High relevance)'. If a fact has no recorded reference, say so.\n"
            "7. CRITICAL LOGGING: At the very bottom of your final output, you must state exactly how many times you used the query tool. Use exactly this format: '\n\n[Diagnostic] Query attempts required: X'"
        ),
        expected_output="A technical and professional answer to the user's question, backed by the specific data retrieved from the database with each fact cited to its literature reference, source type and relevance, ending with a diagnostic count of the query attempts.",
        agent=cypher_agent
    )

    # 4. ASSEMBLE AND RUN
    rag_crew = Crew(
        agents=[cypher_agent],
        tasks=[query_task],
        step_callback=_step_cb,
        task_callback=_task_cb
    )

    result = rag_crew.kickoff(inputs={"user_question": user_question})
    push({"type": "result", "answer": result.raw})
    return result.raw

### Local test
"""
Examples of user_question to test this agent:
-Which device must be protected against rain?
-Which component must be protected against rain?
-which device is vulnerable to risk?
-how to mitigate risk of moisture?
-Which design guidelines come from reference [4]?
-Which High relevance guidelines enable remanufacturing, and what are their sources?
"""
if __name__ == "__main__":
    print("\n==============================================")
    print("GraphRAG System Booted. Ready for queries.")
    print("==============================================\n")
    
    # Create an infinite loop (stop with 'exit'or 'quit') so the user can ask multiple questions
    while True:
        # Ask the user for a question
        human_input = input("Ask the Graph a question (or type 'exit' to 'quit'):\n> ")
        
        # Check if they want to stop
        if human_input.lower() in ['exit', 'quit']:
            print("Shutting down the GraphRAG pipeline. Goodbye!")
            break
            
        print("\nConsulting the Knowledge Graph...\n")
        
        # Pass the input dynamically to the Crew.   The key "user_question" matches the {user_question} placeholder in the Task
        result = run_rag_pipeline(human_input)
        
        # Print the final answer and loop back around
        print("\n==================================")
        print("FINAL ANSWER:")
        print(result)
        print("==================================\n")