import sys
from pathlib import Path

# Fix for Windows console UnicodeEncodeError
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from crewai import Agent, Task, Crew, LLM
from src.tools.pydantic_schemas import KnowledgeGraphSchema

my_llm = LLM(model="gemini/gemini-3.1-flash-lite") 

# the Canonicalisation Agent (The Data Engineer)
canonicalisation_agent = Agent(
    role="Graph Entity Resolution Specialist",
    goal="Clean, standardise, and canonicalise raw knowledge graph extractions to eliminate duplicates and synonyms.",
    backstory="""You are a senior data engineer specialising in entity resolution and ontology alignment. 
    Your job is to examine raw, fragmented data extractions, merge matching concepts (synonyms), 
    enforce uniform naming conventions, and format the output into a precise, structurally sound knowledge graph schema.""",
    llm=my_llm,
    verbose=True
)

# the Mapping and Validation Task
canonicalise_task = Task(
    description="""
    You are given a raw, unverified extraction of nodes and relationships:
    
    {raw_extraction}
    
    Perform the following processing steps:
    1. Entity Resolution: Identify and merge duplicate concepts or obvious synonyms. Resolve them into a single canonical entity using the clearest engineering term (e.g., merge 'industrial glue', 'heavy glue', and 'adhesives' into a single concept 'adhesive').
    2. ID Standardisation: Force all node 'id' fields to be lower snake_case (e.g., 'non_permanent_joint').
    3. Label Standardisation: Enforce strict Title Case for all node 'label' values (e.g., 'Component', 'Material').
    4. Structural Integrity: Every 'source' and 'target' in the edges list MUST exactly match an 'id' that exists in your nodes list. Do not leave hanging relationships.
    """,
    expected_output="A thoroughly resolved, schema-compliant knowledge graph structure containing clean nodes and edges.",
    agent=canonicalisation_agent,
    output_pydantic=KnowledgeGraphSchema  # CrewAI forces validation against written contract
)

# Local Test Loop 
if __name__ == "__main__":
    # Mocking messy data extractions: First test example
    mock_messy_input1 = """
    Nodes:
    - id: "Industrial Glue", label: "material"
    - id: "Adhesives", label: "Component_Type"
    - id: "modular screws", label: "hardware_parts"
    - id: "heavy-duty industrial glue", label: "bonding_agent"
    
    Edges:
    - "Industrial Glue" is technically an alternative to "Adhesives"
    - "modular screws" replaces "heavy-duty industrial glue"
    """
    
    # Mocking messy data extractions: Second test example
    mock_messy_input2 = """
    Nodes:
    - id: "Rubber Seals", label: "part"
    - id: "silicone gasket", label: "Component_Type"
    - id: "O-ring seal", label: "hardware"
    - id: "ip 67 waterproof rating", label: "Standard"
    - id: "IP-67", label: "certification"
    - id: "High Humidity", label: "Condition"
    - id: "outdoor rain exposure", label: "weather"
    - id: "moisture ingress", label: "Risk"
    - id: "main circuit board", label: "electronic"
    - id: "electrical PCB", label: "electronic"
    
    Edges:
    - "Rubber Seals" prevents "moisture ingress"
    - "silicone gasket" is required for "IP-67"
    - "ip 67 waterproof rating" protects "electrical PCB"
    - "O-ring seal" seals against "High Humidity"
    - "main circuit board" must survive "outdoor rain exposure"
    """

    # Assemble the pipeline segment
    crew = Crew(
        agents=[canonicalisation_agent],
        tasks=[canonicalise_task],
        verbose=True
    )
    
    print("Executing Canonicalisation Agent test block...\n")
    mock_messy_input = mock_messy_input2                                # choose test example
    result = crew.kickoff(inputs={"raw_extraction": mock_messy_input})
    
    # Extract the instantiated Pydantic object
    validated_graph = result.pydantic 
    
    print("\n" + "="*40)
    print("--- CHANNELS VALIDATED BY PYDANTIC ---")
    print(f"Total Canonicalised Nodes: {len(validated_graph.nodes)}")
    
    print("\nNodes:")
    for node in validated_graph.nodes:
        print(f"  • ID: [ {node.id} ] | Label: [ {node.label} ]")
        
    print("\nEdges (Relationships):")
    for edge in validated_graph.edges:
        print(f"  • ({edge.source}) -[:{edge.relation}]-> ({edge.target})")
    print("="*40)