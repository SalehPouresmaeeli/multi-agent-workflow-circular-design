import sys
from pathlib import Path
# Fix for Windows console UnicodeEncodeError (emojis and box-drawing chars)
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
####################################

from crewai import Agent, Task, Crew
from src.tools.pydantic_schemas import KnowledgeGraphSchema
from src.tools.llm_config import get_llm

my_llm = get_llm("extraction_agent", temperature=0)

# Extraction Agent: parses raw text into nodes and edges (no deduplication)
extraction_agent = Agent(
    role="Lead Systems Engineering NLP Architect",
    goal="Translate unstructured data into a rigorous, Component-Centric Knowledge Graph using a strict ontological schema.",
    backstory=(
        """You are a world-class expert in Natural Language Processing and Mechanical Systems Engineering. 
        You specialise in parsing Design for Excellence (DfReX) and Circular Economy literature. 
        You excel at bridging the gap between high-level Circular Economy strategies (like Remanufacturing or Recycling) 
        and the low-level physical design decisions required to achieve them.
        You do not just read text; you deconstruct it into its fundamental physical realities. 
        You possess a deep understanding of Bills of Materials (BOMs), instinctively knowing the difference 
        between a physical 'ProductComponent' (e.g., a casing), the 'Material' it is made from (e.g., ABS plastic), 
        and the 'StructuralFeature' designed into it (e.g., a snap-fit).
        You are uncompromisingly strict about data validation. You never invent categories or relationships outside 
        of your approved Pydantic schema, preferring to explicitly flag unknown concepts using the 'UnmappedEntity' and 
        'UNMAPPED_RELATION' escape hatches rather than risk corrupting the database architecture."""
    ),
    llm=my_llm, # Ensure this points to the LLM instance you defined at the top of your file
    verbose=True,
    allow_delegation=False
)

guidelines_extraction_task = Task(
    description="""
    You have received Circular Design guidelines in JSON format. Each record contains the fields:
    'Lifecycle activity', 'DfReX', 'Product requirement', 'Design criteria', 'Ref', 'Source Type', 'Relevance'.

    {raw_extraction}

    Your task is to parse this data into a strictly validated Component-Centric Knowledge Graph.

    CRITICAL EXTRACTION RULES:

    1. THE ARCHITECTURE:
       - Map each 'DfReX' value to a CircularStrategy node and each 'Lifecycle activity' to a LifecycleActivity node.
       - Connect them using: (CircularStrategy) -[PART_OF_ACTIVITY]-> (LifecycleActivity).
       - If a physical component, feature, or process directly enables the circular strategy, use:
         (ProductComponent | StructuralFeature | LifecycleProcess) -[ENABLES_STRATEGY]-> (CircularStrategy).

    2. EXTRACT ALL ENTITY TYPES: Identify every entity in 'Product requirement' and 'Design criteria' and assign
       it to exactly one of the following labels:
       - 'CircularStrategy'   — A DfReX strategy (e.g., 'Remanufacturing', 'Recycling').
       - 'LifecycleActivity'  — A high-level lifecycle phase (e.g., 'Design to recover value').
       - 'LifecycleProcess'   — An operational step (e.g., 'Disassembly', 'Cleaning', 'Testing').
       - 'ProductComponent'   — A distinct physical part (e.g., 'Battery module', 'Casing', 'PCB').
       - 'Material'           — A substance or compound (e.g., 'ABS plastic', 'Epoxy adhesive', 'Copper').
       - 'StructuralFeature'  — A geometry designed into a part (e.g., 'Snap-fit', 'Chamfer', 'Threaded hole').
       - 'Tool'               — An instrument used in a process (e.g., 'Screwdriver', 'Pry bar', 'Solvent').
       - 'InformationTag'     — A data carrier on a component (e.g., 'RFID tag', 'Material stamp', 'Wear indicator').
       - 'PhysicalCondition'  — A degradation state or environmental factor (e.g., 'Corrosion', 'Moisture ingress').
       - 'Constraint'         — A required or forbidden property (e.g., 'Non-permanent', 'Nondestructive', 'Tool-free').

    3. ID FORMATTING: Force all node 'id' fields to be lower_snake_case (e.g., 'lithium_ion_battery', 'moisture_ingress').

    4. PHYSICAL COMPOSITION & BOM HIERARCHY:
       - A component physically contains another: (ProductComponent) -[HAS_PART]-> (ProductComponent).
       - A component is made from a substance: (ProductComponent) -[COMPOSED_OF]-> (Material).
       - A component has a geometry designed into it: (ProductComponent) -[INTEGRATES_FEATURE]-> (StructuralFeature).
       - Two components or tools share the same interface/standard: (ProductComponent | Tool) -[STANDARDISED_WITH]-> (ProductComponent | Tool).

    5. DESIGN INTENT, PROCESS INTERACTIONS & STATES:
       Design directives:
       - Something is mandated: -[REQUIRES]->.
       - Something is banned or discouraged: -[AVOIDS]->.
       - A feature makes another item unnecessary: (StructuralFeature) -[ELIMINATES_NEED_FOR]-> (Tool | Material | ProductComponent).
       - An entity must have a property: -[MUST_BE]-> (Constraint).
       - An entity must NOT have a property: -[MUST_NOT_BE]-> (Constraint).

       Process interactions:
       - A component or material needs a process step: (ProductComponent | Material) -[REQUIRES_PROCESS]-> (LifecycleProcess).
       - A process, component, or feature needs a tool: (LifecycleProcess | ProductComponent | StructuralFeature) -[REQUIRES_TOOL]-> (Tool).
       - A feature or component makes a process easier: (StructuralFeature | InformationTag | ProductComponent) -[FACILITATES]-> (LifecycleProcess).
       - A material or feature makes a process harder: (Material | StructuralFeature) -[HINDERS]-> (LifecycleProcess).

       Environmental & condition states:
       - A component or material is susceptible to a condition: (ProductComponent | Material) -[VULNERABLE_TO]-> (PhysicalCondition).
       - A material or feature protects against a condition: (Material | StructuralFeature | ProductComponent) -[RESISTS]-> (PhysicalCondition).
       - A tag or feature signals a condition: (InformationTag | StructuralFeature) -[INDICATES_CONDITION]-> (PhysicalCondition).

    6. THE ESCAPE HATCHES (CRITICAL):
       Only use an escape hatch AFTER exhausting all options above.
       The ONLY valid node labels are:
         CircularStrategy, LifecycleActivity, LifecycleProcess, ProductComponent, Material,
         StructuralFeature, Tool, InformationTag, PhysicalCondition, Constraint, UnmappedEntity
       The ONLY valid relation types are:
         PART_OF_ACTIVITY, ENABLES_STRATEGY, HAS_PART, COMPOSED_OF, INTEGRATES_FEATURE,
         STANDARDISED_WITH, REQUIRES, AVOIDS, ELIMINATES_NEED_FOR, REQUIRES_PROCESS,
         REQUIRES_TOOL, FACILITATES, HINDERS, VULNERABLE_TO, RESISTS, INDICATES_CONDITION,
         MUST_BE, MUST_NOT_BE, UNMAPPED_RELATION
       - If an entity truly cannot fit any allowed label: use "UnmappedEntity" and add {"proposed_label": "YourGuess"} to the node's 'properties'.
       - If a relationship truly cannot fit any allowed relation: use "UNMAPPED_RELATION" and add {"proposed_relation": "YourGuess"} to the edge's 'properties'.

    7. PROVENANCE TRACKING (CRITICAL): Every edge MUST carry the lineage of the record it was extracted from.
       Map 'Ref' → reference_id, 'Source Type' → source_type, 'Relevance' → relevance, copying the values verbatim.
       - If the same relationship is supported by several records whose Ref, Source Type or Relevance differ
         (e.g., a PART_OF_ACTIVITY edge shared by every record of a strategy), output one edge per distinct
         lineage combination. Never merge them into a single edge.
       - Use an empty string only when the value is absent in the source record.

    OUTPUT FORMAT:
    You must strictly conform to the provided KnowledgeGraphSchema.
    """,
    expected_output="A strictly validated Systems Engineering Graph matching the Pydantic schema.",
    agent=extraction_agent,
    output_pydantic=KnowledgeGraphSchema
)


# Local Test Loop
if __name__ == "__main__":
    # Single record — tests basic entity extraction and provenance mapping
    mock_messy_input1 = """
    [
        {
            "Lifecycle activity": "Design to recover value",
            "DfReX": "Remanufacturing",
            "Product requirement": "Select non-permanent joints",
            "Design criteria": "Use non-permanent joint types rather than adhesives",
            "Ref": "[4]",
            "Source Type": "Academic paper",
            "Relevance": "High"
        }
    ]
    """

    # Multiple records sharing a DfReX strategy — tests synonym-prone entities
    # (rubber seal / silicone gasket / O-ring are all sealing components; IP-67 / ip67 are the same standard)
    mock_messy_input2 = """
    [
        {
            "Lifecycle activity": "Design to recover value",
            "DfReX": "Remanufacturing",
            "Product requirement": "Ensure IP67 waterproof protection",
            "Design criteria": "Use rubber seals to prevent moisture ingress into the main circuit board",
            "Ref": "[5]",
            "Source Type": "Industry standard",
            "Relevance": "High"
        },
        {
            "Lifecycle activity": "Design to recover value",
            "DfReX": "Remanufacturing",
            "Product requirement": "Ensure IP67 waterproof protection",
            "Design criteria": "Install silicone gaskets between housing components to meet IP-67 ingress protection rating",
            "Ref": "[6]",
            "Source Type": "Academic paper",
            "Relevance": "High"
        },
        {
            "Lifecycle activity": "Design to recover value",
            "DfReX": "Remanufacturing",
            "Product requirement": "Ensure IP67 waterproof protection",
            "Design criteria": "Use O-ring seals to protect the electrical PCB from outdoor rain exposure and high humidity",
            "Ref": "[7]",
            "Source Type": "Academic paper",
            "Relevance": "Medium"
        }
    ]
    """

    # Records spanning two DfReX strategies and two lifecycle activities —
    # tests Rule 1 (architecture mapping) and mixed Material / StructuralFeature extraction
    mock_messy_input3 = """
    [
        {
            "Lifecycle activity": "Design to recover value",
            "DfReX": "Remanufacturing",
            "Product requirement": "Prevent corrosion and rust",
            "Design criteria": "Use corrosion-resistant materials such as stainless steel or aluminium alloys for fasteners",
            "Ref": "[4]",
            "Source Type": "Academic paper",
            "Relevance": "High"
        },
        {
            "Lifecycle activity": "Design to extend life",
            "DfReX": "Repair",
            "Product requirement": "Ensure replaceability of wear parts",
            "Design criteria": "Design snap-fit connectors to allow tool-free replacement of battery modules",
            "Ref": "[8]",
            "Source Type": "Industry report",
            "Relevance": "High"
        }
    ]
    """

    # Import here to avoid circular import (canonicalisation_agent.py imports from this module)
    from src.agents.canonicalisation_agent import canonicalisation_agent, canonicalise_task

    # Assemble the pipeline segment
    crew = Crew(
        agents=[extraction_agent, canonicalisation_agent],
        tasks=[guidelines_extraction_task, canonicalise_task],
        verbose=True
    )

    print("Executing Extraction → Canonicalisation pipeline...\n")
    mock_messy_input = mock_messy_input3                                # choose test example (1, 2, or 3)
    result = crew.kickoff(inputs={"raw_extraction": mock_messy_input})

    # Extract the final Pydantic object from the canonicalisation task
    validated_graph = canonicalise_task.output.pydantic

    print("\n" + "="*40)
    print("--- CANONICALISED GRAPH VALIDATED BY PYDANTIC ---")
    print(f"Total Canonicalised Nodes: {len(validated_graph.nodes)}")
    
    print("\nNodes:")
    for node in validated_graph.nodes:
        print(f"  • ID: [ {node.id} ] | Label: [ {node.label} ]")
        print(f"    └── Merged from raw terms: {node.synonyms}")
        
    print("\nEdges (Relationships):")
    for edge in validated_graph.edges:
        print(f"  • ({edge.source}) -[:{edge.relation}]-> ({edge.target})")
        print(f"    └── Lineage: ref={edge.reference_id!r} | source_type={edge.source_type!r} | relevance={edge.relevance!r}")
    print("="*40)