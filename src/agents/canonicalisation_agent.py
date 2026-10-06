import sys
from pathlib import Path
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
####################################

from crewai import Agent, Task
from src.tools.pydantic_schemas import KnowledgeGraphSchema
from src.tools.llm_config import get_llm
from src.agents.knowledge_extraction_agent import guidelines_extraction_task

# Its own LLM instance (rather than reusing the extraction agent's) so this agent can be
# pointed at a different model independently — see the "canonicalisation_agent" key in llm_config.json.
my_llm = get_llm("canonicalisation_agent", temperature=0)

# Canonicalisation Agent: resolves duplicates, standardises IDs/labels, enforces structural integrity
canonicalisation_agent = Agent(
    role="Graph Entity Resolution Specialist",
    goal="Clean, standardise, and canonicalise raw knowledge graph extractions to eliminate duplicates and synonyms.",
    backstory="""You are a senior data engineer specialising in entity resolution and ontology alignment,
    with deep expertise in Circular Economy and mechanical systems engineering terminology.
    Your job is to examine raw knowledge graph extractions, identify concepts that refer to the same physical or
    conceptual entity, merge them into a single canonical node, and enforce uniform formatting.
    You apply engineering domain knowledge to make correct merge decisions: 'ABS plastic' and
    'acrylonitrile butadiene styrene' are always the same material and must be merged; 'rubber seal',
    'silicone gasket', and 'O-ring seal' may be synonyms for the same sealing component or distinct parts
    depending on context — you apply judgement. You know 'disassembly' and 'teardown' refer to the same
    LifecycleProcess. You are conservative: when uncertain whether two terms are truly equivalent, keep them
    separate to avoid losing information.""",
    llm=my_llm,
    verbose=True,
    allow_delegation=False
)

canonicalise_task = Task(
    description="""
    You are given the raw knowledge graph extraction produced by the extraction agent in your context.

    --- EXISTING GRAPH NODES (from all previous ingestion runs) ---
    {existing_nodes}
    --- END OF EXISTING NODES ---

    CROSS-BATCH RESOLUTION RULE (apply this BEFORE any other step):
    For every node in the new extraction, check whether it refers to the same physical or conceptual
    entity as any existing node listed above. If it does, do NOT create a new node — instead, reuse
    the existing node's exact 'id' and 'label'. Add the new raw text string to that node's 'synonyms'
    list. Apply the same conservative judgement as for intra-batch merges: only reuse an existing
    node if the match is unambiguous.

    Perform the following processing steps:

    1. ENTITY RESOLUTION (CRITICAL): Identify and merge synonyms, alternate spellings, and
       abbreviations that refer to the exact same physical or conceptual entity into a SINGLE canonical node.
       - Use domain knowledge: 'IP-67', 'IP67', and 'ip 67 waterproof rating' are the same standard.
       - Be conservative: only merge entities that are truly the same concept. Entities that are merely
         related, or whose equivalence is ambiguous, must be kept separate.
       - When you merge nodes, you MUST re-route ALL edges from the old nodes to the canonical node's id.
         NEVER delete an edge because its endpoint was merged — always reroute it.

    2. ID Standardisation: Force all node 'id' fields to be lower_snake_case (e.g., 'non_permanent_joint').

    3. Label Standardisation: Enforce strict PascalCase for all node 'label' values. The ONLY valid labels are:
       CircularStrategy, LifecycleActivity, LifecycleProcess, ProductComponent, Material,
       StructuralFeature, Tool, InformationTag, PhysicalCondition, Constraint, UnmappedEntity
       Do NOT invent new label names or insert spaces (e.g., 'Product Component' is WRONG; 'ProductComponent' is correct).

    4. Structural Integrity: After merging, verify that every 'source' and 'target' in the edges list exactly
       matches an 'id' that exists in your final nodes list.
       - If a dangling edge references a node merged away, reroute it to the canonical node's id.
       - NEVER delete an edge because its endpoint was merged.

    5. Traceability Mapping: For every merged node, populate its 'synonyms' list with ALL original raw text
       strings that were collapsed into it. For nodes that were not merged, leave 'synonyms' empty.

    6. Edge Provenance (CRITICAL): Every edge carries its literature lineage in 'reference_id', 'source_type' and 'relevance'.
       - Copy these three values exactly as they appear in the extraction. Never blank, alter, or invent them.
       - When you reroute an edge to a canonical node, keep its lineage values unchanged.
       - Only collapse edges that are identical in source, target, relation AND all three lineage values.
         Edges that share source, target and relation but differ in any lineage value are separate evidence — keep every one of them.

    OUTPUT FORMAT:
    You must strictly conform to the provided KnowledgeGraphSchema. All label and relation values must be
    drawn from the allowed lists above. Do not invent labels or relation types outside the schema.
    """,
    expected_output="A thoroughly resolved, schema-compliant knowledge graph structure containing clean nodes with embedded lineage, and edges that preserve their literature lineage (reference_id, source_type, relevance).",
    agent=canonicalisation_agent,
    context=[guidelines_extraction_task],
    output_pydantic=KnowledgeGraphSchema
)
