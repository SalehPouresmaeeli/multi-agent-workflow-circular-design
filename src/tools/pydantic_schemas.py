# To find correct path of files
import sys
from pathlib import Path
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
####################################

from typing import Any, List, Literal
from pydantic import BaseModel, Field

# Strict Ontological Lists based on Systems Engineering
AllowedNodeLabels = Literal[
    # --- High-Level Architecture ---
    "LifecycleActivity",  # e.g., 'Design to recover value'
    "CircularStrategy",   # e.g., 'Remanufacturing', 'Recycling'
    "LifecycleProcess",   # e.g., 'Disassembly', 'Cleaning', 'Testing'
    
    # --- Physical Entities & Features ---
    "ProductComponent",   # e.g., 'Battery module', 'Casing', 'PCB'
    "Material",           # e.g., 'ABS Plastic', 'Epoxy', 'Copper'
    "StructuralFeature",  # e.g., 'Snap-fit', 'Chamfer', 'Threaded hole'
    "Tool",               # e.g., 'Screwdriver', 'Pry bar', 'Solvent'
    
    # --- Modifiers & States ---
    "InformationTag",     # e.g., 'RFID', 'Material stamp', 'Wear indicator'
    "PhysicalCondition",  # e.g., 'Corrosion', 'Moisture', 'Thermal stress'
    "Constraint",         # e.g., 'Non-permanent', 'Nondestructive'
    
    # --- The Escape Hatch ---
    "UnmappedEntity"      # Use ONLY when the concept strictly defies the above categories
]

AllowedRelations = Literal[
    # --- Strategy & Hierarchy ---
    "PART_OF_ACTIVITY",   # (CircularStrategy) -> (LifecycleActivity)
    "ENABLES_STRATEGY",   # (LifecycleProcess | ProductComponent | StructuralFeature) -> (CircularStrategy)
    "HAS_PART",           # (ProductComponent) -> (ProductComponent) [The BOM hierarchy]

    # --- Physical Composition & Features ---
    "COMPOSED_OF",        # (ProductComponent) -> (Material)
    "INTEGRATES_FEATURE", # (ProductComponent) -> (StructuralFeature)
    "STANDARDISED_WITH",  # (ProductComponent | Tool) -> (ProductComponent | Tool)

    # --- Directives & Substitutions ---
    "REQUIRES",           # General positive rule
    "AVOIDS",             # (CircularStrategy | ProductComponent | StructuralFeature) -> (Material | Tool | ProductComponent) 
    "ELIMINATES_NEED_FOR",# (StructuralFeature) -> (Tool | Material | ProductComponent)

    # --- Process Interactions ---
    "REQUIRES_PROCESS",   # (ProductComponent | Material) -> (LifecycleProcess)
    "REQUIRES_TOOL",      # (LifecycleProcess | ProductComponent| StructuralFeature) -> (Tool)
    "FACILITATES",        # (StructuralFeature | InformationTag | ProductComponent) -> (LifecycleProcess)
    "HINDERS",            # (Material | StructuralFeature) -> (LifecycleProcess)

    # --- Environment & Wear ---
    "VULNERABLE_TO",      # (ProductComponent | Material) -> (PhysicalCondition)
    "RESISTS",            # (Material | StructuralFeature | ProductComponent) -> (PhysicalCondition)
    "INDICATES_CONDITION",# (InformationTag | StructuralFeature) -> (PhysicalCondition)

    # --- States ---
    "MUST_BE",            # Connects entities to a (Constraint)
    "MUST_NOT_BE",        # Connects entities to a (Constraint)
    
    # --- The Escape Hatch ---
    "UNMAPPED_RELATION"   # Use ONLY when the relationship strictly defies the above categories
]

# Node Schema
# Schema representing a single entity in the Knowledge Graph.
class GraphNode(BaseModel):
    id: str = Field(
        ..., 
        description="A unique, lower_snake_case identifier for the node."
    )
    label: AllowedNodeLabels = Field(
        ..., 
        description="The strict category of the entity."
    )
    synonyms: List[str] = Field(
        default_factory=list, 
        description="A list of the original messy raw text strings from the raw data that map to this entity."
    )
    properties: dict = Field(
        default_factory=dict, 
        description="Any additional attributes belonging to this node. Keep it flat. If label is 'UnmappedEntity', this MUST contain a 'proposed_label' key."
    )

# Edge Schema
# Schema representing a directional relationship between two nodes.
class GraphEdge(BaseModel):
    source: str = Field(
        ..., 
        description="The exact ID of the source node."
    )
    target: str = Field(
        ..., 
        description="The exact ID of the target node."
    )
    relation: AllowedRelations = Field(
        ..., 
        description="The strict relationship type."
    )
    # --- Edge Properties (Provenance Tracking) ---
    # Required (no default) so an agent cannot silently drop them; an empty string is still valid when the source record has no value.
    # Each edge carries the lineage of ONE source: a relationship backed by several sources is repeated once per distinct lineage.
    reference_id: str = Field(
        ...,
        description="The literature reference number, copied verbatim from the source record (e.g., '[4]'). Empty string only if absent."
    )
    source_type: str = Field(
        ...,
        description="The type of literature, copied verbatim from the source record (e.g., 'Academic paper'). Empty string only if absent."
    )
    relevance: str = Field(
        ...,
        description="The relevance or importance level of the guideline, copied verbatim (e.g., 'High', 'Medium', 'Low', 'None'). Empty string only if absent."
    )
    properties: dict = Field(
        default_factory=dict, 
        description="Additional edge attributes. If relation is 'UNMAPPED_RELATION', this MUST contain a 'proposed_relation' key."
    )

class KnowledgeGraphSchema(BaseModel):
    """The final, validated output expected from the Canonicalisation Agent."""
    
    nodes: list[GraphNode] = Field(
        ..., 
        description="A comprehensive list of all nodes extracted from the text."
    )
    edges: list[GraphEdge] = Field(
        ..., 
        description="A list of all relationships connecting the extracted nodes."
    )