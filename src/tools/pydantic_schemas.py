from typing import Any
from pydantic import BaseModel, Field

class GraphNode(BaseModel):
    """Schema representing a single entity in the Knowledge Graph."""
    
    id: str = Field(
        ..., 
        description="The unique, canonicalised name of the entity (e.g., 'lithium_ion_battery'). Must be lower snake_case."
    )
    label: str = Field(
        ..., 
        description="The category or type of the node (e.g., 'Component', 'Material'). Must be Title Case."
    )
    properties: dict[str, Any] = Field(
        default_factory=dict,
        description="Any additional attributes belonging to this node. Keep it flat."
    )

class GraphEdge(BaseModel):
    """Schema representing a directional relationship between two nodes."""
    
    source: str = Field(
        ..., 
        description="The 'id' of the starting node. Must exactly match a node id."
    )
    target: str = Field(
        ..., 
        description="The 'id' of the destination node. Must exactly match a node id."
    )
    relation: str = Field(
        ..., 
        description="The type of relationship connecting the nodes (e.g., 'MADE_OF', 'REQUIRES'). Must be UPPER_SNAKE_CASE."
    )
    properties: dict[str, Any] = Field(
        default_factory=dict,
        description="Any additional attributes describing the relationship, such as weight or confidence."
    )

class KnowledgeGraphSchema(BaseModel):
    """The final, validated output expected from the Canonicalisation Agent."""
    
    nodes: list[GraphNode] = Field(
        ..., 
        description="A comprehensive list of all canonicalised nodes extracted from the text."
    )
    edges: list[GraphEdge] = Field(
        ..., 
        description="A list of all relationships connecting the extracted nodes."
    )