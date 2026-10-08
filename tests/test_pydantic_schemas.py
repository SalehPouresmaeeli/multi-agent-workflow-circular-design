### Tests for src/tools/pydantic_schemas.py — the knowledge graph schema and its fixed vocabulary.
### The agents' output is validated against these models, so anything they reject never reaches Neo4j.

import pytest
from pydantic import ValidationError
from src.tools.pydantic_schemas import GraphNode, GraphEdge, KnowledgeGraphSchema


def make_edge(**overrides):
    """A valid edge; tests override one field at a time to check that field's rules."""
    fields = {
        "source": "non_permanent_joints",
        "target": "remanufacturing",
        "relation": "ENABLES_STRATEGY",
        "reference_id": "[4]",
        "source_type": "Academic paper",
        "relevance": "High",
    }
    fields.update(overrides)
    return GraphEdge(**fields)


def test_valid_graph_is_accepted():
    graph = KnowledgeGraphSchema(
        nodes=[
            GraphNode(id="non_permanent_joints", label="StructuralFeature", synonyms=["detachable joints"]),
            GraphNode(id="remanufacturing", label="CircularStrategy"),
        ],
        edges=[make_edge()],
    )

    assert len(graph.nodes) == 2
    assert graph.edges[0].reference_id == "[4]"


def test_node_defaults_to_no_synonyms_and_no_properties():
    node = GraphNode(id="casing", label="ProductComponent")

    assert node.synonyms == []
    assert node.properties == {}


@pytest.mark.parametrize("label", ["Component", "product_component", "Product Component", ""])
def test_node_label_outside_vocabulary_is_rejected(label):
    with pytest.raises(ValidationError):
        GraphNode(id="casing", label=label)


@pytest.mark.parametrize("relation", ["PROTECTS", "made_of", "Requires", ""])
def test_edge_relation_outside_vocabulary_is_rejected(relation):
    with pytest.raises(ValidationError):
        make_edge(relation=relation)


def test_unmapped_escape_hatches_are_allowed():
    node = GraphNode(id="mystery", label="UnmappedEntity", properties={"proposed_label": "Standard"})
    edge = make_edge(relation="UNMAPPED_RELATION", properties={"proposed_relation": "CERTIFIED_BY"})

    assert node.label == "UnmappedEntity"
    assert edge.relation == "UNMAPPED_RELATION"


@pytest.mark.parametrize("missing_field", ["reference_id", "source_type", "relevance"])
def test_edge_without_source_record_field_is_rejected(missing_field):
    # Source tracking is required, so an agent cannot silently drop where a fact came from
    fields = {
        "source": "a", "target": "b", "relation": "REQUIRES",
        "reference_id": "[4]", "source_type": "Academic paper", "relevance": "High",
    }
    del fields[missing_field]

    with pytest.raises(ValidationError):
        GraphEdge(**fields)


def test_empty_source_record_values_are_allowed():
    # An empty string is valid when the guideline record itself has no value
    edge = make_edge(reference_id="", source_type="", relevance="")

    assert edge.reference_id == ""
