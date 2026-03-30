"""encoder: bipartite GNN encoder for Chemical Reaction Networks."""
from crn_surrogate.encoder.bipartite_gnn import BipartiteGNNEncoder, CRNContext
from crn_surrogate.encoder.graph_utils import (
    EDGE_FEAT_DIM,
    BipartiteEdges,
    EdgeFeature,
    build_bipartite_edges,
)
from crn_surrogate.encoder.message_passing import (
    AttentiveMessagePassingLayer,
    SumMessagePassingLayer,
)
from crn_surrogate.encoder.tensor_repr import (
    CRNTensorRepr,
    PropensityType,
    crn_to_tensor_repr,
    tensor_repr_to_crn,
)

__all__ = [
    "AttentiveMessagePassingLayer",
    "BipartiteGNNEncoder",
    "BipartiteEdges",
    "CRNContext",
    "CRNTensorRepr",
    "EDGE_FEAT_DIM",
    "EdgeFeature",
    "PropensityType",
    "SumMessagePassingLayer",
    "build_bipartite_edges",
    "crn_to_tensor_repr",
    "tensor_repr_to_crn",
]
