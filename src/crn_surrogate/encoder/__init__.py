"""encoder: bipartite GNN encoder for Chemical Reaction Networks."""

from crn_surrogate.encoder.bipartite_gnn import BipartiteGNNEncoder, CRNContext
from crn_surrogate.encoder.graph_utils import (
    EDGE_FEAT_DIM,
    BipartiteEdges,
    BipartiteGraphBuilder,
    EdgeFeature,
)
from crn_surrogate.encoder.protocol_encoder import ProtocolEncoder
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
    "BipartiteEdges",
    "BipartiteGNNEncoder",
    "BipartiteGraphBuilder",
    "CRNContext",
    "CRNTensorRepr",
    "EDGE_FEAT_DIM",
    "EdgeFeature",
    "PropensityType",
    "ProtocolEncoder",
    "SumMessagePassingLayer",
    "crn_to_tensor_repr",
    "tensor_repr_to_crn",
]
