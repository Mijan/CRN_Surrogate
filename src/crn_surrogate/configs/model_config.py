from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from crn_surrogate.crn.crn import CRN


@dataclass(frozen=True)
class EncoderConfig:
    """Configuration for the bipartite GNN encoder.

    Attributes:
        d_model: Hidden dimension for all node embeddings.
        n_layers: Number of bipartite message-passing rounds.
        n_propensity_types: Size of the propensity type embedding table.
        max_propensity_params: Max number of kinetic parameters per reaction.
        max_species: Upper bound on species count for identity embeddings.
        type_embed_dim: Dimension allocated to the propensity type embedding in
            ReactionEmbedding. Defaults to d_model // 4; the remainder
            (d_model - type_embed_dim) goes to the parameter projection.
        dropout: Dropout probability applied inside message-passing layers (0.0 = disabled).
        context_dropout: Dropout applied to the pooled context vector before it is
            passed to the SDE. Regularises the bottleneck that carries CRN identity.
        use_attention: If True, use AttentiveMessagePassingLayer instead of
            SumMessagePassingLayer. Disabled by default for initial experiments.
    """

    d_model: int = 64
    n_layers: int = 3  # 3 rounds of message passing is sufficient for small CRNs
    n_propensity_types: int = 7  # embedding table size
    max_propensity_params: int = 8
    max_species: int = 32
    type_embed_dim: int = 0  # 0 = auto: set to d_model // 4 in __post_init__
    dropout: float = 0.0
    context_dropout: float = 0.0
    use_attention: bool = False

    def __post_init__(self) -> None:
        if self.type_embed_dim == 0:
            object.__setattr__(self, "type_embed_dim", self.d_model // 4)
        if self.type_embed_dim >= self.d_model:
            raise ValueError(
                f"type_embed_dim ({self.type_embed_dim}) must be strictly "
                f"less than d_model ({self.d_model})"
            )

    def __repr__(self) -> str:
        return (
            f"EncoderConfig(d_model={self.d_model}, n_layers={self.n_layers}, "
            f"n_propensity_types={self.n_propensity_types}, "
            f"max_propensity_params={self.max_propensity_params})"
        )


@dataclass(frozen=True)
class SDEConfig:
    """Configuration for the neural SDE.

    Use SDEConfig.from_crn(crn) to automatically set n_noise_channels = n_reactions,
    which matches the Chemical Langevin Equation structure (one Wiener process per reaction).

    Attributes:
        d_model: Hidden dimension for conditioning context.
        d_hidden: Hidden dimension inside the drift/diffusion ConditionedMLPs.
        n_noise_channels: Number of independent noise channels (Wiener processes).
        n_hidden_layers: Number of FiLM-conditioned hidden layers in each network.
        clip_state: Whether to clamp state to [0, ∞) after each Euler-Maruyama step.
        mlp_dropout: Dropout applied after each hidden-layer activation in the
            drift and diffusion ConditionedMLPs (0.0 = disabled).
    """

    d_model: int = 64
    d_hidden: int = 128
    n_noise_channels: int = 8  # override with from_crn() for CLE-correct noise dim
    n_hidden_layers: int = 2
    clip_state: bool = True  # clamp X >= 0 after each Euler-Maruyama step
    d_protocol: int = 0  # protocol encoder output dim; 0 means no protocol conditioning
    mlp_dropout: float = 0.0

    def __post_init__(self) -> None:
        if self.n_hidden_layers < 1:
            raise ValueError(
                f"n_hidden_layers must be >= 1, got {self.n_hidden_layers}"
            )

    @classmethod
    def from_crn(
        cls,
        crn: "CRN",
        d_model: int = 64,
        d_hidden: int = 128,
        n_hidden_layers: int = 2,
        clip_state: bool = True,
        d_protocol: int = 0,
    ) -> "SDEConfig":
        """Create SDEConfig with n_noise_channels = crn.n_reactions.

        This matches the Chemical Langevin Equation where each reaction drives
        one independent Wiener process.

        Args:
            crn: CRN definition whose n_reactions sets n_noise_channels.
            d_model: Hidden dimension for conditioning.
            d_hidden: Hidden dimension inside the drift/diffusion ConditionedMLPs.
            n_hidden_layers: Number of FiLM-conditioned hidden layers per network.
            clip_state: Whether to clamp state to [0, ∞) after each step.
            d_protocol: Protocol encoder output dim; 0 means no protocol conditioning.

        Returns:
            SDEConfig with n_noise_channels set to crn.n_reactions.
        """
        return cls(
            d_model=d_model,
            d_hidden=d_hidden,
            n_noise_channels=crn.n_reactions,
            n_hidden_layers=n_hidden_layers,
            clip_state=clip_state,
            d_protocol=d_protocol,
        )

    def __repr__(self) -> str:
        return (
            f"SDEConfig(d_model={self.d_model}, d_hidden={self.d_hidden}, "
            f"n_noise_channels={self.n_noise_channels}, "
            f"n_hidden_layers={self.n_hidden_layers}, clip_state={self.clip_state})"
        )


@dataclass(frozen=True)
class ProtocolEncoderConfig:
    """Configuration for the DeepSets protocol encoder.

    Attributes:
        d_event: Hidden dimension for the per-event MLP.
        d_protocol: Output embedding dimension (must match SDEConfig.d_protocol).
        n_layers: Number of hidden layers in the per-event MLP.
        max_input_species: Max number of distinct external input species.
        species_embed_dim: Dimension of the per-species learned embedding.
    """

    d_event: int = 32
    d_protocol: int = 64
    n_layers: int = 2
    max_input_species: int = 16
    species_embed_dim: int = 8


@dataclass(frozen=True)
class ModelConfig:
    """Top-level model configuration."""

    encoder: EncoderConfig = EncoderConfig()
    sde: SDEConfig = SDEConfig()

    def __repr__(self) -> str:
        return f"ModelConfig(encoder={self.encoder!r}, sde={self.sde!r})"
