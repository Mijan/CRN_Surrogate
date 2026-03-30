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
        dropout: Dropout probability (0.0 = disabled).
    """

    d_model: int = 64
    n_layers: int = 3  # 3 rounds of message passing is sufficient for small CRNs
    n_propensity_types: int = 4  # embedding table size
    max_propensity_params: int = 4
    max_species: int = 32
    type_embed_dim: int = 0  # 0 = auto: set to d_model // 4 in __post_init__
    dropout: float = 0.0

    def __post_init__(self) -> None:
        if self.type_embed_dim == 0:
            object.__setattr__(self, "type_embed_dim", self.d_model // 4)

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
    """

    d_model: int = 64
    d_hidden: int = 128
    n_noise_channels: int = 16  # override with from_crn() for CLE-correct noise dim
    clip_state: bool = True  # clamp X >= 0 after each Euler-Maruyama step

    @classmethod
    def from_crn(
        cls,
        crn: "CRN",
        d_model: int = 64,
        d_hidden: int = 128,
        clip_state: bool = True,
    ) -> "SDEConfig":
        """Create SDEConfig with n_noise_channels = crn.n_reactions.

        This matches the Chemical Langevin Equation where each reaction drives
        one independent Wiener process.

        Args:
            crn: CRN definition whose n_reactions sets n_noise_channels.
            d_model: Hidden dimension for conditioning.
            d_hidden: Hidden dimension inside the drift/diffusion MLPs.
            clip_state: Whether to clamp state to [0, ∞) after each step.

        Returns:
            SDEConfig with n_noise_channels set to crn.n_reactions.
        """
        return cls(
            d_model=d_model,
            d_hidden=d_hidden,
            n_noise_channels=crn.n_reactions,
            clip_state=clip_state,
        )

    def __repr__(self) -> str:
        return (
            f"SDEConfig(d_model={self.d_model}, d_hidden={self.d_hidden}, "
            f"n_noise_channels={self.n_noise_channels}, clip_state={self.clip_state})"
        )


@dataclass(frozen=True)
class ModelConfig:
    """Top-level model configuration."""

    encoder: EncoderConfig = EncoderConfig()
    sde: SDEConfig = SDEConfig()

    def __repr__(self) -> str:
        return f"ModelConfig(encoder={self.encoder!r}, sde={self.sde!r})"
