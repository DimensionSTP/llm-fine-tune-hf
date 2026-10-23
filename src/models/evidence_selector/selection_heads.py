from typing import Dict, List, Any
import math

import torch
from torch import nn


class SelectionProjection(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        config: Dict[str, Any],
    ) -> None:
        super().__init__()
        if config["kind"] != "ffn" or config["activation"] != "gelu":
            raise ValueError("Only the approved FFN/GELU projection is implemented.")
        intermediate_size = hidden_size * config["hidden_ratio"]
        output_size = hidden_size * config["output_ratio"]
        if (
            not intermediate_size.is_integer()
            or not output_size.is_integer()
            or min(
                intermediate_size,
                output_size,
            )
            < 1
        ):
            raise ValueError(
                "Projection ratios must produce positive integer dimensions."
            )
        if config["dropout"] != 0.0:
            raise ValueError("Projection dropout is fixed at zero in this contract.")
        self.output_size = int(output_size)
        self.layers = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(
                hidden_size,
                int(intermediate_size),
            ),
            nn.GELU(),
            nn.Linear(
                int(intermediate_size),
                self.output_size,
            ),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        return self.layers(hidden_states)


class BilinearScores(nn.Module):
    def __init__(
        self,
        dimension: int,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty(
                dimension,
                dimension,
            ),
        )
        self.bias = nn.Parameter(torch.zeros(1))
        nn.init.xavier_uniform_(self.weight)
        self.scale = math.sqrt(dimension)

    def forward(
        self,
        queries: torch.Tensor,
        candidates: torch.Tensor,
    ) -> torch.Tensor:
        return (
            queries
            @ self.weight
            @ candidates.transpose(
                -1,
                -2,
            )
        ) / self.scale + self.bias


def pool_endpoints(
    hidden_states: torch.Tensor,
    positions: List[int],
) -> torch.Tensor:
    if hidden_states.ndim != 2:
        raise ValueError("Endpoint pooling expects one document [L,H].")
    if any(
        type(position) is not int or not 0 <= position < hidden_states.shape[0]
        for position in positions
    ):
        raise ValueError("A readout position is outside the document.")
    indices = torch.tensor(
        positions,
        dtype=torch.long,
        device=hidden_states.device,
    )
    return hidden_states.index_select(
        0,
        indices,
    )
