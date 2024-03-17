# Copyright © 2023-2024 Apple Inc.

import math
from typing import Optional, Union
import torch
from torch import nn
from torch import Tensor
from einops import rearrange, repeat


class CAPE1d(nn.Module):
    def __init__(
        self,
        d_model: int,
        max_global_shift: float = 0.0,
        max_local_shift: float = 0.0,
        max_global_scaling: float = 1.0,
        normalize: bool = False,
        freq_scale: float = 1.0,
        batch_first: bool = True,
        positions_delta: float = None,
    ):
        super().__init__()

        assert (
            max_global_shift >= 0
        ), f"""Max global shift is {max_global_shift},
        but should be >= 0."""
        assert (
            max_local_shift >= 0
        ), f"""Max local shift is {max_local_shift},
        but should be >= 0."""
        assert (
            max_global_scaling >= 1
        ), f"""Global scaling is {max_global_scaling},
        but should be >= 1."""

        self.max_global_shift = max_global_shift
        self.max_local_shift = max_local_shift
        self.max_global_scaling = max_global_scaling
        self.normalize = normalize
        self.freq_scale = freq_scale
        self.batch_first = batch_first
        self.positions_delta = positions_delta

        freq = self.freq_scale * torch.exp(
            -2.0 * torch.floor(torch.arange(d_model) / 2) * (math.log(1e4) / d_model)
        )
        self.register_buffer("freq", freq)

        _sin2cos_phase_shift = torch.pi / 2.0
        cos_shifts = _sin2cos_phase_shift * (torch.arange(d_model) % 2)
        self.register_buffer("cos_shifts", cos_shifts)

        self.register_buffer("content_scale", Tensor([math.sqrt(d_model)]))

    def forward(self, x: Tensor, x_lengths: Optional[Tensor] = None) -> Tensor:
        return (x * self.content_scale) + self.compute_pos_emb(
            x, x_lengths, self.positions_delta
        )

    def compute_pos_emb(
        self,
        x: Tensor,
        x_lengths: Optional[Tensor] = None,
        positions_delta: Optional[Union[int, Tensor]] = None,
    ) -> Tensor:
        if self.batch_first:
            batch_size, n_tokens, _ = x.shape  # b, t, c
        else:
            n_tokens, batch_size, _ = x.shape  # t, b, c

        positions = repeat(
            torch.arange(n_tokens), "t -> new_axis t", new_axis=batch_size
        ).to(x)

        if x_lengths is not None:
            padding_mask = positions > x_lengths[:, None]
            positions[padding_mask] = float("nan")

        if positions_delta is None:
            positions_delta = 1
        else:
            if torch.is_tensor(positions_delta) and len(positions_delta.shape) == 1:
                positions_delta = rearrange(positions_delta, "b -> b 1")
            positions *= positions_delta

        if self.normalize:
            positions -= torch.nanmean(positions, axis=1, keepdim=True)

        positions = self.augment_positions(positions, positions_delta)

        positions = rearrange(positions, "b t -> b t 1")
        product = positions * self.freq.to(x)

        pos_emb = torch.sin(product + self.cos_shifts.to(x))

        if not self.batch_first:
            pos_emb = rearrange(pos_emb, "b t c -> t b c")

        pos_emb = torch.nan_to_num(pos_emb, nan=0)

        return pos_emb

    def augment_positions(
        self, positions: Tensor, positions_delta: Optional[Union[int, Tensor]] = None
    ):
        if self.training:
            batch_size, n_tokens = positions.shape

            if self.max_global_shift:
                delta = torch.FloatTensor(batch_size, 1).uniform_(
                    -self.max_global_shift, self.max_global_shift
                )
                delta = delta.to(positions.device)
            else:
                delta = 0

            if self.max_local_shift:
                epsilon = self.max_local_shift
                delta_local = torch.FloatTensor(batch_size, n_tokens)
                delta_local = delta_local.uniform_(-epsilon, epsilon)
                delta_local = delta_local.to(positions.device)
                if positions_delta is not None:
                    if (
                        torch.is_tensor(positions_delta)
                        and len(positions_delta.shape) == 1
                    ):
                        positions_delta = rearrange(positions_delta, "b -> b 1")
                    delta_local *= positions_delta
            else:
                delta_local = 0

            if self.max_global_scaling > 1.0:
                log_lambdas = torch.FloatTensor(batch_size, 1)
                log_lambdas = log_lambdas.uniform_(
                    -math.log(self.max_global_scaling),
                    math.log(self.max_global_scaling),
                )
                log_lambdas = log_lambdas.to(positions.device)
            else:
                log_lambdas = torch.zeros(1).to(positions.device)

            positions = (positions + delta + delta_local) * torch.exp(log_lambdas)

        return positions

    def set_content_scale(self, content_scale: float):
        self.content_scale = Tensor([content_scale])
