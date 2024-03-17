# Copyright © 2023-2024 Apple Inc.

import torch
from torch import nn
from einops import rearrange, repeat, reduce


# offset and width should be num_masks x batch_size
def _band_mask(size: int, offset: int, width: int) -> torch.Tensor:
    """Compute mask for the audio bands"""
    num_masks = offset.shape[0]
    batch_size = offset.shape[1]
    offset = rearrange(offset, "n b -> n b 1")
    width = rearrange(width, "n b -> n b 1")
    # num_masks x batch_size x size
    mask = rearrange(torch.arange(size), "s -> 1 1 s")
    mask = repeat(mask, "n b s -> n (repeat b) s", repeat=batch_size)
    mask = repeat(mask, "n b s -> (repeat n) b s", repeat=num_masks)
    mask = (mask >= offset) & (mask < (offset + width))
    mask = reduce(mask, "n b s -> b s", "max")
    return mask


def length_to_mask(lengths: torch.Tensor, max_length: int) -> torch.Tensor:
    """Convert length vector to the masked vector (ignore padded tokens out of the length)"""
    indices = rearrange(torch.arange(max_length), "t -> 1 t").to(lengths.device)
    return indices < rearrange(lengths, "b -> b 1")  # (batch_size, max_length)


# x: NxWxC length: N
def masked_mean2d(
    x: torch.Tensor, x_length: torch.Tensor, return_mask: bool = False
) -> torch.Tensor:
    """average tensor and return its mean excluding the padded tokens"""
    x_mask = rearrange(length_to_mask(x_length, x.shape[1]), "b t -> b t 1")
    C = x.shape[2]
    scale = 1 / (C * torch.maximum(torch.tensor(1), x_length.clone().detach()))
    scale = scale.to(x.dtype)
    mean = reduce(x * x_mask, "b t c -> b", "sum") * scale
    if return_mask:
        return mean, x_mask
    else:
        return mean


class SpecAugment(nn.Module):
    """SpecAugment implementation, see https://arxiv.org/abs/1904.08779, w/o time warping"""

    def __init__(
        self,
        num_freq_masks: int,
        freq_mask_max_width: int,
        num_time_masks: int,
        time_mask_max_width: int,
        time_mask_width_ratio: float,
        avg_mask_strategy: bool = False,
    ) -> None:
        super().__init__()
        self.num_freq_masks = num_freq_masks
        self.freq_mask_max_width = freq_mask_max_width
        self.num_time_masks = num_time_masks
        self.time_mask_max_width = time_mask_max_width
        self.time_mask_width_ratio = time_mask_width_ratio
        self.avg_mask_strategy = avg_mask_strategy

    def forward(self, input: torch.Tensor, input_length: torch.Tensor) -> torch.Tensor:
        batch_size = input.shape[0]
        input_max_length = input.shape[1]
        num_channels = input.shape[2]

        res = input
        if self.avg_mask_strategy:
            fill_value, length_mask = masked_mean2d(
                input, input_length, return_mask=True
            )
            fill_value = rearrange(fill_value, "b -> b 1 1")
        else:
            fill_value = 0.0
            length_mask = length_to_mask(input_length, input_max_length)
            length_mask = rearrange(length_mask, "b t -> b t 1")

        # freq mask
        assert self.freq_mask_max_width < num_channels, "invalid freq_mask_max_width"
        f_widths = torch.randint(
            0, self.freq_mask_max_width, (self.num_freq_masks, batch_size)
        )
        ### better implementation??
        f_offsets_max = num_channels - f_widths
        f_offsets = torch.zeros(self.num_freq_masks, batch_size)
        for i in range(self.num_freq_masks):
            for j in range(batch_size):
                f_offsets[i][j] = torch.randint(0, f_offsets_max[i][j], (1,))[0]
        mask = _band_mask(num_channels, f_offsets, f_widths)
        mask = rearrange(mask, "b l -> b 1 l")
        res = torch.where(mask.to(input.device), fill_value, res)

        # time mask
        time_mask_width = torch.minimum(
            torch.tensor(self.time_mask_max_width),
            input_length.clone().detach() * self.time_mask_width_ratio,
        ).to(torch.int32)
        t_widths = torch.zeros(self.num_time_masks, batch_size)
        for i in range(self.num_time_masks):
            for j in range(batch_size):
                t_widths[i][j] = torch.randint(0, time_mask_width[j], (1,))[0]
        t_offsets_max = input_length - t_widths.to(input.device)
        t_offsets = torch.zeros(self.num_time_masks, batch_size)
        for i in range(self.num_time_masks):
            for j in range(batch_size):
                t_offsets[i][j] = torch.randint(
                    0, int(t_offsets_max[i][j].item()), (1,)
                )[0]

        mask = _band_mask(input_max_length, t_offsets, t_widths)
        mask = rearrange(mask, "b l -> b l 1")
        res = torch.where(mask.to(input.device), fill_value, res)
        res = torch.where(length_mask.to(input.device), res, 0)

        return res
