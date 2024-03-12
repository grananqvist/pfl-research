import numpy as np

from pfl.internal.ops.pytorch_ops import get_default_device
from pfl.metrics import Summed, Weighted


import math
from typing import Optional, Union
import torch
from torch import nn
from torch import Tensor
from einops import rearrange, repeat, reduce
import enum
import torch.nn.functional as F
import logging
import copy
from .asr_features import LogMelSpectrumCalculator, sliding_window_output_length, length_masked_normalize2d, masked_mean2d, length_to_mask, LengthMaskedNorm2d
    

# offset and width should be num_masks x batch_size
def _band_mask(size, offset, width):
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


# feature extraction
class AudioPreprocessor(nn.Module):
    def __init__(self, filters=80, hz=16000, window_ms=25, stride_ms=10) -> None:
        super().__init__()
        self.window = hz * (window_ms / 1000)
        self.stride = hz * (stride_ms / 1000)
        log_mel_fn = LogMelSpectrumCalculator(filters, hz)
        self.log_mel_fn_batch = torch.vmap(log_mel_fn)  # batch

    def forward(self, x, x_length):
        x = self.log_mel_fn_batch(x)
        x_length = sliding_window_output_length(self.window, self.stride, x_length)
        x = length_masked_normalize2d(x, x_length)
        return x, x_length
    

class SpecAugment(nn.Module):
    def __init__(
        self,
        num_freq_masks,
        freq_mask_max_width,
        num_time_masks,
        time_mask_max_width,
        time_mask_width_ratio,
        avg_mask_strategy,
    ) -> None:
        super().__init__()
        self.num_freq_masks = num_freq_masks
        self.freq_mask_max_width = freq_mask_max_width
        self.num_time_masks = num_time_masks
        self.time_mask_max_width = time_mask_max_width
        self.time_mask_width_ratio = time_mask_width_ratio
        self.avg_mask_strategy = avg_mask_strategy

    def forward(self, input, input_length):
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
    

class CAPE1d(nn.Module):
    def __init__(self, 
                 d_model: int,
                 max_global_shift: float = 0.0,
                 max_local_shift: float = 0.0,
                 max_global_scaling: float = 1.0,
                 normalize: bool = False,
                 freq_scale: float = 1.0,
                 batch_first: bool = True,
                 positions_delta: float = None):
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


class AsrSelfAttention(nn.Module):
    def __init__(
        self,
        emb_dim: int = 768,
        num_heads: int = 4,
        dropout: float = 0.3,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.dropout = dropout

        self._wqkv = nn.Linear(
            self.emb_dim,
            3 * self.emb_dim,
            bias=bias,
        )

        self._wf = nn.Linear(
            self.emb_dim,
            self.emb_dim,
            bias=bias,
        )
        
        self._init_parameters()

    def _init_parameters(self):
        nn.init.uniform_(
            self._wqkv.weight, -((1.0 / self.emb_dim) ** 0.5), (1.0 / self.emb_dim) ** 0.5
        )
        nn.init.uniform_(
            self._wf.weight, -((1.0 / self.emb_dim) ** 0.5), (1.0 / self.emb_dim) ** 0.5
        )

    def _get_mask(self, q, k, kv_length):
        # mask should be of size (b h tq tk)
        assert q.shape[:2] == k.shape[:2]
        b, h, tq, _hc = q.shape
        b, h, tk, _hc = k.shape
        padding_mask = repeat(
            torch.arange(tk, device=k.device), "tk -> b h tq tk", b=b, h=h, tq=tq
        )
        padding_mask = padding_mask < rearrange(kv_length, "b -> b 1 1 1")
        return padding_mask

    def forward(self, q, k, v, kv_length):
        q, k, v = rearrange(
            self._wqkv(k),
            "b t (qkv h hc) -> qkv b h t hc",
            qkv=3,
            h=self.num_heads,
            hc=self.emb_dim // self.num_heads,
        )

        attn_mask = self._get_mask(q, k, kv_length)
            
        if self.training:
            dropout_p = self.dropout
        else:
            dropout_p = 0.0

        result = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=False
        )

        result = rearrange(result, "b h t hc -> b t (h hc)")
        return self._wf(result)
    


class AsrTransformerBlock(nn.Module):
    def __init__(
        self,
        emb_dim: int,
        mlp_dim: int,
        num_heads: int,
        dropout: float,
        layer_dropout: float,
        ln_norm_type: str = "pre",
        bias: bool = False,
        ln_epsilon: float = 1e-05,
    ):
        """
        The ASR transformer encoder block initialization.
        Args:
            emb_dim (int): the embedded dimension
            mlp_dim (int): the hidden dimention of the feed-forward network
            num_heads (int): number of heads for self-attention
            dropout (float): dropout ratio
            layer_dropout (float): layer dropout ratio
            ln_norm_type (LayerNormType): when to do layernorm, LayerNormType.PRE or LayerNormType.POST
            bias (bool): whether or not to include bias in self-attention and mlp
            ln_epsilon (float): value added to layernorm for numerical stability
        """
        super().__init__()
        self.emb_dim = emb_dim
        self.mlp_dim = mlp_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.layer_dropout = layer_dropout
        self.ln_norm_type = ln_norm_type

        # normalizations
        self._ln1 = nn.LayerNorm(
            self.emb_dim,
            eps=ln_epsilon,
        )
        self._ln2 = nn.LayerNorm(
            self.emb_dim,
            eps=ln_epsilon,
        )

        # self attention
        self._self_attention = AsrSelfAttention(
            self.emb_dim,
            self.num_heads,
            self.dropout,
            bias=bias,
        )
        # mlp block
        self._w1 = nn.Linear(
            self.emb_dim,
            self.mlp_dim,
            bias=bias,
        )
        self._w2 = nn.Linear(
            self.mlp_dim,
            self.emb_dim,
            bias=bias,
        )
        self._do = nn.Dropout(dropout)
        self._init_parameters()

    def _init_parameters(self):
        nn.init.uniform_(
            self._w1.weight, -((1.0 / self.emb_dim) ** 0.5), (1.0 / self.emb_dim) ** 0.5
        )
        nn.init.uniform_(
            self._w2.weight, -((1.0 / self.mlp_dim) ** 0.5), (1.0 / self.mlp_dim) ** 0.5
        )

    def _mlp(self, x):
        x = self._w1(x)
        x = F.relu(x)
        x = self._do(x)
        x = self._w2(x)

        return x

    def forward(self, x, x_length):
        if self.training:
            do_layer_drop = torch.rand((1,), device=x.device) < self.layer_dropout
        else:
            do_layer_drop = False
        if do_layer_drop:
            if self.ln_norm_type == "post":
                x = self._ln2(self._ln1(x))
            elif self.ln_norm_type == "pre":
                pass  # x=x
        else:
            if self.ln_norm_type == "post":
                x = self._ln1(self._self_attention(x, x, x, x_length) + x)
                x = self._ln2(self._mlp(x) + x)
            elif self.ln_norm_type == "pre":
                y = self._ln1(x)
                x = self._self_attention(y, y, y, x_length) + x
                x = self._mlp(self._ln2(x)) + x
        return x
    

class Conv1dSubsampling(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, kernel: int = 7, stride: int = 3
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.kernel = kernel
        self.stride = stride
        self.dilation = 1
        self.padding = kernel // 2

        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(kernel,),
            stride=(stride,),
            padding=self.padding,
        )
        self.glu = nn.GLU(dim=-1)

    def _init_parameters(self):
        nn.init.uniform_(
            self.conv.weight,
            -((3.0 / (self.in_channels * self.kernel)) ** 0.5),
            (3.0 / (self.in_channels * self.kernel)) ** 0.5,
        )
        nn.init.uniform_(
            self.conv.bias,
            -((1.0 / (self.in_channels * self.kernel)) ** 0.5),
            (1.0 / (self.in_channels * self.kernel)) ** 0.5,
        )

    def forward(self, x, x_length):
        x = rearrange(x, "b t c -> b c t")
        x = self.conv(x)
        x = rearrange(x, "b c t -> b t c")
        x = self.glu(x)
        x_length = (
            x_length
            + 2 * self.padding
            - self.kernel
            - (self.kernel - 1) * (self.dilation - 1)
        ) // self.stride + 1
        return x, x_length
    

class AsrEncoder(nn.Module):
    def __init__(
        self,
        nlabel: int,
        in_channels: int = 80,
        kernel: int = 7,
        stride: int = 3,
        conv_dim: int = 1536,
        pos_embedding_layer = None,
        dropout: float = 0.3,
        layer_dropout: float = 0.3,
        emb_dim: int = 768,
        num_heads: int = 4,
        mlp_dim: int = 3072,
        n_blocks: int = 36,
        bias: bool = False,
        ln_epsilon: float = 1e-5,
        ln_norm_type: str = "pre",
    ):
        super().__init__()
        self.emb_dim = emb_dim
        self._conv_subsample = Conv1dSubsampling(
            in_channels, conv_dim, kernel, stride
        )
        self._ln = LengthMaskedNorm2d()
        self.ln_norm_type = ln_norm_type
        self._pos_embedding = pos_embedding_layer
        self._tf_blocks = nn.ModuleList(
            [
                AsrTransformerBlock(
                    emb_dim=self.emb_dim,
                    mlp_dim=mlp_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    layer_dropout=layer_dropout,
                    ln_norm_type=self.ln_norm_type,
                    bias=bias,
                    ln_epsilon=ln_epsilon,
                )
                for _ in range(n_blocks)
            ]
        )
        if self.ln_norm_type == "pre":
            self._ln_final = nn.LayerNorm(
                self.emb_dim,
                eps=ln_epsilon,
            )
        
        self._do = nn.Dropout(dropout)
        self._linear = nn.Linear(
            self.emb_dim,
            nlabel,
        )
        self._init_parameters()

    def _init_parameters(self):
        nn.init.uniform_(
            self._linear.weight,
            -((3.0 / (self.emb_dim)) ** 0.5),
            (3.0 / (self.emb_dim)) ** 0.5,
        )
        nn.init.uniform_(
            self._linear.bias,
            -((1.0 / (self.emb_dim)) ** 0.5),
            (1.0 / (self.emb_dim)) ** 0.5,
        )

    def _count_parameters(self) -> int:
        return sum([p.numel() for p in self.parameters()])

    def forward(self, x, x_length):
        # expected input (b t c)
        x = self._ln(x, x_length)
        x, x_length = self._conv_subsample(x, x_length)
        x = self._do(x)
        x = self._pos_embedding(x, x_lengths=x_length)
        for block in self._tf_blocks:
            x = block(x, x_length)
        if self.ln_norm_type == "pre":
            x = self._ln_final(x)        
        x = self._linear(x)
        x = x.log_softmax(-1)
        return x, x_length.int()

    def save(self, fname, other_items=None):
        logging.info(f"Saving model file: {fname}")
        checkpoint = {"state_dict": self.state_dict()}
        if other_items is not None:
            for k, v in other_items.items():
                checkpoint[k] = v
        torch.save(checkpoint, fname)

    def load(self, fname):
        logging.info(f"Loading model file: {fname}")
        # first load to cpu or we will run out of memory.
        checkpoint = torch.load(fname, map_location=lambda storage, loc: storage)
        self.load_state_dict(checkpoint["state_dict"])
        other_items = {}
        for k, v in checkpoint.items():
            if k != "model_state_dict":
                other_items[k] = copy.copy(v)
        del checkpoint
        return other_items
    

def make_pytorch_dummy_model(nlabel):
    import torch  # type: ignore
    torch.manual_seed(42)
    from pfl.model.pytorch import PyTorchModel

    class TestModel(torch.nn.Module):
        def __init__(self, nlabel):
            super().__init__()
            self.mfsc_func = AudioPreprocessor()
            self.saug_func = SpecAugment(
                num_freq_masks=2,
                freq_mask_max_width=30,
                num_time_masks=10,
                time_mask_max_width=50,
                time_mask_width_ratio=0.1,
                avg_mask_strategy=False,
            )
            pos_embedding_layer = CAPE1d(
                d_model=768, 
                max_global_shift=30, 
                normalize=True, 
                freq_scale=30, 
                positions_delta=0.03)
            pos_embedding_layer.set_content_scale(1.0)
            self.model = AsrEncoder(pos_embedding_layer=pos_embedding_layer, nlabel=nlabel)
            self.loss_func = nn.CTCLoss(reduction="none", zero_infinity=True)

        def forward(self, x, x_length, is_eval=False):
            x, x_length = self.mfsc_func(x, x_length)
            if not is_eval:
                x = self.saug_func(x, x_length)
            return self.model(x, x_length)

        def loss(self,
                 input,
                 target,
                 input_length,
                 target_length,
                 user_id,
                 transcript,
                 is_eval=False):
            if is_eval:
                self.eval()
            else:
                self.train()
            
            output, output_length = self(input, input_length, is_eval=is_eval)
            output = rearrange(output, "b t c -> t b c")
            return self.loss_func(output, target, output_length.long(), target_length.long()).mean()

        @torch.no_grad()
        def metrics(self, input, target, input_length, target_length,
                    user_id, transcript):
            loss_value = self.loss(input,
                                   target,
                                   input_length,
                                   target_length,
                                   user_id,
                                   transcript,
                                   is_eval=True)
            num_samples = input.shape[0]
            # print('input.shape:', input.shape, np.prod(input.shape))
            output = {
                'loss': Weighted(loss_value, num_samples),
                'num_samples': Summed(num_samples),
            }
            print('calculated metrics:', output)
            return output

    pytorch_model = TestModel(nlabel=nlabel).to(get_default_device())

    print('pytorch_model:', pytorch_model)
    pytorch_total_params = sum(p.numel() for p in pytorch_model.parameters())
    print('Total params:', pytorch_total_params)

    return pytorch_model
