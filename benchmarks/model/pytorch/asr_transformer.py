# Copyright © 2023-2024 Apple Inc.

import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange, repeat, reduce

# from pfl.internal.ops.pytorch_ops import get_default_device
from pfl.metrics import Summed, Weighted

from .saug import SpecAugment, length_to_mask
from .cape1d import CAPE1d


def length_masked_normalize2d(x, x_length, epsilon=1e-7, return_mask=False):
    x_mask = rearrange(length_to_mask(x_length, x.shape[1]), "b t -> b t 1")
    C = x.shape[2]
    scale = 1 / (C * torch.maximum(torch.tensor(1), x_length.clone().detach()))
    scale = scale.to(x.dtype)
    mean = reduce(x * x_mask, "b t c -> b", "sum") * scale
    x = x - rearrange(mean, "b -> b 1 1")
    norm = torch.rsqrt(reduce(x * x * x_mask, "b t c -> b", "sum") * scale + epsilon)
    x = torch.where(x_mask, x * rearrange(norm, "b -> b 1 1"), 0)
    if return_mask:
        return x, x_mask
    else:
        return x


class LengthMaskedNorm2d(nn.Module):
    def __init__(self, epsilon: float = 1e-7):
        super().__init__()
        self.epsilon = epsilon
        self.scale = torch.nn.Parameter(torch.ones((1,)))
        self.bias = torch.nn.Parameter(torch.zeros((1,)))

    def forward(self, input, length):
        res, mask = length_masked_normalize2d(input, length, self.epsilon, True)
        res = torch.where(mask, res * self.scale + self.bias, 0)
        return res


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
            self._wqkv.weight,
            -((1.0 / self.emb_dim) ** 0.5),
            (1.0 / self.emb_dim) ** 0.5,
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
        pos_embedding_layer=None,
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
        self._conv_subsample = Conv1dSubsampling(in_channels, conv_dim, kernel, stride)
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


class ASRModel(torch.nn.Module):
    def __init__(self, saug_func, encoder, loss_func):
        super().__init__()
        self.saug_func = saug_func
        self.loss_func = loss_func
        self.encoder = encoder

    def forward(self, x, x_length, is_eval=False):
        x = length_masked_normalize2d(x, x_length)
        if not is_eval:
            x = self.saug_func(x, x_length)
        return self.encoder(x, x_length)

    def loss(
        self,
        input,
        target,
        input_length,
        target_length,
        user_id,
        transcript,
        is_eval=False,
    ):
        if is_eval:
            self.eval()
        else:
            self.train()

        output, output_length = self(input, input_length, is_eval=is_eval)
        output = rearrange(output, "b t c -> t b c")
        return self.loss_func(
            output, target, output_length.long(), target_length.long()
        ).mean()

    @torch.no_grad()
    # loss on the validation
    def metrics(
        self,
        input,
        target,
        input_length,
        target_length,
        user_id,
        transcript,
        is_eval=True,
    ):
        self.eval() if is_eval else self.train()
        loss_value = self.loss(
            input,
            target,
            input_length,
            target_length,
            user_id,
            transcript,
            is_eval=True,
        )
        num_samples = input.shape[0]
        loss_key = "eval-loss" if is_eval else "loss"
        output = {
            loss_key: Weighted(loss_value, num_samples),
            "num_samples": Summed(num_samples),
        }
        print("calculated metrics:", output)
        return output


def create_asr_ctc_model(nlabel):
    saug_func = SpecAugment(
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
        positions_delta=0.03,
    )
    pos_embedding_layer.set_content_scale(1.0)
    encoder = AsrEncoder(pos_embedding_layer=pos_embedding_layer, nlabel=nlabel)
    loss_func = nn.CTCLoss(reduction="none", zero_infinity=True)
    model = ASRModel(
        encoder=encoder,
        saug_func=saug_func,
        loss_func=loss_func,
    )
    print("pytorch_model:", model)
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print("Total params:", pytorch_total_params)
    return model

    # pytorch_model = TestModel(nlabel=nlabel).to(get_default_device())
