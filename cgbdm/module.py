import copy
import math
import sys
from typing import Callable, Optional, Union
import torch
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from torch import Tensor, nn
from cgbdm.arch_transformer import Transformer

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def _gelu2(x):
    return x * F.sigmoid(1.702 * x)

def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    elif activation == "gelu2":
        return _gelu2
    else:
        raise RuntimeError(
            "activation should be relu/gelu/gelu2, not {}".format(activation)
        )

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < self.num_layers - 1:
                x = F.softplus(x)
        return x

class Q_former(nn.Module):
    def __init__(self, in_dim=512, out_dim=1, num_heads=8, num_tokens=1, n_layers=2):
        super().__init__()
        scale = in_dim ** -0.5
        self.num_tokens = num_tokens
        self.scale_emb = nn.Parameter(torch.randn(1, num_tokens, in_dim) * scale)
        self.transformer_blocks = Transformer(
            width=in_dim,
            layers=n_layers,
            heads=num_heads,
        )
        self.ln1 = nn.LayerNorm(in_dim)
        self.ln2 = nn.LayerNorm(in_dim)

        self.out = nn.Sequential(
            nn.Linear(in_dim, in_dim // 2),
            nn.GELU(),
            nn.Linear(in_dim // 2, out_dim),
            nn.Softplus(),
        )

    def forward(self, img):
        # x = torch.cat((img, layout), dim=1)
        x = img
        scale_emb = self.scale_emb.repeat(x.shape[0], 1, 1)
        x = torch.cat([scale_emb, x], dim=1)
        x = self.ln1(x)

        x = x.permute(1, 0, 2)
        x = self.transformer_blocks(x)
        x = x.permute(1, 0, 2)

        x = self.ln2(x[:, :self.num_tokens, :])
        x = self.out(x)
        return x

class SinusoidalPosEmb(nn.Module):
    def __init__(self, num_steps: int, dim: int, rescale_steps: int = 4000):
        super().__init__()
        self.dim = dim
        self.num_steps = float(num_steps)
        self.rescale_steps = float(rescale_steps)

    def forward(self, x: Tensor):
        x = x / self.num_steps * self.rescale_steps
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class _AdaNorm(nn.Module):
    def __init__(
        self, n_embd: int, max_timestep: int, emb_type: str = "adalayernorm_abs"
    ):
        super().__init__()
        if "abs" in emb_type:
            self.emb = SinusoidalPosEmb(max_timestep, n_embd)
        elif "mlp" in emb_type:
            self.emb = nn.Sequential(
                Rearrange("b -> b 1"),
                nn.Linear(1, n_embd // 2),
                nn.ReLU(),
                nn.Linear(n_embd // 2, n_embd),
            )
        else:
            self.emb = nn.Embedding(max_timestep, n_embd)
        self.silu = nn.SiLU()
        self.linear = nn.Linear(n_embd, n_embd * 2)


class AdaLayerNorm(_AdaNorm):
    def __init__(
        self, n_embd: int, max_timestep: int, emb_type: str = "adalayernorm_abs"
    ):
        super().__init__(n_embd, max_timestep, emb_type)
        self.layernorm = nn.LayerNorm(n_embd, elementwise_affine=False)

    def forward(self, x: Tensor, timestep: int):
        timestep_emb = self.emb(timestep)
        timestep_emb = self.silu(timestep_emb)
        timestep_emb = self.linear(timestep_emb)
        emb = timestep_emb.unsqueeze(1)
        scale, shift = torch.chunk(emb, 2, dim=2)
        x = self.layernorm(x) * (1 + scale) + shift
        return x


class AdaInsNorm(_AdaNorm):
    def __init__(
        self, n_embd: int, max_timestep: int, emb_type: str = "adalayernorm_abs"
    ):
        super().__init__(n_embd, max_timestep, emb_type)
        self.instancenorm = nn.InstanceNorm1d(n_embd)

    def forward(self, x, timestep):
        emb = self.linear(self.silu(self.emb(timestep))).unsqueeze(1)
        scale, shift = torch.chunk(emb, 2, dim=2)
        x = (
            self.instancenorm(x.transpose(-1, -2)).transpose(-1, -2) * (1 + scale)
            + shift
        )
        return x


class Block(nn.Module):
    """an unassuming Transformer block"""

    def __init__(
        self,
        d_model=512,
        nhead=8,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
        batch_first: bool = True,
        device=None,
        dtype=None,
        # extension for diffusion
        diffusion_steps: int = 100,
        timestep_type: str = 'adalayernorm',
    ) -> None:
        super().__init__()

        layer_norm_eps = 1e-5  # fixed

        self.diffusion_steps = diffusion_steps
        self.timestep_type = timestep_type

        factory_kwargs = {"device": device, "dtype": dtype}
        self.self_attn = torch.nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=batch_first, **factory_kwargs
        )

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, **factory_kwargs)

        if timestep_type is not None:
            if "adalayernorm" in timestep_type:
                self.norm1 = AdaLayerNorm(d_model, diffusion_steps, timestep_type)
            elif "adainnorm" in timestep_type:
                self.norm1 = AdaInsNorm(d_model, diffusion_steps, timestep_type)
        else:
            self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm4 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        if isinstance(activation, str):
            self.activation = _get_activation_fn(activation)
        else:
            self.activation = activation

    def forward(
        self,
        src: Tensor,
        img: Tensor,
        cgb_w: Tensor,
        salbox_encode: Tensor,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        timestep: Tensor = None,
    ) -> Tensor:
        x = src

        if self.timestep_type is not None:
            x = self.norm1(x, timestep)
        else:
            x = self.norm1(x)
        x = x + self._sa_block(x, src_mask, src_key_padding_mask)

        if img is not None:
            x = self.norm2(x)
            if cgb_w is not None:
                x = x + cgb_w * self._ca_block(x, img, img, None, src_key_padding_mask)
            else:
                x = x + self._ca_block(x, img, img, None, src_key_padding_mask)

        if salbox_encode is not None:
            x = self.norm3(x)
            # if cgb_w is not None:
            #     x = x + cgb_w * self._ca_block(x, detect_encode, detect_encode, None, src_key_padding_mask)
            # else:
            x = x + self._ca_block(x, salbox_encode, salbox_encode, None, src_key_padding_mask)

        x = x + self._ff_block(self.norm4(x))
        return x

    # self-attention block
    def _sa_block(
        self,
        x: Tensor,
        attn_mask: Optional[Tensor],
        key_padding_mask: Optional[Tensor],
    ) -> Tensor:
        x = self.self_attn(
            x, x, x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )[0]
        return self.dropout1(x)

    def _ca_block(
        self,
        x: Tensor,
        k: Tensor,
        v: Tensor,
        attn_mask: Optional[Tensor],
        key_padding_mask: Optional[Tensor],
    ) -> Tensor:
        output = self.self_attn(
            x, k, v,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )[0]
        return self.dropout2(output)

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout3(x)


class LayoutModule(nn.Module):
    __constants__ = ["norm"]

    def __init__(self,
                 num_layers: int,
                 if_encoder: bool,
                 dim_seq: int = 8,
                 dim_model: int = 512,
                 nhead: int = 8,
                 dim_feedforward: int = 1024,
                 diffusion_steps: int = 1000,
                 max_elem: int = 16,
                 timestep_type: str = 'adalayernorm',
                 device: Optional[str] = None):
        super(LayoutModule, self).__init__()

        self.max_elem = max_elem
        self.device = device
        if if_encoder:
            self.mlp = MLP(input_dim=dim_seq, hidden_dim=dim_model, output_dim=dim_model, num_layers=3).to(device)
        else:
            self.mlp = MLP(input_dim=dim_model, hidden_dim=dim_model, output_dim=dim_seq, num_layers=3).to(device)
        self.pos_encoder = SinusoidalPosEmb(num_steps=max_elem, dim=dim_model).to(device)

        encoder_layer = Block(d_model=dim_model, nhead=nhead, dim_feedforward=dim_feedforward, diffusion_steps=diffusion_steps, timestep_type=timestep_type)
        self.layers = _get_clones(encoder_layer, num_layers).to(device)
        self.num_layers = num_layers
        self.if_encoder = if_encoder

    def forward(
        self,
        src: Tensor,
        img_encode,
        cgb_w,
        salbox_encode,
        timestep: Tensor = None,
    ) -> Tensor:

        if self.if_encoder:
            output = F.softplus(self.mlp(src))
            pos_i = torch.tensor([i for i in range(self.max_elem)]).to(self.device)
            pos_embed = self.pos_encoder(pos_i)
            output = output + pos_embed
        else:
            output = src

        for i, mod in enumerate(self.layers):
            output = mod(
                output,
                img_encode,
                cgb_w,
                salbox_encode,
                src_mask=None,
                src_key_padding_mask=None,
                timestep=timestep,
            )
            if i < self.num_layers - 1:
                output = F.softplus(output)

        if not self.if_encoder:
            output = self.mlp(output)
        return output

