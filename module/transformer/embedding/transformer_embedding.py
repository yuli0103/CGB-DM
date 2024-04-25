"""
@author : Hyunwoong
@when : 2019-10-22
@homepage : https://github.com/gusdnd852
"""

from torch import nn
import torch

from transformer.embedding.positional_encoding import PositionalEncoding
# from embedding.token_embeddings import TokenEmbedding
import math

class SinusoidalPosEmb(nn.Module):
    def __init__(self, num_steps: int, dim: int, rescale_steps: int = 4000):
        super().__init__()
        self.dim = dim
        self.num_steps = float(num_steps)
        self.rescale_steps = float(rescale_steps)

    def forward(self, x):
        x = x / self.num_steps * self.rescale_steps
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class TransformerEmbedding(nn.Module):
    """
    token embedding + positional encoding (sinusoid)
    positional encoding can give positional information to network
    """

    def __init__(self, d_model, drop_prob, max_elem, device):
        """
        class for word embedding that included positional information

        :param vocab_size: size of vocabulary
        :param d_model: dimensions of model
        """
        super(TransformerEmbedding, self).__init__()
        self.device = device
        self.layer_norm_eps = 1e-6
        # self.tok_emb = nn.Linear(8, d_model).to(device)
        # self.silu = nn.SiLU()
        self.pos_emb = PositionalEncoding(d_model, max_elem, device)
        self.drop_out = nn.Dropout(p=drop_prob)
        self.LayerNorm = nn.LayerNorm(d_model, eps=self.layer_norm_eps)

    def forward(self, x):

        pos_emb = self.pos_emb(x)
        return self.drop_out(self.LayerNorm(x + pos_emb))

