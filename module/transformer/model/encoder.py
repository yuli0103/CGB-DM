"""
@author : Hyunwoong
@when : 2019-12-18
@homepage : https://github.com/gusdnd852
"""
from torch import nn

from transformer.blocks.encoder_layer import EncoderLayer
from transformer.embedding.transformer_embedding import TransformerEmbedding

class Encoder(nn.Module):

    def __init__(self, d_model, ffn_hidden, n_head, n_layers, drop_prob, max_elem, device):
        super().__init__()
        self.emb = TransformerEmbedding(d_model=d_model, device=device, max_elem=max_elem, drop_prob=drop_prob)
        self.layers = nn.ModuleList([EncoderLayer(d_model=d_model,
                                                  ffn_hidden=ffn_hidden,
                                                  n_head=n_head,
                                                  drop_prob=drop_prob)
                                     for _ in range(n_layers)])

    def forward(self, x, src_mask):
        x = self.emb(x)
        for layer in self.layers:
            x = layer(x, src_mask)
        return x