"""
@author : Hyunwoong
@when : 2019-12-18
@homepage : https://github.com/gusdnd852
"""
import torch.nn.functional as F
from torch import nn

from transformer.blocks.decoder_layer import DecoderLayer
from transformer.embedding.transformer_embedding import TransformerEmbedding


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
                x = F.relu(x)
        return x

class Decoder(nn.Module):
    def __init__(self, max_len, d_model, ffn_hidden, n_head, n_layers, drop_prob, diffusion_steps, device):
        super().__init__()
        self.emb = TransformerEmbedding(d_model=d_model,
                                        drop_prob=drop_prob,
                                        max_len=max_len,
                                        device=device)

        self.layers = nn.ModuleList([DecoderLayer(d_model=d_model,
                                                  ffn_hidden=ffn_hidden,
                                                  n_head=n_head,
                                                  drop_prob=drop_prob,
                                                  diffusion_steps=diffusion_steps)
                                     for _ in range(n_layers)])
        self.fc = nn.Linear(in_features=d_model, out_features=8).to(device)



    def forward(self, trg, enc_src, timesteps, trg_mask, src_mask):
        trg = self.emb(trg)
        for layer in self.layers:
            trg = layer(trg, enc_src, timesteps, trg_mask, src_mask)
        # pass to LM head
        # output = self.linear(trg)
        output = self.fc(trg)
        return output
