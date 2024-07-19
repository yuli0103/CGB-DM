from torch import nn
from cgbdm.vit import ViT
from cgbdm.module import LayoutModule, MLP, Q_former
from typing import Optional


class LayoutModel(nn.Module):
    def __init__(self,
                 num_layers: int = 4,
                 dim_seq: int = 8,
                 dim_model: int = 512,
                 n_head: int = 8,
                 dim_feedforward: int = 1024,
                 diffusion_steps: int = 1000,
                 max_elem: int = 16,
                 device: Optional[str] = None):
        super(LayoutModel, self).__init__()

        common_params = {
            'dim_seq': dim_seq,
            'dim_model': dim_model,
            'nhead': n_head,
            'dim_feedforward': dim_feedforward,
            'diffusion_steps': diffusion_steps,
            'max_elem': max_elem,
            'device': device
        }

        self.img_encoder = ViT(image_size=[384,256],patch_size=32,channels=4,
                               dim=512,depth=6,heads=8,mlp_dim=2048,dropout=0.1,emb_dropout=0.1).to(device)

        self.layout_encoder = LayoutModule(
            num_layers=num_layers // 2,
            if_encoder=True,
            **common_params
        )
        self.layout_decoder = LayoutModule(
            num_layers=num_layers,
            if_encoder=False,
            **common_params
        )
        self.cgbwp = Q_former(in_dim=dim_model, num_tokens=1).to(device)
        self.salbox_encoder = MLP(input_dim=4, hidden_dim=dim_model, output_dim=dim_model, num_layers=3).to(device)

        # self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, layout, image, sal_box, timestep):
        img_encode = self.img_encoder(image)
        salbox_encode = self.salbox_encoder(sal_box)
        layout_encode = self.layout_encoder(layout, None, None, None, timestep)
        cgb_w = self.cgbwp(img_encode)
        output = self.layout_decoder(layout_encode, img_encode, cgb_w, salbox_encode, timestep)

        return output


