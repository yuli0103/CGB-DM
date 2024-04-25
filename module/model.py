from torch import nn
from module.image_encoder.vit import ViT
from module.scale_encoder import ScaleEncoder
from module.layout_module import Layout_module, MLP
import logging

logger = logging.getLogger(__name__)


class My_Model(nn.Module):
    def __init__(self, num_layers=4,
                       dim_seq=8,
                       dim_transformer=512,
                       n_head=8,
                       dim_feedforward=1024,
                       diffusion_step=1000,
                       max_elem=16,
                       device='cuda',
                       ):
        super().__init__()
        self.device = device
        self.max_elem = max_elem
        '''1. pretrain ViT'''
        # self.pretrain_image_encoder = timm.create_model('vit_base_patch32_384',
        #                                                 pretrained=True,
        #                                                 pretrained_cfg_overlay=dict(file='/mnt/data/ly24/model_weight/pytorch_model.bin', custom_load=False))
        # self.requires_grad(self.pretrain_image_encoder)
        '''2. simpleViT '''
        # self.img_encoder = SimpleViT(image_size=256,patch_size=32,channels=4,num_classes=256,
        #                              dim=1024,depth=12,heads=16,mlp_dim=2048,)
        '''3. 大ViT'''
        self.img_encoder = ViT(image_size=[256,256],patch_size=16,channels=4,num_classes=512,
                               dim=512,depth=6,heads=8,mlp_dim=2048,dropout=0.1,emb_dropout=0.1).to(device)
        '''4. resnetfpn + transformer encoder'''
        # self.img_encoder = ResnetFeatureExtractor(backbone="resnet50", d_model=512, head="transformer")
        '''5. pretrain CLIP ViT'''
        # self.img_encoder = FrozenOpenCLIPImageEmbedder(device=self.device)
        # self.img_adapter = Transformer(width=768,
        #                                layers=2,
        #                                heads=12,)
        # self.proj = nn.Parameter(torch.randn(768, 512) * (768 ** -0.5))
        # self.scale_encoder = MoE_vit(dim=512)

        self.layout_encoder = Layout_module(num_layers=num_layers // 2,
                                            dim_seq=dim_seq,
                                            dim_transformer=dim_transformer,
                                            nhead=n_head,
                                            dim_feedforward=dim_feedforward,
                                            diffusion_step=diffusion_step,
                                            max_elem=max_elem,
                                            trans_in=True,
                                            device=self.device)

        self.scale_encoder = ScaleEncoder(in_dim=dim_transformer, num_tokens=1).to(device)

        self.detectbox_encoder = MLP(input_dim=4, hidden_dim=dim_transformer, output_dim=dim_transformer, num_layers=3).to(device)

        self.layout_decoder = Layout_module(num_layers=num_layers,
                                            dim_seq=dim_seq,
                                            dim_transformer=dim_transformer,
                                            nhead=n_head,
                                            dim_feedforward=dim_feedforward,
                                            diffusion_step=diffusion_step,
                                            max_elem=max_elem,
                                            trans_in=False,
                                            device=self.device)

        # self.apply(self._init_weights)
        # logger.log(f"number of parameters: %e{sum(p.numel() for p in self.parameters())}")
    def requires_grad(self, model, flag=False):
        """
        Set requires_grad flag for all parameters in a model.
        """
        for p in model.parameters():
            p.requires_grad = flag

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


    def forward(self, layout, image, detect_box, timestep):
        # with torch.no_grad():
        #     img_encode = self.pretrain_image_encoder.forward_features(image)
        # img_encode = self.img_adapter(img_encode)
        # img_encode = img_encode @ self.proj
        # weight = self.scale_encoder(img_encode)

        img_encode = self.img_encoder(image)
        weight = self.scale_encoder(img_encode)
        detectbox_encode = self.detectbox_encoder(detect_box)
        layout_encode = self.layout_encoder(layout, None, None, None, timestep)
        output = self.layout_decoder(layout_encode, img_encode, weight, detectbox_encode, timestep)

        return output


