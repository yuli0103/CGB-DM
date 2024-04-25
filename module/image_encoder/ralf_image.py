import sys

import torch.nn as nn
import timm
import torch
import torch.nn.functional as F
from module.arch_transformer import Transformer
from module.MoE_vision.MoE_vit import MoE_vit
class ResnetBackbone(nn.Module):
    def __init__(self, max_elem=16):
        # CNN backbone
        super(ResnetBackbone, self).__init__()

        resnet = timm.create_model("resnet50")
        # resnet_weight = torch.load("model_weight/resnet50_a1_0-14fe96d1.pth")
        # resnet.load_state_dict(resnet_weight)
        ch = [1024, 2048]

        resnet.conv1 = nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        res_chi = list(resnet.children())
        self.resnet_tilconv4 = nn.Sequential(*res_chi[:7])
        self.resnet_conv5 = res_chi[7]

        ## FPN
        self.fpn_conv11_4 = nn.Conv2d(ch[0], 256, 1, 1, 0)
        self.fpn_conv11_5 = nn.Conv2d(ch[1], 256, 1, 1, 0)
        self.fpn_conv33 = nn.Conv2d(256, 256, 3, 1, 1)
        # self.proj = nn.Conv2d(512, 8 * max_elem, 1, 1, 0)
        self.fc_h0 = nn.Linear(256, 4 *max_elem)

    def forward(self, img):
        # Multi-sacle feature
        resnet_f4 = self.resnet_tilconv4(img)
        resnet_f5 = self.resnet_conv5(resnet_f4)

        resnet_f4p = self.fpn_conv11_4(resnet_f4)
        resnet_f5p = self.fpn_conv11_5(resnet_f5)
        resnet_f5up = F.interpolate(resnet_f5p, size=resnet_f4p.shape[2:], mode="nearest")
        resnet_fused = torch.concat([resnet_f5up, self.fpn_conv33(resnet_f5up + resnet_f4p)], dim=1)
        # resnet_proj = self.proj(resnet_fused)
        resnet_flat = resnet_fused.flatten(start_dim=-2)
        h0 = self.fc_h0(resnet_flat).permute(0, 2, 1)
        return h0

class ralf_img_encoder(nn.Module):
    def __init__(self,width=512,num_classes=256,layers=6,heads=8):
        super().__init__()
        self.resnet = ResnetBackbone(max_elem=16)
        self.encoder = Transformer(width=width,layers=layers,heads=heads)
        # self.mlp_head = nn.Linear(width, num_classes)
        # self.scale_encoder = MoE_vit(dim=num_classes)

    def forward(self, img):
        img_encode = self.resnet(img)
        img_encode = self.encoder(img_encode)
        # img_encode = self.mlp_head(img_encode)
        # weight = self.scale_encoder(img_encode)
        return img_encode

# img = torch.randn(32, 4, 256, 256)
# model = rdam_img_encoder()
# model(img)