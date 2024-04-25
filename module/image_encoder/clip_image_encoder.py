import sys

import torch.nn as nn
import torch
import kornia
import open_clip
# torch.cuda.set_device(2)
class AbstractEncoder(nn.Module):
    def __init__(self):
        super().__init__()

    def encode(self, *args, **kwargs):
        raise NotImplementedError

class FrozenOpenCLIPImageEmbedder(AbstractEncoder):
    """
    Uses the OpenCLIP vision transformer encoder for images
    """

    def __init__(self, arch="ViT-B-32", version="laion2b_s32b_b79k", device="cuda", max_length=77,
                 freeze=True, layer="pooled", antialias=True, ucg_rate=0., only_cls=True, use_proj=True,
                 use_shuffle=False, mask_ratio=0.0):
        super().__init__()
        torch.cuda.set_device(device)
        model, _, _ = open_clip.create_model_and_transforms(arch, device=device,
                                                            pretrained="/home/ly24/code/py_model/checkpoints/open_clip/open_clip_vit_b_32_pytorch_model.bin",)
        del model.transformer
        self.model = model
        self.mask_ratio = mask_ratio
        # self.patch_dropout = PatchDropout(prob=patch_dropout, exclude_first_token=True) if patch_dropout > 0.0 else nn.Identity()

        self.device = device
        self.max_length = max_length
        if freeze:
            self.freeze()
        self.layer = layer
        if self.layer == "penultimate":
            raise NotImplementedError()
            self.layer_idx = 1

        self.antialias = antialias

        self.register_buffer('mean', torch.Tensor([0.48145466, 0.4578275, 0.40821073]), persistent=False)
        self.register_buffer('std', torch.Tensor([0.26862954, 0.26130258, 0.27577711]), persistent=False)
        self.ucg_rate = ucg_rate
        self.only_cls = only_cls
        self.use_proj = use_proj
        self.use_shuffle = use_shuffle

    def preprocess(self, x):
        # input [-1,1]
        # normalize to [0,1]
        x = kornia.geometry.resize(x, (224, 224),
                                   interpolation='bicubic', align_corners=True,
                                   antialias=self.antialias)
        x = (x + 1.) / 2.
        # renormalize according to clip
        x = kornia.enhance.normalize(x, self.mean, self.std)
        return x

    def freeze(self):
        self.model = self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, image, no_dropout=False):
        ## image: b c h w
        z = self.encode_with_vision_transformer(image)
        return z

    def encode_with_vision_transformer(self, x):
        x = self.preprocess(x)

        # to patches - whether to use dual patchnorm - https://arxiv.org/abs/2302.01327v1
        # if self.model.visual.input_patchnorm:
        #     # einops - rearrange(x, 'b c (h p1) (w p2) -> b (h w) (c p1 p2)')
        #     x = x.reshape(x.shape[0], x.shape[1], self.model.visual.grid_size[0], self.model.visual.patch_size[0],
        #                   self.model.visual.grid_size[1], self.model.visual.patch_size[1])
        #     x = x.permute(0, 2, 4, 1, 3, 5)
        #     x = x.reshape(x.shape[0], self.model.visual.grid_size[0] * self.model.visual.grid_size[1], -1)
        #     x = self.model.visual.patchnorm_pre_ln(x)
        #     x = self.model.visual.conv1(x)
        # else:
        x = self.model.visual.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]

        # class embeddings and positional embeddings
        x = torch.cat(
            [self.model.visual.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype,
                                                                         device=x.device),
             x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.model.visual.positional_embedding.to(x.dtype)

        # a patch_dropout of 0. would mean it is disabled and this function would do nothing but return what was passed in
        x = self.model.visual.patch_dropout(x)
        x = self.model.visual.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.model.visual.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        return x

# from PIL import Image
# import numpy as np
# example_image = Image.open("/home/ly24/code/py_model/Dataset/pku/train/inpainted_image/0.png").convert('RGB')
# example_tensor = torch.tensor(np.array(example_image), dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0
#
# # 初始化编码器
# encoder = FrozenOpenCLIPImageEmbedder()
#
# # 将图像输入编码器
# image_features = encoder.forward(example_tensor)
#
# # 打印特征向量的形状
# print(f"Image features shape: {image_features.shape}")