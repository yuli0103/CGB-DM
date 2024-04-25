import logging
import torch
import torch.nn as nn
import timm
from torchvision import transforms
import os
import numpy as np
from PIL import Image
from einops import repeat
logger = logging.getLogger(__name__)
class _TimmVGGWrapper(nn.Module):
    """
    Wrapper class to adjust singleton pattern of SingletonVGG.
    """

    def __init__(self) -> None:
        super().__init__()
        backbone_tag = "hf_hub:timm/vgg16.tv_in1k"
        logger.info(f"Loading timm model from {backbone_tag=}")
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = timm.create_model(backbone_tag, pretrained=True, num_classes=0).to(
            torch.device(device_name)
        )
        self.model.eval()
        data_config = timm.data.resolve_model_data_config(self.model)
        # transform = timm.data.create_transform(**data_config, is_training=False)
        # transform = [
        #     t for t in transform.transforms if not isinstance(t, transforms.ToTensor)
        # ]
        # self.transform = transforms.Compose(transform)
        self.transform = transforms.Compose(
            [
                transforms.Resize(
                    (224, 224), transforms.InterpolationMode.BICUBIC, antialias=True
                ),
                transforms.Normalize(mean=data_config["mean"], std=data_config["std"]),
            ]
        )
        logger.info(f"Transform of VGG: {self.transform}")

    def forward(self, images):
        assert images.ndim == 4
        h = torch.stack([self.transform(image) for image in images])
        h = self.model(h)
        return h
class SingletonTimmVGG(object):
    """
    Follow singleton pattern to avoid loading VGG16 multiple times.
    """

    def __new__(cls):  # type: ignore
        if not hasattr(cls, "instance"):
            cls.instance = _TimmVGGWrapper()
        return cls.instance  # type: ignore

# from image_encoder.rshm_vgg import SingletonTimmVGG
def compute_rshm(img_names, img_size, clses, boxes, test_bg_dir):
    """
    Measure the occlusion levels of key subjects (denoted as R_{shm}).
    We feed the salient images with or without layout regions masked
    into a pretrained VGG16, and calculate L2 distance between their output logits.
    Lower values are generally better (in 0.0 - 1.0 range).
    """
    # unlike compute_saliency_aware_metrics, do batch processing as much as possible
    vgg16 = SingletonTimmVGG()  # type: ignore
    w, h = img_size
    layout_masks = []
    images = []
    for idx, name in enumerate(img_names):
        rgb_image = Image.open(os.path.join(test_bg_dir, name)).convert("RGB").resize((w, h))
        rgb_image = torch.from_numpy(np.array(rgb_image) / 255).float()
        images.append(rgb_image)

        cls = np.array(clses[idx].cpu(), dtype=int)
        box = np.array(boxes[idx].cpu(), dtype=int)
        mask = (cls > 0).reshape(-1)
        mask_box = box[mask]

        bbox_mask = torch.full((1, h, w), fill_value=False)
        for mb in mask_box:
            xl, yl, xr, yr = mb
            bbox_mask[yl:yr, xl:xr] = True
        layout_masks.append(bbox_mask)

    images = torch.stack(images, dim=0)
    layout_masks = torch.stack(layout_masks)  # type: ignore
    layout_masks = repeat(layout_masks, "b 1 h w -> b c h w", c=3)
    images_masked = images.clone()
    images_masked[layout_masks] = 0.5  # mask by gray values, is it ok?

    with torch.no_grad():
        logits = vgg16(images.cuda()).detach().cpu()  # type: ignore
        logits_masked = vgg16(images_masked.cuda()).detach().cpu()  # type: ignore

    dist = torch.linalg.vector_norm(
        logits_masked - logits, dim=1
    )  # L2 dist from (B, 1000)
    return {"R_{shm} (vgg distance)": dist.tolist()}