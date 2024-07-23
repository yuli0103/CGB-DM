<h1 align="center"> CGB-DM: Content and Graphic Balance Layout Generation with Transformer-based Diffusion Model</h1>

<div align="center">

 <a href=''><img src='https://img.shields.io/badge/arxiv-paper-red'></a> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
 <a href='https://yuli0103.github.io/CGB-DM.github.io/'><img src='https://img.shields.io/badge/Project-Page-Green'></a> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
 
_**[Yu Li](https://github.com/yuli0103/), [Yifan Chen](), [Gongye Liu](https://github.com/GongyeLiu), [Jie Wu](), [Yujiu Yang*](https://scholar.google.com/citations?user=4gH3sxsAAAAJ&hl=zh-CN&oi=ao)**_

Tsinghua University
<br>
(* corresponding authors)

</div>

<p align="center">
  <img width="100%" src="docs/teaser.jpg">
</p>

## Setup
 Create a conda enviroment:

```bash

```


## Dataset & Checkpoint
### Download data
   
   Here we provide links to our organized `pku` and `cgl` datasets <url>, which include inpainted images, saliency maps, ground truth labels, and detected saliency bounding box data.
   ```
   dataset/
   ├─ pku/
   │  ├─ csv/
   │  │  ├─ train.csv/
   │  │  ├─ train_sal.csv/
   │  │  ├─ ...
   │  ├─ train/
   │  │  ├─ inpaint/
   │  │  ├─ saliency/
   │  │  ├─ saliency_sub/
   │  ├─ test_anno/
   │  │  ├─ ...
   │  ├─ test_unanno/
   │  │  ├─ image_canvas/
   │  │  ├─ saliency/
   │  │  ├─ saliency_sub/
   │  ├─ val/
   │  │  ├─ ...
   ├─ cgl/
   ├─ ...
   ```
### Download pre-trained weights
   
   Links from <>, which include the weights for CGB-DM (Ours), as well as the weights for the saliency detection algorithms ISNet and BASNet.
   
### Preprocess with your data
   
- Image inpainting: run `generate_inpaint_img.py` and specify the `input_dir`, `mask_dir`, and `output_dir`.
- Saliency detection: run `saliency_detection.py` and specify the `WEIGHT_ROOT`.
- Detect saliency bounding box: run `generate_sal_box.py` and specify the `input_dir`,  and `output_dir`.

## Usage
### Modify the configuration file
In the `configs/*.yaml` files, you need to replace some paths with your own. This includes:

- `paths.base` (dataset path)
- `base_check_dir` (directory to save checkpoints)
- `imgname_order_dir` (directory to load image names for metric calculation)
- `save_imgs_dir` (directory to save rendered images)

### Training
Run the commands in terminal
```bash
# You can choose the training dataset and task
python scripts/train.py --gpuid 0 --dataset pku --task uncond
```

### Inference 
Run the commands in terminal
```bash
# You can choose the test dataset, type and corresponding task
python scripts/test.py --gpuid 0 --dataset pku --anno unanno --task uncond --check_path ''
```
The meaning of `anno` is to select either annotated or unannotated test sets. It is important to note that unannotated test sets can only be used for `uncond` tasks, as they lack ground truth labels.
### Inference with a single image
Run the commands in terminal
```bash
python scripts/run_single_image.py --gpuid 0 --render_style pku --image_path ''  --check_path ''
```
`render_style` includes pku and cgl. 

In `image_path`, select the test image, and in `check_path`, select the model weights.

## Citation

```tex
@misc{li2024cgbdmcontentgraphicbalance,
      title={CGB-DM: Content and Graphic Balance Layout Generation with Transformer-based Diffusion Model}, 
      author={Yu Li and Yifan Chen and Gongye Liu and Jie Wu and Yujiu Yang},
      year={2024},
      eprint={2407.15233},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2407.15233}, 
     }
```
