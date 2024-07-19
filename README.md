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
conda env create -n desigen python=3.8
conda activate desigen
pip install -r requirements.txt
```


## Dataset & Checkpoint
1. Download data
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
2. Download pre-trained weights from <>, which include the weights for CGB-DM (Ours), as well as the weights for the saliency detection algorithms ISNet and BASNet.
3. how to preprocess with your data
   (1) inpaint
   (2) saliency detection
   (3) get saliency detection box

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

```
