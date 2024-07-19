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

1. Create a conda enviroment:

```bash
conda env create -n desigen python=3.8
conda activate desigen
pip install -r requirements.txt
```

2. Download [Bastnet](https://github.com/xuebinqin/BASNet) checkpoint into `saliency` folder.


## Dataset & Pre-process


## Usage

### Training

```bash
cd background/
# config accelerate first
accelerate config
# train the background generator
sh train.sh
```


### Testing and Evaluating

Generate background images with the prompts on validation set and evaluate them by the proposed metrics: FID, Salient Ratio and CLIP Score.

The pretrained saliency dectection mode can be download on [Basnet](https://github.com/xuebinqin/BASNet) and placed on `salliency` directory.

```bash
sh test.sh
```
### Testing with a single image




## Citation

```tex

```
