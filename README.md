# VLDet: Learning Object-Language Alignments for Open-Vocabulary Object Detection

<p align="center"> <img src='docs/readme.jpeg' align="center" height="200px"> </p>

> [**Learning Object-Language Alignments for Open-Vocabulary Object Detection**](arXiv Link),               
> Author List                
> *arXiv technical report ([arXiv Link](arXiv Link))*   

## Installation

### Requirements
- Linux or macOS with Python ≥ 3.7
- PyTorch ≥ 1.9.
  Install them together at [pytorch.org](https://pytorch.org) to make sure of this. Note, please check
  PyTorch version matches that is required by Detectron2.
- Detectron2: follow [Detectron2 installation instructions](https://detectron2.readthedocs.io/tutorials/install.html).

### Example conda environment setup
```bash
conda create --name VLDet python=3.7 -y
conda activate VLDet
conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch-lts -c nvidia

# under your working directory

git clone https://github.com/clin1223/VLDet.git
cd VLDet
cd detectron2
pip install -e .
cd ..
pip install -r requirements.txt
```

## Features
- Directly learn an open-vocabulary object detector from image-text pairs by formulating the task as a bipartite matching problem.

- State-of-the-art results on Open-vocabulary LVIS and Open-vocabulary COCO.

- Scaling and extending novel object vocabulary easily.


## Benchmark evaluation and training

Please first [prepare datasets](prepare_datasets.md).

The VLDet models are finetuned on the corresponding [Box-Supervised models](https://drive.google.com/drive/folders/1ngb1mBOUvFpkcUM7D3bgIkMdUj2W5FUa?usp=sharing) (indicated by MODEL.WEIGHTS in the config files). Please train or download the Box-Supervised model and place them under VLDet_ROOT/models/ before training the VLDet models.

To train a model, run

```
python train_net.py --num-gpus 8 --config-file /path/to/config/name.yaml
``` 

To evaluate a model with a trained/ pretrained model, run 

```
python train_net.py --num-gpus 8 --config-file /path/to/config/name.yaml --eval-only MODEL.WEIGHTS /path/to/weight.pth
``` 

Download the trained network weights [here](https://drive.google.com/drive/folders/1ngb1mBOUvFpkcUM7D3bgIkMdUj2W5FUa?usp=sharing).

| OV_COCO  | box mAP50 | box mAP50_novel |
|----------|-----------|-----------------|
| [config_RN50](configs/VLDet_OVCOCO_CLIP_R50_1x_caption.yaml) | 45.8      | 32.0            |

| OV_LVIS       | mask mAP_all | mask mAP_novel |
| ------------- | ------------ | -------------- |
| [config_RN50](configs/VLDet_LbaseCCcap_CLIP_R5021k_640b64_2x_ft4x_caption.yaml)   | 30.1         | 21.7           |
| [config_Swin-B](configs/VLDet_LbaseI_CLIP_SwinB_896b32_2x_ft4x_caption.yaml) | 38.1         | 26.3           |
 

## Citation

If you find this project useful for your research, please use the following BibTeX entry.

## Acknowledgement
This repository was built on top of [Detectron2](https://github.com/facebookresearch/detectron2), [Detic](https://github.com/facebookresearch/Detic.git), [RegionCLIP](https://github.com/microsoft/RegionCLIP.git) and [OVR-CNN](https://github.com/alirezazareian/ovr-cnn). We thank for their hard work.