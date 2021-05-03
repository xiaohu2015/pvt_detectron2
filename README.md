# pvt_detectron2

Pyramid Vision Transformer for Object Detection by detectron2

This repo contains the supported code and configuration files to reproduce object detection results of [Pyramid Vision Transformer: A Versatile Backbone for Dense Prediction without Convolutions](https://arxiv.org/pdf/2102.12122.pdf). It is based on [detectron2](https://github.com/facebookresearch/detectron2).


## Results and Models

### RetinaNet

| Backbone | Pretrain | Lr Schd | box mAP | mask mAP | #params | FLOPs | config | log | model |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |:---: |
| PVT-Small | ImageNet-1K | 1x | 41.6| - | 34.2M | 226G | [config](configs/pvt/pvt_small_FPN_1x.yaml) | - | - |

***The box mAP (41.6 vs 40.4) is better than implementation of the mmdetection version (need checked?)***


## Usage
Please refer to [get_started.md](https://detectron2.readthedocs.io/en/latest/tutorials/getting_started.html) for installation and dataset preparation.

note: you need convert the original pretrained weights to d2 format by [convert_to_d2.py](convert_to_d2.py)

## References
- [PVT](https://github.com/whai362/PVT)
- [detectron2](https://github.com/facebookresearch/detectron2)
