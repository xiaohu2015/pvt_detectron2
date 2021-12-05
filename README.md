# pvt_detectron2

Pyramid Vision Transformer for Object Detection by detectron2, together with [Conditional Positional Encodings for Vision Transformers](https://arxiv.org/abs/2102.10882) and [Twins: Revisiting the Design of Spatial Attention in Vision Transformers](https://arxiv.org/pdf/2104.13840.pdf).

This repo contains the supported code and configuration files to reproduce object detection results of [Pyramid Vision Transformer: A Versatile Backbone for Dense Prediction without Convolutions](https://arxiv.org/pdf/2102.12122.pdf). It is based on [detectron2](https://github.com/facebookresearch/detectron2).


## Results and Models

### RetinaNet

| Backbone | Pretrain | Lr Schd | box mAP | mask mAP | #params | FLOPs | config | log | model |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |:---: |
| PVT-Small | ImageNet-1K | 1x | 41.6| - | 34.2M | 226G | [config](configs/pvt/pvt_small_FPN_1x.yaml) | - | [model](https://github.com/xiaohu2015/pvt_detectron2/releases/download/v0.5/retinanet_pvt_small_1k.pth) |
| PCPVT-Small | ImageNet-1K | 1x | 44.2| - | 34.4M | 226G | [config](configs/pvt/pcpvt_small_FPN_1x.yaml) | - | [model](https://github.com/xiaohu2015/pvt_detectron2/releases/download/v0.9/retinanet_pcpvt_small_coco.pth) |
| Twins-SVT-Small | ImageNet-1K | 1x | 43.1| - | 34.3M | 209G | [config](configs/pvt/gvt_small_FPN_1x.yaml) | - | [model](https://github.com/xiaohu2015/pvt_detectron2/releases/download/v1.0/retinanet_gvt_small_coco.pth) |
| PVTV2-b2 | ImageNet-1K | 1x | | - |  |  | [config](configs/pvt/pvtv2_b2_FPN_1x.yaml) | - | [model](-) |




***The box mAP (41.6 vs 40.4) is better than implementation of the mmdetection version (need checked?)***

The performance gap maybe lie in the training strategy of `resize`:

```
# The resize in mmdetection is single scale:
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]

# The resize in detectron2 for retinanet is multi-scale:

# Size of the smallest side of the image during training
_C.INPUT.MIN_SIZE_TRAIN = (640, 672, 704, 736, 768, 800)
# Sample size of smallest side by choice or random selection from range give by
# INPUT.MIN_SIZE_TRAIN
_C.INPUT.MIN_SIZE_TRAIN_SAMPLING = "choice"
# Maximum size of the side of the image during training
_C.INPUT.MAX_SIZE_TRAIN = 1333
# Size of the smallest side of the image during testing. Set to zero to disable resize in testing.
_C.INPUT.MIN_SIZE_TEST = 800
# Maximum size of the side of the image during testing
_C.INPUT.MAX_SIZE_TEST = 1333
# Mode for flipping images used in data augmentation during training
# choose one of ["horizontal, "vertical", "none"]
_C.INPUT.RANDOM_FLIP = "horizontal"
```


## Usage
Please refer to [get_started.md](https://detectron2.readthedocs.io/en/latest/tutorials/getting_started.html) for installation and dataset preparation.

note: you need convert the original pretrained weights to d2 format by [convert_to_d2.py](convert_to_d2.py)

## References
- [PVT](https://github.com/whai362/PVT)
- [detectron2](https://github.com/facebookresearch/detectron2)
- [CPVT](https://github.com/Meituan-AutoML/CPVT)
- [Twins](https://github.com/Meituan-AutoML/Twins)
