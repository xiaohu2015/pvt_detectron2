# -*- coding: utf-8 -*-                                                                                       

from detectron2.config import CfgNode as CN

def add_pvt_config(cfg):
    # PVT backbone
    cfg.MODEL.PVT = CN()
    cfg.MODEL.PVT.OUT_FEATURES = ["stage1", "stage2", "stage3", "stage4"]

    cfg.MODEL.PVT.PATCH_SIZE = 4 
    cfg.MODEL.PVT.EMBED_DIMS = [64, 128, 320, 512]
    cfg.MODEL.PVT.NUM_HEADS = [1, 2, 5, 8]
    cfg.MODEL.PVT.MLP_RATIOS = [8, 8, 4, 4]
    cfg.MODEL.PVT.DEPTHS = [3, 4, 6, 3]
    cfg.MODEL.PVT.SR_RATIOS = [8, 4, 2, 1]
    
    cfg.MODEL.PVT.WSS = [7, 7, 7, 7]
    
    # addation
    cfg.SOLVER.OPTIMIZER = "AdamW"
