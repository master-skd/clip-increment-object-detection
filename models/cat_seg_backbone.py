import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange

from mmdet.models.builder import BACKBONES
from mmcv.runner import BaseModule
from mmcv.cnn import build_norm_layer

from detectron2.config import get_cfg
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import CfgNode as CN

def add_deeplab_config(cfg):
    cfg.MODEL.DEEPLAB = CN()
    cfg.MODEL.DEEPLAB.NORM = "BN"
    cfg.MODEL.DEEPLAB.PROJECT_NAME = "deeplab"
    cfg.MODEL.DEEPLAB.USE_ASPP = True
    cfg.MODEL.DEEPLAB.ASPP_DROPOUT = 0.1
    cfg.MODEL.DEEPLAB.ASPP_ATROUS_RATES = [6, 12, 18]

from cat_seg import add_cat_seg_config

@BACKBONES.register_module()
class CATSegCLIPViT(BaseModule):
    def __init__(self, 
                 config_path, 
                 checkpoint_path, 
                 class_json_path,
                 out_indices=[3, 5, 7, 11], 
                 norm_cfg=None, 
                 width=768,
                 **kwargs):
        super().__init__(**kwargs)

        self.vit_layers = out_indices

        cfg = get_cfg()
        add_deeplab_config(cfg)
        add_cat_seg_config(cfg)
        cfg.merge_from_file(config_path)
        cfg.MODEL.DEVICE = 'cuda:0'
        cfg.MODEL.SEM_SEG_HEAD.TRAIN_CLASS_JSON = class_json_path
        cfg.MODEL.SEM_SEG_HEAD.TEST_CLASS_JSON = class_json_path

        self.checkpoint_path = checkpoint_path
        self.full_model = build_model(cfg)
        DetectionCheckpointer(self.full_model).resume_or_load(checkpoint_path, resume=False)
        

        # 2. 清除源码自带的 Hooks (因为它只抓 3 和 7，不够我们要的)
        # 必须先清空，否则会有冲突
        for m in self.full_model.sem_seg_head.predictor.clip_model.visual.transformer.resblocks:
            m._forward_hooks.clear()

        # 3. 注册我们需要的所有 Hooks
        self.clip_feats = {}    # 存 3, 5, 7, 11
        self.decoder_feat = None # 存 Decoder2

        # A. 注册 CLIP Hooks
        target_layers = out_indices
        resblocks = self.full_model.sem_seg_head.predictor.clip_model.visual.transformer.resblocks

        # print(f"resblocks: {len(resblocks)}")
        
        for idx in target_layers:
            # 使用闭包绑定 idx
            def get_clip_hook(layer_idx):
                def hook_fn(_module, _input, output):
                    # output: [N+1, B, C] -> 去掉 CLS -> 变回 BCHW
                    # 硬编码 H=24 (384/16)，因为你输入固定 384
                    feat = output[1:, :, :] 
                    feat = rearrange(feat, "(H W) B C -> B C H W", H=24)
                    self.clip_feats[layer_idx] = feat
                return hook_fn
            
            resblocks[idx].register_forward_hook(get_clip_hook(idx))

        # B. 注册ln_post hook (11层出来的dense feature)
        self.ln_post_feat = None
        self.full_model.sem_seg_head.predictor.clip_model.visual.ln_post.register_forward_hook(
            lambda _m, _i, o: setattr(self, 'ln_post_feat', o)
        )

        # # C. 注册 Decoder Hook
        # self.full_model.sem_seg_head.predictor.transformer.decoder2.register_forward_hook(
        #     lambda _m, _i, o: setattr(self, 'decoder_feat', o)
        # )

        # D. 注册投影后的dense feature
        self.full_model.sem_seg_head.predictor.clip_model.visual.register_forward_hook(
            lambda _m, _i, o: setattr(self, 'dense_map', o)
        )

        self.interpolate1 = nn.Sequential(
            nn.ConvTranspose2d(width, width, kernel_size=2, stride=2),
            build_norm_layer(norm_cfg, width)[1] if norm_cfg else nn.Identity(),
            nn.GELU(),
            nn.ConvTranspose2d(width, width, kernel_size=2, stride=2),
        )
        # 上采样2倍
        self.interpolate2 = nn.ConvTranspose2d(width, width, kernel_size=2, stride=2)
        self.interpolate3 = nn.Identity()
        self.interpolate4 = nn.MaxPool2d(kernel_size=2, stride=2)

    def init_weights(self):
        DetectionCheckpointer(self.full_model).resume_or_load(self.checkpoint_path, resume=False)
        self.full_model.eval()
        for param in self.full_model.parameters():
            param.requires_grad = False

    def train(self, mode=True):
        self.training = mode
        self.full_model.train(mode)
        self.interpolate1.train(mode)
        self.interpolate2.train(mode)
        self.interpolate3.train(mode)
        self.interpolate4.train(mode)
        return self

    def forward(self, x):
        bs, _, h, w = x.shape
        self.clip_feats = {}
        self.decoder_feat = None

        # clip前向传播
        with torch.no_grad():
            clip_out = self.full_model.sem_seg_head.predictor.clip_model.encode_image(x, dense=True)

        # 准备decoder的输出
        feat3 = self.clip_feats[3]
        feat7 = self.clip_feats[7]
        feat11 = self.ln_post_feat[:, 1:, :]
        self.clip_feats[11] = rearrange(feat11, "B (H W) C -> B C H W", H=24)
        res3 = rearrange(clip_out[:, 1:, :], "B (H W) C -> B C H W", H=24)
        res4 = self.full_model.upsample1(feat3)
        res5 = self.full_model.upsample2(feat7)

        # 运行decoder
        features_dict = {'res5': res5, "res4": res4, "res3": res3}
        with torch.no_grad():
            logits, topk_indices = self.full_model.sem_seg_head(clip_out, features_dict)

        outs = [self.clip_feats[3],
            self.clip_feats[5],
            self.clip_feats[7],
            self.clip_feats[11],]

        # outs = [self.clip_feats[11]]

        for idx, out in enumerate(outs):
            interpolate = getattr(self, f"interpolate{idx+1}")
            outs[idx] = interpolate(out.detach())

        outs.append(logits)
        outs.append(topk_indices)
        # outs.append(corr)

        if not self.training:
            x = self.dense_map[:, 1:]
            x = F.normalize(x, dim=-1)
            feature_map = x.view(bs, h // 16, w // 16, -1).permute(0, 3, 1, 2)
        else:
            feature_map = None
        outs.append(feature_map)

        return tuple(outs)
