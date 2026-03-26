import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models.builder import NECKS
from mmcv.runner import BaseModule, auto_fp16
from mmcv.cnn import ConvModule, build_norm_layer

@NECKS.register_module()
class SimpleFPN(BaseModule):
    """
    完美复刻 Detectron2 官方 ViTDet 配置。
    Input: [feat_last] (ViT 最后一层输出)
    Output: (P2, P3, P4, P5, P6)
    """

    def __init__(self,
                 in_channels,      # ViT 输出通道 (e.g. 768)
                 out_channels,     # FPN 输出通道 (e.g. 256)
                 scale_factors=[4.0, 2.0, 1.0, 0.5], # 对应 P2, P3, P4, P5
                 num_outs=5,       # P2-P6 共5层
                 norm_cfg=dict(type='GN', num_groups=1), # 模拟 LN
                 init_cfg=None):
        super(SimpleFPN, self).__init__(init_cfg)
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.scale_factors = scale_factors
        self.num_outs = num_outs
        self.stages = nn.ModuleList()

        # --- 构建 P2, P3, P4, P5 ---
        for scale in scale_factors:
            layers = []
            
            # 1. 几何变换 (Resizing)
            if scale == 4.0: # P2: 4x 上采样 (2x -> GELU -> 2x)
                layers.extend([
                    nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2),
                    build_norm_layer(norm_cfg, in_channels // 2)[1],
                    nn.GELU(),
                    nn.ConvTranspose2d(in_channels // 2, in_channels // 4, kernel_size=2, stride=2)
                ])
                curr_dim = in_channels // 4
            elif scale == 2.0: # P3: 2x 上采样
                layers.append(nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2))
                curr_dim = in_channels // 2
            elif scale == 1.0: # P4: 保持
                curr_dim = in_channels
            elif scale == 0.5: # P5: 2x 下采样
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
                curr_dim = in_channels
            else:
                raise NotImplementedError(f"Scale {scale} not supported")

            # 2. 特征精修 (1x1 Conv + 3x3 Conv)
            layers.append(ConvModule(curr_dim, out_channels, 1, norm_cfg=norm_cfg, act_cfg=None))
            layers.append(ConvModule(out_channels, out_channels, 3, padding=1, norm_cfg=norm_cfg, act_cfg=None))

            self.stages.append(nn.Sequential(*layers))

        # --- 构建 Top Block (P6) ---
        # 对应配置中的 top_block=L(LastLevelMaxPool)()
        # 它作用于 P5 (list中的最后一个) 之上
        if num_outs > len(scale_factors):
            self.top_block = nn.MaxPool2d(kernel_size=1, stride=2, padding=0)
        else:
            self.top_block = None

    @auto_fp16()
    def forward(self, inputs):
        # 确保输入只取最后一层
        bottom_feat = inputs[-1]

        results = []
        # 生成 P2-P5
        for stage in self.stages:
            results.append(stage(bottom_feat))

        # 生成 P6 (LastLevelMaxPool)
        if self.top_block is not None:
            # P6 是对 P5 (results[-1]) 进行 MaxPool
            p6 = self.top_block(results[-1])
            results.append(p6)

        return tuple(results)
