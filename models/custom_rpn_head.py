from typing import List, Optional, Sequence
import numpy as np

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmdet.models.builder import HEADS
from mmdet.models.dense_heads import RPNHead
from mmdet.core import bbox2result

@HEADS.register_module()
class CustomRPNHead(RPNHead):
    """RPN head that processes class-decoupled multi-scale features.

    接受形状为 ``[B, T, C, H, W]`` 的类别解耦特征，每个类别独立进行
    anchor matching，仅使用对应类别的 gt 做 IoU 正负样本划分，并在生成
    proposals 后再在图像维度进行聚合。
    """

    def __init__(self,
                 norm_cfg=None,
                 *args,
                 **kwargs):
        self.norm_cfg = norm_cfg
        super().__init__(*args, **kwargs)

    def _init_layers(self):
        """Initialize layers of the head."""
        if self.num_convs > 1:
            rpn_convs = []
            for i in range(self.num_convs):
                if i == 0:
                    in_channels = self.in_channels
                else:
                    in_channels = self.feat_channels
                rpn_convs.append(
                    ConvModule(
                        in_channels,
                        self.feat_channels,
                        3,
                        padding=1,
                        inplace=False,
                        norm_cfg=self.norm_cfg))
            self.rpn_conv = nn.Sequential(*rpn_convs)
        else:
            self.rpn_conv = nn.Conv2d(
                self.in_channels, self.feat_channels, 3, padding=1)
        self.rpn_cls = nn.Conv2d(self.feat_channels,
                                 self.num_base_priors * self.cls_out_channels,
                                 1)
        self.rpn_reg = nn.Conv2d(self.feat_channels, self.num_base_priors * 4,
                                 1)

    def forward(self, feats):
        reshaped_feats = []
        self._cached_num_classes = None
        for feat in feats:
            if feat.dim() == 5:
                B, T, C, H, W = feat.shape
                if self._cached_num_classes is None:
                    self._cached_num_classes = T
                else:
                    assert self._cached_num_classes == T, \
                        f"Expected {self._cached_num_classes} classes, got {T}"
                reshaped_feat = feat.reshape(B * T, C, H, W)
                reshaped_feats.append(reshaped_feat)
            elif feat.dim() == 4:
                reshaped_feats.append(feat)
            else:
                raise ValueError(f"Unexpected feature dimension: {feat.dim()}, shape: {feat.shape}")
        return super().forward(tuple(reshaped_feats))

    def _infer_num_classes(self, cls_scores, img_metas):
        batch_size = len(img_metas)          # 原始bsz
        total_batch = cls_scores[0].size(0)  # 实际用于rpn的bsz=bsz*T
        assert total_batch % batch_size == 0, \
            f"Total batch {total_batch} cannot be divided by num images {batch_size}"
        return total_batch // batch_size

    def _repeat_items(self, items: Optional[Sequence], num_repeats: int):
        if items is None:
            return None
        repeated = []
        for item in items:
            repeated.extend([item] * num_repeats)
        return repeated

    def _split_gt_bboxes_by_class(self,
                                  gt_bboxes: List[torch.Tensor],
                                  gt_labels: List[torch.Tensor],
                                  num_classes: int):
        split_bboxes = []
        for bboxes, labels in zip(gt_bboxes, gt_labels):
            assert labels is not None, \
                'gt_labels is required for class-aware RPN training.'
            if labels.numel() == 0:
                empty = bboxes.new_zeros((0, 4))
                split_bboxes.extend([empty.clone() for _ in range(num_classes)])
                continue
            for cls_id in range(num_classes):
                mask = labels == cls_id
                if mask.any():
                    split_bboxes.append(bboxes[mask])
                else:
                    split_bboxes.append(bboxes.new_zeros((0, 4)))
        return split_bboxes

    def loss(self,
             cls_scores,
             bbox_preds,
             gt_bboxes,
             gt_labels,
             img_metas,
             gt_bboxes_ignore=None):
        """Override loss to expand GTs per class before调用标准RPN损失."""
        num_classes = self._infer_num_classes(cls_scores, img_metas)
        expanded_img_metas = self._repeat_items(img_metas, num_classes)
        expanded_gt_bboxes = self._split_gt_bboxes_by_class(
            gt_bboxes, gt_labels, num_classes)
        expanded_gt_bboxes_ignore = self._repeat_items(
            gt_bboxes_ignore, num_classes)

        losses = super().loss(
            cls_scores,
            bbox_preds,
            expanded_gt_bboxes,
            expanded_img_metas,
            gt_bboxes_ignore=expanded_gt_bboxes_ignore)
        return losses

    def _merge_proposals(self,
                         proposal_list: List[torch.Tensor],
                         batch_size: int,
                         num_classes: int,
                         max_per_img: int,
                         add_class_labels: bool = False):
        """Merge proposals from different classes.
        
        Args:
            proposal_list: List of proposals, each is (N, 5) [x1, y1, x2, y2, score]
            batch_size: Number of images
            num_classes: Number of classes
            max_per_img: Maximum proposals per image
            add_class_labels: If True, add class labels to proposals (N, 6) [x1, y1, x2, y2, score, cls]
        
        Returns:
            If add_class_labels=False: List of (N, 5) proposals
            If add_class_labels=True: List of (N, 6) proposals with class labels
        """
        merged = []
        ptr = 0
        for _ in range(batch_size):
            class_props = proposal_list[ptr:ptr + num_classes]
            ptr += num_classes
            if len(class_props) == 0:
                if add_class_labels:
                    merged.append(proposal_list[0].new_zeros((0, 6)))
                else:
                    merged.append(proposal_list[0].new_zeros((0, 5)))
                continue
            
            # Add class labels to each proposal
            if add_class_labels:
                labeled_props = []
                for cls_id, props in enumerate(class_props):
                    if props.size(0) > 0:
                        # Add class label: (N, 5) -> (N, 6)
                        cls_labels = props.new_full((props.size(0), 1), cls_id, dtype=torch.float)
                        props_with_cls = torch.cat([props, cls_labels], dim=1)
                        labeled_props.append(props_with_cls)
                if len(labeled_props) > 0:
                    merged_props = torch.cat(labeled_props, dim=0)
                else:
                    merged_props = class_props[0].new_zeros((0, 6))
            else:
                merged_props = torch.cat(class_props, dim=0)
            
            if merged_props.size(0) > max_per_img > 0:
                scores = merged_props[:, -2] if add_class_labels else merged_props[:, -1]
                _, topk_inds = scores.topk(max_per_img)
                merged_props = merged_props[topk_inds]
            merged.append(merged_props)
        return merged

    def get_bboxes(self,
                   cls_scores,
                   bbox_preds,
                   score_factors=None,
                   img_metas=None,
                   cfg=None,
                   rescale=False,
                   with_nms=True,
                   **kwargs):
        num_classes = self._infer_num_classes(cls_scores, img_metas)
        expanded_img_metas = self._repeat_items(img_metas, num_classes)
        cfg = self.test_cfg if cfg is None else cfg
        proposal_list = super().get_bboxes(
            cls_scores,
            bbox_preds,
            score_factors=score_factors,
            img_metas=expanded_img_metas,
            cfg=cfg,
            rescale=rescale,
            with_nms=with_nms,
            **kwargs)
        batch_size = len(img_metas)
        return self._merge_proposals(
            proposal_list, batch_size, num_classes, cfg.max_per_img)
    
    def simple_test_rpn(self, x, img_metas):
        """Test RPN and return proposals for each class independently.
        
        Args:
            x: List of (B, T, C, H, W) class-decoupled features
            img_metas: List of image meta info
        
        Returns:
            List of proposals per image, each image contains a list of 
            num_classes proposals. Each proposal is a tensor of shape (N, 5)
            [x1, y1, x2, y2, score] for that class.
        """
        rpn_outs = self(x)
        num_classes = self._infer_num_classes(rpn_outs[0], img_metas)
        batch_size = len(img_metas)
        
        # Get proposals from each class
        expanded_img_metas = self._repeat_items(img_metas, num_classes)
        cfg = self.test_cfg
        
        # Get proposals from each class (without merging)
        proposal_list_per_class = super().get_bboxes(
            *rpn_outs,
            img_metas=expanded_img_metas,
            cfg=cfg,
            rescale=False,
            with_nms=True)
        
        # Group proposals by image: each image has num_classes proposals
        results = []
        for img_id in range(batch_size):
            img_proposals = []
            for cls_id in range(num_classes):
                class_idx = img_id * num_classes + cls_id
                if class_idx < len(proposal_list_per_class):
                    # Each proposal is (N, 5) [x1, y1, x2, y2, score]
                    props = proposal_list_per_class[class_idx]
                    img_proposals.append(props)
                else:
                    # Empty proposal for this class
                    img_proposals.append(proposal_list_per_class[0].new_zeros((0, 5)))
            results.append(img_proposals)
        
        return results
