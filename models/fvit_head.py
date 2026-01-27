import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from einops import rearrange

from mmdet.models.roi_heads import StandardRoIHead, ConvFCBBoxHead
from mmdet.models.builder import HEADS, build_roi_extractor
from mmdet.core import bbox2roi, bbox2result
from mmcv.runner import force_fp32
from mmdet.core import multiclass_nms


@HEADS.register_module()
class FViTBBoxHead(ConvFCBBoxHead):
    def __init__(self,
                fixed_temperature=0,
                learned_temperature=50.0,
                class_embed=None,
                seen_classes=None,
                all_classes=None,
                vlm_temperature=100.0,
                alpha=0.2,
                beta=0.45,
                learn_bg=False,
                **kwargs):
        super(FViTBBoxHead, self).__init__(**kwargs)
        if fixed_temperature != 0:
            self.detect_temperature = fixed_temperature
        else:
            self.detect_temperature = torch.nn.Parameter(
                torch.tensor(learned_temperature))
        # 固定参数，仅在推理时使用
        self.vlm_temperature = vlm_temperature
        self.alpha = alpha
        self.beta = beta
        self.learn_bg = learn_bg

        assert class_embed is not None, 'class embed is None'
        self.seen_classes = json.load(open(seen_classes)) + ['background']
        self.all_classes = json.load(open(all_classes)) + ['background']
        idx = [self.all_classes.index(seen) for seen in self.seen_classes]

        # 创建类别掩码，标记未见类别的位置
        self.base_idx = torch.zeros(len(self.all_classes), dtype=bool)
        self.base_idx[idx] = True
        self.novel_idx = self.base_idx == False

        # 构建text embedding 矩阵
        class_embed = torch.load(class_embed)
        all_embed = [class_embed[name] for name in self.all_classes]
        all_embed = torch.stack(all_embed, dim=0).permute(1, 0).contiguous()
        all_embed = F.normalize(all_embed, p=2, dim=0)

        self.register_buffer('all_embeddings', all_embed)
        if learn_bg:
            self.bg_embedding = nn.Parameter(all_embed[:, -1:])

    @property
    def all_embed(self):
        if self.learn_bg:
            bg_embed = F.normalize(self.bg_embedding, dim=0)
            return torch.cat([self.all_embeddings[:, :-1], bg_embed], dim=1)
        else:
            return self.all_embeddings

    def forward(self, x, vlm_box_feats=None):
        # x: ROIAlign提取的特征, [N, C, H, W], N为proposal的数量, H、W为RoI Align后的空间尺寸
        # vlm_box_feats: vlm roi特征, [N, C], 仅推理时使用
        all_embed = self.all_embed.type_as(x)

        # 从backbone的高层CLIP特征中，针对每个proposal提取的区域特征，仅在推理时使用
        if vlm_box_feats is not None:
            assert not self.training
            # 归一化, 即后续做cosine similarity
            normalized_vlm_box_feats = F.normalize(vlm_box_feats, dim=-1, p=2)
        else:
            normalized_vlm_box_feats = None

        # 共享卷积层, conv(C, C, 3, 1)
        if self.num_shared_convs > 0:
            for conv in self.shared_convs:
                x = conv(x)  # [N, C, H, W]

        # 共享全连接层
        if self.num_shared_fcs > 0:
            if self.with_avg_pool:
                x = self.avg_pool(x)  # [N, C, 1, 1]

            x = x.flatten(1)  # [N, C]

            for fc in self.shared_fcs:
                x = self.relu(fc(x))
        # 上述作用是提取分类和回归任务都需要的公共特征
        # [N, C], 与前面的C不一样，前面的C是bbox_roi_extractor的输出通道, 此时的C需要与text embedding的通道维度一致

        # separate branches
        x_cls = x
        x_reg = x

        # 分类分支
        for conv in self.cls_convs:
            x_cls = conv(x_cls)
        if x_cls.dim() > 2:
            if self.with_avg_pool:
                x_cls = self.avg_pool(x_cls)
            x_cls = x_cls.flatten(1)
        for fc in self.cls_fcs:
            x_cls = self.relu(fc(x_cls))  # FIXME: should discard the last relu
        # [N, C]

        # 回归分支
        for conv in self.reg_convs:
            x_reg = conv(x_reg)
        if x_reg.dim() > 2:
            if self.with_avg_pool:
                x_reg = self.avg_pool(x_reg)
            x_reg = x_reg.flatten(1)
        for fc in self.reg_fcs:
            x_reg = self.relu(fc(x_reg))
        # [N, C]

        # 回归分支的输出, [N, 4]
        bbox_pred = self.fc_reg(x_reg) if self.with_reg else None
        # 通过归一化来计算cosine similarity
        normalized_x_cls = x_cls / x_cls.norm(dim=-1, keepdim=True)
        # [N, C], 每个proposal对每个类别的得分
        cls_score = normalized_x_cls @ all_embed * self.detect_temperature

        if not self.training and normalized_vlm_box_feats is not None:
            cls_score = cls_score.softmax(dim=-1)
            vlm_score = normalized_vlm_box_feats @ all_embed * self.vlm_temperature
            vlm_score = vlm_score.softmax(dim=-1)

            cls_score[:, self.base_idx] = cls_score[:, self.base_idx] ** (
                    1 - self.alpha) * vlm_score[:, self.base_idx] ** self.alpha
            cls_score[:, self.novel_idx] = cls_score[:, self.novel_idx] ** (
                    1 - self.beta) * vlm_score[:, self.novel_idx] ** self.beta

        return cls_score, bbox_pred

    @force_fp32(apply_to=('cls_score', 'bbox_pred'))
    def get_bboxes(self,
                    rois,
                    cls_score,
                    bbox_pred,
                    img_shape,
                    scale_factor,
                    rescale=False,
                    cfg=None):
        # rois: [N, 5], [batch_idx, x1, y1, x2, y2], 该样本的所有proposal的bbox坐标
        # cls_score: [N, class_num], 每个proposal对每个类别的得分
        # bbox_pred: [N, 4], 每个proposal的边界框回归
        # img_shape: [H, W], 图像的形状
        # scale_factor: [scale_w, scale_h], 图像的缩放比例
        # rescale: 是否重新缩放边界框
        # cfg: 配置
        # some loss (Seesaw loss..) may have custom activation
        if self.custom_cls_channels:
            scores = self.loss_cls.get_activation(cls_score)
        else:
            if self.training:
                scores = F.softmax(cls_score,
                                   dim=-1) if cls_score is not None else None
            else:
                scores = cls_score
        # bbox_pred would be None in some detector when with_reg is False,
        # e.g. Grid R-CNN.
        if bbox_pred is not None:
            bboxes = self.bbox_coder.decode(rois[..., 1:],
                                            bbox_pred,
                                            max_shape=img_shape)
        else:
            bboxes = rois[:, 1:].clone()
            if img_shape is not None:
                bboxes[:, [0, 2]].clamp_(min=0, max=img_shape[1])
                bboxes[:, [1, 3]].clamp_(min=0, max=img_shape[0])

        if rescale and bboxes.size(0) > 0:
            scale_factor = bboxes.new_tensor(scale_factor)
            bboxes = (bboxes.view(bboxes.size(0), -1, 4) / scale_factor).view(
                bboxes.size()[0], -1)

        # nms操作
        if cfg is None:
            return bboxes, scores
        else:
            det_bboxes, det_labels = multiclass_nms(bboxes, scores,
                                                    cfg.score_thr, cfg.nms,
                                                    cfg.max_per_img)
            return det_bboxes, det_labels


@HEADS.register_module()
class FViTRoIHead(StandardRoIHead):
    def __init__(self, vlm_roi_extractor=None, **kwargs):
        super().__init__(**kwargs)
        # 只从stride=16提取1*1全局特征
        self.vlm_roi_extractor = build_roi_extractor(vlm_roi_extractor)

    def simple_test(self,
                    x,
                    proposal_list,
                    img_metas,
                    vlm_feat=None,
                    proposals=None,
                    rescale=False):
        assert self.with_bbox, 'Bbox head must be implemented.'

        det_bboxes, det_labels = self.simple_test_bboxes(
            x, img_metas, proposal_list, self.test_cfg, rescale=rescale, vlm_feat=vlm_feat)

        bbox_results = [
            bbox2result(det_bboxes[i], det_labels[i],
                        self.bbox_head.num_classes)
            for i in range(len(det_bboxes))
        ]

        if not self.with_mask:
            return bbox_results
        else:
            segm_results = self.simple_test_mask(
                x, img_metas, det_bboxes, det_labels, rescale=rescale)
            return list(zip(bbox_results, segm_results))

    # 核心检测逻辑
    def simple_test_bboxes(self,
                           x,
                           img_metas,
                           proposals,
                           rcnn_test_cfg,
                           vlm_feat=None,
                           rescale=False):
        # x: 5个tensor, 来自FPN的输出[B, C, H, W], C为FPN输出的channel
        # proposals: 所有的proposal的bbox坐标 [N, 4]
        # vlm_feat: 来自vlm的特征, [B, C, H, W], C为vlm输出的channel
        
        rois = bbox2roi(proposals)
        # [N, 5], [batch_idx, x1, y1, x2, y2]

        if rois.shape[0] == 0:
            batch_size = len(proposals)
            det_bbox = rois.new_zeros(0, 5)
            det_label = rois.new_zeros((0, ), dtype=torch.long)
            if rcnn_test_cfg is None:
                det_bbox = det_bbox[:, :4]
                det_label = rois.new_zeros(
                    (0, self.bbox_head.fc_cls.out_features))
            # There is no proposal in the whole batch
            return [det_bbox] * batch_size, [det_label] * batch_size

        # 前向传播
        bbox_results = self._bbox_forward(x, rois, vlm_feat=vlm_feat)
        img_shapes = tuple(meta['img_shape'] for meta in img_metas)
        scale_factors = tuple(meta['scale_factor'] for meta in img_metas)

        # split batch bbox prediction back to each image
        # 从bbox_head中得到的
        cls_score = bbox_results['cls_score']   # [N, class_num], 分类得分
        bbox_pred = bbox_results['bbox_pred']   # [N, 4], 边界框回归
        # 每张图的proposal数量
        num_proposals_per_img = tuple(len(p) for p in proposals)
        # 将rois和cls_score拆分回每张图
        rois = rois.split(num_proposals_per_img, 0)
        cls_score = cls_score.split(num_proposals_per_img, 0)

        # some detector with_reg is False, bbox_pred will be None
        if bbox_pred is not None:
            # TODO move this to a sabl_roi_head
            # the bbox prediction of some detectors like SABL is not Tensor
            if isinstance(bbox_pred, torch.Tensor):
                bbox_pred = bbox_pred.split(num_proposals_per_img, 0)
            else:
                bbox_pred = self.bbox_head.bbox_pred_split(
                    bbox_pred, num_proposals_per_img)
        else:
            bbox_pred = (None, ) * len(proposals)

        # apply bbox post-processing to each image individually
        det_bboxes = []
        det_labels = []
        for i in range(len(proposals)):
            if rois[i].shape[0] == 0:
                # There is no proposal in the single image
                det_bbox = rois[i].new_zeros(0, 5)
                det_label = rois[i].new_zeros((0, ), dtype=torch.long)
                if rcnn_test_cfg is None:
                    det_bbox = det_bbox[:, :4]
                    det_label = rois[i].new_zeros(
                        (0, self.bbox_head.fc_cls.out_features))

            else:
                det_bbox, det_label = self.bbox_head.get_bboxes(
                    rois[i],
                    cls_score[i],
                    bbox_pred[i],
                    img_shapes[i],
                    scale_factors[i],
                    rescale=rescale,
                    cfg=rcnn_test_cfg)
            det_bboxes.append(det_bbox)
            det_labels.append(det_label)
        return det_bboxes, det_labels

    def _bbox_forward(self, x, rois, vlm_feat=None):
        """Box head forward function used in both training and testing."""
        # x: 5个tensor, 来自FPN的输出[B, C, H, W], C为FPN输出的channel
        # rois: [N, 5], [batch_idx, x1, y1, x2, y2]
        # vlm_feat: 来自vlm的特征, [B, C, H, W], C为vlm输出的channel
        # TODO: a more flexible way to decide which feature maps to use
        # ROIAlign从多尺度特征图中针对每个proposal提取区域特征
        # [N, C, 7, 7]
        bbox_feats = self.bbox_roi_extractor(
            x[:self.bbox_roi_extractor.num_inputs], rois)
        if self.with_shared_head:
            bbox_feats = self.shared_head(bbox_feats)
        # [N, C, 7, 7]
        vlm_roi_feats = None
        if vlm_feat is not None:
            vlm_roi_feats = self.vlm_roi_extractor([vlm_feat], rois)[..., 0, 0]
        cls_score, bbox_pred = self.bbox_head(bbox_feats, vlm_roi_feats)
        bbox_results = dict(
            cls_score=cls_score, bbox_pred=bbox_pred, bbox_feats=bbox_feats)
        return bbox_results

@HEADS.register_module()
class FViTTransferBBoxHead(FViTBBoxHead):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fixed_temperature = kwargs['fixed_temperature']

    def forward(self, x, vlm_box_feats=None):

        all_embed = self.all_embed.type_as(x)

        if vlm_box_feats is not None:
            normalized_vlm_box_feats = F.normalize(vlm_box_feats, dim=-1, p=2)
        else:
            # origin vlm
            normalized_vlm_box_feats = None

        if self.num_shared_convs > 0:
            for conv in self.shared_convs:
                x = conv(x)

        if self.num_shared_fcs > 0:
            if self.with_avg_pool:
                x = self.avg_pool(x)

            x = x.flatten(1)

            for fc in self.shared_fcs:
                x = self.relu(fc(x))

        # separate branches
        x_cls = x
        x_reg = x

        for conv in self.cls_convs:
            x_cls = conv(x_cls)
        if x_cls.dim() > 2:
            if self.with_avg_pool:
                x_cls = self.avg_pool(x_cls)
            x_cls = x_cls.flatten(1)
        for fc in self.cls_fcs:
            x_cls = self.relu(fc(x_cls))
            # x_cls = fc(x_cls)

        for conv in self.reg_convs:
            x_reg = conv(x_reg)
        if x_reg.dim() > 2:
            if self.with_avg_pool:
                x_reg = self.avg_pool(x_reg)
            x_reg = x_reg.flatten(1)
        for fc in self.reg_fcs:
            x_reg = self.relu(fc(x_reg))

        bbox_pred = self.fc_reg(x_reg) if self.with_reg else None
        normalized_x_cls = x_cls / x_cls.norm(dim=-1, keepdim=True)

        assert not self.training
        cls_score = normalized_x_cls @ all_embed * self.fixed_temperature
        cls_score = cls_score.softmax(dim=-1)
        vlm_score = normalized_vlm_box_feats @ all_embed * self.vlm_temperature
        vlm_score = vlm_score.softmax(dim=-1)

        cls_score = cls_score ** (1 - self.alpha) * vlm_score ** self.alpha

        return cls_score, bbox_pred
