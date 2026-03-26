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
class FViTBBoxHeadDist(ConvFCBBoxHead):
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
        super(FViTBBoxHeadDist, self).__init__(**kwargs)
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
        self.seen_classes = json.load(open(seen_classes))
        self.all_classes = json.load(open(all_classes))
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
            cls_score = cls_score.sigmoid()
            vlm_score = normalized_vlm_box_feats @ all_embed * self.vlm_temperature
            vlm_score = vlm_score.sigmoid()

            cls_score[:, self.base_idx] = cls_score[:, self.base_idx] ** (
                    1 - self.alpha) * vlm_score[:, self.base_idx] ** self.alpha
            cls_score[:, self.novel_idx] = cls_score[:, self.novel_idx] ** (
                    1 - self.beta) * vlm_score[:, self.novel_idx] ** self.beta

        return cls_score, bbox_pred, x_cls

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
        # if self.custom_cls_channels:
        #     scores = self.loss_cls.get_activation(cls_score)
        # else:
        #     if self.training:
        #         scores = F.softmax(cls_score,
        #                            dim=-1) if cls_score is not None else None
        #     else:
        #         scores = cls_score
        scores = cls_score.sigmoid()
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
class FViTRoIHeadDist(StandardRoIHead):
    def __init__(self, vlm_roi_extractor=None, **kwargs):
        super().__init__(**kwargs)
        # 只从stride=16提取1*1全局特征
        self.vlm_roi_extractor = build_roi_extractor(vlm_roi_extractor)

    def compute_roi_distillation_loss(self, student_results, teacher_results, old_end):
        losses = {}
        s_cls_score = student_results.get('cls_score', None)
        t_cls_score = teacher_results.get('cls_score', None)
        
        # 可选：如果回归框飘了，可以把回归也蒸馏上
        s_bbox_pred = student_results.get('bbox_pred', None)
        t_bbox_pred = teacher_results.get('bbox_pred', None)

        if s_cls_score is None or t_cls_score is None:
            return losses

        # 只取前 old_end 个旧类的 Logits
        s_old_logit = s_cls_score[:, :old_end]
        t_old_logit = t_cls_score[:, :old_end]

        with torch.no_grad():
            # Teacher 的绝对概率打分 [N, old_end]
            t_prob = torch.sigmoid(t_old_logit)
            
            # --- 【极其关键：防幻觉掩码】 ---
            # 如果 Student 在这块区域极其强烈地认为是新类（比如狗，概率>0.3），
            # 就不要强迫它去拟合 Teacher 在这块区域对旧类（比如猫）的底噪瞎猜。
            s_new_logit = s_cls_score[:, old_end:]
            s_new_prob = torch.sigmoid(s_new_logit)
            max_new_prob, _ = s_new_prob.max(dim=1)
            distill_mask = max_new_prob < 0.30

        # =======================================================
        # 1. 纯粹的 Logit 蒸馏 (BCE)
        # =======================================================
        if distill_mask.sum() > 0:
            # 只对没有严重幻觉冲突的框进行蒸馏
            s_logit_sel = s_old_logit[distill_mask]
            t_prob_sel = t_prob[distill_mask]

            # 直接使用 BCEWithLogitsLoss，Student 的输出必须完美复现 Teacher 的概率！
            # 这是多标签分类的蒸馏标准做法
            loss_cls = F.binary_cross_entropy_with_logits(s_logit_sel, t_prob_sel, reduction='none')
            
            # 对类别维度取 mean，对 N 维度取 mean，然后乘一个较高的权重（比如 5.0）
            losses['loss_roi_dist_cls'] = loss_cls.mean() * 5.0
        else:
            losses['loss_roi_dist_cls'] = s_old_logit.sum() * 0.0

        # =======================================================
        # 2. 回归蒸馏 (保底)
        # =======================================================
        if s_bbox_pred is not None and t_bbox_pred is not None:
            with torch.no_grad():
                t_conf, t_label_old = t_prob.max(dim=1)
                valid_mask_reg = (t_conf > 0.45) & distill_mask

            if valid_mask_reg.sum() > 0:
                s_bbox = s_bbox_pred[valid_mask_reg]
                t_bbox = t_bbox_pred[valid_mask_reg]
                t_label_sel = t_label_old[valid_mask_reg]

                M = s_bbox.shape[0]
                idx = torch.arange(M, device=s_bbox.device)

                if s_bbox.shape[1] > 4:
                    s_target = s_bbox.view(M, -1, 4)[idx, t_label_sel]
                    t_target = t_bbox.view(M, -1, 4)[idx, t_label_sel]
                else:
                    s_target = s_bbox
                    t_target = t_bbox

                loss_bbox_per = F.smooth_l1_loss(s_target, t_target, reduction='none').sum(dim=1)
                losses['loss_roi_dist_bbox'] = loss_bbox_per.mean()
            else:
                losses['loss_roi_dist_bbox'] = s_bbox_pred.sum() * 0.0

        return losses

    def forward_train(self, 
                      x, 
                      img_metas, 
                      proposal_list, 
                      gt_bboxes, 
                      gt_labels, 
                      gt_bboxes_ignore=None, 
                      gt_masks=None, 

                      x_old=None,
                      teacher_roi_head=None,
                      teacher_vlm_feats=None,
                      old_end=19,

                      **kwargs):
        num_imgs = len(img_metas)
        if gt_bboxes_ignore is None:
            gt_bboxes_ignore = [None for _ in range(num_imgs)]

        sampling_results = []
        for i in range(num_imgs):
            assign_result = self.bbox_assigner.assign(
                proposal_list[i], gt_bboxes[i], gt_bboxes_ignore[i],
                gt_labels[i])
            sampling_result = self.bbox_sampler.sample(
                assign_result,
                proposal_list[i],
                gt_bboxes[i],
                gt_labels[i],
                feats=[lvl_feat[i][None] for lvl_feat in x])
            sampling_results.append(sampling_result)

        rois = bbox2roi([res.bboxes for res in sampling_results])

        bbox_results = self._bbox_forward(x, rois)
        bbox_targets = self.bbox_head.get_targets(
            sampling_results, gt_bboxes, gt_labels, self.train_cfg)

        loss_bbox = self.bbox_head.loss(bbox_results['cls_score'],
                                        bbox_results['bbox_pred'], rois,
                                        *bbox_targets)
        losses = dict()
        losses.update(loss_bbox)

        # =======================================================
        # 2. 绕过采样的 VIP 蒸馏通道 (拯救旧类饥饿)
        # =======================================================
        if self.training and (teacher_roi_head is not None):
            # 为了防止显存爆炸，我们取每张图 RPN 最靠前的 500 个高质量 proposals
            # 这些 proposals 里绝对包含了旧类的框！
            topk_proposals = [p[:500] for p in proposal_list]
            proposal_rois = bbox2roi(topk_proposals)
            
            # Student 对这些全量 proposals 提取特征并打分
            student_prop_results = self._bbox_forward(x, proposal_rois)
            
            with torch.no_grad():
                # Teacher 对这些全量 proposals 提取特征并打分
                teacher_prop_results = teacher_roi_head._bbox_forward(x_old, proposal_rois, vlm_feat=teacher_vlm_feats)
            
            # 在全量 proposals 上算蒸馏，彻底解决“见不到旧类”的问题！
            loss_dist = self.compute_roi_distillation_loss(
                student_prop_results, 
                teacher_prop_results, 
                old_end
            )
            losses.update(loss_dist)

        return losses

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
        cls_score, bbox_pred, x_cls = self.bbox_head(bbox_feats, vlm_roi_feats)
        bbox_results = dict(
            cls_score=cls_score, bbox_pred=bbox_pred, x_cls=x_cls)
        return bbox_results
