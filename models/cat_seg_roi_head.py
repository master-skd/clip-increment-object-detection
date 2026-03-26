import json
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from einops import rearrange

from mmdet.models.roi_heads import StandardRoIHead, ConvFCBBoxHead
from mmdet.models.builder import HEADS, build_roi_extractor
from mmdet.core import bbox2roi, bbox2result, multiclass_nms
from mmcv.runner import force_fp32
from mmcv.cnn import ConvModule
from mmdet.models.losses import accuracy

@HEADS.register_module()
class CatSegRoIHead(StandardRoIHead):
    def __init__(self, cat_seg_roi_extractor, **kwargs):
        super().__init__(**kwargs)
        if cat_seg_roi_extractor is not None:
            self.cat_seg_roi_extractor = build_roi_extractor(cat_seg_roi_extractor)

    def forward_train(self,
                      x,
                      img_metas,
                      proposal_list,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      cat_seg_feats=None,
                      **kwargs):
        # 分配正负样本逻辑
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

        # 传入cat_seg_feats
        bbox_results = self._bbox_forward(x, rois, cat_seg_feats=cat_seg_feats)
        bbox_targets = self.bbox_head.get_targets(
            sampling_results, gt_bboxes, gt_labels, self.train_cfg)

        loss_bbox = self.bbox_head.loss(bbox_results['cls_score'],
                                        bbox_results['bbox_pred'], rois,
                                        *bbox_targets)
        losses = dict()
        losses.update(loss_bbox)

        return losses

    def _bbox_forward(self, x, rois, cat_seg_feats):
        """
        :param x: FPN features
        :param rois: [N, 5]
        :param cat_seg_feats: [bsz, class_num, c, h, w]
        """
        # FPN特征提取(用于回归)
        bbox_feats = self.bbox_roi_extractor(
            x[:self.bbox_roi_extractor.num_inputs], rois
        )
        # bbox_feats: [N, 256, 7, 7]

        # cat-seg特征提取(用于分类)
        B, T, C, H, W = cat_seg_feats.shape
        flattened_seg_feats = cat_seg_feats.reshape(B, T*C, H, W)
        cat_seg_roi_feats = self.cat_seg_roi_extractor(
            [flattened_seg_feats], rois
        )
        # cat_seg_roi_feats: [N, T*C, 7, 7]

        N_rois = bbox_feats.shape[0]
        # 将cat-seg特征按类别维度拆分并池化
        cat_seg_roi_feats = cat_seg_roi_feats.view(N_rois, T, C, 7, 7)

        cls_score, bbox_pred = self.bbox_head(bbox_feats, cat_seg_roi_feats)

        bbox_results = dict(
            cls_score=cls_score, bbox_pred=bbox_pred, bbox_feats=bbox_feats
        )
        return bbox_results
    
    def simple_test(self, x, proposal_list, img_metas, cat_seg_feats=None, rescale=False):
        assert self.with_bbox, "Bbox head must be implemented."
        det_bboxes, det_labels = self.simple_test_bboxes(
            x, img_metas, proposal_list, self.test_cfg,
            cat_seg_feats=cat_seg_feats, rescale=rescale)
        
        bbox_results = [
            bbox2result(det_bboxes[i], det_labels[i], self.bbox_head.num_classes)
            for i in range(len(det_bboxes))
        ]

        if not self.with_mask:
            return bbox_results
        else:
            segm_results = self.simple_test_mask(
                x, img_metas, det_bboxes, det_labels, rescale=rescale)
            return list(zip(bbox_results, segm_results))

    def simple_test_bboxes(self,
                           x,
                           img_metas,
                           proposals,
                           rcnn_test_cfg,
                           cat_seg_feats=None,
                           rescale=False):
        """Test only det bboxes without augmentation."""
        rois = bbox2roi(proposals)
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

        bbox_results = self._bbox_forward(x, rois, cat_seg_feats=cat_seg_feats)
        img_shapes = tuple(meta['img_shape'] for meta in img_metas)
        scale_factors = tuple(meta['scale_factor'] for meta in img_metas)

        # 拆分batch数据
        cls_score = bbox_results['cls_score']
        bbox_pred = bbox_results['bbox_pred']
        num_proposals_per_img = tuple(len(p) for p in proposals)
        rois = rois.split(num_proposals_per_img, dim=0)
        cls_score = cls_score.split(num_proposals_per_img, dim=0)
        if bbox_pred is not None:
            if isinstance(bbox_pred, torch.Tensor):
                bbox_pred = bbox_pred.split(num_proposals_per_img, dim=0)
            else:
                bbox_pred = self.bbox_head.bbox_pred_split(
                    bbox_pred, num_proposals_per_img)
        else:
            bbox_pred = (None, ) * len(proposals)

        # 逐图应用后处理NMS
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

@HEADS.register_module()
class CatSegBBoxHead(ConvFCBBoxHead):
    def __init__(self,
                 cat_seg_dim=32,
                 text_dim=512,
                #  num_cat_convs=2,
                 num_cat_fcs=1,
                 logit_scale=50.0,
                 class_embed=None,
                 seen_classes=None,
                 all_classes=None,
                 learn_bg=False,
                 init_cfg=None,
                 **kwargs):
        if 'reg_class_agnostic' in kwargs:
            assert kwargs['reg_class_agnostic'] is True, "Must be class agnostic for incremental learning"
        else:
            kwargs['reg_class_agnostic'] = True

        super().__init__(**kwargs)
        self.learn_bg = learn_bg

        self.cat_shared_fcs = nn.ModuleList()
        last_layer_dim = (cat_seg_dim + self.conv_out_channels) * self.roi_feat_area
        for _ in range(num_cat_fcs):
            fc_out_dim = text_dim
            self.cat_shared_fcs.append(
                nn.Linear(last_layer_dim, fc_out_dim)
            )
            last_layer_dim = fc_out_dim

        # 可学习温度参数
        self.logit_scale = nn.Parameter(torch.tensor(logit_scale))

        # 类别embedding
        assert class_embed is not None, "class_embed must be provided"
        self.seen_classes = json.load(open(seen_classes)) + ['background']
        self.all_classes = json.load(open(all_classes)) + ['background']

        class_embed = torch.load(class_embed)
        all_embed = [class_embed[name] for name in self.all_classes]
        all_embed = torch.stack(all_embed, dim=0).permute(1, 0).contiguous()
        all_embed = F.normalize(all_embed, p=2, dim=-1)
        self.register_buffer('all_embeddings', all_embed)

        if learn_bg:
            self.bg_embedding = nn.Parameter(all_embed[:, -1:].clone().contiguous())

        if init_cfg is None:
            self.init_cfg += [
                dict(
                    type='Xavier',
                    distribution='uniform',
                    override=[
                        dict(name='cat_shared_fcs'),
                    ]
                ),
            ]

    @property
    def all_embed(self):
        if self.learn_bg:
            bg_embed = F.normalize(self.bg_embedding, dim=0)
            return torch.cat([self.all_embeddings[:, :-1], bg_embed], dim=1)
        else:
            return self.all_embeddings

    def forward(self, x_fpn, x_catseg):
        """
        :param x_fpn: [N, c_fpn, 7, 7] 来自[fpn]
        :param x_catseg: [N, T, c_catseg, 7, 7] 来自[cat-seg]
        """
        all_embed = self.all_embed.type_as(x_fpn)

        x = x_fpn
        # 共享卷积层
        if self.num_shared_convs > 0:
            for conv in self.shared_convs:
                x = conv(x)
        x_conv = x

        # 共享全连接层
        if self.num_shared_fcs > 0:
            if self.with_avg_pool:
                x = self.avg_pool(x)

            x = x.flatten(1)

            for fc in self.shared_fcs:
                x = self.relu(fc(x))

        # part1: 回归分支
        x_reg = x
        for conv in self.reg_convs:
            x_reg = conv(x_reg)
        if x_reg.dim() > 2:
            if self.with_avg_pool:
                x_reg = self.avg_pool(x_reg)
            x_reg = x_reg.flatten(1)
        for fc in self.reg_fcs:
            x_reg = self.relu(fc(x_reg))
        bbox_pred = self.fc_reg(x_reg) if self.with_reg else None

        N, T, C_cat, H, W = x_catseg.shape
        C_fpn = x_conv.shape[1]
        # 广播FPN特征
        x_conv_expanded = x_conv.unsqueeze(1).expand(-1, T, -1, -1, -1)
        # 通道拼接
        x_fusion = torch.cat([x_conv_expanded, x_catseg], dim=2)  # [N, T, C_fpn+C_cat, 7, 7]

        x_fusion = x_fusion.reshape(N*T, C_cat+C_fpn, H, W)
        # x_fusion = x_fusion.reshape(N*T, C_cat, H, W)
        if self.with_avg_pool:
            x_fusion = self.avg_pool(x_fusion)
        x_fusion = x_fusion.flatten(1)
        for fc in self.cat_shared_fcs:
            x_fusion = self.relu(fc(x_fusion))
        
        x_fusion = x_fusion.reshape(N, T, -1)  # [N, T, text_dim]
        normalized_x_cls = x_fusion / x_fusion.norm(dim=-1, keepdim=True)
        target_embeds = all_embed.permute(1, 0).unsqueeze(0)
        cls_score = (normalized_x_cls * target_embeds).sum(dim=-1) * self.logit_scale

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
        # 使用sigmoid激活
        scores = F.softmax(cls_score, dim=-1)
        scores = scores[:, :-1]  # 去掉背景分数
        if bbox_pred is not None:
            bboxes = self.bbox_coder.decode(
                rois[..., 1:], bbox_pred, max_shape=img_shape
            )
        else:
            bboxes = rois[:, 1:].clone()
            if img_shape is not None:
                bboxes[:, [0, 2]].clamp_(min=0, max=img_shape[1])
                bboxes[:, [1, 3]].clamp_(min=0, max=img_shape[0])

        if rescale and bboxes.size(0) > 0:
            scale_factor = bboxes.new_tensor(scale_factor)
            bboxes = (bboxes.view(bboxes.size(0), -1, 4) / scale_factor).view(
                bboxes.size()[0], -1)

        if cfg is None:
            return bboxes, scores
        else:
            det_bboxes, det_labels = multiclass_nms(
                bboxes,
                scores,
                cfg.score_thr,
                cfg.nms,
                cfg.max_per_img)
            return det_bboxes, det_labels
        
@HEADS.register_module()
class CatSegMaskRoIHead(StandardRoIHead):
    def __init__(self, vlm_roi_extractor=None, **kwargs):
        super().__init__(**kwargs)
        self.vlm_roi_extractor = build_roi_extractor(vlm_roi_extractor)
        # self.old_end = old_end          # 0-18 old => old_end=19
        # self.detach_old = detach_old    # True: 旧类通道不回传梯度

    def _apply_mask_to_fpn(self, x_fpn, x_mask, topk_indices=None, x_fpn_old=None, old_end=19):
        weighted_feats = []
        for lvl, lvl_new in enumerate(x_fpn):
            B, C, H, W = lvl_new.shape

            # 1. Mask 处理
            if x_mask.shape[-2:] != (H, W):
                mask_resized = F.interpolate(x_mask, size=(H, W), mode='bilinear', align_corners=False)
            else:
                mask_resized = x_mask
            # mask_resized = mask_resized.clamp(-10, 10)
            mask_sig = torch.sigmoid(mask_resized)

            # 2. 特征路由 (Hybrid Routing) - 关键回归
            base_new = lvl_new.unsqueeze(1) # [B, 1, C, H, W]

            # 只有在提供了 Teacher FPN 且 提供了索引时，才启用替换
            # if (x_fpn_old is not None) and (topk_indices is not None):
            #     lvl_old = x_fpn_old[lvl]
            #     base_old = lvl_old.unsqueeze(1)

            #     # 判定旧类 (Novel/Task1): indices < old_end
            #     old_mask = (topk_indices < old_end).view(B, -1, 1, 1, 1)
                
            #     # === 核心逻辑 ===
            #     base = torch.where(old_mask, base_old, base_new)
            # else:
            base = base_new.expand(B, x_mask.size(1), C, H, W)

            weighted = base * mask_sig.unsqueeze(2)
            weighted_feats.append(rearrange(weighted, 'B K C H W -> B (K C) H W'))

        return tuple(weighted_feats)

    # def compute_roi_distillation_loss(self, student_results, teacher_results, old_end):
    #     """
    #     计算 R-CNN 阶段的 ERD 蒸馏损失 (支持 Softmax 分类头)
        
    #     Args:
    #         student_results (dict): Student 的 RoI 预测结果，包含 'cls_score' 和 'bbox_pred'
    #         teacher_results (dict): Teacher 的 RoI 预测结果
    #         old_end (int): 旧类别数量 (不含背景)。例如 Task1 有 40 类，则 old_end=40。
    #                         假设 Student 的通道排列为 [0~39 (旧), 40~79 (新), 80 (背景)]
    #                         假设 Teacher 的通道排列为 [0~39 (旧), 40 (背景)]
    #     """
    #     losses = {}
    #     s_cls_score = student_results['cls_score']
    #     t_cls_score = teacher_results['cls_score']
        
    #     # 基础检查
    #     if s_cls_score is None or t_cls_score is None:
    #         return losses

    #     # -------------------------------------------------
    #     # 1. 计算 Teacher 的统计量与弹性权重 (ERD Core)
    #     # -------------------------------------------------
    #     with torch.no_grad():
    #         T = 1.0 # 蒸馏温度
    #         # Teacher 的 Softmax 概率
    #         t_probs = F.softmax(t_cls_score / T, dim=1) # [N, old_end + 1]
            
    #         # 获取 Teacher 的最大概率和对应的 Label
    #         # 注意：这里我们在所有类别(含背景)中找最大值，以正确判断 Teacher 的置信度
    #         t_max_prob, t_label = t_probs.max(dim=1)
            
    #         # === 策略 A: 动态阈值 (Statistical Selection) ===
    #         batch_mean = t_max_prob.mean()
    #         batch_std = t_max_prob.std()
            
    #         # 动态阈值：选取显著高于平均水平的样本
    #         dynamic_threshold = batch_mean + 0.5 * batch_std 
    #         # 保底阈值，防止所有样本都被选中引入噪声
    #         dynamic_threshold = torch.max(dynamic_threshold, torch.tensor(0.01, device=t_max_prob.device))

    #         # 生成 Mask：只有 Teacher 确信的样本才参与蒸馏
    #         valid_mask = t_max_prob > dynamic_threshold
            
    #         # === 策略 B: 弹性权重 (Elastic Weighting) ===
    #         # 置信度越高，权重越大 (Power Law)
    #         max_p = t_max_prob.max().clamp(min=1e-6)
    #         distill_weight = torch.pow(t_max_prob / max_p, 2.0)
            
    #         # 应用 Mask
    #         distill_weight = distill_weight * valid_mask.float()

    #         # num_cls_selected = valid_mask.sum().item()
            
    #         # # 计算参与回归蒸馏的数量 (需满足: 通过阈值 AND 是旧类前景)
    #         # # 注意: t_label < old_end 意味着是前景 (假设 old_end 是背景的索引)
    #         # is_fg_for_print = t_label < old_end
    #         # num_reg_selected = (valid_mask & is_fg_for_print).sum().item()
            
    #         # # 打印 (格式: [总RoI数] -> [分类蒸馏数] | [回归蒸馏数])
    #         # print(f"[Distill Debug] Total: {len(t_max_prob)} | "
    #         #       f"Cls Selected: {num_cls_selected} | "
    #         #       f"Reg Selected: {num_reg_selected}")

    #         # 如果本 Batch 没有有效样本，直接返回 0
    #         if distill_weight.sum() == 0:
    #             return {
    #                 'loss_roi_dist_cls': s_cls_score.sum() * 0.0,
    #                 'loss_roi_dist_bbox': student_results['bbox_pred'].sum() * 0.0
    #             }

    #     # -------------------------------------------------
    #     # 2. Class Distillation (概率聚合策略 - 关键修改)
    #     # -------------------------------------------------
    #     # 目的：让 Student [旧类, 新类+背景] 的概率分布 去拟合 Teacher [旧类, 背景]
        
    #     # 1. 计算 Student 的概率
    #     s_probs = F.softmax(s_cls_score / T, dim=1) # [N, old_end + num_new + 1]
        
    #     # 2. 拆分并聚合
    #     # 旧类别部分 (0 ~ old_end-1)
    #     s_probs_old = s_probs[:, :old_end] 
        
    #     # 广义背景部分：将 Student 的 [所有新类] + [背景] 加起来
    #     # 对应 Teacher 的 [背景] 通道
    #     s_probs_new_bg_aggregated = s_probs[:, old_end:].sum(dim=1, keepdim=True)
        
    #     # 3. 拼接成对齐后的分布 [N, old_end + 1]
    #     s_probs_aligned = torch.cat([s_probs_old, s_probs_new_bg_aggregated], dim=1)
        
    #     # 4. 计算 KL 散度
    #     # F.kl_div 输入要求：input 是 log_softmax，target 是 probabilities
    #     # 所以我们需要对 s_probs_aligned 取 log
    #     loss_cls_per_sample = F.kl_div(
    #         torch.log(s_probs_aligned + 1e-8), 
    #         t_probs, 
    #         reduction='none'
    #     ).sum(dim=1) # [N]
        
    #     # 5. 应用 ERD 权重
    #     losses['loss_roi_dist_cls'] = (loss_cls_per_sample * distill_weight).sum() / distill_weight.sum()

    #     # -------------------------------------------------
    #     # 3. BBox Distillation (ERD 加权回归)
    #     # -------------------------------------------------
    #     s_bbox_pred = student_results['bbox_pred']
    #     t_bbox_pred = teacher_results['bbox_pred']

    #     if s_bbox_pred is not None and t_bbox_pred is not None:
    #         # 前景掩码：只有 Teacher 认为是前景物体时，才蒸馏 BBox
    #         # 假设 Teacher 的背景类索引是 old_end (即最后一个通道)
    #         is_fg = t_label < old_end
            
    #         # 只有同时满足 [ERD阈值筛选] 和 [Teacher认为是前景] 两个条件，才计算回归损失
    #         bbox_weight = distill_weight * is_fg.float()
            
    #         if bbox_weight.sum() > 0:
    #             N = s_bbox_pred.shape[0]
    #             indices = torch.arange(N, device=s_bbox_pred.device)
                
    #             # --- 对齐 Student 的框 ---
    #             if s_bbox_pred.shape[1] > 4: 
    #                 # Class Specific: 取 Teacher 预测类别对应的通道
    #                 s_target_box = s_bbox_pred.view(N, -1, 4)[indices, t_label]
    #             else:
    #                 # Class Agnostic: 直接取
    #                 s_target_box = s_bbox_pred

    #             # --- 对齐 Teacher 的框 ---
    #             if t_bbox_pred.shape[1] > 4:
    #                 t_target_box = t_bbox_pred.view(N, -1, 4)[indices, t_label]
    #             else:
    #                 t_target_box = t_bbox_pred

    #             # 计算 Smooth L1 Loss
    #             loss_bbox_per_sample = F.smooth_l1_loss(
    #                 s_target_box, 
    #                 t_target_box, 
    #                 reduction='none'
    #             ).sum(dim=1) # Sum over coordinates [dx, dy, dw, dh]
                
    #             # 应用权重
    #             losses['loss_roi_dist_bbox'] = (loss_bbox_per_sample * bbox_weight).sum() / bbox_weight.sum()
    #         else:
    #             losses['loss_roi_dist_bbox'] = s_bbox_pred.sum() * 0.0

    #     return losses

    def compute_roi_distillation_loss_previous(self, student_results, teacher_results, old_end):
        losses = {}
        s_cls_score = student_results.get('cls_score', None)
        t_cls_score = teacher_results.get('cls_score', None)
        s_bbox_pred = student_results.get('bbox_pred', None)
        t_bbox_pred = teacher_results.get('bbox_pred', None)
        if s_cls_score is None or t_cls_score is None:
            return losses
        
        # ------ 1) Teacher gating + ERD weights(old-only) -------
        with torch.no_grad():
            T = 2.0
            # ! 原先
            # Teacher probs over [old + bg]
            # t_probs_all = F.softmax(t_cls_score / T, dim=1)
            # t_bg_prob = t_probs_all[:, old_end]

            # # Teacher old-subspace probs (conditional distillation uses old-subspace softmax)
            # t_probs_old = F.softmax(t_cls_score[:, :old_end] / T, dim=1)
            # t_fg_max, t_label_old = t_probs_old.max(dim=1)

            # # ERD-style dynamic threshold (bud based on old fg confidence)
            # batch_mean = t_fg_max.mean()
            # batch_std = t_fg_max.std(unbiased=False)
            # dynamic_threshold = batch_mean + 0.5 * batch_std
            # dynamic_threshold = torch.max(dynamic_threshold, torch.tensor(0.2, device=t_fg_max.device))

            # # Key: must be old-foreground AND stronger than bg (avoid bg-definition conflict)
            # valid_mask = (t_fg_max > dynamic_threshold) & (t_fg_max > t_bg_prob)

            # ! 修改
            # 1) old-subspace softmax: 用于distill_weight和label
            t_probs_old = F.softmax(t_cls_score[:, :old_end] / T, dim=1)
            t_fg_max, t_label_old = t_probs_old.max(dim=1)
            # 2) logit margin: 用于避免bg冲突（同一个logit空间比较）
            t_old_logit_max,_ = t_cls_score[:, :old_end].max(dim=1)
            t_bg_logit = t_cls_score[:, old_end]
            margin = t_old_logit_max - t_bg_logit

            # 3)用quantile控制蒸馏比例
            q = 0.8
            thr = torch.quantile(margin, 0.8)
            valid_mask = margin > thr

            # Elastic weights: confidence power-law
            max_p = t_fg_max.max().clamp(min=1e-6)
            distill_weight = torch.pow(t_fg_max / max_p, 2.0) * valid_mask.float()

            sel = int(valid_mask.sum().item())
            total = int(valid_mask.numel())
            # print(f"[ROI-ERD-Q] total={total} sel={sel} sel_ratio={sel/total:.3f} q={q:.2f} thr={thr.item():.3f} "
            #     f"margin(mean)={margin.mean().item():.3f} margin(std)={margin.std(unbiased=False).item():.3f} "
            #     f"fgmax(mean)={t_fg_max.mean().item():.3f} fgmax(std)={t_fg_max.std(unbiased=False).item():.3f}")

            # ===== DEBUG PRINT (low frequency) =====
            # if self.training and (torch.rand(1).item() < 0.01):  # 1% 概率，避免刷屏
            #     num_total = t_fg_max.numel()
            #     num_sel = valid_mask.sum().item()
            #     w_sum = distill_weight.sum().item()
            #     w_max = distill_weight.max().item() if num_total > 0 else 0.0
            #     w_mean = distill_weight.mean().item() if num_total > 0 else 0.0

            #     print(
            #         f"[ROI-ERD-GATE] total={num_total} sel={num_sel} "
            #         f"thr={dynamic_threshold.item():.3f} "
            #         f"fgmax(mean={batch_mean.item():.3f}, std={batch_std.item():.3f}, max={t_fg_max.max().item():.3f}) "
            #         f"bgprob(mean={t_bg_prob.mean().item():.3f}, max={t_bg_prob.max().item():.3f}) "
            #         f"w(sum={w_sum:.3f}, mean={w_mean:.3f}, max={w_max:.3f})"
            #     )

            # No selected sampled -> return 0 (keep graph)
            if distill_weight.sum() == 0:
                losses['loss_roi_dist_cls'] = s_cls_score.sum() * 0.0
                if s_bbox_pred is not None:
                    losses['loss_roi_dist_bbox'] = s_bbox_pred.sum() * 0.0
                return losses
            
        # ------ 2) Class distillation: KL in old subspace only -------
        s_old = s_cls_score[valid_mask][:, :old_end]
        t_old = t_cls_score[valid_mask][:, :old_end]
        w = distill_weight[valid_mask]

        # old-subspace distributions
        t_old_prob = F.softmax(t_old / T, dim=1)
        s_old_logp = F.log_softmax(s_old / T, dim=1)
        loss_cls_per = F.kl_div(s_old_logp, t_old_prob, reduction='none').sum(dim=1) * (T * T)
        losses['loss_roi_dist_cls'] = (loss_cls_per * w).sum() / (w.sum() + 1e-6)

        # if self.training and (torch.rand(1).item() < 0.01):
        #     print(
        #         f"[ROI-ERD-LOSS] M={w.numel()} "
        #         f"loss_cls={losses['loss_roi_dist_cls'].item():.4f} "
        #         f"loss_cls_per(mean={loss_cls_per.mean().item():.4f}, max={loss_cls_per.max().item():.4f}) "
        #         f"w(mean={w.mean().item():.3f}, max={w.max().item():.3f})"
        #     )

        # -------- 3) BBox distillation: only on selected pseudo-old RoIs --------
        if s_bbox_pred is not None and t_bbox_pred is not None:
            s_bbox = s_bbox_pred[valid_mask]
            t_bbox = t_bbox_pred[valid_mask]

            # teacher label within old subspace
            with torch.no_grad():
                t_label_sel = t_label_old[valid_mask]  # [M]

            if w.sum() > 0:
                M = s_bbox.shape[0]
                idx = torch.arange(M, device=s_bbox.device)

                # Student alignment
                if s_bbox.shape[1] > 4:
                    s_target = s_bbox.view(M, -1, 4)[idx, t_label_sel]
                else:
                    s_target = s_bbox

                # Teacher alignment
                if t_bbox.shape[1] > 4:
                    t_target = t_bbox.view(M, -1, 4)[idx, t_label_sel]
                else:
                    t_target = t_bbox

                loss_bbox_per = F.smooth_l1_loss(s_target, t_target, reduction='none').sum(dim=1)  # [M]
                losses['loss_roi_dist_bbox'] = (loss_bbox_per * w).sum() / (w.sum() + 1e-6)
            else:
                losses['loss_roi_dist_bbox'] = s_bbox_pred.sum() * 0.0

        return losses

    def compute_roi_distillation_loss(self, student_results, teacher_results, old_end):
        losses = {}
        s_cls_score = student_results.get('cls_score', None)
        t_cls_score = teacher_results.get('cls_score', None)
        s_bbox_pred = student_results.get('bbox_pred', None)
        t_bbox_pred = teacher_results.get('bbox_pred', None)
        if s_cls_score is None or t_cls_score is None:
            return losses

        # teacher: [old] ; student: [old + new]
        s_old_logit_all = s_cls_score[:, :old_end]   # [N, old]
        t_old_logit_all = t_cls_score[:, :old_end]   # [N, old]

        # ------ 1) Teacher gating + ERD weights (old-only, sigmoid-space) ------
        with torch.no_grad():
            T = 1.0

            # teacher old prob（逐类独立）
            t_old_prob_all = torch.sigmoid(t_old_logit_all / T)    # [N, old]

            # 用 old 子空间 max prob 作为 “foreground confidence”
            t_conf, t_label_old = t_old_prob_all.max(dim=1)        # [N], [N]

            # ===== ERD 动态阈值 =====
            lam = 0.5
            # thr_floor = 0.20   # 你原先 ROI 用过 0.2，可以先保留
            mean = t_conf.mean()
            std = t_conf.std(unbiased=False)
            # dynamic_threshold = torch.max(mean + lam * std, t_conf.new_tensor(thr_floor))
            dynamic_threshold = torch.max(mean + lam * std, t_conf.new_tensor(0.20))

            valid_mask = (t_conf > dynamic_threshold)              # [N]

            # Elastic weights（沿用你原先的 power-law）
            gamma = 2.0
            max_p = t_conf.max()
            # distill_weight = (t_conf / max_p).pow(gamma) * valid_mask.float()  # [N]
            if max_p > 0.3:
                # 只有当图里确实有比较确信的旧类时，才使用 ERD 的弹性放大
                distill_weight = (t_conf / max_p).pow(gamma) * valid_mask.float()
            else:
                # 图里全是底噪，不准做除法放大！直接用原始概率压制权重
                distill_weight = t_conf.pow(gamma) * valid_mask.float()

            # 没选到就返回 0（保持图）
            if distill_weight.sum() == 0:
                losses['loss_roi_dist_cls'] = s_cls_score.sum() * 0.0
                if s_bbox_pred is not None:
                    losses['loss_roi_dist_bbox'] = s_bbox_pred.sum() * 0.0
                return losses

        # ------ 2) Class distillation: BCE(sigmoid) in old subspace only ------
        s_old = s_old_logit_all[valid_mask]          # [M, old]
        t_old = t_old_logit_all[valid_mask]          # [M, old]
        w = distill_weight[valid_mask]               # [M]

        with torch.no_grad():
            t_target = torch.sigmoid(t_old / T)      # [M, old]

        # BCEWithLogits：输入 logits，target 是 [0,1] 概率
        bce = F.binary_cross_entropy_with_logits(s_old / T, t_target, reduction='none')  # [M, old]

        # 对 old 维度求 mean（更稳，不会因为 old 类数变化导致尺度漂）
        loss_cls_per = bce.mean(dim=1) * (T * T)     # [M]
        losses['loss_roi_dist_cls'] = (loss_cls_per * w).sum() / (w.sum() + 1e-6)
        losses['loss_roi_dist_cls'] = losses['loss_roi_dist_cls'] * 0.1

        # ------ 3) BBox distillation: 仍然只对选中的 RoIs ------
        if s_bbox_pred is not None and t_bbox_pred is not None:
            # 增加对回归的严格过滤：只有同时满足动态阈值 AND 置信度绝对值大于 0.45 的框，才蒸馏回归
            with torch.no_grad():
                valid_mask_reg = valid_mask & (t_conf > 0.45)
            
            w_reg = distill_weight[valid_mask_reg]

            # 只有当存在符合严格要求的高质量框时，才计算回归损失
            if w_reg.sum() > 0:
                s_bbox = s_bbox_pred[valid_mask_reg]
                t_bbox = t_bbox_pred[valid_mask_reg]

                with torch.no_grad():
                    t_label_sel = t_label_old[valid_mask_reg]  # [M_reg]

                M_reg = s_bbox.shape[0]
                idx = torch.arange(M_reg, device=s_bbox.device)

                # Student alignment
                if s_bbox.shape[1] > 4:
                    s_target = s_bbox.view(M_reg, -1, 4)[idx, t_label_sel]
                else:
                    s_target = s_bbox

                # Teacher alignment
                if t_bbox.shape[1] > 4:
                    t_target = t_bbox.view(M_reg, -1, 4)[idx, t_label_sel]
                else:
                    t_target = t_bbox

                loss_bbox_per = F.smooth_l1_loss(s_target, t_target, reduction='none').sum(dim=1)  # [M_reg]
                losses['loss_roi_dist_bbox'] = (loss_bbox_per * w_reg).sum() / (w_reg.sum() + 1e-6)
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
                      cat_seg_feats=None,
                      topk_indices=None,

                      x_old=None,
                      cat_seg_logits_old=None,
                      teacher_indices=None,
                      teacher_roi_head=None,
                      old_end=19,
                      **kwargs):
        # 分配正负样本逻辑
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

        # if (x_old is not None) and (cat_seg_logits_old is not None) and (teacher_indices is not None):
            
        #     with torch.no_grad():
        #         # --- 1. 直接复用你的匹配逻辑 (完全照搬) ---
        #         s_idx_exp = topk_indices.unsqueeze(2)
        #         t_idx_exp = teacher_indices.unsqueeze(1)
        #         match_matrix = (s_idx_exp == t_idx_exp)
        #         is_old = s_idx_exp < old_end
        #         valid_match = match_matrix & is_old
        #         matches = torch.nonzero(valid_match, as_tuple=True)
        #         b_idx, s_ch, t_ch = matches

        #         # --- 2. 准备 Soft Protection Mask ---
        #         # 初始化一个全 0 的 Mask [B, H, W]
        #         # device 保持一致
        #         B_size, _, H_size, W_size = cat_seg_logits_old.shape
        #         spatial_protection_mask = torch.zeros((B_size, H_size, W_size), 
        #                                               device=cat_seg_logits_old.device)

        #         if b_idx.numel() > 0:
        #             # 提取 Teacher 对应通道的 Mask Logits [N_match, H, W]
        #             t_mask_logits = cat_seg_logits_old[b_idx, t_ch]
                    
        #             # 转为概率 (0~1) -> 这就是我们要的"软权重"
        #             t_mask_prob = torch.sigmoid(t_mask_logits)
                    
        #             # --- 3. 聚合 Mask (Scatter Max) ---
        #             # 因为一个 batch 里可能匹配到多个旧类 (比如既有猫又有狗)
        #             # 我们取它们概率的最大值，作为该像素的最终保护力度
                    
        #             # 为了效率，我们遍历本次 batch 中涉及到的 b_idx
        #             unique_b = torch.unique(b_idx)
        #             for b in unique_b:
        #                 # 找到属于这个 sample 的所有匹配 mask
        #                 mask_indices = (b_idx == b)
        #                 probs_in_batch = t_mask_prob[mask_indices] # [M, H, W]
                        
        #                 # 取最大值: 只要任意一个旧类说这里是它，我们就保护这里
        #                 if probs_in_batch.shape[0] > 0:
        #                     max_prob, _ = probs_in_batch.max(dim=0)
        #                     spatial_protection_mask[b] = max_prob
                
        #         # 增加通道维度 [B, H, W] -> [B, 1, H, W]
        #         spatial_protection_mask = spatial_protection_mask.unsqueeze(1)

        #     # --- 4. 定义 Soft Hook ---
        #     def get_soft_gradient_mask_hook(mask_resized):
        #         def hook(grad):
        #             # [关键改动] 软截断
        #             # grad * (1.0 - probability)
        #             # 概率越高，保留的梯度越少
        #             return grad * (1.0 - mask_resized)
        #         return hook

        #     # --- 5. 注册 Hook ---
        #     new_x = []
        #     for feat in x:
        #         if feat.requires_grad:
        #             B, C, H, W = feat.shape
        #             # 插值到当前特征层大小
        #             mask_resized = F.interpolate(spatial_protection_mask, size=(H, W), mode='bilinear', align_corners=False).detach()
        #             # 注册
        #             feat.register_hook(get_soft_gradient_mask_hook(mask_resized))
        #         new_x.append(feat)
        #     x = tuple(new_x)
        # 传入cat_seg_feats
        # topk_indices = self._inject_gt_to_topk(topk_indices, gt_labels, k_old=5)
        bbox_results = self._bbox_forward(x, rois, cat_seg_feats=cat_seg_feats, topk_indices=topk_indices, x_old=x_old, old_end=old_end)
        bbox_targets = self.bbox_head.get_targets(
            sampling_results, gt_bboxes, gt_labels, self.train_cfg)

        loss_bbox = self.bbox_head.loss(bbox_results['cls_score'],
                                        bbox_results['bbox_pred'], rois,
                                        *bbox_targets)
        losses = dict()
        losses.update(loss_bbox)

        # 先初始化为 0 (防 DDP 报错)
        pseudo_zero = bbox_results['cls_score'].sum() * 0.0
        losses['loss_roi_dist_cls'] = pseudo_zero
        # losses['loss_roi_dist_bbox'] = pseudo_zero

        if (teacher_roi_head is not None) and (x_old is not None):
            with torch.no_grad():
                # 移除 if distill_mask.any(): 判断
                # 对所有 RoI 进行 Teacher 推理
                teacher_bbox_results = teacher_roi_head._bbox_forward(
                    x_old, 
                    rois, 
                    cat_seg_feats=cat_seg_logits_old,
                    topk_indices=teacher_indices,
                    old_end=old_end
                )
            
            # 计算全样本 KL Loss
            loss_dist = self.compute_roi_distillation_loss(
                bbox_results, 
                teacher_bbox_results, 
                old_end
            )
            
            losses.update(loss_dist)

        return losses

    def _bbox_forward(self, x, rois, cat_seg_feats, topk_indices=None, x_old=None, old_end=19, vlm_feat=None):
        """
        :param x: FPN features
        :param rois: [N, 5]
        :param cat_seg_feats: [bsz, class_num, h, w]
        """
        x_fusion = self._apply_mask_to_fpn(x, cat_seg_feats, topk_indices=topk_indices, x_fpn_old=x_old,
            old_end=old_end)  # tuple of [B, T*C, H, W]
        bbox_feats = self.bbox_roi_extractor(
            x_fusion[:self.bbox_roi_extractor.num_inputs], rois
        )
        vlm_roi_feats = None
        if vlm_feat is not None:
            vlm_roi_feats = self.vlm_roi_extractor([vlm_feat], rois)[..., 0, 0]
        cls_score, bbox_pred = self.bbox_head(bbox_feats, vlm_roi_feats, rois, topk_indices)
        bbox_results = dict(
            cls_score=cls_score, bbox_pred=bbox_pred, bbox_feats=bbox_feats
        )
        return bbox_results

    def simple_test(self, x, proposal_list, img_metas, cat_seg_feats=None, topk_indices=None, vlm_feat=None, rescale=False):
        assert self.with_bbox, "Bbox head must be implemented."
        det_bboxes, det_labels = self.simple_test_bboxes(
            x, img_metas, proposal_list, self.test_cfg,
            cat_seg_feats=cat_seg_feats, topk_indices=topk_indices, rescale=rescale, vlm_feat=vlm_feat)
        
        bbox_results = [
            bbox2result(det_bboxes[i], det_labels[i], self.bbox_head.num_classes)
            for i in range(len(det_bboxes))
        ]

        if not self.with_mask:
            return bbox_results
        else:
            segm_results = self.simple_test_mask(
                x, img_metas, det_bboxes, det_labels, rescale=rescale)
            return list(zip(bbox_results, segm_results))
    
    def simple_test_bboxes(self,
                           x,
                           img_metas,
                           proposals,
                           rcnn_test_cfg,
                           cat_seg_feats=None,
                           topk_indices=None,
                           vlm_feat=None,
                           rescale=False):
        """Test only det bboxes without augmentation."""
        rois = bbox2roi(proposals)
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

        bbox_results = self._bbox_forward(x, rois, cat_seg_feats=cat_seg_feats, topk_indices=topk_indices, vlm_feat=vlm_feat)
        img_shapes = tuple(meta['img_shape'] for meta in img_metas)
        scale_factors = tuple(meta['scale_factor'] for meta in img_metas)

        # 拆分batch数据
        cls_score = bbox_results['cls_score']
        bbox_pred = bbox_results['bbox_pred']
        num_proposals_per_img = tuple(len(p) for p in proposals)
        rois = rois.split(num_proposals_per_img, dim=0)
        cls_score = cls_score.split(num_proposals_per_img, dim=0)
        if bbox_pred is not None:
            if isinstance(bbox_pred, torch.Tensor):
                bbox_pred = bbox_pred.split(num_proposals_per_img, dim=0)
            else:
                bbox_pred = self.bbox_head.bbox_pred_split(
                    bbox_pred, num_proposals_per_img)
        else:
            bbox_pred = (None, ) * len(proposals)

        # 逐图应用后处理NMS
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
    
@HEADS.register_module()
class CatSegMaskBBoxHead(ConvFCBBoxHead):
    def __init__(self, 
                 num_classes,
                 text_dim,
                 logit_scale=50.0,
                 class_embed=None,
                 seen_classes=None,
                 all_classes=None ,
                 learn_bg=True,
                    
                 vlm_temperature=100.0,
                 alpha=0.2,
                 beta=0.45,
                 
                #  old_end=None,
                #  is_incremental=False,
                 **kwargs):
        super().__init__(**kwargs)

        if self.fc_out_channels != text_dim:
            raise ValueError(
                f"fc_out_channels ({self.fc_out_channels}) must match "
                f"text_dim ({text_dim}) when no projector is used!"
            )

        self.learn_bg = learn_bg
        self.num_classes = num_classes

        # 可学习温度参数
        self.logit_scale = nn.Parameter(torch.tensor(logit_scale))
        self.logit_scale.requires_grad_(False)

        # 固定参数，仅在推理时用
        self.vlm_temperature = vlm_temperature
        self.alpha = alpha
        self.beta = beta

        # 类别embedding
        assert class_embed is not None, "class_embed must be provided"
        # self.seen_classes = json.load(open(seen_classes)) + ['background']
        # self.all_classes = json.load(open(all_classes)) + ['background']
        self.seen_classes = json.load(open(seen_classes))
        self.all_classes = json.load(open(all_classes))
        idx = [self.all_classes.index(seen) for seen in self.seen_classes]

        # 创建类别掩码，标记novel类的位置
        self.base_idx = torch.zeros(len(self.all_classes), dtype=bool)
        self.base_idx[idx] = True
        self.novel_idx = self.base_idx == False

        class_embed = torch.load(class_embed)
        all_embed = [class_embed[name] for name in self.all_classes]
        all_embed = torch.stack(all_embed, dim=0).permute(1, 0).contiguous()
        all_embed = F.normalize(all_embed, p=2, dim=-1)
        self.register_buffer('all_embeddings', all_embed)

        if learn_bg:
            self.bg_embedding = nn.Parameter(all_embed[:, -1:].clone().contiguous())

    @property
    def all_embed(self):
        if self.learn_bg:
            bg_embed = F.normalize(self.bg_embedding, dim=0)
            return torch.cat([self.all_embeddings[:, :-1], bg_embed], dim=1)
        else:
            return self.all_embeddings

    def forward(self, x, vlm_box_feats=None, rois=None, topk_indices=None):
        all_embed = self.all_embed.type_as(x)

        if vlm_box_feats is not None:
            assert not self.training
            normalized_vlm_box_feats = F.normalize(vlm_box_feats, dim=-1, p=2)
        else:
            normalized_vlm_box_feats = None

        # ---------------------------------------------------
        # 1. Student Forward
        # ---------------------------------------------------
        feat = x
        # Shared Convs
        if self.num_shared_convs > 0:
            for conv in self.shared_convs:
                feat = conv(feat)
        # Shared FCs
        if self.num_shared_fcs > 0:
            if self.with_avg_pool:
                feat = self.avg_pool(feat)
            feat = feat.flatten(1)
            for fc in self.shared_fcs:
                feat = self.relu(fc(feat))
        
        # Cls Branch
        x_cls = feat
        if self.num_cls_convs > 0:
            for conv in self.cls_convs:
                x_cls = conv(x_cls)
        if x_cls.dim() > 2:
            x_cls = x_cls.flatten(1)
        if self.num_cls_fcs > 0:
            for fc in self.cls_fcs:
                x_cls = self.relu(fc(x_cls))
        
        norm_student = F.normalize(x_cls, p=2, dim=-1, eps=1e-6)
        cls_score = (norm_student @ all_embed) * self.logit_scale
        
        # Reg Branch
        x_reg = feat
        if self.num_reg_convs > 0:
            for conv in self.reg_convs:
                x_reg = conv(x_reg)
        if x_reg.dim() > 2:
            if self.with_avg_pool:
                x_reg = self.avg_pool(x_reg)
            x_reg = x_reg.flatten(1)
        if self.num_reg_fcs > 0:
            for fc in self.reg_fcs:
                x_reg = self.relu(fc(x_reg))
        
        bbox_pred = self.fc_reg(x_reg) if self.with_reg else None

        if not self.training and normalized_vlm_box_feats is not None:
            cls_score = cls_score.sigmoid()
            vlm_score = (normalized_vlm_box_feats @ all_embed) * self.vlm_temperature
            vlm_score = vlm_score.sigmoid()

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
        # scores = F.softmax(cls_score, dim=-1)
        # scores = scores[:, :-1]  # 去掉背景分数
        scores = cls_score.sigmoid()
        if bbox_pred is not None:
            bboxes = self.bbox_coder.decode(
                rois[..., 1:], bbox_pred, max_shape=img_shape
            )
        else:
            bboxes = rois[:, 1:].clone()
            if img_shape is not None:
                bboxes[:, [0, 2]].clamp_(min=0, max=img_shape[1])
                bboxes[:, [1, 3]].clamp_(min=0, max=img_shape[0])
            bboxes = bboxes.view(bboxes.size(0), -1)

        if rescale and bboxes.size(0) > 0:
            scale_factor = bboxes.new_tensor(scale_factor)
            bboxes = (bboxes.view(bboxes.size(0), -1, 4) / scale_factor).view(
                bboxes.size()[0], -1)

        if cfg is None:
            return bboxes, scores
        else:
            det_bboxes, det_labels = multiclass_nms(
                bboxes,
                scores,
                cfg.score_thr,
                cfg.nms,
                cfg.max_per_img)
            return det_bboxes, det_labels
