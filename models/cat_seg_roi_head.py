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
            if (x_fpn_old is not None) and (topk_indices is not None):
                lvl_old = x_fpn_old[lvl]
                base_old = lvl_old.unsqueeze(1)

                # 判定旧类 (Novel/Task1): indices < old_end
                old_mask = (topk_indices < old_end).view(B, -1, 1, 1, 1)
                
                # === 核心逻辑 ===
                # 旧类(Novel)强制用 Teacher -> 保证 AP 回到 20+
                # 新类(Base)用 Student -> 保证学习新知识
                base = torch.where(old_mask, base_old, base_new)
            else:
                base = base_new.expand(B, x_mask.size(1), C, H, W)

            weighted = base * mask_sig.unsqueeze(2)
            weighted_feats.append(rearrange(weighted, 'B K C H W -> B (K C) H W'))

        return tuple(weighted_feats)

    # def compute_roi_distillation_loss(self, student_results, teacher_results, old_end):
    #     losses = {}
        
    #     # 1. 获取 Logits
    #     s_cls_score = student_results['cls_score']
    #     t_cls_score = teacher_results['cls_score']
        
    #     if s_cls_score is None or t_cls_score is None:
    #         return losses

    #     # 2. 截取旧类通道 (例如 0~18)
    #     # 【关键】只拿出旧的前景类别，绝对不要包含背景通道！
    #     # 如果包含背景，Teacher 会把新类物体当成背景，KL Loss 就会强迫 Student 也把它当背景。
    #     s_logits_old = s_cls_score[:, :old_end]
    #     t_logits_old = t_cls_score[:, :old_end]

    #     # 3. 生成 Mask: 只蒸馏 Teacher "看懂了" (置信度高) 的样本
    #     with torch.no_grad():
    #         # 这里可以用 softmax 或 sigmoid 看 Teacher 对旧类的信心
    #         # 既然我们要用 KL (Softmax)，这里用 Softmax 比较一致
    #         t_probs_all = F.softmax(t_cls_score, dim=1) 
            
    #         # 拿到 Teacher 在旧类前景通道上的最大概率
    #         # 注意：t_probs_all 包含了背景，我们看它是否觉得"是旧类前景"
    #         t_max_prob, _ = t_probs_all[:, :old_end].max(dim=1)
            
    #         # 阈值筛选：只有 Teacher 确信这是旧类 (p > 0.4) 时，Student 才需要模仿
    #         # 对于新类物体，Teacher 看到的旧类概率通常很低，会被过滤掉 -> 保护新类学习
    #         valid_mask = t_max_prob > 0.5

    #     # 4. 计算 KL Divergence Loss
    #     if valid_mask.sum() > 0:
    #         # T = 3.0  # 温度系数，KL 蒸馏的标配，让分布更平滑，关注暗知识
            
    #         # 取出有效样本
    #         s_valid = s_logits_old[valid_mask]
    #         t_valid = t_logits_old[valid_mask]

    #         # Student 输出 Log Softmax
    #         s_log_probs = F.log_softmax(s_valid, dim=1)
            
    #         # Teacher 输出 Softmax (作为 Target)
    #         with torch.no_grad():
    #             t_probs = F.softmax(t_valid, dim=1)

    #         # 计算 KL Loss
    #         # reduction='batchmean' 是 KL 散度的标准用法，会自动除以 batch size
    #         loss_dist_cls = F.kl_div(s_log_probs, t_probs, reduction='batchmean')
            
    #         # 权重建议：KL Loss 的梯度通常比 MSE 柔和，可以给适当的权重 (如 2.0 ~ 5.0)
    #         losses['loss_roi_dist_cls'] = loss_dist_cls * 1.0
    #     else:
    #         losses['loss_roi_dist_cls'] = s_logits_old.sum() * 0.0

    #     # 5. 回归蒸馏 (保持不变，同样只针对有效样本)
    #     # s_bbox_pred = student_results['bbox_pred']
    #     # t_bbox_pred = teacher_results['bbox_pred']

    #     # if s_bbox_pred is not None and t_bbox_pred is not None:
    #     #      if s_bbox_pred.shape == t_bbox_pred.shape:
    #     #         if valid_mask.sum() > 0:
    #     #             loss_dist_bbox = F.smooth_l1_loss(
    #     #                 s_bbox_pred[valid_mask], 
    #     #                 t_bbox_pred[valid_mask]
    #     #             )
    #     #             losses['loss_roi_dist_bbox'] = loss_dist_bbox * 2.0
    #     #         else:
    #     #             losses['loss_roi_dist_bbox'] = s_bbox_pred.sum() * 0.0
            
    #     return losses

    def compute_roi_distillation_loss(self, student_results, teacher_results, old_end):
        losses = {}
        s_cls_score = student_results['cls_score']
        t_cls_score = teacher_results['cls_score']
        if s_cls_score is None or t_cls_score is None:
            return losses
        
        # 1. 拆解组件
        t_logits_old = t_cls_score[:, :old_end]
        s_logits_new_bg = s_cls_score[:, old_end:].detach()

        # 2. 数值对齐
        with torch.no_grad():
            t_mean = t_logits_old.mean(dim=1, keepdim=True)
            t_std = t_logits_old.std(dim=1, keepdim=True)
            s_mean = s_logits_new_bg.mean(dim=1, keepdim=True)
            s_std = s_logits_new_bg.std(dim=1, keepdim=True)
            
            # 将 Teacher 的分布拉伸到 Student 的水平
            t_logits_old_aligned = (t_logits_old - t_mean) / (t_std + 1e-6) * s_std + s_mean

        # 3. 直接拼接
        target_logits = torch.cat([t_logits_old_aligned, s_logits_new_bg], dim=1)

        # 4. 计算全维度KL Loss
        T = 1.0
        loss_dist = F.kl_div(
            F.log_softmax(s_cls_score / T, dim=1),
            F.softmax(target_logits / T, dim=1),
            reduction='batchmean'
        )

        losses['loss_roi_dist_cls'] = loss_dist * 2.0
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

        if (x_old is not None) and (cat_seg_logits_old is not None) and (teacher_indices is not None):
            
            with torch.no_grad():
                # --- 1. 直接复用你的匹配逻辑 (完全照搬) ---
                s_idx_exp = topk_indices.unsqueeze(2)
                t_idx_exp = teacher_indices.unsqueeze(1)
                match_matrix = (s_idx_exp == t_idx_exp)
                is_old = s_idx_exp < old_end
                valid_match = match_matrix & is_old
                matches = torch.nonzero(valid_match, as_tuple=True)
                b_idx, s_ch, t_ch = matches

                # --- 2. 准备 Soft Protection Mask ---
                # 初始化一个全 0 的 Mask [B, H, W]
                # device 保持一致
                B_size, _, H_size, W_size = cat_seg_logits_old.shape
                spatial_protection_mask = torch.zeros((B_size, H_size, W_size), 
                                                      device=cat_seg_logits_old.device)

                if b_idx.numel() > 0:
                    # 提取 Teacher 对应通道的 Mask Logits [N_match, H, W]
                    t_mask_logits = cat_seg_logits_old[b_idx, t_ch]
                    
                    # 转为概率 (0~1) -> 这就是我们要的"软权重"
                    t_mask_prob = torch.sigmoid(t_mask_logits)
                    
                    # --- 3. 聚合 Mask (Scatter Max) ---
                    # 因为一个 batch 里可能匹配到多个旧类 (比如既有猫又有狗)
                    # 我们取它们概率的最大值，作为该像素的最终保护力度
                    
                    # 为了效率，我们遍历本次 batch 中涉及到的 b_idx
                    unique_b = torch.unique(b_idx)
                    for b in unique_b:
                        # 找到属于这个 sample 的所有匹配 mask
                        mask_indices = (b_idx == b)
                        probs_in_batch = t_mask_prob[mask_indices] # [M, H, W]
                        
                        # 取最大值: 只要任意一个旧类说这里是它，我们就保护这里
                        if probs_in_batch.shape[0] > 0:
                            max_prob, _ = probs_in_batch.max(dim=0)
                            spatial_protection_mask[b] = max_prob
                
                # 增加通道维度 [B, H, W] -> [B, 1, H, W]
                spatial_protection_mask = spatial_protection_mask.unsqueeze(1)

            # --- 4. 定义 Soft Hook ---
            def get_soft_gradient_mask_hook(mask_resized):
                def hook(grad):
                    # [关键改动] 软截断
                    # grad * (1.0 - probability)
                    # 概率越高，保留的梯度越少
                    return grad * (1.0 - mask_resized)
                return hook

            # --- 5. 注册 Hook ---
            new_x = []
            for feat in x:
                if feat.requires_grad:
                    B, C, H, W = feat.shape
                    # 插值到当前特征层大小
                    mask_resized = F.interpolate(spatial_protection_mask, size=(H, W), mode='bilinear', align_corners=False).detach()
                    # 注册
                    feat.register_hook(get_soft_gradient_mask_hook(mask_resized))
                new_x.append(feat)
            x = tuple(new_x)
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
    
    def _inject_gt_to_topk(self, topk_indices, gt_labels, k_old=3):
        if topk_indices is None:
            return None
        topk = topk_indices.clone()
        B, K = topk.shape
        device = topk.device

        for i in range(B):
            gt = torch.unique(gt_labels[i]).to(device=device, dtype=torch.long)
            # 去掉非法
            gt = gt[gt >= 0]
            # 逐个注入缺失的 gt
            for g in gt.tolist():
                if (topk[i] == g).any():
                    continue
                # 优先替换 new 槽位（从尾到 k_old）
                replaced = False
                for pos in range(K - 1, k_old - 1, -1):
                    if topk[i, pos].item() not in gt.tolist():
                        topk[i, pos] = g
                        replaced = True
                        break
                if not replaced:
                    # 实在塞不进就跳过（一般不会发生）
                    pass
        return topk


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
        self.seen_classes = json.load(open(seen_classes)) + ['background']
        self.all_classes = json.load(open(all_classes)) + ['background']
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
        
        # self.old_end = old_end
        # self.is_incremental = is_incremental
        # self.teacher_branch = None
        # if self.is_incremental:
        #     # 此时 student 的权重可能还是随机的，没关系，我们先复制结构
        #     # 这样 MMCV 就能监测到这些参数的存在，不会报错
        #     self.teacher_branch = nn.ModuleDict({
        #         'shared_convs': copy.deepcopy(self.shared_convs),
        #         'shared_fcs': copy.deepcopy(self.shared_fcs),
        #         'cls_convs': copy.deepcopy(self.cls_convs),
        #         'cls_fcs': copy.deepcopy(self.cls_fcs)
        #     })
            
        #     # 顺便直接把 Teacher 冻结住
        #     self.teacher_branch.eval()
        #     for param in self.teacher_branch.parameters():
        #         param.requires_grad = False

    # def init_weights(self):
    #     super().init_weights()

    #     # =======================================================
    #     # 修改点 2: 这里只做权重数值的同步 (State Dict Copy)
    #     # =======================================================
    #     if self.is_incremental and self.teacher_branch is not None:
    #         print(f"==> [CatSegHead] Incremental Mode: Syncing Teacher weights from initialized Student...")
            
    #         # 使用 load_state_dict 把 Student 的权重值覆盖到 Teacher 上
    #         self.teacher_branch['shared_convs'].load_state_dict(self.shared_convs.state_dict())
    #         self.teacher_branch['shared_fcs'].load_state_dict(self.shared_fcs.state_dict())
    #         self.teacher_branch['cls_convs'].load_state_dict(self.cls_convs.state_dict())
    #         self.teacher_branch['cls_fcs'].load_state_dict(self.cls_fcs.state_dict())
            
    #         print(f"==> [CatSegHead] Teacher weights synced and frozen.")
    #     else:
    #         print(f"==> [CatSegHead] Base Mode: No Teacher used.")

    # def get_channel_mask(self, x, topk_indices):
    #     """生成旧类通道掩码 (仅在 Incremental 模式下使用)"""
    #     N, KC, H, W = x.shape
    #     K = topk_indices.shape[1]
    #     C = KC // K
    #     is_old = topk_indices < self.old_end
    #     mask_old = is_old.unsqueeze(-1).expand(-1, -1, C).reshape(N, -1)
    #     mask_old = mask_old.view(N, KC, 1, 1).type_as(x)
    #     return mask_old
    
    # def forward_teacher(self, x):
    #     """
    #     Teacher 前向传播
    #     修改：必须同时返回 cls_score 和 bbox_pred
    #     """
    #     # 1. Shared Parts
    #     if self.num_shared_convs > 0:
    #         for conv in self.teacher_branch['shared_convs']:
    #             x = conv(x)
    #     if self.num_shared_fcs > 0:
    #         if self.with_avg_pool:
    #             x = self.avg_pool(x)
    #         x = x.flatten(1)
    #         for fc in self.teacher_branch['shared_fcs']:
    #             x = self.relu(fc(x))
        
    #     # 2. Cls Branch
    #     x_cls = x
    #     for conv in self.teacher_branch['cls_convs']:
    #         x_cls = conv(x_cls)
    #     if x_cls.dim() > 2:
    #         x_cls = x_cls.flatten(1)
    #     for fc in self.teacher_branch['cls_fcs']:
    #         x_cls = self.relu(fc(x_cls))
            
    #     # 3. Reg Branch (新增)
    #     x_reg = x
    #     # 假设 teacher 结构和 student 一样，也有 reg_convs/fcs
    #     # 如果你的 teacher_branch 构建时没 copy reg部分，需要去 init 里加上
    #     if 'reg_convs' in self.teacher_branch:
    #          for conv in self.teacher_branch['reg_convs']:
    #             x_reg = conv(x_reg)
    #     if x_reg.dim() > 2:
    #         if self.with_avg_pool:
    #             x_reg = self.avg_pool(x_reg)
    #         x_reg = x_reg.flatten(1)
    #     if 'reg_fcs' in self.teacher_branch:
    #         for fc in self.teacher_branch['reg_fcs']:
    #             x_reg = self.relu(fc(x_reg))

    #     return x_cls, x_reg

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

        # if self.is_incremental and self.training and self.teacher_branch is not None:
        #     with torch.no_grad():
        #         # 获取 Teacher 的 Cls 和 Reg 特征
        #         x_out_teacher_cls, x_out_teacher_reg = self.forward_teacher(x)
                
        #         # 计算 Teacher Logits
        #         norm_teacher = F.normalize(x_out_teacher_cls, p=2, dim=-1, eps=1e-6)
        #         scores_teacher = (norm_teacher @ all_embed) * self.logit_scale
                
        #         # 计算 Teacher BBoxes
        #         bboxes_teacher = self.fc_reg(x_out_teacher_reg) if self.with_reg else None
            
        #     # 缓存起来，loss 里用
        #     self.latest_teacher_scores = scores_teacher
        #     self.latest_teacher_bboxes = bboxes_teacher
        # else:
        #     self.latest_teacher_scores = None
        #     self.latest_teacher_bboxes = None

        # 只在topk空间里学习
        # if self.training:
        #     cls_score = self._mask_logits_by_topk(cls_score, rois, topk_indices)

        if not self.training and normalized_vlm_box_feats is not None:
            cls_score = cls_score.softmax(dim=-1)
            vlm_score = (normalized_vlm_box_feats @ all_embed) * self.vlm_temperature
            vlm_score = vlm_score.softmax(dim=-1)

            cls_score[:, self.base_idx] = cls_score[:, self.base_idx] ** (
                1 - self.alpha) * vlm_score[:, self.base_idx] ** self.alpha
            cls_score[:, self.novel_idx] = cls_score[:, self.novel_idx] ** (
                    1 - self.beta) * vlm_score[:, self.novel_idx] ** self.beta

        return cls_score, bbox_pred
    
    # def loss(self,
    #          cls_score,
    #          bbox_pred,
    #          rois,
    #          labels,
    #          label_weights,
    #          bbox_targets,
    #          bbox_weights,
    #          reduction_override=None):
    #     # ---------------------------------------------------
    #     # 1. 常规 Loss (基于 Merged GT)
    #     # ---------------------------------------------------
    #     # 重要：因为您已经把伪标签加入了 GT，这里会正常计算旧类的 CrossEntropy Loss。
    #     # 请确保 config 里的 loss_cls 没有屏蔽旧类！
    #     losses = super().loss(cls_score,
    #          bbox_pred,
    #          rois,
    #          labels,
    #          label_weights,
    #          bbox_targets,
    #          bbox_weights,
    #          reduction_override=None)

    #     # ---------------------------------------------------
    #     # 2. 蒸馏 Loss (辅助旧类)
    #     # ---------------------------------------------------
    #     if self.is_incremental and self.training and self.latest_teacher_scores is not None:
            
    #         # --- A. 筛选 Mask ---
    #         with torch.no_grad():
    #             t_probs = F.softmax(self.latest_teacher_scores, dim=1)
    #             novel_prob_sum = t_probs[:, self.novel_idx].sum(dim=1)
    #             # 筛选 Teacher 确信是旧类的样本
    #             valid_mask = novel_prob_sum > 0.3 

    #         if valid_mask.sum() > 0:
    #             # --- B. 分类蒸馏 (KL) ---
    #             T = 3.0
    #             s_logits_old = cls_score[valid_mask][:, self.novel_idx]
    #             t_logits_old = self.latest_teacher_scores[valid_mask][:, self.novel_idx]
                
    #             losses['loss_dist_cls'] = F.kl_div(
    #                 F.log_softmax(s_logits_old / T, dim=1),
    #                 F.softmax(t_logits_old / T, dim=1),
    #                 reduction='batchmean'
    #             ) * (T**2) * 2.0

    #             # --- C. 回归蒸馏 (Smooth L1) - [修复点] ---
    #             if bbox_pred is not None and self.latest_teacher_bboxes is not None:
                    
    #                 # 判断是 Class Agnostic (N, 4) 还是 Class Specific (N, 4*C)
    #                 if bbox_pred.shape[1] == 4:
    #                     # === 情况 1: 类别无关回归 (Class Agnostic) ===
    #                     # 此时不分新旧类通道，所有类别共用一组回归参数
    #                     # 我们直接蒸馏有效样本的回归值即可
                        
    #                     s_bbox_target = bbox_pred[valid_mask]
    #                     t_bbox_target = self.latest_teacher_bboxes[valid_mask]
                        
    #                     losses['loss_dist_bbox'] = F.smooth_l1_loss(
    #                         s_bbox_target, 
    #                         t_bbox_target, 
    #                         reduction='mean'
    #                     ) * 1.0
                        
    #                 else:
    #                     # === 情况 2: 类别特定回归 (Class Specific) ===
    #                     # 此时 shape 是 [N, 4 * num_classes]
    #                     # 需要 reshape 并只切出旧类通道
                        
    #                     num_bbox_classes = bbox_pred.shape[1] // 4
    #                     s_bbox = bbox_pred.reshape(-1, num_bbox_classes, 4)
    #                     t_bbox = self.latest_teacher_bboxes.reshape(-1, num_bbox_classes, 4)
                        
    #                     # 取 Valid 样本 + Novel 通道
    #                     s_bbox_target = s_bbox[valid_mask][:, self.novel_idx, :]
    #                     t_bbox_target = t_bbox[valid_mask][:, self.novel_idx, :]
                        
    #                     losses['loss_dist_bbox'] = F.smooth_l1_loss(
    #                         s_bbox_target, 
    #                         t_bbox_target, 
    #                         reduction='mean'
    #                     ) * 1.0

    #             # 清理缓存
    #             self.latest_teacher_scores = None
    #             self.latest_teacher_bboxes = None

    #     return losses

    # def forward(self, x, vlm_box_feats=None, rois=None, topk_indices=None):
    #     # x: [N, KC, 7, 7]
    #     all_embed = self.all_embed.type_as(x)

    #     if vlm_box_feats is not None:
    #         assert not self.training
    #         normalized_vlm_box_feats = F.normalize(vlm_box_feats, dim=-1, p=2)
    #     else:
    #         normalized_vlm_box_feats = None

    #     use_teacher_branck = self.is_incremental and self.training and (self.teacher_branch is not None)
    #     # =======================================================
    #     # 场景 A: Task 1 (Base Training) - 纯 Student 跑法
    #     # =======================================================
    #     if not use_teacher_branck:
    #         # 正常的 ConvFCHead 流程
    #         # ... (复用父类逻辑或手动写一遍) ...
    #         feat = x
    #         if self.num_shared_convs > 0:
    #             for conv in self.shared_convs:
    #                 feat = conv(feat)
    #         if self.num_shared_fcs > 0:
    #             if self.with_avg_pool:
    #                 feat = self.avg_pool(feat)
    #             feat = feat.flatten(1)
    #             for fc in self.shared_fcs:
    #                 feat = self.relu(fc(feat))
            
    #         # Cls Branch
    #         x_cls = feat
    #         for conv in self.cls_convs:
    #             x_cls = conv(x_cls)
    #         if x_cls.dim() > 2:
    #             x_cls = x_cls.flatten(1)
    #         for fc in self.cls_fcs:
    #             x_cls = self.relu(fc(x_cls))
            
    #         norm_cls = F.normalize(x_cls, p=2, dim=-1, eps=1e-6)
    #         cls_score = (norm_cls @ all_embed) * self.logit_scale
            
    #         # Reg Branch
    #         x_reg = feat
    #         # ... (Reg logic) ...
    #         for conv in self.reg_convs:
    #             x_reg = conv(x_reg)
    #         if x_reg.dim() > 2:
    #             if self.with_avg_pool:
    #                 x_reg = self.avg_pool(x_reg)
    #             x_reg = x_reg.flatten(1)
    #         for fc in self.reg_fcs:
    #             x_reg = self.relu(fc(x_reg))
    #         bbox_pred = self.fc_reg(x_reg) if self.with_reg else None
            
    #         return cls_score, bbox_pred

    #     # =======================================================
    #     # 场景 B: Task 2+ (Incremental) - 双流 + 通道掩码
    #     # =======================================================
        
    #     # 1. 准备掩码
    #     if topk_indices is not None and rois is not None:
    #         batch_inds = rois[:, 0].long()
    #         # 兼容 indices 和 x 的 batch 对齐
    #         current_indices = topk_indices[batch_inds] if topk_indices.shape[0] != x.shape[0] else topk_indices
    #         mask_old = self.get_channel_mask(x, current_indices)
    #         mask_new = 1.0 - mask_old
    #     else:
    #         # 兜底：如果没有 index，默认全开
    #         mask_old = torch.ones_like(x)
    #         mask_new = torch.ones_like(x)

    #     # 2. Teacher Forward (Old Channels Only)
    #     x_teacher_in = x * mask_old
    #     with torch.no_grad():
    #         x_out_teacher = self.forward_teacher(x_teacher_in)
    #         norm_teacher = F.normalize(x_out_teacher, p=2, dim=-1, eps=1e-6)
    #         scores_teacher = (norm_teacher @ all_embed) * self.logit_scale

    #     # 3. Student Forward (Gradient Detach on Old Channels)
    #     x_student_in = (x * mask_old).detach() + (x * mask_new)
        
    #     feat = x_student_in
    #     # ... Shared Layers ...
    #     if self.num_shared_convs > 0:
    #         for conv in self.shared_convs:
    #             feat = conv(feat)
    #     if self.num_shared_fcs > 0:
    #         if self.with_avg_pool:
    #             feat = self.avg_pool(feat)
    #         feat = feat.flatten(1)
    #         for fc in self.shared_fcs:
    #             feat = self.relu(fc(feat))
        
    #     # ... Cls Branch ...
    #     x_cls = feat
    #     for conv in self.cls_convs:
    #         x_cls = conv(x_cls)
    #     if x_cls.dim() > 2:
    #         x_cls = x_cls.flatten(1)
    #     for fc in self.cls_fcs:
    #         x_cls = self.relu(fc(x_cls))
        
    #     norm_student = F.normalize(x_cls, p=2, dim=-1, eps=1e-6)
    #     scores_student = (norm_student @ all_embed) * self.logit_scale

    #     # 4. Stitching
    #     cls_score = scores_student.clone()
    #     cls_score[:, self.novel_idx] = scores_teacher[:, self.novel_idx]

    #     # 5. Reg Branch (Student Only)
    #     x_reg = feat
    #     for conv in self.reg_convs:
    #         x_reg = conv(x_reg)
    #     if x_reg.dim() > 2:
    #         if self.with_avg_pool:
    #             x_reg = self.avg_pool(x_reg)
    #         x_reg = x_reg.flatten(1)
    #     for fc in self.reg_fcs:
    #         x_reg = self.relu(fc(x_reg))
    #     bbox_pred = self.fc_reg(x_reg) if self.with_reg else None

    #     if not self.training and normalized_vlm_box_feats is not None:
    #         cls_score = cls_score.softmax(dim=-1)
    #         vlm_score = (normalized_vlm_box_feats @ all_embed) * self.vlm_temperature
    #         vlm_score = vlm_score.softmax(dim=-1)

    #         cls_score[:, self.base_idx] = cls_score[:, self.base_idx] ** (
    #             1 - self.alpha) * vlm_score[:, self.base_idx] ** self.alpha
    #         cls_score[:, self.novel_idx] = cls_score[:, self.novel_idx] ** (
    #                 1 - self.beta) * vlm_score[:, self.novel_idx] ** self.beta

    #     return cls_score, bbox_pred

    @force_fp32(apply_to=('cls_score', 'bbox_pred'))
    def get_bboxes(self,
                   rois,
                   cls_score,
                   bbox_pred,
                   img_shape,
                   scale_factor,
                   rescale=False,
                   cfg=None):
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
