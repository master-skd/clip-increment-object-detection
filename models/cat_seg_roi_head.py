import json
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

        # 解耦一致性蒸馏(Decoupled Consistency Distillation)
        # if (x_old is not None) and (cat_seg_logits_old is not None) and (teacher_indices is not None):
        #     # 1.匹配矩阵计算
        #     # topk_indices: [B, K]
        #     # teacher_indices: [B, K]

        #     # 扩展维度以便广播对比 [B, K, 1] == [B, 1, K]
        #     s_idx_exp = topk_indices.unsqueeze(2)
        #     t_idx_exp = teacher_indices.unsqueeze(1)

        #     # 找到相同的类别id
        #     # match_matrix[b, i, j]为True表示student的第i个通道和teacher的第j个通道是同一个类
        #     match_matrix = (s_idx_exp == t_idx_exp)  # [B, K, K]

        #     # 2.限制在旧类范围内(只蒸馏旧类)
        #     is_old = s_idx_exp < old_end
        #     valid_match = match_matrix & is_old  # [B, K, K]

        #     # 3. 提取匹配对索引
        #     # matches: (batch_idx, student_channel_idx, teacher_channel_idx)
        #     matches = torch.nonzero(valid_match, as_tuple=True)
        #     b_idx, s_ch, t_ch = matches

        #     # 4. 计算Mask Loss
        #     if b_idx.numel() > 0:
        #         # student拿s_ch通道, teacher拿t_ch通道
        #         s_mask_logits = cat_seg_feats[b_idx, s_ch]  # [num_matches, h, w]
        #         t_mask_logits = cat_seg_logits_old[b_idx, t_ch]  # [num_matches, h, w]

        #         # s_mask_prob = torch.sigmoid(s_mask_logits)
        #         t_mask_prob = torch.sigmoid(t_mask_logits)
        #         # losses['loss_mask_kd'] = F.mse_loss(s_mask_prob, t_mask_prob) * 100.0
        #         loss_mask_kd = F.binary_cross_entropy_with_logits(s_mask_logits, t_mask_prob)
        #         losses['loss_mask_kd'] = loss_mask_kd * 10.0
        #     else:
        #         losses['loss_mask_kd'] = cat_seg_feats.sum() * 0.0

            # # FPN蒸馏(按对应类别对齐Visual Features)
            # loss_fpn_kd = 0.
            # if b_idx.numel() > 0:
            #     with torch.no_grad():
            #         # 我们不使用全局Max，而是根据匹配到的t_ch，把对应的Teacher mask拿出来
            #         # 只有这些区域才是Student正在关注且属于旧类的区域

            #         # 取出匹配的Teacher mask [N_match, H, W]
            #         matched_masks = torch.sigmoid(cat_seg_logits_old[b_idx, t_ch])

            #         # 我们需要把这些散落的mask还原回Batch维度[B, 1, H, W]
            #         B, _, H, W = cat_seg_logits_old.shape
            #         spatial_weight = torch.zeros((B, H, W), device=cat_seg_logits_old.device)

            #         # 使用scatter_reduce将Mask聚合回对应的batch
            #         unique_b = torch.unique(b_idx)
            #         for b in unique_b:
            #             # 找到属于这个batch的所有匹配
            #             mask_indices = (b_idx == b)
            #             mask_in_batch = matched_masks[mask_indices]
            #             # 取最大值作为该batch的权重图
            #             if mask_in_batch.shape[0] > 0:
            #                 spatial_weight[b], _ = mask_in_batch.max(dim=0)
            #         spatial_weight = spatial_weight.unsqueeze(1)  # [B,1,H,W]

            #     # 开始逐层蒸馏
            #     for i, (feat_s, feat_t) in enumerate(zip(x, x_old)):
            #         B_f, C_f, H_f, W_f = feat_s.shape
            #         weight = F.interpolate(spatial_weight, size=(H_f, W_f), mode='bilinear', align_corners=False).detach()
                    
            #         # [修改 2] 使用 Cosine Similarity 替代 MSE
            #         # CLIP 特征的核心是角度。MSE 过于严苛且数值不稳定。
            #         # Cosine Loss = 1 - cos(a, b). 范围 [0, 2]
                    
            #         # Normalize (在 Channel 维度)
            #         feat_s_norm = F.normalize(feat_s, dim=1)
            #         feat_t_norm = F.normalize(feat_t, dim=1)
                    
            #         # 计算 Cosine Similarity [B, H, W]
            #         cos_sim = (feat_s_norm * feat_t_norm).sum(dim=1, keepdim=True)
                    
            #         # Loss = (1 - cos) * weight
            #         # 只有在旧物体区域，才要求方向一致
            #         loss_level = ((1.0 - cos_sim) * weight).sum() / (weight.sum() + 1e-6)
                    
            #         loss_fpn_kd += loss_level
            #     loss_fpn_kd /= len(x)  # 平均各层
            #     losses['loss_fpn_kd'] = loss_fpn_kd * 5.0
            # else:
            #     losses['loss_fpn_kd'] = x[0].sum() * 0.0

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

    @property
    def all_embed(self):
        if self.learn_bg:
            bg_embed = F.normalize(self.bg_embedding, dim=0)
            return torch.cat([self.all_embeddings[:, :-1], bg_embed], dim=1)
        else:
            return self.all_embeddings
        
    def _mask_logits_by_topk(self, cls_logits, rois, topk_indices):
        """
        :param cls_logits: [N, C_all+1]
        :param rois: [N, 5]
        :param topk_indices: [B, K]
        """
        if (rois is None) or (topk_indices is None):
            return cls_logits
        
        device = cls_logits.device
        topk = topk_indices.to(device=device, dtype=torch.long)

        img_ids = rois[:, 0].to(device=device, dtype=torch.long)
        cand = topk[img_ids]

        N, C = cls_logits.shape
        allowed = torch.zeros((N, C), dtype=torch.bool, device=device)
        allowed.scatter_(1, cand.clamp_(0, C - 1), True)
        allowed[:, C-1] = True  # bg永远允许，最后一类

        fill = -1e4 if cls_logits.dtype in (torch.float16, torch.bfloat16) else -1e9
        return cls_logits.masked_fill(~allowed, fill)

    def forward(self, x, vlm_box_feats=None, rois=None, topk_indices=None):
        all_embed = self.all_embed.type_as(x)

        if vlm_box_feats is not None:
            assert not self.training
            normalized_vlm_box_feats = F.normalize(vlm_box_feats, dim=-1, p=2)
        else:
            normalized_vlm_box_feats = None

        N, _, _, _ = x.shape
        if self.num_shared_convs > 0:
            for conv in self.shared_convs:
                x = conv(x)

        if self.num_shared_fcs > 0:
            if self.with_avg_pool:
                x = self.avg_pool(x)

            x = x.flatten(1)

            for fc in self.shared_fcs:
                x = self.relu(fc(x))
        
        # 回归分支
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

        # 分类分支
        x_cls = x
        for conv in self.cls_convs:
            x_cls = conv(x_cls)
        if x_cls.dim() > 2:
            if self.with_avg_pool:
                x_cls = self.avg_pool(x_cls)
            x_cls = x_cls.flatten(1)
        for fc in self.cls_fcs:
            x_cls = self.relu(fc(x_cls))
        normalized_x_cls = F.normalize(x_cls, p=2, dim=-1, eps=1e-6)
        target_embeds = all_embed
        cls_score = (normalized_x_cls @ target_embeds) * self.logit_scale

        # 只在topk空间里学习
        # if self.training:
        #     cls_score = self._mask_logits_by_topk(cls_score, rois, topk_indices)

        if not self.training and normalized_vlm_box_feats is not None:
            cls_score = cls_score.softmax(dim=-1)
            vlm_score = (normalized_vlm_box_feats @ target_embeds) * self.vlm_temperature
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
