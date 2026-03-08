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

@HEADS.register_module()
class CatSegMoERoIHead(StandardRoIHead):
    def __init__(self, vlm_roi_extractor=None, **kwargs):
        super().__init__(**kwargs)
        self.vlm_roi_extractor = build_roi_extractor(vlm_roi_extractor)

    def _apply_mask_to_fpn(self, x_fpn, x_mask):
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

                    #   x_old=None,
                    #   cat_seg_logits_old=None,
                    #   teacher_indices=None,
                    #   teacher_roi_head=None,
                    #   old_end=19,
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

        bbox_results = self._bbox_forward(x, rois, cat_seg_feats=cat_seg_feats, topk_indices=topk_indices)
        bbox_targets = self.bbox_head.get_targets(
            sampling_results, gt_bboxes, gt_labels, self.train_cfg)

        loss_bbox = self.bbox_head.loss(bbox_results['cls_score'],
                                        bbox_results['bbox_pred'], rois,
                                        *bbox_targets)
        losses = dict()
        losses.update(loss_bbox)

        return losses
    

    # def _bbox_forward(self, x, rois, cat_seg_feats, topk_indices=None, vlm_feat=None):
    #     """
    #     :param x: FPN features
    #     :param rois: [N, 5]
    #     :param cat_seg_feats: [bsz, class_num, h, w]
    #     """
    #     x_fusion = self._apply_mask_to_fpn(x, cat_seg_feats, topk_indices=topk_indices)  # tuple of [B, T*C, H, W]
    #     bbox_feats = self.bbox_roi_extractor(
    #         x_fusion[:self.bbox_roi_extractor.num_inputs], rois
    #     )
    #     vlm_roi_feats = None
    #     if vlm_feat is not None:
    #         vlm_roi_feats = self.vlm_roi_extractor([vlm_feat], rois)[..., 0, 0]
    #     # =========================================================
    #     # 将 Image 级别的 topk_indices 转换为 Proposal 级别的
    #     # =========================================================
    #     # topk_indices 是 [B, K]，rois 的第 0 列是 batch_idx
    #     # 我们需要把它扩展成 [N, K]，其中 N 是当前 batch 的总 Proposal 数量
    #     if topk_indices is not None:
    #         batch_inds = rois[:, 0].long()
    #         proposal_topk_indices = topk_indices[batch_inds] # shape: [N, K]
    #     else:
    #         proposal_topk_indices = None

    #     cls_score, bbox_pred = self.bbox_head(bbox_feats, vlm_roi_feats, proposal_topk_indices)
    #     bbox_results = dict(
    #         cls_score=cls_score, bbox_pred=bbox_pred, bbox_feats=bbox_feats
    #     )
    #     return bbox_results

    def _bbox_forward(self, x, rois, cat_seg_feats, topk_indices=None, vlm_feat=None):
        """
        核心改造：先分别做 RoIAlign，再在 RoI 级别进行特征乘法分解。
        """
        # ---------------------------------------------------------
        # Step 1: 对原始 FPN 特征做 RoIAlign 
        # 输出维度: [N, C, 7, 7]  (C 通常是 256)
        # ---------------------------------------------------------
        bbox_feats = self.bbox_roi_extractor(
            x[:self.bbox_roi_extractor.num_inputs], rois
        )

        # ---------------------------------------------------------
        # Step 2: 构建 Mask 金字塔，并对其做 RoIAlign
        # ---------------------------------------------------------
        mask_pyramid = []
        for lvl_feat in x[:self.bbox_roi_extractor.num_inputs]:
            B, C, H, W = lvl_feat.shape
            
            if cat_seg_feats.shape[-2:] != (H, W):
                mask_resized = F.interpolate(
                    cat_seg_feats, size=(H, W), mode='bilinear', align_corners=False
                )
            else:
                mask_resized = cat_seg_feats
                
            mask_sig = torch.sigmoid(mask_resized)
            mask_pyramid.append(mask_sig)

        # =========================================================
        # 【关键修复】：临时动态修改 Extractor 的通道数占位符
        # =========================================================
        original_channels = self.bbox_roi_extractor.out_channels
        K_mask = mask_pyramid[0].shape[1] # 获取当前 Mask 的类别通道数 (10)
        
        self.bbox_roi_extractor.out_channels = K_mask # 临时改为 10
        
        # 此时 Extractor 会正确分配 [N, 10, 7, 7] 的张量容器
        mask_roi_feats = self.bbox_roi_extractor(tuple(mask_pyramid), rois)
        
        self.bbox_roi_extractor.out_channels = original_channels # 用完后立刻恢复为 256

        # ---------------------------------------------------------
        # Step 3: 在 RoI 级别进行张量乘法分解 (极大地节省显存！)
        # ---------------------------------------------------------
        N_roi, C_feat, H_roi, W_roi = bbox_feats.shape
        K_mask = mask_roi_feats.shape[1]

        # bbox_feats:     [N, C, 7, 7] -> [N, 1, C, 7, 7]
        # mask_roi_feats: [N, K, 7, 7] -> [N, K, 1, 7, 7]
        # 广播相乘后:      [N, K, C, 7, 7]
        weighted_roi_feats = bbox_feats.unsqueeze(1) * mask_roi_feats.unsqueeze(2)

        # 融合后展平类别和特征通道，适配后续 BBoxHead 的输入要求: [N, K*C, 7, 7]
        fused_bbox_feats = weighted_roi_feats.view(N_roi, K_mask * C_feat, H_roi, W_roi)

        # ---------------------------------------------------------
        # Step 4: VLM 特征处理与 Top-K 索引映射
        # ---------------------------------------------------------
        vlm_roi_feats = None
        if vlm_feat is not None:
            vlm_roi_feats = self.vlm_roi_extractor([vlm_feat], rois)[..., 0, 0]

        # 将 Image 级别的 topk_indices 映射为 Proposal 级别的
        if topk_indices is not None:
            batch_inds = rois[:, 0].long()
            proposal_topk_indices = topk_indices[batch_inds] # shape: [N, K]
        else:
            proposal_topk_indices = None

        # 送入修改过参数量优化的 CatSegMoEBBoxHead
        cls_score, bbox_pred = self.bbox_head(
            fused_bbox_feats, vlm_roi_feats, proposal_topk_indices
        )
        
        bbox_results = dict(
            cls_score=cls_score, bbox_pred=bbox_pred, bbox_feats=fused_bbox_feats
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
class CatSegMoEBBoxHead(ConvFCBBoxHead):
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
                 
                 old_end=None,
                 topk=None,
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

        self.old_end = old_end
        self.topk = topk
        
        # =========================================================
        # 核心改造：放弃原有的巨大 FC，构建针对每个类别的“微型专家”
        # =========================================================
        # feat_dim 是经过 RoIAlign 展平后的特征维度。
        # 原来是 [N, 256, 7, 7] 展平，现在因为你送进来的是拆分好的 Top-K 单类特征，
        # 所以维度依然是 256 * 7 * 7 (假设 FPN 输出是 256 维)
        self.avg_pool_cls = nn.AdaptiveAvgPool2d((1, 1))
        feat_dim = int(self.in_channels / self.topk)
        hidden_dim = 128 # 专家内部隐藏层维度，可调
        
        self.class_experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(feat_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, text_dim) # 投影到 Text Embedding 所在的语义空间！
            ) for _ in range(self.num_classes)
        ])
        
        # 增量学习的绝对防线：物理冻结所有旧类专家
        if self.old_end > 0:
            for i in range(self.old_end):
                for p in self.class_experts[i].parameters():
                    p.requires_grad = False

        self.bg_expert = nn.Sequential(
            nn.Linear(feat_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, text_dim)
        )

    @property
    def all_embed(self):
        if self.learn_bg:
            bg_embed = F.normalize(self.bg_embedding, dim=0)
            return torch.cat([self.all_embeddings[:, :-1], bg_embed], dim=1)
        else:
            return self.all_embeddings

    def forward(self, x, vlm_box_feats=None, topk_indices=None):
        all_embed = self.all_embed.type_as(x)

        if vlm_box_feats is not None:
            assert not self.training
            normalized_vlm_box_feats = F.normalize(vlm_box_feats, dim=-1, p=2)
        else:
            normalized_vlm_box_feats = None
        # x 的维度目前是：[N, K * 256, 7, 7] (因为你在 RoIHead 里 rearrange 了)
        N = x.shape[0]
        K = self.topk
        C = x.shape[1] // K  # C 就是 256
        H, W = x.shape[2], x.shape[3]
        
        # 1. 解开 rearranged 的特征，恢复出 [N, K, 256, 7, 7]
        # 然后展平最后三维得到 [N, K, 256*7*7]
        x_unfolded = x.view(N, K, C, H, W).flatten(2) # shape: [N, K, feat_dim]
        
        # 2. 初始化分类得分矩阵为极小值 (代表概率接近0)
        # [N, num_classes]
        x_pooled = self.avg_pool_cls(x_unfolded.view(N * K, C, H, W))
        # 展平并变回 [N, K, 256]
        x_cls_feat = x_pooled.view(N, K, C) 
        
        mean_feat = x_cls_feat.mean(dim=1) # shape: [N, 256]
        
        # 送入 bg_expert 得到投影，并做归一化
        bg_projected = self.bg_expert(mean_feat) # [N, text_dim]
        bg_norm = F.normalize(bg_projected, p=2, dim=-1, eps=1e-6)
        
        # 算出一个涵盖所有 num_classes 的底分矩阵
        # 这个操作保证了所有类别的计算图都是连通的，永远不会发生梯度断裂！
        base_score = (bg_norm @ all_embed) * self.logit_scale # [N, num_classes]
        
        # 使用 clone() 是为了防止后面的原地索引赋值导致 PyTorch 反向传播报错
        cls_score = base_score.clone()
        
        # 3. 动态路由打分 (GPU 并行优化版)
        if topk_indices is not None:
            # 找到当前 Batch 中所有被激活的独特类别
            unique_classes = torch.unique(topk_indices)
            
            for cls_id in unique_classes:
                if cls_id >= self.num_classes:
                    continue
                    
                # 找到 [N, K] 中所有等于当前 cls_id 的位置掩码
                mask = (topk_indices == cls_id) # shape: [N, K], bool
                
                # 提取出属于该类别的所有特征，实现 Batch 化处理
                # feats 维度: [M, feat_dim]，其中 M 是当前 Batch 中预测为该类的总数
                feats = x_cls_feat[mask] 
                
                # 1. 专家提取 (一次性处理 M 个特征，充分利用 GPU 并行)
                projected_feat = self.class_experts[cls_id](feats) # [M, text_dim]
                
                # 2. L2 归一化
                projected_feat_norm = F.normalize(projected_feat, p=2, dim=-1, eps=1e-6)
                
                # 3. 提取对应的 Text Embedding 并算相似度
                target_text_embed = all_embed[:, cls_id] # [text_dim]
                score = (projected_feat_norm @ target_text_embed) * self.logit_scale # [M]
                
                # 4. 将得分填回 cls_score 的对应位置
                # 获取 mask 为 True 的第一维度索引 (即 N 维度上的索引 i)
                batch_indices = mask.nonzero(as_tuple=True)[0] 
                cls_score[batch_indices, cls_id] = score
        else:
            # 万一没有 topk_indices (比如某些初始化检查)，为了防报错
            pass

        # ---------------------------------------------------
        # 回归分支 (Reg Branch) - 简化处理
        # ---------------------------------------------------
        # 由于输入是 K 个类的融合特征，我们可以把它们简单平均，或者用一个小的注意力池化，
        # 然后送进原有的回归分支。这里用平均操作，把 [N, K*256, 7, 7] 压缩回 [N, 256, 7, 7]
        x_reg = x  
        
        # 正常经过父类帮你建好的 2560 -> 256 的卷积层
        if self.num_shared_convs > 0:
            for conv in self.shared_convs:
                x_reg = conv(x_reg)
                
        if self.num_shared_fcs > 0:
            if self.with_avg_pool:
                x_reg = self.avg_pool(x_reg)
            x_reg = x_reg.flatten(1)
            for fc in self.shared_fcs:
                x_reg = self.relu(fc(x_reg))

        # 专属回归分支
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
