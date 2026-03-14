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

        bbox_results = self._bbox_forward(x, rois, cat_seg_feats=cat_seg_feats)
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

    def _bbox_forward(self, x, rois, cat_seg_feats, vlm_feat=None):
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


        # 送入修改过参数量优化的 CatSegMoEBBoxHead
        cls_score, bbox_pred = self.bbox_head(
            fused_bbox_feats, 
            bbox_feats,
            vlm_roi_feats
        )
        
        bbox_results = dict(
            cls_score=cls_score, bbox_pred=bbox_pred, bbox_feats=fused_bbox_feats
        )
        return bbox_results

    def simple_test(self, x, proposal_list, img_metas, cat_seg_feats=None, vlm_feat=None, rescale=False):
        assert self.with_bbox, "Bbox head must be implemented."
        det_bboxes, det_labels = self.simple_test_bboxes(
            x, img_metas, proposal_list, self.test_cfg,
            cat_seg_feats=cat_seg_feats, rescale=rescale, vlm_feat=vlm_feat)
        
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

        bbox_results = self._bbox_forward(x, rois, cat_seg_feats=cat_seg_feats, vlm_feat=vlm_feat)
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
    
class ResidualExpert(nn.Module):
    def __init__(self, feat_dim, hidden_dim, text_dim):
        super().__init__()
        self.fc1 = nn.Linear(feat_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, text_dim)
        self.shortcut = nn.Linear(feat_dim, text_dim, bias=False)

    def forward(self, x):
        return self.fc2(self.act(self.norm(self.fc1(x)))) + self.shortcut(x)
    
class ConvExpert(nn.Module):
    def __init__(self, in_channels=256, hidden_channels=64, out_dim=512):
        super().__init__()
        # 1x1卷积降维
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden_channels)
        self.act1 = nn.ReLU(inplace=True)

        # 3x3卷积提取特征
        self.conv2 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(hidden_channels)
        self.act2 = nn.ReLU(inplace=True)

        # 1x1卷积升维
        self.conv3 = nn.Conv2d(hidden_channels, in_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(in_channels)
        self.act3 = nn.ReLU(inplace=True)

        # 投影到text embedding语义空间
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.proj = nn.Linear(in_channels, out_dim)

    def forward(self, x_2d):
        identity = x_2d 
        
        # 空间特征提取
        out = self.act1(self.bn1(self.conv1(x_2d)))
        out = self.act2(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        
        # 引入残差连接 (极其有利于梯度回传)
        out = self.act3(out + identity)
        
        # 池化并展平: [N, 256, 7, 7] -> [N, 256, 1, 1] -> [N, 256]
        pooled_feat = self.global_pool(out).flatten(1)
        
        # 最终线性投影到 512 维
        final_embed = self.proj(pooled_feat)
        
        return final_embed

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
                 current_classes=None,
                #  topk=None,
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
        all_embed = F.normalize(all_embed, p=2, dim=0)
        self.register_buffer('all_embeddings', all_embed)

        if learn_bg:
            self.bg_embedding = nn.Parameter(all_embed[:, -1:].clone().contiguous())

        self.old_end = old_end
        # self.topk = topk
        
        # =========================================================
        # 核心改造：放弃原有的巨大 FC，构建针对每个类别的“微型专家”
        # =========================================================
        # feat_dim 是经过 RoIAlign 展平后的特征维度。
        # 原来是 [N, 256, 7, 7] 展平，现在因为你送进来的是拆分好的 Top-K 单类特征，
        # 所以维度依然是 256 * 7 * 7 (假设 FPN 输出是 256 维)
        # self.avg_pool_cls = nn.AdaptiveAvgPool2d((1, 1))
        feat_dim = self.in_channels
        hidden_dim = 64 # 专家内部隐藏层维度，可调
        
        # self.class_experts = nn.ModuleList([
        #     nn.Sequential(
        #         ResidualExpert(feat_dim, hidden_dim, text_dim)
        #     ) for _ in range(self.num_classes)
        # ])

        self.class_experts = nn.ModuleList([
            nn.Sequential(
                ConvExpert(in_channels=feat_dim, hidden_channels=hidden_dim, out_dim=text_dim)
            ) for _ in range(self.num_classes)
        ])
        
        # 增量学习的绝对防线：物理冻结所有旧类专家
        if self.old_end > 0:
            for i in range(self.old_end):
                for p in self.class_experts[i].parameters():
                    p.requires_grad = False

    @property
    def all_embed(self):
        if self.learn_bg:
            bg_embed = F.normalize(self.bg_embedding, dim=0)
            return torch.cat([self.all_embeddings[:, :-1], bg_embed], dim=1)
        else:
            return self.all_embeddings

    def forward(self, x, fpn_feats, vlm_box_feats=None):
        all_embed = self.all_embed.type_as(x)

        if vlm_box_feats is not None:
            assert not self.training
            normalized_vlm_box_feats = F.normalize(vlm_box_feats, dim=-1, p=2)
        else:
            normalized_vlm_box_feats = None
        
        if self.training and self.old_end is not None and self.old_end > 0:
            # Task 2 训练阶段：只激活新类
            active_classes = list(range(self.old_end, self.num_classes))
        else:
            # 测试阶段 (或 Task 1 训练)：激活所有类
            active_classes = list(range(self.num_classes))

        # x 的维度目前是：[N, K * 256, 7, 7] (因为你在 RoIHead 里 rearrange 了)
        N = x.shape[0]
        num_active = len(active_classes)
        C = x.shape[1] // num_active  # C 就是 256
        H, W = x.shape[2], x.shape[3]
        
        x_unfolded = x.view(N, num_active, C, H, W)
        
        # x_pooled = self.avg_pool_cls(x_unfolded.flatten(0, 1)).view(N, num_active, C)
        # ==========================================
        
        cls_score = torch.full((N, self.num_classes), -10.0, device=x.device, dtype=x.dtype)
        
        # 专属专家精准打分
        for i, cls_id in enumerate(active_classes):
            if cls_id >= self.num_classes:
                continue
            
            cls_feats = x_unfolded[:, i, ...] 

            projected_feat = self.class_experts[cls_id](cls_feats) # [N, 256] @ [256, 128] 完美契合！
            projected_feat_norm = F.normalize(projected_feat, p=2, dim=-1, eps=1e-6)
            
            # 取出文本 Embedding 算余弦相似度
            target_text_embed = all_embed[:, cls_id] # [text_dim]
            score = (projected_feat_norm @ target_text_embed) * self.logit_scale # [N]
            
            # 填入矩阵对应的位置
            cls_score[:, cls_id] = score

        # ---------------------------------------------------
        # 回归分支 (Reg Branch) - 简化处理
        # ---------------------------------------------------
        x_reg = fpn_feats
        
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
