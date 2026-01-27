import json
import torch
import torch.nn as nn
import torch.nn.functional as F
# import torch.utils.checkpoint as cp
# from einops import rearrange

from mmdet.models.roi_heads import StandardRoIHead, ConvFCBBoxHead
from mmdet.models.builder import HEADS, build_roi_extractor
from mmdet.core import bbox2roi, bbox2result, multiclass_nms
from mmcv.runner import force_fp32
from mmcv.cnn import ConvModule
# from mmdet.models.losses import accuracy

@HEADS.register_module()
class CatSeg3DRoIHead(StandardRoIHead):
    def __init__(self, vlm_roi_extractor=None, mask_roi_extractor=None, **kwargs):
        super().__init__(**kwargs)
        self.vlm_roi_extractor = build_roi_extractor(vlm_roi_extractor)
        self.mask_roi_extractor = build_roi_extractor(mask_roi_extractor)  # NEW

    def forward_train(self,
                      x,
                      img_metas,
                      proposal_list,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      cat_seg_feats=None,
                      topk_indices=None,   # [B,K]  每图的K个slot对应的global类id
                      x_old=None,          # teacher FPN tuple(list)
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

        bbox_results = self._bbox_forward(
            x, rois,
            cat_seg_feats=cat_seg_feats,
            topk_indices=topk_indices,
            x_old=x_old,
            old_end=old_end
        )

        bbox_targets = self.bbox_head.get_targets(
            sampling_results, gt_bboxes, gt_labels, self.train_cfg)

        loss_bbox = self.bbox_head.loss(
            bbox_results['cls_score'],
            bbox_results['bbox_pred'],
            rois, *bbox_targets
        )

        losses = dict()
        losses.update(loss_bbox)
        return losses

    def _bbox_forward(self, x, rois,
                  cat_seg_feats=None,      # [B,K,h,w] 或 [B,T,h,w]
                  topk_indices=None,       # [B,K]
                  x_old=None,              # teacher FPN tuple
                  old_end=19,
                  vlm_feat=None):

        # -------- 0) 保证 mask 是 K 通道且顺序与 topk_indices 对齐 --------
        if (topk_indices is not None) and (cat_seg_feats is not None) and (cat_seg_feats.size(1) != topk_indices.size(1)):
            # cat_seg_feats: [B,T,h,w] -> gather -> [B,K,h,w]
            cat_seg_feats = torch.gather(
                cat_seg_feats, 1,
                topk_indices[..., None, None].expand(-1, -1, cat_seg_feats.size(-2), cat_seg_feats.size(-1))
            )

        assert cat_seg_feats is not None
        B, K, h0, w0 = cat_seg_feats.shape

        # teacher fpn (optional)
        x_old_use = None
        if (x_old is not None) and self.training:
            x_old_use = x_old[:self.bbox_roi_extractor.num_inputs]

        x_use = x[:self.bbox_roi_extractor.num_inputs]

        # -------- 1) per-k 做：FPN 空间 fuse -> ROIAlign --------
        roi_feats = []
        for k in range(K):
            # gate_k: [B,1,h,w]
            mask_k = cat_seg_feats[:, k:k+1, :, :]

            fused_pyr = []
            for lvl, feat_new in enumerate(x_use):
                B1, C, H, W = feat_new.shape

                # resize gate to this level
                if mask_k.shape[-2:] != (H, W):
                    mk = F.interpolate(mask_k, size=(H, W), mode='bilinear', align_corners=False)
                else:
                    mk = mask_k

                mk = mk.clamp(-10, 10)
                gate = torch.sigmoid(mk / 2.0)  # [B,1,H,W]

                # old/new slot mask per image: [B,1,1,1]
                if topk_indices is not None:
                    old_sel_bool = (topk_indices[:, k] < old_end).view(B, 1, 1, 1)   # bool
                else:
                    old_sel_bool = None

                # old slot 的 gate detach（用 float 权重）
                if self.training and (old_sel_bool is not None):
                    old_sel_f = old_sel_bool.to(gate.dtype)                          # float
                    gate = gate * (1.0 - old_sel_f) + gate.detach() * old_sel_f

                # choose base (teacher for old slot, student for new slot)
                base = feat_new
                if (x_old_use is not None) and (old_sel_bool is not None):
                    feat_old = x_old_use[lvl].to(dtype=feat_new.dtype)               # [B,C,H,W]
                    base = torch.where(old_sel_bool, feat_old, feat_new)             # OK: condition is bool

                fused = base * gate                                    # [B,C,H,W]
                fused_pyr.append(fused)

            fused_pyr = tuple(fused_pyr)

            # ROIAlign（不扩 rois！）
            roi_k = self.bbox_roi_extractor(fused_pyr, rois)           # [N,C,ph,pw]
            roi_feats.append(roi_k)

        # stack -> [N,K,C,ph,pw]
        roi_feat_5d = torch.stack(roi_feats, dim=1)

        # -------- 2) vlm roi feats（按你的原逻辑）--------
        vlm_roi_feats = None
        if vlm_feat is not None:
            vlm_roi_feats = self.vlm_roi_extractor([vlm_feat], rois)[..., 0, 0]

        cls_score, bbox_pred = self.bbox_head(roi_feat_5d, vlm_roi_feats)

        return dict(cls_score=cls_score, bbox_pred=bbox_pred, bbox_feats=roi_feat_5d)

    
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
class CatSeg3DBBoxHead(ConvFCBBoxHead):
    def __init__(self,
               num_classes,
               text_dim,
               logit_scale=50.0,
               class_embed=None,
               seen_classes=None,
               all_classes=None,
               learn_bg=True,
               
               vlm_temperature=100.0,
               alpha=0.2,
               beta=0.45,

               # ---- 3D 特有控制项 ----
               k_dim=None,        # 固定 top-k（pad_len），pool_k=False 时必须提供
               pool_k=False,       # True: pool 时聚合 K；False: 保留K到pool后再投影
               mix_k=False,       # True: 允许 kernel 在 K 维>1（需要排序才更合理）
               **kwargs):
        super().__init__(**kwargs)

        self.conv_cfg_3d = dict(type='Conv3d')

        self.pool_k = pool_k
        self.k_dim = k_dim
        self.mix_k = mix_k

        # 你现在不排序，建议 mix_k=False（不在K维做局部卷积）
        if self.mix_k:
            self.kconv_kernel = (3, 3, 3)
            self.kconv_pad = (1, 1, 1)
        else:
            self.kconv_kernel = (1, 3, 3)
            self.kconv_pad = (0, 1, 1)

        # shared
        self.shared_convs, self.shared_fcs, shared_last_dim = self._add_conv_fc_branch_3d(
            self.num_shared_convs, self.num_shared_fcs, self.in_channels, is_shared=True
        )
        self.shared_out_channels = shared_last_dim

        # cls/reg branches
        self.cls_convs, self.cls_fcs, self.cls_last_dim = self._add_conv_fc_branch_3d(
            self.num_cls_convs, self.num_cls_fcs, self.shared_out_channels
        )
        self.reg_convs, self.reg_fcs, self.reg_last_dim = self._add_conv_fc_branch_3d(
            self.num_reg_convs, self.num_reg_fcs, self.shared_out_channels
        )

        if self.with_avg_pool:
            if self.pool_k:
                # 聚 K,H,W -> (1,1,1)
                self.avg_pool3d = nn.AdaptiveAvgPool3d((1, 1, 1))
                self.k_projector = None
            else:
                # 保留 K 到 pool 后：输出 (K,1,1)，再用线性层压回 C
                assert self.k_dim is not None, "pool_k=False requires fixed k_dim (pad_len)."
                self.avg_pool3d = nn.AdaptiveAvgPool3d((self.k_dim, 1, 1))
                # 注意：pool 后 flatten 会变成 C*K，所以加个投影回 C，后续 fc 维度不改
                C0 = self.conv_out_channels if self.num_shared_convs > 0 else self.in_channels
                self.k_projector = nn.Linear(C0 * self.k_dim, C0)
        else:
            # 不建议：会把 K*H*W 全展平，维度很大
            self.avg_pool3d = None
            self.k_projector = None

        self.learn_bg = learn_bg
        self.num_classes = num_classes

        # 可学习温度参数
        self.logit_scale = nn.Parameter(torch.tensor(logit_scale))

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

        # 最终把 cls feature 映射到 512
        self.fc_cls = nn.Linear(self.cls_last_dim, text_dim)

        # bbox reg
        out_dim_reg = (4 if self.reg_class_agnostic else 4 * (len(self.all_classes)-1))
        self.fc_reg = nn.Linear(self.reg_last_dim, out_dim_reg)

    @property
    def all_embed(self):
        if self.learn_bg:
            bg = F.normalize(self.bg_embedding, dim=0)
            return torch.cat([self.all_embeddings[:, :-1], bg], dim=1)
        return self.all_embeddings

    def _add_conv_fc_branch_3d(self, num_convs, num_fcs, in_channels, is_shared=False):
        last_dim = in_channels

        convs = nn.ModuleList()
        for i in range(num_convs):
            conv_in = last_dim if i == 0 else self.conv_out_channels
            # 不排序时：先用(1,3,3)不混K维，避免顺序假设
            convs.append(ConvModule(
                conv_in, self.conv_out_channels,
                kernel_size=(1, 3, 3),
                padding=(0, 1, 1),
                conv_cfg=self.conv_cfg_3d,
                norm_cfg=self.norm_cfg,
                act_cfg=dict(type='ReLU')
            ))
            last_dim = self.conv_out_channels

        fcs = nn.ModuleList()
        for i in range(num_fcs):
            fc_in = last_dim if i == 0 else self.fc_out_channels
            fcs.append(nn.Linear(fc_in, self.fc_out_channels))
            last_dim = self.fc_out_channels

        return convs, fcs, last_dim
    
    def _pool_and_flatten(self, x):
        """
        x: [N,C,K,H,W] or [N,C,H,W] or [N,feat]
        return: [N,feat]
        """
        if x.dim() == 5:
            if self.with_avg_pool:
                x = self.avg_pool3d(x)           # [N,C,1,1,1] or [N,C,K,1,1]
                x = x.flatten(1)                 # [N, C] or [N, C*K]
                if (self.k_projector is not None):
                    x = self.k_projector(x)      # [N, C]
            else:
                x = x.flatten(1)
        elif x.dim() == 4:
            # 兼容：如果哪天你又传回 4D
            x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return x
    
    def forward(self, x5d, vlm_box_feats=None):
        all_embed = self.all_embed.type_as(x5d)

        if vlm_box_feats is not None:
            assert not self.training
            normalized_vlm_box_feats = F.normalize(vlm_box_feats, dim=-1, p=2)
        else:
            normalized_vlm_box_feats = None

        assert x5d.dim() == 5  # [N, K, C, pd, pw]
        N, K, C, H, W = x5d.shape

        # Conv3d expects [N, C, K, H, W]
        x = x5d.permute(0, 2, 1, 3, 4).contiguous()

        # shared convs
        if self.num_shared_convs > 0:
            for conv in self.shared_convs:
                x = conv(x)

        if self.num_shared_fcs > 0:
            x = self._pool_and_flatten(x)  # [N,feat]
            for fc in self.shared_fcs:
                x = self.relu(fc(x))

        # separate branches
        x_cls = x
        x_reg = x

        # ---- cls branch convs ----
        for conv in self.cls_convs:
            x_cls = conv(x_cls)
        # conv 后进入 fc 前的 flatten
        if x_cls.dim() > 2:
            x_cls = self._pool_and_flatten(x_cls)
        for fc in self.cls_fcs:
            x_cls = self.relu(fc(x_cls))

        # ---- reg branch convs ----
        for conv in self.reg_convs:
            x_reg = conv(x_reg)
        if x_reg.dim() > 2:
            x_reg = self._pool_and_flatten(x_reg)
        for fc in self.reg_fcs:
            x_reg = self.relu(fc(x_reg))

        # ===== bbox reg =====
        bbox_pred = self.fc_reg(x_reg) if self.with_reg else None

        # ===== open-vocab cls =====
        x_cls = self.fc_cls(x_cls)                         # -> [N,512]
        normalized_x_cls = F.normalize(x_cls, p=2, dim=-1, eps=1e-6)
        cls_score = (normalized_x_cls @ all_embed) * self.logit_scale

        # ===== 推理时融合 VLM =====
        if (not self.training) and (normalized_vlm_box_feats is not None):
            cls_prob = cls_score.softmax(dim=-1)
            vlm_score = (normalized_vlm_box_feats @ all_embed) * self.vlm_temperature
            vlm_prob = vlm_score.softmax(dim=-1)

            cls_prob[:, self.base_idx] = cls_prob[:, self.base_idx] ** (1 - self.alpha) * vlm_prob[:, self.base_idx] ** self.alpha
            cls_prob[:, self.novel_idx] = cls_prob[:, self.novel_idx] ** (1 - self.beta) * vlm_prob[:, self.novel_idx] ** self.beta

            # 你原代码是返回 prob（后面 get_bboxes 里再 softmax 一次会重复）
            # 建议你要么这里返回 logit（log），要么 get_bboxes 里别再 softmax
            cls_score = cls_prob  # 保持和你原来一致

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
