import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmdet.core import images_to_levels, multi_apply
from mmdet.models.builder import HEADS
from mmdet.models.dense_heads.rpn_head import RPNHead
from mmdet.models.builder import build_roi_extractor

@HEADS.register_module()
class CatSegRPNHead(RPNHead):
    def __init__(self, **kwargs):
        if 'loss_cls' in kwargs:
            kwargs['loss_cls']['use_sigmoid'] = True
        super(CatSegRPNHead, self).__init__(**kwargs)

    def _init_layers(self):
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
                        inplace=False
                    )
                )
            self.rpn_conv = nn.Sequential(*rpn_convs)
        else:
            self.rpn_conv = nn.Conv2d(self.in_channels, self.feat_channels, 3, padding=1)
        
        self.rpn_reg = nn.Conv2d(self.feat_channels, self.num_base_priors * 4, 1)

    def forward(self, feats, seg_logits=None):
        assert seg_logits is not None, "seg_logits is required"
        semantic_map, _ = seg_logits.max(dim=1, keepdim=True)
        cls_scores = []
        bbox_preds = []
        for x in feats:
            x = self.rpn_conv(x)
            x = F.relu(x, inplace=False)

            # 回归分支
            rpn_bbox_pred = self.rpn_reg(x)

            # 分类分支
            target_h, target_w = x.shape[-2:]
            rpn_cls_score_feat = F.interpolate(
                semantic_map, 
                size=(target_h, target_w),
                mode='bilinear',
                align_corners=False
            )
            num_anchors = self.num_base_priors
            rpn_cls_score = rpn_cls_score_feat.expand(-1, num_anchors, -1, -1)
            cls_scores.append(rpn_cls_score)
            bbox_preds.append(rpn_bbox_pred)
        return cls_scores, bbox_preds

    def loss(self,
             cls_scores,
             bbox_preds,
             gt_bboxes,
             img_metas,
             gt_bboxes_ignore=None):
        losses = super(RPNHead, self).loss(
            cls_scores,
            bbox_preds,
            gt_bboxes,
            None,
            img_metas,
            gt_bboxes_ignore=gt_bboxes_ignore
        )

        # 切断分类loss的梯度
        losses['loss_cls'] = cls_scores[0].new_tensor(0.0)
        return dict(
            loss_rpn_cls=losses['loss_cls'],
            loss_rpn_bbox=losses['loss_bbox']
        )
    
    def forward_train(self,
                      x,
                      img_metas,
                      gt_bboxes,
                      gt_labels=None,
                      gt_bboxes_ignore=None,
                      proposal_cfg=None,
                      seg_logits=None,
                      **kwargs):
        outs = self(x, seg_logits)
        if gt_labels is None:
            loss_inputs = outs + (gt_bboxes, img_metas)
        else:
            loss_inputs = outs + (gt_bboxes, gt_labels, img_metas)
        losses = self.loss(*loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)

        if proposal_cfg is None:
            return losses
        else:
            proposal_list = self.get_bboxes(
                *outs, img_metas=img_metas, cfg=proposal_cfg)
            return losses, proposal_list
        
    def simple_test_rpn(self, x, img_metas, seg_logits=None):
        rpn_outs = self(x, seg_logits)
        proposal_list = self.get_bboxes(*rpn_outs, img_metas=img_metas)
        return proposal_list
    
@HEADS.register_module()
class CatSegPseudoLabelRPNHead(RPNHead):
    def __init__(self, 
                 split_idx=None, 
                 cost_roi_extractor=None, # 接收 Config 传进来的配置
                 **kwargs):
        super(CatSegPseudoLabelRPNHead, self).__init__(**kwargs)
        self.split_idx = split_idx
        assert split_idx is not None, "必须提供split_idx区分任务阶段"
        if cost_roi_extractor is not None:
            self.cost_roi_extractor = build_roi_extractor(cost_roi_extractor)
        else:
            self.cost_roi_extractor = None

    def _init_layers(self):
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
                        inplace=False
                    )
                )
            self.rpn_conv = nn.Sequential(*rpn_convs)
        else:
            self.rpn_conv = nn.Conv2d(self.in_channels, self.feat_channels, 3, padding=1)

        self.rpn_cls = nn.Conv2d(self.feat_channels,
                                 self.num_base_priors * self.cls_out_channels,
                                 1)
        self.rpn_reg = nn.Conv2d(self.feat_channels, self.num_base_priors * 4, 1)

    def _get_dynamic_threshold(self, cost_volume, gt_bboxes):
        """
        Args:
            cost_volume: [K, H, W]
            gt_bboxes: [N, 4]
        Returns:
            float: 动态计算的绝对阈值
        """
        # 默认保守阈值 (如果没有 GT 或 GT 太小)
        DEFAULT_THRESH = 0.65
        
        if gt_bboxes is None or gt_bboxes.size(0) == 0:
            return DEFAULT_THRESH

        # 1. 构造 GT RoI [0, x1, y1, x2, y2]
        # 都在单张图上操作，batch_ind 为 0
        batch_inds = gt_bboxes.new_zeros(gt_bboxes.size(0), 1)
        gt_rois = torch.cat([batch_inds, gt_bboxes], dim=1)
        
        # 2. 提取 GT 区域特征
        # cost_volume: [K, H, W] -> [1, K, H, W]
        # 注意：必须确保 self.cost_roi_extractor 已经初始化
        input_feats = [cost_volume.unsqueeze(0)]
        gt_feats = self.cost_roi_extractor(input_feats, gt_rois)
        
        # 3. 计算 GT 的置信度
        # [N_gt, K, 1, 1] -> [N_gt, K]
        gt_scores_raw = gt_feats.view(gt_bboxes.size(0), -1)
        gt_probs = gt_scores_raw.sigmoid()
        
        # 取每个 GT 最自信的那个类别的分数 (Max over Classes)
        # 代表 "这个物体有多像一个前景"
        gt_max_scores, _ = gt_probs.max(dim=1)
        
        # 4. 统计分布，确定下限
        # 逻辑：我们要找的伪标签，其置信度不应该比 "最差的那 10% 的真 GT" 还要低太多
        def safe_quantile(vals, q):
             if vals.numel() < 2: return vals.mean()
             return torch.quantile(vals, q)
        
        # 取 P10 (第 10% 分位数) 作为基准
        p10_score = safe_quantile(gt_max_scores, 0.1)
        
        # 5. 设定动态阈值
        # 稍微放宽一点点 (0.9 倍)，允许伪标签比真 GT 稍微弱一点
        dynamic_thresh = p10_score * 0.9
        
        # 6. 安全截断 (Clamping)
        # 防止阈值太低引入背景，或太高导致漏检
        # Lower bound 0.5: 即使全图都很黑，低于 0.5 的大概率还是背景
        # Upper bound 0.8: 即使全图都很亮，高于 0.8 的要求太苛刻了
        final_thresh = torch.clamp(dynamic_thresh, min=0.5, max=0.8)
        
        return final_thresh.item()

    def _get_pseudo_targets_single(self, 
                                   labels, 
                                   label_weights, 
                                   bbox_weights, 
                                   anchors, 
                                   cost_volume, # [K, H, W]
                                   img_meta,
                                   gt_bboxes=None):
        
        if self.split_idx == 0 or cost_volume is None or self.cost_roi_extractor is None:
            return labels, label_weights, bbox_weights

        # ... (维度处理代码不变) ...
        if labels.dim() > 1: labels = labels.reshape(-1)
        if label_weights.dim() > 1: label_weights = label_weights.reshape(-1)
        if bbox_weights.dim() > 2: bbox_weights = bbox_weights.reshape(-1, 4)

        # 1. 找出背景样本
        bg_label = self.num_classes 
        neg_mask = (labels == bg_label) & (label_weights > 0)
        
        if not neg_mask.any():
            return labels, label_weights, bbox_weights

        # 2. 提取负样本特征
        neg_anchors = anchors[neg_mask]
        
        # 构造 RoI 并提取
        N_neg = neg_anchors.size(0)
        batch_inds = neg_anchors.new_zeros(N_neg, 1) 
        rois = torch.cat([batch_inds, neg_anchors], dim=1)
        
        input_feats = [cost_volume.unsqueeze(0)] 
        roi_feats = self.cost_roi_extractor(input_feats, rois)
        sampled_scores = roi_feats.view(N_neg, -1).sigmoid()

        # 分离新旧类分数
        if self.split_idx >= sampled_scores.shape[1]:
             score_prev, _ = sampled_scores.max(dim=1)
             score_curr = torch.zeros_like(score_prev)
        else:
             score_prev, _ = sampled_scores[:, :self.split_idx].max(dim=1)
             score_curr, _ = sampled_scores[:, self.split_idx:].max(dim=1)

        # =======================================================
        # 【核心修改 1】计算动态阈值
        # =======================================================
        # 利用真 GT 测算当前图片的 "体温"
        dynamic_abs_thresh = self._get_dynamic_threshold(cost_volume, gt_bboxes)
        
        # 如果整张图没有任何伪标签达到这个动态门槛，直接返回
        if score_prev.max() < dynamic_abs_thresh:
            return labels, label_weights, bbox_weights

        # 结合相对分位数 (双重保险)
        def safe_quantile(vals, q):
            if vals.numel() < 10: return vals.mean()
            return torch.quantile(vals, q)
            
        # 既要超过动态绝对阈值，又要排在前面
        thresh_quantile = safe_quantile(score_prev, 0.85) 
        final_thresh_prev = max(thresh_quantile, dynamic_abs_thresh)
        
        # 避嫌阈值 (如果像新类，就不要)
        THRESH_CURR_LOW = 0.3
        
        is_pseudo = (score_prev > final_thresh_prev) & (score_curr < THRESH_CURR_LOW)
        
        # =======================================================
        # 【核心修改 2】GT 碰撞检测 (Collision Detection)
        # =======================================================
        # 如果伪标签和真 GT 重叠，那是万万不能要的
        valid_candidates = None
        
        if is_pseudo.any():
            neg_indices = torch.nonzero(neg_mask, as_tuple=True)[0]
            candidate_indices = neg_indices[is_pseudo]
            
            if gt_bboxes is not None and gt_bboxes.size(0) > 0:
                # 计算 IoU
                from mmdet.core.bbox.iou_calculators import bbox_overlaps
                candidate_anchors = anchors[candidate_indices]
                ious = bbox_overlaps(candidate_anchors, gt_bboxes) # [N_pseudo, N_gt]
                max_ious, _ = ious.max(dim=1)
                
                # 只保留 IoU 极小的 (纯背景区域)
                keep_mask = max_ious < 0.05 
                valid_candidates = candidate_indices[keep_mask]
            else:
                valid_candidates = candidate_indices
        
        if valid_candidates is None or valid_candidates.numel() == 0:
            return labels, label_weights, bbox_weights

        # =======================================================
        # 【核心修改 3】数量配额 (Quota)
        # =======================================================
        # 即使动态阈值很完美，也要防止数量爆炸淹没真 GT
        MAX_QUOTA = 32 
        
        if valid_candidates.numel() > MAX_QUOTA:
            # 取分数最高的 Top-K
            # 因为 valid_candidates 是原 anchors 的索引，我们需要找到对应的 score
            # 这里的 score_prev 对应的是 neg_mask 里的顺序
            # 这是一个索引映射问题。为了简单且高效，我们这里直接随机采样
            # 因为经过了动态阈值筛选，剩下的质量应该都达标了
            
            # 使用随机采样来保持多样性，或者...
            # 如果想严谨取 Top-K，需要反查 score，比较繁琐，随机采样足够了
            perm = torch.randperm(valid_candidates.numel())[:MAX_QUOTA]
            final_indices = valid_candidates[perm]
        else:
            final_indices = valid_candidates
            
        # 5. 应用修改
        labels[final_indices] = 0        
        label_weights[final_indices] = 1.0 
        bbox_weights[final_indices, :] = 0.0 
        
        return labels, label_weights, bbox_weights

    def forward_train(self,
                      x,
                      img_metas,
                      gt_bboxes,
                      gt_labels=None,
                      gt_bboxes_ignore=None,
                      proposal_cfg=None,
                      seg_logits=None,
                      **kwargs):
        outs = self(x)
        if gt_labels is None:
            loss_inputs = outs + (gt_bboxes, None, img_metas)
        else:
            loss_inputs = outs + (gt_bboxes, gt_labels, img_metas)
        losses = self.loss(*loss_inputs, seg_logits=seg_logits, gt_bboxes_ignore=gt_bboxes_ignore)

        if proposal_cfg is None:
            return losses
        else:
            proposal_list = self.get_bboxes(
                *outs, img_metas=img_metas, cfg=proposal_cfg)
            return losses, proposal_list

    def loss(self,
             cls_scores,
             bbox_preds,
             gt_bboxes,
             gt_labels,
             img_metas,
             gt_bboxes_ignore=None,
             seg_logits=None):
        
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        device = cls_scores[0].device
        batch_size = len(img_metas)

        # 1. Anchor List: [Img][Level]
        anchor_list, valid_flag_list = self.get_anchors(
            featmap_sizes, img_metas, device=device)

        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1
        
        # 2. Targets: [Level] (Target Tensor)
        cls_reg_targets = self.get_targets(
            anchor_list,
            valid_flag_list,
            gt_bboxes,
            img_metas,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=gt_labels,
            label_channels=label_channels)
            
        if cls_reg_targets is None:
            return None
            
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets

        # =======================================================
        # 3. 核心修复: 兼容 [B, N] 和 [B*N] 两种 Target 格式
        # =======================================================
        if self.split_idx is not None and self.split_idx > 0 and seg_logits is not None:
            
            new_labels_list = []
            new_label_weights_list = []
            new_bbox_weights_list = []

            # 遍历每个 Level
            for lvl in range(len(labels_list)):
                
                # 获取该 Level 的原始 Target
                curr_labels = labels_list[lvl]
                curr_lws = label_weights_list[lvl]
                curr_bws = bbox_weights_list[lvl]
                
                # --- 维度适配逻辑 ---
                # 情况 A: Target 是 [Batch, N] (你的报错显示是这种情况)
                if curr_labels.dim() == 2 and curr_labels.size(0) == batch_size:
                    reshaped_labels = curr_labels
                    reshaped_lws = curr_lws
                    reshaped_bws = curr_bws # [B, N, 4]
                # 情况 B: Target 是 [Batch*N] (Flattened)
                else:
                    total_anchors = curr_labels.size(0)
                    assert total_anchors % batch_size == 0
                    num_per_img = total_anchors // batch_size
                    
                    reshaped_labels = curr_labels.reshape(batch_size, num_per_img)
                    reshaped_lws = curr_lws.reshape(batch_size, num_per_img)
                    reshaped_bws = curr_bws.reshape(batch_size, num_per_img, 4)

                processed_labels = []
                processed_lws = []
                processed_bws = []

                # 遍历 Batch
                for img_id in range(batch_size):
                    
                    # 取出单张图的 Target
                    # [N]
                    l_img = reshaped_labels[img_id] 
                    lw_img = reshaped_lws[img_id]
                    bw_img = reshaped_bws[img_id]
                    
                    # 取出单张图的 Anchor
                    # anchor_list 是 [Img][Level]，所以取 [img_id][lvl]
                    anchor_img = anchor_list[img_id][lvl] 
                    
                    # 校验维度 (防止 Anchor 和 Target 不对齐)
                    if l_img.size(0) != anchor_img.size(0):
                         # 如果对不上，尝试 flatten anchor
                         # 有时候 anchor 也是 list of tensor? 不，get_anchors 返回 tensor
                         raise RuntimeError(f"Img {img_id} Lvl {lvl} Mismatch: Label {l_img.size()} vs Anchor {anchor_img.size()}")

                    # 取出 Cost Volume
                    cost_vol_img = seg_logits[img_id]

                    # 挖掘
                    curr_gt_bboxes = gt_bboxes[img_id]
                    l, lw, bw = self._get_pseudo_targets_single(
                        l_img,
                        lw_img,
                        bw_img,
                        anchor_img,
                        cost_vol_img,
                        img_metas[img_id],
                        gt_bboxes=curr_gt_bboxes
                    )
                    
                    processed_labels.append(l)
                    processed_lws.append(lw)
                    processed_bws.append(bw)
                
                # 重新 Flatten 并保存 ([B*N])
                new_labels_list.append(torch.cat(processed_labels))
                new_label_weights_list.append(torch.cat(processed_lws))
                new_bbox_weights_list.append(torch.cat(processed_bws))

            # 替换数据
            labels_list = new_labels_list
            label_weights_list = new_label_weights_list
            bbox_weights_list = new_bbox_weights_list

            # 更新统计
            num_total_pos = sum([(l == 0).sum().item() for l in labels_list])
            num_total_neg = sum([(l == self.num_classes).sum().item() for l in labels_list])

        num_total_samples = (
            num_total_pos + num_total_neg if self.sampling else num_total_pos)

        num_valid_reg_samples = 0
        for bw in bbox_weights_list:
            # bw: [N, 4]
            if bw.numel() > 0:
                # 只要坐标有权重，就算有效样本
                num_valid_reg_samples += (bw.sum(dim=1) > 0).float().sum().item()
        
        # 防止除以 0
        avg_factor_bbox = max(num_valid_reg_samples, 1.0)

        # -------------------------------------------------------------
        # 步骤 C: 调用 multi_apply
        # -------------------------------------------------------------
        concat_anchor_list = []
        for lvl in range(len(labels_list)):
            anchors_at_lvl = [anchor_list[img_id][lvl] for img_id in range(batch_size)]
            concat_anchor_list.append(torch.cat(anchors_at_lvl))

        # 注意这里调用的是 loss_single_custom
        losses_cls, losses_bbox = multi_apply(
            self.loss_single_custom, 
            cls_scores,
            bbox_preds,
            concat_anchor_list,
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            num_total_samples=num_total_samples, # 传给分类
            avg_factor_bbox=avg_factor_bbox      # 传给回归
        )
            
        return dict(loss_rpn_cls=losses_cls, loss_rpn_bbox=losses_bbox)

    def loss_single_custom(self, cls_score, bbox_pred, anchors, labels, label_weights,
                           bbox_targets, bbox_weights, num_total_samples, avg_factor_bbox):
        """
        自定义的单层 Loss 计算。
        关键修改：loss_bbox 的 avg_factor 使用传入的 avg_factor_bbox (仅统计真 GT)，
        而不是 num_total_samples (真 GT + 伪标签)。
        """
        # 1. Classification Loss (保持源码逻辑)
        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)
        cls_score = cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels)
        
        # 分类依然使用总样本数 (含伪标签) 进行平均，这是对的
        loss_cls = self.loss_cls(
            cls_score, labels, label_weights, avg_factor=num_total_samples)
            
        # 2. Regression Loss (应用 avg_factor_bbox)
        bbox_targets = bbox_targets.reshape(-1, 4)
        bbox_weights = bbox_weights.reshape(-1, 4)
        bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
        
        if self.reg_decoded_bbox:
            anchors = anchors.reshape(-1, 4)
            bbox_pred = self.bbox_coder.decode(anchors, bbox_pred)
        
        # 【核心修改点】
        # 源码: avg_factor=num_total_samples
        # 修改: avg_factor=avg_factor_bbox
        # 作用: 排除掉伪标签的分母稀释效应，让真 GT 的梯度恢复正常
        loss_bbox = self.loss_bbox(
            bbox_pred,
            bbox_targets,
            bbox_weights,
            avg_factor=avg_factor_bbox) 

        return loss_cls, loss_bbox
