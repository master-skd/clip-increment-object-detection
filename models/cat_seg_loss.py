import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models.builder import LOSSES  # 引入注册器

@LOSSES.register_module()
class SegOrthogonalLoss(nn.Module):
    def __init__(self, loss_weight=1.0):
        super(SegOrthogonalLoss, self).__init__()
        self.loss_weight = loss_weight

    def forward(self, logits):
        """
        Args:
            logits: (B, T, H, W) - 来自 CatSegEvaCLIPViT 的输出
        """
        # 1. 激活函数处理 (可选，Sigmoid 将 logits 映射到 0-1，使正交更有意义)
        probs = torch.sigmoid(logits) 
        
        # 2. 展平空间维度: (B, T, H, W) -> (B, T, H*W)
        B, T, H, W = probs.shape
        probs_flat = probs.view(B, T, -1)
        
        # 3. 归一化 (Cosine Similarity 风格) - 可选，防止数值过大
        probs_norm = F.normalize(probs_flat, p=2, dim=2)
        
        # 4. 计算 Gram 矩阵: (B, T, HW) @ (B, HW, T) -> (B, T, T)
        gram_matrix = torch.bmm(probs_norm, probs_norm.transpose(1, 2))
        
        # 5. 构建对角掩码 (只惩罚非对角线元素)
        eye = torch.eye(T, device=logits.device).unsqueeze(0).expand(B, -1, -1)
        
        # 6. 计算 Loss: 最小化非对角线元素的平方和
        # (gram_matrix * (1 - eye)) 提取了非对角元素
        ortho_loss = ((gram_matrix * (1 - eye)) ** 2).sum() / (B * T * (T - 1) + 1e-6)
        
        return self.loss_weight * ortho_loss