import torch
import torch.nn as nn
import math
from typing import Optional, Tuple
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding

import os
from mmdet.models.builder import BACKBONES
from mmcv.runner import BaseModule
from torch.nn import functional as F
from mmcv.utils.logging import print_log
from mmcv.cnn import build_norm_layer
import transformers
import math


class RotaryEmbedding2D(nn.Module):
    def __init__(self, dim, base=10000):
        """
        Args:
            dim (int): 需要被旋转的向量维度。在Attention中，这通常是 head_dim。
                       `dim` 必须是4的倍数。
        """
        super().__init__()
        if dim % 4 != 0:
            raise ValueError(f"RoPE的维度 `dim` 必须是4的倍数, 但收到了 {dim}")
            
        self.dim = dim
        self.base = base
        
        # 计算旋转频率，并注册为buffer。
        # self.inv_freq 的形状为 [dim / 4]
        self.register_buffer(
            "inv_freq",
            1.0 / (self.base ** (torch.arange(0, self.dim // 2, 2).float() / (self.dim // 2)))
        )

    def forward(self, q, k):
        """
        Args:
            q, k (torch.Tensor): 形状为 [B, H, N, D] 的张量,
                                  B=batch_size, H=num_heads, N=num_patches, D=head_dim.
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: 旋转后的 q, k 
        """
        B, H, N, D = q.shape
        
        # 1. 动态计算当前输入的网格尺寸 (H_grid, W_grid)
        grid_h = int(math.sqrt(N))
        grid_w = N // grid_h
        if grid_h * grid_w != N:
             # 对于非正方形的图像块网格，可以采用更复杂的逻辑，这里简化处理
             raise ValueError(f"序列长度 {N} 无法分解为整数网格。")
        
        # 2. 创建位置索引网格 [H_grid, W_grid]
        pos_h, pos_w = torch.meshgrid(
            torch.arange(grid_h, device=q.device, dtype=self.inv_freq.dtype),
            torch.arange(grid_w, device=q.device, dtype=self.inv_freq.dtype),
            indexing='ij'
        )

        # 3. 计算h和w维度的旋转频率
        #    使用einsum进行高效的外积计算
        #    freqs_h/w 的形状: [N, D/4]
        freqs_h = torch.einsum("..., i -> ...i", pos_h.flatten(), self.inv_freq)
        freqs_w = torch.einsum("..., i -> ...i", pos_w.flatten(), self.inv_freq)
        
        # 4. 拼接h和w的频率，并计算sin/cos值
        #    freqs 的形状: [N, D/2]
        #    emb 的形状:   [1, 1, N, D/2] -> [1, 1, N, D]
        freqs = torch.cat((freqs_h, freqs_w), dim=-1)
        emb = torch.cat((freqs, freqs), dim=-1) # [N, D]
        emb_cos = emb.cos()[None, None, :, :]  # 扩展维度以进行广播
        emb_sin = emb.sin()[None, None, :, :]

        # 5. 定义旋转函数并应用
        def rotate_half(x):
            """将最后一个维度减半，交换前后部分并给前部分符号取反"""
            x1 = x[..., : D // 2]
            x2 = x[..., D // 2 :]
            return torch.cat((-x2, x1), dim=-1)

        q_rotated = (q * emb_cos) + (rotate_half(q) * emb_sin)
        k_rotated = (k * emb_cos) + (rotate_half(k) * emb_sin)

        return q_rotated, k_rotated


class Rope2D(nn.Module):
    def __init__(self, dim, base=10000):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("维度必须是偶数，以便为H和W维度平分")
        self.dim_half = dim // 2
        self.rope_h = LlamaRotaryEmbedding(self.dim_half, base=base)
        self.rope_w = LlamaRotaryEmbedding(self.dim_half, base=base)

    def forward(self, x: torch.Tensor):
        shape = x.shape
        if x.ndim != 4:
            raise ValueError("Rope2D 输入张量必须是4D [B, H, N, D]")
        B, n_heads, N, D = x.shape
        if N == 0: return x # 如果序列为空，直接返回
        grid_h = int(math.sqrt(N))
        grid_w = N // grid_h
        if grid_h * grid_w != N:
             raise ValueError(f"序列长度 {N} 无法分解为整数网格。")
        pos_h_ids = torch.arange(grid_h, device=x.device).repeat_interleave(grid_w)
        pos_w_ids = torch.arange(grid_w, device=x.device).repeat(grid_h)
        x_h = x[..., :self.dim_half]
        x_w = x[..., self.dim_half:]
        cos_h, sin_h = self.rope_h(x_h, position_ids=pos_h_ids)
        cos_w, sin_w = self.rope_w(x_w, position_ids=pos_w_ids)
        def apply_rotary_pos_emb(q, cos, sin):
            def rotate_half(x):
                x1 = x[..., : x.shape[-1] // 2]
                x2 = x[..., x.shape[-1] // 2 :]
                return torch.cat((-x2, x1), dim=-1)
            return (q * cos) + (rotate_half(q) * sin)
        x_h_rotated = apply_rotary_pos_emb(x_h, cos_h, sin_h)
        x_w_rotated = apply_rotary_pos_emb(x_w, cos_w, sin_w)
        x_rotated = torch.cat((x_h_rotated, x_w_rotated), dim=-1)
        return x_rotated.reshape(shape)

class RoPECLIPAttention(nn.Module):
    """
    一个从CLIPAttention改造而来的、使用2D RoPE的Multi-headed attention模块。
    **此版本能够正确处理包含 [CLS] token 的输入序列。**
    """
    def __init__(self, embed_dim: int, num_heads: int, attention_dropout: float = 0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(f"embed_dim ({self.embed_dim}) 必须能被 num_heads ({self.num_heads}) 整除。")
        self.rope = RotaryEmbedding2D(dim=self.head_dim)
        self.dropout = attention_dropout
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        causal_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Input shape: Batch x SequenceLength (1 + num_patches) x Channel
        """
        bsz, tgt_len, _ = hidden_states.shape

        # 1. 线性投影 Q, K, V
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # 2. 将Q,K,V切分为多头
        # [B, N, C] -> [B, N, H, D] -> [B, H, N, D]
        query_states = query_states.view(bsz, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        #    序列的第一个token是[CLS]
        cls_query, patch_query = query_states[:, :, :1, :], query_states[:, :, 1:, :]
        cls_key, patch_key = key_states[:, :, :1, :], key_states[:, :, 1:, :]

        # 只旋转图像块部分
        # patch_query_rotated = self.rope(patch_query)
        # patch_key_rotated = self.rope(patch_key)
        patch_query_rotated, patch_key_rotated = self.rope(patch_query, patch_key)
        
        # 重新拼接
        query_states = torch.cat((cls_query, patch_query_rotated), dim=2)
        key_states = torch.cat((cls_key, patch_key_rotated), dim=2)

        # 4. 计算注意力分数
        scale = self.head_dim ** -0.5
        attn_weights = torch.matmul(query_states, key_states.transpose(-1, -2)) * scale

        # 应用attention masks
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        if causal_attention_mask is not None:
             attn_weights = attn_weights + causal_attention_mask

        # 5. Softmax和Dropout
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        # 6. 计算输出
        attn_output = torch.matmul(attn_probs, value_states)

        # 7. 合并多头
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)

        # 8. 最终的线性投影
        attn_output = self.out_proj(attn_output)
        
        attn_weights_reshaped = attn_weights if output_attentions else None

        return attn_output, attn_weights_reshaped


from rope import RoPECLIPAttention
@BACKBONES.register_module()
class FGCLIPViT_ROPE(BaseModule):
    """
    在Attention中加入ROPE的FGCLIP
    """
    def __init__(self, model_name, pretrained, out_indices=[3, 5, 7, 11], norm_cfg=None):
        super().__init__()
        self.vit_layers = out_indices
        self.model_name = model_name
        self.pretrained = pretrained

        if self.pretrained and os.path.exists(self.pretrained):
            print_log(f"[FG-CLIP] Loading model '{self.pretrained}' using transformers.AutoModel")
            clip_model = transformers.AutoModel.from_pretrained(self.pretrained, local_files_only=True)
        else:
            print_log(f"[FG-CLIP] Create model without pretrained (or path missing): {self.pretrained}")
            config_path = os.path.join("/hy-tmp/liuzihan/CLIPSelf/F-ViT/checkpoints/", self.model_name)
            model_config = transformers.AutoConfig.from_pretrained(config_path, local_files_only=True)
            clip_model = transformers.AutoModel.from_config(model_config)

        # HF CLIP's vision model is in the 'vision_model' attribute
        self.visual = clip_model.vision_model
        # 为FG-CLIP保留 visual_projection 属性
        self.visual_projection = clip_model.visual_projection
        # add rope
        self.inject_rope_into_vit()

        # 获取维度信息
        self.width = width = self.visual.config.hidden_size
        self.patch_size = self.visual.config.patch_size

        print_log(f"FG-CLIP initialized: width={self.width}, patch_size={self.patch_size}, "
                  f"blocks={len(self.visual.encoder.layers)}")

        # 上采样和下采样层 (与其它类保持一致)
        self.interpolate1 = nn.Sequential(
            nn.ConvTranspose2d(width, width, kernel_size=2, stride=2),
            build_norm_layer(norm_cfg, width)[1] if norm_cfg else nn.Identity(),
            nn.GELU(),
            nn.ConvTranspose2d(width, width, kernel_size=2, stride=2),
        )
        self.interpolate2 = nn.Sequential(
            nn.ConvTranspose2d(width, width, kernel_size=2, stride=2),
        )
        self.interpolate3 = nn.Identity()
        self.interpolate4 = nn.MaxPool2d(kernel_size=2, stride=2)

    def init_weights(self):
        """从预训练模型加载权重并冻结。"""
        if self.pretrained and os.path.exists(self.pretrained):
            print_log(f"[FG-CLIP init_weights] Loading from: {self.pretrained}")
            clip_model = transformers.AutoModel.from_pretrained(self.pretrained, local_files_only=True)
        else:
            print_log(f"[FG-CLIP init_weights]: No pretrained weights found at {self.pretrained}")
            config_path = os.path.join("/hy-tmp/liuzihan/CLIPSelf/F-ViT/checkpoints/", self.model_name)
            model_config = transformers.AutoConfig.from_pretrained(config_path, local_files_only=True)
            clip_model = transformers.AutoModel.from_config(model_config)

        # 加载视觉部分的权重
        load_result = self.visual.load_state_dict(clip_model.vision_model.state_dict(), strict=False)
        print_log(f"[FG-CLIP init_weights] Load state_dict result: {load_result}")

        # 加载 visual_projection 权重
        vp_load_result = self.visual_projection.load_state_dict(clip_model.visual_projection.state_dict(), strict=True)
        print_log(f"[FG-CLIP init_weights] Load visual_projection result: {vp_load_result}")
        
        # load attn weights
        self.load_attn(clip_model=clip_model)

        # 冻结视觉编码器参数
        for param in self.visual.parameters():
            param.requires_grad = False
        print_log("FG-CLIP visual encoder frozen")

        # 冻结 visual_projection 参数
        for param in self.visual_projection.parameters():
            param.requires_grad = False
        print_log("FG-CLIP visual projection frozen")
    
    def load_attn(self, clip_model):
        for i, (self_encoder_layer, clip_encoder_layer) \
            in enumerate(zip(self.visual.encoder.layers, clip_model.vision_model.encoder.layers)):

            orig_attn = clip_encoder_layer.self_attn
            rope_attn = self_encoder_layer.self_attn

            k_load_result = rope_attn.k_proj.load_state_dict(orig_attn.k_proj.state_dict(), strict=True)
            v_load_result = rope_attn.v_proj.load_state_dict(orig_attn.v_proj.state_dict(), strict=True)
            q_load_result = rope_attn.q_proj.load_state_dict(orig_attn.q_proj.state_dict(), strict=True)
            out_load_result = rope_attn.out_proj.load_state_dict(orig_attn.out_proj.state_dict(), strict=True)
            print_log(f"[FG-CLIP init_weights] Load attn{i} result: \
                k: {k_load_result} v: {v_load_result} q: {q_load_result}, out: {out_load_result}")


    def inject_rope_into_vit(self):
        """
        用RoPEAttention替换其标准的Attention模块，并移除旧的位置编码。
        """
        print("开始将RoPE注入ViT模型...")
        model = self.visual
        # 遍历模型的所有Block
        for i, clip_encoder_layer in enumerate(model.encoder.layers):
            # 1. 创建一个新的RoPEAttention实例，参数与旧的Attention层完全一致
            orig_attn = clip_encoder_layer.self_attn
            rope_attn = RoPECLIPAttention(
                embed_dim=orig_attn.embed_dim, 
                num_heads=orig_attn.num_heads, 
                attention_dropout=orig_attn.dropout
            )
            # 3. 用新的RoPEAttention模块替换旧模块
            clip_encoder_layer.self_attn = rope_attn
        print(f"Attention层已替换为RoPEAttention。")

        # 4. 移除模型中不再需要的可学习绝对位置编码
        model.embeddings.position_embedding = None
        print("  - 旧的 `positional_embedding` 已被移除。")

        # 5. 修改模型的forward_features方法，以跳过位置编码的相加操作
        # def forward_features_with_rope(self, x):
        #     x = self.patch_embed(x)
        #     # 将 [CLS] token 拼接到图像块序列
        #     x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        #     # **原版ViT在这里会加上 self.positional_embedding，我们直接跳过**
        #     x = self.pos_drop(x)
        #     x = self.blocks(x)
        #     x = self.norm(x)
        #     return x
        
        # 使用"方法绑定"动态地替换实例的forward方法
        # model.forward_features = types.MethodType(forward_features_with_rope, model)
        # print("  - 模型的 `forward_features` 方法已更新。")
        print("模型改造完成！现在它使用2D RoPE并支持可变分辨率。")

    def train(self, mode=True):
        print(f"Set train mode for FG-CLIP: {mode}", flush=True)
        self.training = mode
        self.visual.train(False)  # 保持visual在eval模式
        self.visual_projection.train(False)
        self.interpolate1.train(mode)
        self.interpolate2.train(mode)
        self.interpolate3.train(mode)
        self.interpolate4.train(mode)
        return self

    def forward(self, x):
        visual = self.visual
        bs, _, H, W = x.shape
        h = H // self.patch_size
        w = W // self.patch_size

        with torch.no_grad():
            # hidden_states = visual.embeddings(x) 

            ## skip absolute pos embedding 
            # hidden_states = visual.embeddings.patch_embedding(x)
            hidden_states = self.embedding(visual, x)

            # 5. 循环遍历 Transformer Blocks
            transformer_blocks = visual.encoder.layers
            outs = []
            for i, blk in enumerate(transformer_blocks[:-1]):
                # HF Transformer Block 的输出是一个元组
                layer_outputs = blk(hidden_states,                     
                                attention_mask=None,
                                causal_attention_mask=None)
                hidden_states = layer_outputs[0]
                
                if i in self.vit_layers:
                    outs.append(self._expand_x(hidden_states, h, w))

            # 6. 对最后一个 block 使用 _forward_without_attn
            hidden_states = self.forward_without_attn(hidden_states)
            
            if (len(transformer_blocks) - 1) in self.vit_layers:
                outs.append(self._expand_x(hidden_states, h, w))
            
            # 7. 在推理模式下生成最终的密集特征图 (feature_map)
            if not self.training:
                # 7.1 移除 [CLS] token，只保留 patch tokens
                patch_tokens = hidden_states[:, 1:]
                
                # 7.2 应用最终的 LayerNorm
                patch_tokens = visual.post_layernorm(patch_tokens)
                
                # 7.4 L2 归一化
                patch_tokens = F.normalize(patch_tokens, dim=-1)
                
                # 7.5 重塑为 (B, C, H, W) 格式的特征图
                feature_map = patch_tokens.view(bs, h, w, -1).permute(0, 3, 1, 2)
            else:
                feature_map = None

        # 8. 断言并对中间层特征图进行插值
        assert len(outs) == len(self.vit_layers), f"Expected {len(self.vit_layers)} output layers, got {len(outs)}"
        
        for idx, out in enumerate(outs):
            interpolate = getattr(self, f"interpolate{idx + 1}")
            outs[idx] = interpolate(out.detach())

        # 9. 添加 feature_map 并返回
        outs.append(feature_map)

        return tuple(outs)

    def _expand_x(self, x, h, w):
        """将序列格式转换为空间格式，并移除 [CLS] token。"""
        # 标准CLIP模型包含 [CLS] token，在转换为特征图时需要移除
        x = x[:, 1:].permute(0, 2, 1).contiguous()
        x = x.view(-1, self.width, h, w)
        return x

    def embedding(self, visual, x):
        bs, _, _, _ = x.shape
        
        target_dtype = visual.embeddings.patch_embedding.weight.dtype
        patch_embeds = visual.embeddings.patch_embedding(x.to(dtype=target_dtype))  # shape = [*, width, grid, grid]
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)

        class_embeds = visual.embeddings.class_embedding.expand(bs, 1, -1)
        hidden_states = torch.cat([class_embeds, patch_embeds], dim=1)
        hidden_states = visual.pre_layrnorm(hidden_states)

        return hidden_states
    
    def forward_without_attn(self, x):
        # get last layer 
        residual = x
        x = self.visual.encoder.layers[-1].layer_norm1(x)

        x = F.linear(input=x, weight=self.visual.encoder.layers[-1].self_attn.v_proj.weight, bias=self.visual.encoder.layers[-1].self_attn.v_proj.bias)
        x = self.visual.encoder.layers[-1].self_attn.out_proj(x)
        x = residual+x


        x = self.visual.encoder.layers[-1].layer_norm2(x)
        x = self.visual.encoder.layers[-1].mlp(x)
        x = residual + x

        return x


# if __name__ == "__main__":
#     attn = RoPECLIPAttention(embed_dim=768, num_heads=16, attention_dropout=0.)
#     x = torch.rand([1, 197, 768])

#     print(x.shape)
#     with torch.no_grad():
#         out = attn.forward(x)
    
#     print(out)