import open_clip
from functools import partial
import os
import torch
from torch import nn
from mmdet.models.builder import BACKBONES
from mmcv.runner import BaseModule
from torch.nn import functional as F
from mmcv.utils.logging import print_log
from mmcv.cnn import build_norm_layer
import transformers
import math

@BACKBONES.register_module()
class EvaCLIPViT(BaseModule):
    def __init__(self, model_name, pretrained, out_indices=[3, 5, 7, 11], norm_cfg=None):
        super().__init__()
        self.vit_layers = out_indices
        self.model_name = model_name
        self.pretrained = pretrained  # the pretrained .pt file
        clip_model = open_clip.create_model(model_name,
                                            pretrained="eva",
                                            cache_dir=pretrained)
        self.embed_dim = embed_dim = clip_model.embed_dim  # output dim
        self.width = width = clip_model.visual.embed_dim
        self.patch_size = patch_size = clip_model.visual.patch_embed.patch_size[0]
        # 上采样4倍
        self.interpolate1 = nn.Sequential(
            nn.ConvTranspose2d(width, width, kernel_size=2, stride=2),
            build_norm_layer(norm_cfg, width)[1] if norm_cfg else nn.Identity(),
            nn.GELU(),
            nn.ConvTranspose2d(width, width, kernel_size=2, stride=2),
        )
        # 上采样2倍
        self.interpolate2 = nn.Sequential(
            nn.ConvTranspose2d(width, width, kernel_size=2, stride=2),
        )
        self.interpolate3 = nn.Identity()
        # 下采样2倍
        self.interpolate4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.visual = clip_model.visual
        # self.interpolate3 = nn.Conv2d(width, width, kernel_size=3, stride=1, padding=1)
        # self.interpolate4 = nn.Conv2d(width, width, kernel_size=3, stride=2, padding=1)

    def init_weights(self):
        clip_model = open_clip.create_model(self.model_name,
                                            pretrained="eva",
                                            cache_dir=self.pretrained,
                                            device="cpu")
        print_log(self.visual.load_state_dict(clip_model.visual.state_dict(), strict=True))
        for param in self.visual.parameters():  # only freeze the CLIP model
            param.requires_grad = False

    def train(self, mode=True):
        print(f"Set train mode for EVA: {mode}", flush=True)
        self.training = mode
        self.visual.train(False)
        self.interpolate1.train(mode)
        self.interpolate2.train(mode)
        self.interpolate3.train(mode)
        self.interpolate4.train(mode)

        return self

    def forward(self, x):
        visual = self.visual
        bs, _, h, w = x.shape
        h = h // visual.patch_embed.patch_size[0]
        w = w // visual.patch_embed.patch_size[1]

        with torch.no_grad():
            x = visual.patch_embed(x)
            batch_size, seq_len, _ = x.size()

            cls_tokens = visual.cls_token.expand(batch_size, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
            x = torch.cat((cls_tokens, x), dim=1)
            if visual.pos_embed is not None:
                x = x + visual.rescale_positional_embedding(out_size=(h, w))
            x = visual.pos_drop(x)

            # a patch_dropout of 0. would mean it is disabled and this function would do nothing but return what was passed in
            if os.getenv('RoPE') == '1':
                if visual.training and not isinstance(visual.patch_dropout, nn.Identity):
                    x, patch_indices_keep = visual.patch_dropout(x)
                    visual.rope.forward = partial(visual.rope.forward, patch_indices_keep=patch_indices_keep)
                else:
                    visual.rope.forward = partial(visual.rope.forward, patch_indices_keep=None)
                    x = visual.patch_dropout(x)
            else:
                x = visual.patch_dropout(x)

            rel_pos_bias = visual.rel_pos_bias() if visual.rel_pos_bias is not None else None

            outs = []
            for i, blk in enumerate(visual.blocks[:-1]):
                x = blk(x, rel_pos_bias=rel_pos_bias)
                if i in self.vit_layers:
                    outs.append(self._expand_x(x, h, w))
            x = visual.blocks[-1].forward_without_attn(x)
            if (len(visual.blocks) - 1) in self.vit_layers:
                outs.append(self._expand_x(x, h, w))
            if not self.training:
                x = x[:, 1:]
                x = visual.norm(x)
                x = visual.head(x)
                assert visual.fc_norm is None
                x = F.normalize(x, dim=-1)  # normalize along last dimension
                feature_map = x.view(bs, h, w, -1).permute(0, 3, 1, 2)
            else:
                feature_map = None

        assert len(outs) == 4
        for idx, out in enumerate(outs):
            interpolate = getattr(self, f"interpolate{idx + 1}")
            outs[idx] = interpolate(out.detach())

        outs.append(feature_map)

        return tuple(outs)

    def _expand_x(self, x, h, w):
        # x: bs q c
        x = x[:, 1:].permute(0, 2, 1).contiguous()
        x = x.view(-1, self.width, h, w)

        return x


@BACKBONES.register_module()
class SigLIPViT(BaseModule):
    """SigLIP Vision Transformer for object detection
    
    适配 SigLIP2 模型 (TimmModel结构)，支持从NPZ文件加载
    
    关键差异：
    - SigLIP 使用 TimmModel 包装 VisionTransformer
    - 没有 cls_token，使用 attn_pool
    - pos_embed 固定，不需要 rescale
    - blocks 简单forward，不需要 rel_pos_bias
    """
    def __init__(self, model_name, pretrained, out_indices=[3, 5, 7, 11], norm_cfg=None):
        super().__init__()
        self.vit_layers = out_indices
        self.model_name = model_name
        self.pretrained = pretrained
        
        if self.pretrained and os.path.exists(self.pretrained):
            print_log(f"[SigLIP2] Loading weights from local file: {self.pretrained}")
            clip_model, _, _ = open_clip.create_model_and_transforms(
                model_name, pretrained=self.pretrained
            )
        else:
            print_log(f"[SigLIP2] Create model without pretrained (or path missing): {self.pretrained}")
            clip_model, _, _ = open_clip.create_model_and_transforms(
                model_name, pretrained=False
            )
        
        # SigLIP 使用 TimmModel，实际ViT在trunk中
        self.visual = clip_model.visual
        
        # 获取维度信息
        self.width = width = self.visual.trunk.embed_dim
        self.embed_dim = getattr(clip_model, 'embed_dim', width)
        self.patch_size = self.visual.trunk.patch_embed.patch_size[0]
        
        print_log(f"SigLIP initialized: width={self.width}, patch_size={self.patch_size}, "
                 f"blocks={len(self.visual.trunk.blocks)}")
        
        # 上采样4倍
        self.interpolate1 = nn.Sequential(
            nn.ConvTranspose2d(width, width, kernel_size=2, stride=2),
            build_norm_layer(norm_cfg, width)[1] if norm_cfg else nn.Identity(),
            nn.GELU(),
            nn.ConvTranspose2d(width, width, kernel_size=2, stride=2),
        )
        # 上采样2倍
        self.interpolate2 = nn.Sequential(
            nn.ConvTranspose2d(width, width, kernel_size=2, stride=2),
        )
        self.interpolate3 = nn.Identity()
        # 下采样2倍
        self.interpolate4 = nn.MaxPool2d(kernel_size=2, stride=2)
    
    def init_weights(self):
        """重新加载预训练权重（参考 EvaCLIPViT）"""
        # 重新创建模型并加载预训练权重到 CPU
        if self.pretrained and os.path.exists(self.pretrained):
            print_log(f"[SigLIP init_weights] Loading from: {self.pretrained}")
            clip_model, _, _ = open_clip.create_model_and_transforms(
                self.model_name, 
                pretrained=self.pretrained,
                device="cpu"
            )
        else:
            print_log(f"[SigLIP init_weights] Warning: No pretrained weights found at {self.pretrained}")
            clip_model, _, _ = open_clip.create_model_and_transforms(
                self.model_name, 
                pretrained=False,
                device="cpu"
            )
        
        # 加载权重到 self.visual（严格模式）
        load_result = self.visual.load_state_dict(clip_model.visual.state_dict(), strict=True)
        print_log(f"[SigLIP init_weights] Load state_dict result: {load_result}")
        
        # 冻结视觉编码器参数
        for param in self.visual.parameters():
            param.requires_grad = False
        print_log("SigLIP visual encoder frozen")

    def train(self, mode=True):
        print(f"Set train mode for SigLIP: {mode}", flush=True)
        self.training = mode
        self.visual.train(False)  # 保持visual在eval模式
        self.interpolate1.train(mode)
        self.interpolate2.train(mode)
        self.interpolate3.train(mode)
        self.interpolate4.train(mode)
        return self

    def forward(self, x):
        """
        用 built-in 前向 + (可选)get_intermediate_layers / forward hooks
        拿到 out_indices 指定层的 token，然后还原 feature maps。
        """
        trunk = self.visual.trunk
        bs, _, H, W = x.shape
        h = H // self.patch_size
        w = W // self.patch_size

        outs = []
        feature_map = None
        #用 forward hook 捕获
        feats = {}
        handles = []

        def make_hook(idx):
            def _hook(_m, _in, out):
                # out: (B, N, C)
                feats[idx] = out
            return _hook

        # 在需要的层注册 hook
        for i in self.vit_layers:
            handles.append(trunk.blocks[i].register_forward_hook(make_hook(i)))

        with torch.no_grad():
            # 直接跑 trunk 的标准前向（会自动做 patch_embed/pos_drop 等）
            _ = trunk(x)
            # 清理 hook（非常重要，避免内存泄露）
            for hnd in handles:
                hnd.remove()

        # 按 self.vit_layers 顺序取出并还原
        for i in self.vit_layers:
            tk = feats[i]
            outs.append(self._expand_x(tk, h, w))

        # eval 的全局向量
        if not self.training:
            with torch.no_grad():
                last_idx = self.vit_layers[-1]
                last_tokens = feats[last_idx]
                z = trunk.norm(last_tokens)
                if getattr(trunk, "attn_pool", None) is not None:
                    z = trunk.attn_pool(z)
                else:
                    z = z.mean(1)
                if getattr(self.visual, "head", None) is not None:
                    z = self.visual.head(z)
                z = F.normalize(z, dim=-1)
                feature_map = z.unsqueeze(-1).unsqueeze(-1).expand(bs, -1, h, w)

        # 检查数量
        assert len(outs) == len(self.vit_layers) == 4, f"Expected 4 output layers, got {len(outs)}"

        for idx, out in enumerate(outs):
            interpolate = getattr(self, f"interpolate{idx + 1}")
            outs[idx] = interpolate(out.detach())

        outs.append(feature_map)

        return tuple(outs)

    def _expand_x(self, x, h, w):
        """将序列格式转换为空间格式
        Args:
            x: [bs, num_patches, dim]
            h, w: 空间维度
        Returns:
            [bs, dim, h, w]
        """
        # SigLIP 没有 cls_token，直接reshape
        bs, num_patches, dim = x.shape
        x = x.permute(0, 2, 1).contiguous()  # [bs, dim, num_patches]
        x = x.view(bs, dim, h, w)  # [bs, dim, h, w]
        return x

@BACKBONES.register_module()
class FGCLIPViT_interpolate(BaseModule):
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
        load_result = self.visual.load_state_dict(clip_model.vision_model.state_dict(), strict=True)
        print_log(f"[FG-CLIP init_weights] Load state_dict result: {load_result}")

        # 加载 visual_projection 权重
        vp_load_result = self.visual_projection.load_state_dict(clip_model.visual_projection.state_dict(), strict=True)
        print_log(f"[FG-CLIP init_weights] Load visual_projection result: {vp_load_result}")

        # 加载 position_embedding 权重
        # self.interpolate_pos_embedding_module(new_input_size=512)

        # 冻结视觉编码器参数
        for param in self.visual.parameters():
            param.requires_grad = False
        print_log("FG-CLIP visual encoder frozen")

        # 冻结 visual_projection 参数
        for param in self.visual_projection.parameters():
            param.requires_grad = False
        print_log("FG-CLIP visual projection frozen")

    
    def interpolate_pos_embedding_module(
        self,
        new_input_size: int
    ):
        """
        对ViT模型中作为 nn.Embedding 模块的位置编码进行2D插值，
        并用一个新的、尺寸正确的 nn.Embedding 模块替换旧的。

        Args:
            model (VisionTransformer): 预训练的ViT模型 (例如 model.visual)。
            new_input_size (int): 新的输入图像正方形边长 (例如 896)。
            
        Returns:
            VisionTransformer: 修改了位置编码模块的模型。
        """
        model = self.visual
        # 1. 定位到旧的Embedding模块和其权重
        if not isinstance(model.embeddings.position_embedding, nn.Embedding):
            raise TypeError(
                f"预期的 model.embeddings.position_embedding 是 nn.Embedding 类型, "
                f"但收到了 {type(model.pos_embed)}."
            )
        
        original_embedding_module = model.embeddings.position_embedding
        # nn.Embedding 的权重存储在 .weight 属性中，这是一个 nn.Parameter
        original_embedding_weight = original_embedding_module.weight
        
        # 权重形状: [num_embeddings, embed_dim]
        original_num_embeddings = original_embedding_module.num_embeddings
        embed_dim = original_embedding_module.embedding_dim
        
        num_patches = original_num_embeddings - 1

        # 2. 计算原始网格尺寸
        original_grid_size = int(math.sqrt(num_patches))
        if original_grid_size * original_grid_size != num_patches:
            raise ValueError(f"原始块数量 {num_patches} 不是一个完全平方数。")

        # 3. 分离 [CLS] token 的嵌入和图像块的嵌入
        cls_token_embed = original_embedding_weight[:1, :]
        patch_pos_embed = original_embedding_weight[1:, :]

        # 4. 重塑图像块嵌入为2D网格形状以便插值
        # [196, 768] -> [14, 14, 768]
        patch_pos_embed = patch_pos_embed.reshape(original_grid_size, original_grid_size, embed_dim)
        # [14, 14, 768] -> [768, 14, 14] -> [1, 768, 14, 14]
        patch_pos_embed = patch_pos_embed.permute(2, 0, 1).unsqueeze(0)

        # 5. 计算新的网格尺寸
        patch_size = self.patch_size
        new_grid_size = new_input_size // patch_size
        
        print(f"位置编码插值: patches: {original_grid_size}x{original_grid_size} -> {new_grid_size}x{new_grid_size}")

        # 6. 执行双三次插值
        interpolated_patch_pos_embed = F.interpolate(
            patch_pos_embed,
            size=(new_grid_size, new_grid_size),
            mode='bicubic',
            align_corners=False
        )

        # 7. 将插值后的结果转换回序列格式
        # [1, 768, 56, 56] -> [768, 56, 56] -> [56, 56, 768] -> [3136, 768]
        interpolated_patch_pos_embed = interpolated_patch_pos_embed.squeeze(0).permute(1, 2, 0)
        interpolated_patch_pos_embed = interpolated_patch_pos_embed.flatten(0, 1)

        # 8. 重新拼接 [CLS] token 嵌入，形成新的完整权重
        new_pos_embed_weight = torch.cat((cls_token_embed, interpolated_patch_pos_embed), dim=0)
        
        # 9. **核心改造**: 创建一个新的 nn.Embedding 模块
        new_num_embeddings = new_pos_embed_weight.shape[0]
        new_pos_embed_module = nn.Embedding(
            num_embeddings=new_num_embeddings,
            embedding_dim=embed_dim
        )
        
        # 10. 将插值后的权重加载到新模块中
        #     使用 .data 直接赋值，或者使用 .load_state_dict()
        with torch.no_grad():
            new_pos_embed_module.weight.copy_(new_pos_embed_weight)
            
        # 11. 用新的模块替换模型中的旧模块
        model.embeddings.position_embedding = new_pos_embed_module
        print(f"成功替换新的 nn.Embedding(num_embeddings={new_num_embeddings}, ...)")

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
        """
        """
        visual = self.visual
        bs, _, H, W = x.shape
        h = H // self.patch_size
        w = W // self.patch_size

        with torch.no_grad():
            hidden_states = visual.embeddings(x, interpolate_pos_encoding=True)
            hidden_states = visual.pre_layrnorm(hidden_states)

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
    
    def forward_without_attn(self, x):
        # get last layer 
        residual = x
        x = self.visual.encoder.layers[-1].layer_norm1(x)

        x = F.linear(input=x, weight=self.visual.encoder.layers[-1].self_attn.v_proj.weight, bias=self.visual.encoder.layers[-1].self_attn.v_proj.bias)
        x = self.visual.encoder.layers[-1].self_attn.out_proj(x)
        x = residual+x

        residual = x
        x = self.visual.encoder.layers[-1].layer_norm2(x)
        x = self.visual.encoder.layers[-1].mlp(x)
        x = residual + x

        return x


@BACKBONES.register_module()
class FGCLIPViT(BaseModule):
    """FG-CLIP Vision Transformer for object detection

    关键差异：
    - 使用 'transformers.AutoModel' 加载模型.
    - 视觉模型在 'vision_model' 属性中.
    - ViT blocks 在 'vision_model.encoder.layers' 中.
    - 存在 [CLS] token, 处理方式需参考 EvaCLIPViT.
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
        load_result = self.visual.load_state_dict(clip_model.vision_model.state_dict(), strict=True)
        print_log(f"[FG-CLIP init_weights] Load state_dict result: {load_result}")

        # 加载 visual_projection 权重
        vp_load_result = self.visual_projection.load_state_dict(clip_model.visual_projection.state_dict(), strict=True)
        print_log(f"[FG-CLIP init_weights] Load visual_projection result: {vp_load_result}")

        # 冻结视觉编码器参数
        for param in self.visual.parameters():
            param.requires_grad = False
        print_log("FG-CLIP visual encoder frozen")

        # 冻结 visual_projection 参数
        for param in self.visual_projection.parameters():
            param.requires_grad = False
        print_log("FG-CLIP visual projection frozen")

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
        """
        """
        visual = self.visual
        bs, _, H, W = x.shape
        h = H // self.patch_size
        w = W // self.patch_size

        with torch.no_grad():
            # 1. Patch Embedding
            # (B, C, H, W) -> (B, D, h, w) -> (B, N, D)
            # patch_embeds = visual.embeddings.patch_embedding(x)
            # patch_embeds = patch_embeds.flatten(2).transpose(1, 2)
            
            # 2. 添加 [CLS] Token
            # class_embeds = visual.embeddings.class_embedding.expand(bs, 1, -1)
            # hidden_states = torch.cat([class_embeds, patch_embeds], dim=1)

            # 3. 添加位置编码
            # hidden_states = hidden_states + visual.embeddings.position_embedding(visual.embeddings.position_ids)
            
            hidden_states = visual.embeddings(x)
            hidden_states = visual.pre_layrnorm(hidden_states)

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

                # 7.3 应用 visual_projection
                patch_tokens = self.visual_projection(patch_tokens)
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
    
    def forward_without_attn(self, x):
        # get last layer 
        residual = x
        x = self.visual.encoder.layers[-1].layer_norm1(x)

        x = F.linear(input=x, weight=self.visual.encoder.layers[-1].self_attn.v_proj.weight, bias=self.visual.encoder.layers[-1].self_attn.v_proj.bias)
        x = self.visual.encoder.layers[-1].self_attn.out_proj(x)
        x = residual+x

        residual = x
        x = self.visual.encoder.layers[-1].layer_norm2(x)
        x = self.visual.encoder.layers[-1].mlp(x)
        x = residual + x

        return x



# if __name__ == "__main__":
#     model = FGCLIPViT_interpolate(model_name="fg-clip-base", pretrained="/hy-tmp/liuzihan/CLIPSelf/F-ViT/checkpoints/fg-clip-base")

#     model.init_weights()
#     model.train(False)

#     x = torch.rand([1, 3, 896, 896])
#     print(x.shape)

#     out = model(x)
#     print(out[-1].shape)
#     print(out[-1])

