from huggingface_hub.inference._generated.types import visual_question_answering
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from timm.layers import PatchEmbed, Mlp, DropPath, to_2tuple, to_ntuple, trunc_normal_, _assert

import json
import open_clip
from functools import partial
import os
from mmdet.models.builder import BACKBONES
from mmcv.runner import BaseModule
from mmcv.utils.logging import print_log
from mmcv.cnn import build_norm_layer

import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig


def window_partition(x, window_size: int):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size: int, H: int, W: int):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x



class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        head_dim (int): Number of channels per head (dim // num_heads if not set)
        window_size (tuple[int]): The height and width of the window.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, appearance_guidance_dim, num_heads, head_dim=None, window_size=7, qkv_bias=True, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = to_2tuple(window_size)  # Wh, Ww
        win_h, win_w = self.window_size
        self.window_area = win_h * win_w
        self.num_heads = num_heads
        head_dim = head_dim or dim // num_heads
        attn_dim = head_dim * num_heads
        self.scale = head_dim ** -0.5

        self.q = nn.Linear(dim + appearance_guidance_dim, attn_dim, bias=qkv_bias)
        self.k = nn.Linear(dim + appearance_guidance_dim, attn_dim, bias=qkv_bias)
        self.v = nn.Linear(dim, attn_dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(attn_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        
        q = self.q(x).reshape(B_, N, self.num_heads, -1).permute(0, 2, 1, 3)
        k = self.k(x).reshape(B_, N, self.num_heads, -1).permute(0, 2, 1, 3)
        v = self.v(x[:, :, :self.dim]).reshape(B_, N, self.num_heads, -1).permute(0, 2, 1, 3)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        if mask is not None:
            num_win = mask.shape[0]
            attn = attn.view(B_ // num_win, num_win, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        window_size (int): Window size.
        num_heads (int): Number of attention heads.
        head_dim (int): Enforce the number of channels per head
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(
            self, dim, appearance_guidance_dim, input_resolution, num_heads=4, head_dim=None, window_size=7, shift_size=0,
            mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.,
            act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, appearance_guidance_dim=appearance_guidance_dim, num_heads=num_heads, head_dim=head_dim, window_size=to_2tuple(self.window_size),
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            cnt = 0
            for h in (
                    slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None)):
                for w in (
                        slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None)):
                    img_mask[:, h, w, :] = cnt
                    cnt += 1
            mask_windows = window_partition(img_mask, self.window_size)  # num_win, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x, appearance_guidance):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)
        if appearance_guidance is not None:
            appearance_guidance = appearance_guidance.view(B, H, W, -1)
            x = torch.cat([x, appearance_guidance], dim=-1)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # num_win*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, x_windows.shape[-1])  # num_win*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)  # num_win*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class SwinTransformerBlockWrapper(nn.Module):
    def __init__(self, dim, appearance_guidance_dim, input_resolution, nheads=4, window_size=5, pad_len=0):
        super().__init__()
        self.block_1 = SwinTransformerBlock(dim, appearance_guidance_dim, input_resolution, num_heads=nheads, head_dim=None, window_size=window_size, shift_size=0)
        self.block_2 = SwinTransformerBlock(dim, appearance_guidance_dim, input_resolution, num_heads=nheads, head_dim=None, window_size=window_size, shift_size=window_size // 2)
        self.guidance_norm = nn.LayerNorm(appearance_guidance_dim) if appearance_guidance_dim > 0 else None

        self.pad_len = pad_len
        self.padding_tokens = nn.Parameter(torch.zeros(1, 1, dim)) if pad_len > 0 else None
        self.padding_guidance = nn.Parameter(torch.zeros(1, 1, appearance_guidance_dim)) if pad_len > 0 and appearance_guidance_dim > 0 else None
    
    def forward(self, x, appearance_guidance):
        """
        Arguments:
            x: B C T H W
            appearance_guidance: B C H W
        """
        B, C, T, H, W = x.shape
        
        x = rearrange(x, 'B C T H W -> (B T) (H W) C')
        if appearance_guidance is not None:
            appearance_guidance = self.guidance_norm(repeat(appearance_guidance, 'B C H W -> (B T) (H W) C', T=T))
        x = self.block_1(x, appearance_guidance)
        x = self.block_2(x, appearance_guidance)
        x = rearrange(x, '(B T) (H W) C -> B C T H W', B=B, T=T, H=H, W=W)
        return x


def elu_feature_map(x):
    return torch.nn.functional.elu(x) + 1


class LinearAttention(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.feature_map = elu_feature_map
        self.eps = eps

    def forward(self, queries, keys, values):
        """ Multi-Head linear attention proposed in "Transformers are RNNs"
        Args:
            queries: [N, L, H, D]
            keys: [N, S, H, D]
            values: [N, S, H, D]
            q_mask: [N, L]
            kv_mask: [N, S]
        Returns:
            queried_values: (N, L, H, D)
        """
        Q = self.feature_map(queries)
        K = self.feature_map(keys)

        v_length = values.size(1)
        values = values / v_length  # prevent fp16 overflow
        KV = torch.einsum("nshd,nshv->nhdv", K, values)  # (S,D)' @ S,V
        Z = 1 / (torch.einsum("nlhd,nhd->nlh", Q, K.sum(dim=1)) + self.eps)
        queried_values = torch.einsum("nlhd,nhdv,nlh->nlhv", Q, KV, Z) * v_length

        return queried_values.contiguous()


class FullAttention(nn.Module):
    def __init__(self, use_dropout=False, attention_dropout=0.1):
        super().__init__()
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, q_mask=None, kv_mask=None):
        """ Multi-head scaled dot-product attention, a.k.a full attention.
        Args:
            queries: [N, L, H, D]
            keys: [N, S, H, D]
            values: [N, S, H, D]
            q_mask: [N, L]
            kv_mask: [N, S]
        Returns:
            queried_values: (N, L, H, D)
        """

        # Compute the unnormalized attention and apply the masks
        QK = torch.einsum("nlhd,nshd->nlsh", queries, keys)
        if kv_mask is not None:
            QK.masked_fill_(~(q_mask[:, :, None, None] * kv_mask[:, None, :, None]), float('-inf'))

        # Compute the attention and the weighted average
        softmax_temp = 1. / queries.size(3)**.5  # sqrt(D)
        A = torch.softmax(softmax_temp * QK, dim=2)
        if self.use_dropout:
            A = self.dropout(A)

        queried_values = torch.einsum("nlsh,nshd->nlhd", A, values)

        return queried_values.contiguous()


class AttentionLayer(nn.Module):
    def __init__(self, hidden_dim, guidance_dim, nheads=8, attention_type='linear'):
        super().__init__()
        self.nheads = nheads
        self.q = nn.Linear(hidden_dim + guidance_dim, hidden_dim)
        self.k = nn.Linear(hidden_dim + guidance_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, hidden_dim)

        if attention_type == 'linear':
            self.attention = LinearAttention()
        elif attention_type == 'full':
            self.attention = FullAttention()
        else:
            raise NotImplementedError
    
    def forward(self, x, guidance):
        """
        Arguments:
            x: B, L, C
            guidance: B, L, C
        """
        q = self.q(torch.cat([x, guidance], dim=-1)) if guidance is not None else self.q(x)
        k = self.k(torch.cat([x, guidance], dim=-1)) if guidance is not None else self.k(x)
        v = self.v(x)

        q = rearrange(q, 'B L (H D) -> B L H D', H=self.nheads)
        k = rearrange(k, 'B S (H D) -> B S H D', H=self.nheads)
        v = rearrange(v, 'B S (H D) -> B S H D', H=self.nheads)

        out = self.attention(q, k, v)
        out = rearrange(out, 'B L H D -> B L (H D)')
        return out


class ClassTransformerLayer(nn.Module):
    def __init__(self, hidden_dim=64, guidance_dim=64, nheads=8, attention_type='linear', pooling_size=(4, 4), pad_len=256) -> None:
        super().__init__()
        self.pool = nn.AvgPool2d(pooling_size) if pooling_size is not None else nn.Identity()
        self.attention = AttentionLayer(hidden_dim, guidance_dim, nheads=nheads, attention_type=attention_type)
        self.MLP = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        self.pad_len = pad_len
        self.padding_tokens = nn.Parameter(torch.zeros(1, 1, hidden_dim)) if pad_len > 0 else None
        self.padding_guidance = nn.Parameter(torch.zeros(1, 1, guidance_dim)) if pad_len > 0 and guidance_dim > 0 else None
    
    def pool_features(self, x):
        """
        Intermediate pooling layer for computational efficiency.
        Arguments:
            x: B, C, T, H, W
        """
        B = x.size(0)
        x = rearrange(x, 'B C T H W -> (B T) C H W')
        x = self.pool(x)
        x = rearrange(x, '(B T) C H W -> B C T H W', B=B)
        return x

    def forward(self, x, guidance):
        """
        Arguments:
            x: B, C, T, H, W
            guidance: B, T, C
        """
        B, C, T, H, W = x.size()
        x_pool = self.pool_features(x)
        *_, H_pool, W_pool = x_pool.size()
        
        if self.padding_tokens is not None:
            orig_len = x.size(2)
            if orig_len < self.pad_len:
                # pad to pad_len
                padding_tokens = repeat(self.padding_tokens, '1 1 C -> B C T H W', B=B, T=self.pad_len - orig_len, H=H_pool, W=W_pool)
                x_pool = torch.cat([x_pool, padding_tokens], dim=2)

        x_pool = rearrange(x_pool, 'B C T H W -> (B H W) T C')
        if guidance is not None:
            if self.padding_guidance is not None:
                if orig_len < self.pad_len:
                    padding_guidance = repeat(self.padding_guidance, '1 1 C -> B T C', B=B, T=self.pad_len - orig_len)
                    guidance = torch.cat([guidance, padding_guidance], dim=1)
            guidance = repeat(guidance, 'B T C -> (B H W) T C', H=H_pool, W=W_pool)

        x_pool = x_pool + self.attention(self.norm1(x_pool), guidance) # Attention
        x_pool = x_pool + self.MLP(self.norm2(x_pool)) # MLP
        
        x_pool = rearrange(x_pool, '(B H W) T C -> (B T) C H W', H=H_pool, W=W_pool)
        x_pool = F.interpolate(x_pool, size=(H, W), mode='bilinear', align_corners=True)
        x_pool = rearrange(x_pool, '(B T) C H W -> B C T H W', B=B)

        if self.padding_tokens is not None:
            if orig_len < self.pad_len:
                x_pool = x_pool[:, :, :orig_len]

        x = x + x_pool # Residual
        return x


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Bottleneck(nn.Module):
    expansion = 4
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class AggregatorLayer(nn.Module):
    def __init__(self, hidden_dim=64, text_guidance_dim=512, appearance_guidance=512, nheads=4, input_resolution=(20, 20), pooling_size=(5, 5), window_size=(10, 10), attention_type='linear', pad_len=256) -> None:
        super().__init__()
        self.swin_block = SwinTransformerBlockWrapper(hidden_dim, appearance_guidance, input_resolution, nheads, window_size)
        self.attention = ClassTransformerLayer(hidden_dim, text_guidance_dim, nheads=nheads, attention_type=attention_type, pooling_size=pooling_size, pad_len=pad_len)

    def forward(self, x, appearance_guidance, text_guidance):
        """
        Arguments:
            x: B C T H W
        """
        x = self.swin_block(x, appearance_guidance)
        x = self.attention(x, text_guidance)
        return x


class AggregatorResNetLayer(nn.Module):
    def __init__(self, hidden_dim=64, appearance_guidance=512) -> None:
        super().__init__()
        self.conv_linear = nn.Conv2d(hidden_dim + appearance_guidance, hidden_dim, kernel_size=1, stride=1)
        self.conv_layer = Bottleneck(hidden_dim, hidden_dim // 4)


    def forward(self, x, appearance_guidance):
        """
        Arguments:
            x: B C T H W
        """
        B, T = x.size(0), x.size(2)
        x = rearrange(x, 'B C T H W -> (B T) C H W')
        appearance_guidance = repeat(appearance_guidance, 'B C H W -> (B T) C H W', T=T)

        x = self.conv_linear(torch.cat([x, appearance_guidance], dim=1))
        x = self.conv_layer(x)
        x = rearrange(x, '(B T) C H W -> B C T H W', B=B)
        return x


class DoubleConv(nn.Module):
    """(convolution => [GN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(mid_channels // 16, mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(mid_channels // 16, mid_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, guidance_channels):
        super().__init__()

        self.up = nn.ConvTranspose2d(in_channels, in_channels - guidance_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x, guidance=None):
        x = self.up(x)
        if guidance is not None:
            T = x.size(0) // guidance.size(0)
            guidance = repeat(guidance, "B C H W -> (B T) C H W", T=T)
            x = torch.cat([x, guidance], dim=1)
        return self.conv(x)


class Aggregator(nn.Module):
    def __init__(self, 
        text_guidance_dim=512,
        text_guidance_proj_dim=128,
        appearance_guidance_dim=512,
        appearance_guidance_proj_dim=128,
        decoder_dims=(64, 32),
        decoder_guidance_dims=(256, 128),
        decoder_guidance_proj_dims=(32, 16),
        num_layers=4,
        nheads=4, 
        hidden_dim=128,
        pooling_size=(6, 6),
        feature_resolution=(24, 24),
        window_size=14,
        attention_type='linear',
        prompt_channel=1,
        pad_len=256,
        # old_end=0,
        # k_old=0
    ) -> None:
        """
        Cost Aggregation Model for CAT-Seg
        Args:
            text_guidance_dim: Dimension of text guidance
            text_guidance_proj_dim: Dimension of projected text guidance
            appearance_guidance_dim: Dimension of appearance guidance
            appearance_guidance_proj_dim: Dimension of projected appearance guidance
            decoder_dims: Upsampling decoder dimensions
            decoder_guidance_dims: Upsampling decoder guidance dimensions
            decoder_guidance_proj_dims: Upsampling decoder guidance projected dimensions
            num_layers: Number of layers for the aggregator
            nheads: Number of attention heads
            hidden_dim: Hidden dimension for transformer blocks
            pooling_size: Pooling size for the class aggregation layer
                          To reduce computation, we apply pooling in class aggregation blocks to reduce the number of tokens during training
            feature_resolution: Feature resolution for spatial aggregation
            window_size: Window size for Swin block in spatial aggregation
            attention_type: Attention type for the class aggregation. 
            prompt_channel: Number of prompts for ensembling text features. Default: 1
            pad_len: Padding length for the class aggregation. Default: 256
                     pad_len enforces the class aggregation block to have a fixed length of tokens for all inputs
                     This means it either pads the sequence with learnable tokens in class aggregation,
                     or truncates the classes with the initial CLIP cosine-similarity scores.
                     Set pad_len to 0 to disable this feature.
            """
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        self.layers = nn.ModuleList([
            AggregatorLayer(
                hidden_dim=hidden_dim, text_guidance_dim=text_guidance_proj_dim, appearance_guidance=appearance_guidance_proj_dim, 
                nheads=nheads, input_resolution=feature_resolution, pooling_size=pooling_size, window_size=window_size, attention_type=attention_type, pad_len=pad_len,
            ) for _ in range(num_layers)
        ])

        self.conv1 = nn.Conv2d(prompt_channel, hidden_dim, kernel_size=7, stride=1, padding=3)

        self.guidance_projection = nn.Sequential(
            nn.Conv2d(appearance_guidance_dim, appearance_guidance_proj_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        ) if appearance_guidance_dim > 0 else None
        
        self.text_guidance_projection = nn.Sequential(
            nn.Linear(text_guidance_dim, text_guidance_proj_dim),
            nn.ReLU(),
        ) if text_guidance_dim > 0 else None

        self.decoder_guidance_projection = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(d, dp, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
            ) for d, dp in zip(decoder_guidance_dims, decoder_guidance_proj_dims)
        ]) if decoder_guidance_dims[0] > 0 else None

        self.decoder1 = Up(hidden_dim, decoder_dims[0], decoder_guidance_proj_dims[0])
        self.decoder2 = Up(decoder_dims[0], decoder_dims[1], decoder_guidance_proj_dims[1])
        self.head = nn.Conv2d(decoder_dims[1], 1, kernel_size=3, stride=1, padding=1)

        self.pad_len = pad_len

        # self.old_end = old_end
        # self.k_old = k_old

    def feature_map(self, img_feats, text_feats):
        # concatenated feature volume for feature aggregation baselines
        img_feats = F.normalize(img_feats, dim=1) # B C H W
        img_feats = repeat(img_feats, "B C H W -> B C T H W", T=text_feats.shape[1])
        text_feats = F.normalize(text_feats, dim=-1) # B T P C
        text_feats = text_feats.mean(dim=-2) # average text features over different prompts
        text_feats = F.normalize(text_feats, dim=-1) # B T C
        text_feats = repeat(text_feats, "B T C -> B C T H W", H=img_feats.shape[-2], W=img_feats.shape[-1])
        return torch.cat((img_feats, text_feats), dim=1) # B 2C T H W

    def correlation(self, img_feats, text_feats):
        img_feats = F.normalize(img_feats, dim=1) # B C H W
        text_feats = F.normalize(text_feats, dim=-1) # B T P C
        corr = torch.einsum('bchw, btpc -> bpthw', img_feats, text_feats)
        return corr

    def corr_embed(self, x): # x: B P T H W
        B = x.shape[0]
        corr_embed = rearrange(x, 'B P T H W -> (B T) P H W')
        corr_embed = self.conv1(corr_embed) # (B T) hidden_dim H W
        corr_embed = rearrange(corr_embed, '(B T) C H W -> B C T H W', B=B) # B hidden_dim T H W
        return corr_embed # B hidden_dim T H W
    
    def corr_projection(self, x, proj):
        corr_embed = rearrange(x, 'B C T H W -> B T H W C')
        corr_embed = proj(corr_embed)
        corr_embed = rearrange(corr_embed, 'B T H W C -> B C T H W')
        return corr_embed

    def upsample(self, x):
        B = x.shape[0]
        corr_embed = rearrange(x, 'B C T H W -> (B T) C H W')
        corr_embed = F.interpolate(corr_embed, scale_factor=2, mode='bilinear', align_corners=True)
        corr_embed = rearrange(corr_embed, '(B T) C H W -> B C T H W', B=B)
        return corr_embed

    def conv_decoder(self, x, guidance):
        B = x.shape[0]
        corr_embed = rearrange(x, 'B C T H W -> (B T) C H W')
        corr_embed = self.decoder1(corr_embed, guidance[0])
        corr_embed = self.decoder2(corr_embed, guidance[1])
        corr_embed = self.head(corr_embed)
        corr_embed = rearrange(corr_embed, '(B T) () H W -> B T H W', B=B)
        return corr_embed
    
    def forward(self, img_feats, text_feats, appearance_guidance):
        """
        Arguments:
            img_feats: (B, C, H, W)
            text_feats: (B, T, P, C) = (B, 2, 4, 512) for CLIP ViT-B/16 with 4 prompt templates
            appearance_guidance: tuple of (B, C, H, W)
        """
        classes = None
        topk_indices = None

        corr = self.correlation(img_feats, text_feats) # shape = [B, P, T, H, W]

        # # !=== 新增: 强制对齐逻辑 ===
        # if force_indices is not None:
        #     classes = force_indices
        #     th_text = F.normalize(text_feats, dim=-1)
        #     gather_index = classes[..., None, None].expand(-1, -1, th_text.size(-2), th_text.size(-1))
        #     th_text = torch.gather(th_text, dim=1, index=gather_index)
        #     img_feats = F.normalize(img_feats, dim=1)
        #     corr = torch.einsum('bchw, bkpc -> bpkhw', img_feats, th_text)
        #     topk_indices = classes
        #     text_feats = th_text

        # elif self.pad_len > 0 and text_feats.size(1) > self.pad_len:
        #     K = self.pad_len
        #     old_end = getattr(self, 'old_end', 19)  # 0-18 old, 19-... new
        #     # 建议 pad_len=8 时 old 保底 2 个
        #     k_old = getattr(self, 'k_old', 4)
        #     k_old = min(k_old, old_end, K)
        #     k_new = K - k_old

        #     # ------- (A) 更稳健的 class score：先 spatial max，再 prompt mean -------
        #     # corr: [B,P,T,H,W] -> spatial max: [B,P,T] -> mean over P: [B,T]
        #     score = corr.amax(dim=(-1, -2)).mean(dim=1)  # [B,T]

        #     # ------- (B) 配额 TopK：old + new -------
        #     score_old = score[:, :old_end]          # [B,old_end]
        #     score_new = score[:, old_end:]          # [B,T-old_end]

        #     idx_old = score_old.topk(k_old, dim=-1, largest=True, sorted=False).indices  # [B,k_old]

        #     if k_new > 0 and score_new.numel() > 0:
        #         idx_new = score_new.topk(min(k_new, score_new.size(1)), dim=-1, largest=True, sorted=False).indices + old_end
        #     else:
        #         idx_new = idx_old.new_zeros((idx_old.size(0), 0))

        #     classes = torch.cat([idx_old, idx_new], dim=-1)  # [B,K]（通常就是8）
        #     # 可选：防御性去重+补齐（一般不会重复，但留着更稳）
        #     # classes = torch.unique(classes, dim=-1)
        #     if classes.size(1) < K:
        #         mask = torch.ones_like(score, dtype=torch.bool)
        #         mask.scatter_(1, classes, False)
        #         fill = score.masked_fill(~mask, -1e9).topk(K - classes.size(1), dim=-1).indices
        #         classes = torch.cat([classes, fill], dim=-1)

        #     # ------- (C) gather text feats -------
        #     th_text = F.normalize(text_feats, dim=-1)  # [B,T,P,C]
        #     th_text = torch.gather(
        #         th_text, dim=1,
        #         index=classes[..., None, None].expand(-1, -1, th_text.size(-2), th_text.size(-1))
        #     )  # [B,K,P,C]

        #     # 重新算 corr（注意这里 th_text 的第二维变成 K，所以 einsum 的符号要对应）
        #     img_feats = F.normalize(img_feats, dim=1)
        #     text_feats = th_text
        #     corr = torch.einsum('bchw, bkpc -> bpkhw', img_feats, th_text)  # [B,P,K,H,W]

        #     # 如果你后面还需要 classes/topk_indices，直接 return / cache
        #     topk_indices = classes  # [B,K]
        # corr = self.feature_map(img_feats, text_feats)

        if self.pad_len > 0 and text_feats.size(1) > self.pad_len:
            avg = corr.permute(0, 2, 1, 3, 4).flatten(-3).max(dim=-1)[0] 
            classes = avg.topk(self.pad_len, dim=-1, sorted=False)[1]
            th_text = F.normalize(text_feats, dim=-1)
            th_text = torch.gather(th_text, dim=1, index=classes[..., None, None].expand(-1, -1, th_text.size(-2), th_text.size(-1)))
            orig_clases = text_feats.size(1)
            img_feats = F.normalize(img_feats, dim=1) # B C H W
            text_feats = th_text
            corr = torch.einsum('bchw, btpc -> bpthw', img_feats, th_text)
        #corr = self.feature_map(img_feats, text_feats)
        corr_embed = self.corr_embed(corr) # shape = [B, T, hidden_dim, H, W]
        # return corr_embed

        projected_guidance, projected_text_guidance, projected_decoder_guidance = None, None, [None, None]
        if self.guidance_projection is not None:
            projected_guidance = self.guidance_projection(appearance_guidance[0]) # res3
        if self.decoder_guidance_projection is not None:
            projected_decoder_guidance = [proj(g) for proj, g in zip(self.decoder_guidance_projection, appearance_guidance[1:])]

        if self.text_guidance_projection is not None:
            text_feats = text_feats.mean(dim=-2) # shape = [B, T, C]
            text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True) # shape = [B, T, C]
            projected_text_guidance = self.text_guidance_projection(text_feats) # shape = [B, T, text_guidance_proj_dim]

        for layer in self.layers:
            corr_embed = layer(corr_embed, projected_guidance, projected_text_guidance) # shape = [B, hidden_dim, T, H, W]

        logit = self.conv_decoder(corr_embed, projected_decoder_guidance) # shape = [B, T, C, H, W]
        # if classes is not None:
        #     # 处理五维情况: 需要在 T 维度上进行 scatter
        #     # 创建全 -100 的输出: (B, orig_classes, C, H, W)
        #     out = torch.full((logit.size(0), orig_classes, logit.size(2), logit.size(3), logit.size(4)), -100., device=logit.device)
        #     # scatter 操作: 在 dim=1 (类别维) 上根据 classes 索引放置 logit
        #     # classes: (B, pad_len) -> (B, pad_len, 1, 1, 1) -> (B, pad_len, C, H, W)
        #     out.scatter_(dim=1, 
        #                 index=classes[..., None, None, None].expand(-1, -1, logit.size(2), logit.size(3), logit.size(4)), 
        #                 src=logit)
        #     logit = out
        return logit


class PromptOrder:
    def __init__(self) -> None:
        super().__init__()
        self.template_list = [
            'a bad photo of a {}.',
    'a photo of many {}.',
    'a sculpture of a {}.',
    'a photo of the hard to see {}.',
    'a low resolution photo of the {}.',
    'a rendering of a {}.',
    'graffiti of a {}.',
    'a bad photo of the {}.',
    'a cropped photo of the {}.',
    'a tattoo of a {}.',
    'the embroidered {}.',
    'a photo of a hard to see {}.',
    'a bright photo of a {}.',
    'a photo of a clean {}.',
    'a photo of a dirty {}.',
    'a dark photo of the {}.',
    'a drawing of a {}.',
    'a photo of my {}.',
    'the plastic {}.',
    'a photo of the cool {}.',
    'a close-up photo of a {}.',
    'a black and white photo of the {}.',
    'a painting of the {}.',
    'a painting of a {}.',
    'a pixelated photo of the {}.',
    'a sculpture of the {}.',
    'a bright photo of the {}.',
    'a cropped photo of a {}.',
    'a plastic {}.',
    'a photo of the dirty {}.',
    'a jpeg corrupted photo of a {}.',
    'a blurry photo of the {}.',
    'a photo of the {}.',
    'a good photo of the {}.',
    'a rendering of the {}.',
    'a {} in a video game.',
    'a photo of one {}.',
    'a doodle of a {}.',
    'a close-up photo of the {}.',
    'a photo of a {}.',
    'the origami {}.',
    'the {} in a video game.',
    'a sketch of a {}.',
    'a doodle of the {}.',
    'a origami {}.',
    'a low resolution photo of a {}.',
    'the toy {}.',
    'a rendition of the {}.',
    'a photo of the clean {}.',
    'a photo of a large {}.',
    'a rendition of a {}.',
    'a photo of a nice {}.',
    'a photo of a weird {}.',
    'a blurry photo of a {}.',
    'a cartoon {}.',
    'art of a {}.',
    'a sketch of the {}.',
    'a embroidered {}.',
    'a pixelated photo of a {}.',
    'itap of the {}.',
    'a jpeg corrupted photo of the {}.',
    'a good photo of a {}.',
    'a plushie {}.',
    'a photo of the nice {}.',
    'a photo of the small {}.',
    'a photo of the weird {}.',
    'the cartoon {}.',
    'art of the {}.',
    'a drawing of the {}.',
    'a photo of the large {}.',
    'a black and white photo of a {}.',
    'the plushie {}.',
    'a dark photo of a {}.',
    'itap of a {}.',
    'graffiti of the {}.',
    'a toy {}.',
    'itap of my {}.',
    'a photo of a cool {}.',
    'a photo of a small {}.',
    'a tattoo of the {}.',
        ]

    def prompt(self, class_names):
        """
        为每个类别生成提示
        
        Args:
            class_names: 类别名称列表，例如 ['cat', 'dog'] 或单个字符串 'cat'
            
        Returns:
            提示列表，每个类别对应一个模板列表
            例如：[['a photo of a cat.', ...], ['a photo of a dog.', ...]]
        """
        # 如果输入是单个字符串，转换为列表
        if isinstance(class_names, str):
            class_names = [class_names]
        # class_names += ['background']  # 添加背景类别
        
        # 为每个类别生成所有模板的提示
        all_prompts = []
        for class_name in class_names:
            class_prompts = [template.format(class_name) for template in self.template_list]
            all_prompts.append(class_prompts)
            
        return all_prompts


@BACKBONES.register_module()
class CatSegEvaCLIPViT(BaseModule):
    def __init__(self, 
        model_name,                    # clip模型名称
        pretrained,                    # clip模型预训练权重路径
        class_names,                   # 类别名称
        text_guidance_dim,             # 文本guidance投影前维度
        text_guidance_proj_dim,        # 文本guidance投影后维度
        appearance_guidance_dim,       # 外观guidance投影前维度
        appearance_guidance_proj_dim,  # 外观guidance投影后维度
        decoder_dims,                  # 解码器维度
        decoder_guidance_dims,         # 解码器guidance维度
        decoder_guidance_proj_dims,    # 解码器guidance投影维度
        num_layers,                    # 解码器层数
        nheads,                        # 解码器注意力头数
        hidden_dim,                    # 解码器隐藏维度
        pooling_size,                  # 解码器池化大小
        feature_resolution,            # 特征分辨率
        window_size,                   # 窗口大小
        attention_type,                # 注意力类型
        out_indices,                   # appearance guidance 索引
        cat_seg_indices,
        # pad_len,
        # old_end,
        # k_old,
        norm_cfg,                      # 归一化配置
        **kwargs                       # ⭐ 接受额外参数（包括with_cp）
    ):
        # ⭐ 提取 with_cp 参数，传递其余参数给父类
        self.with_cp = kwargs.pop('with_cp', False)
        # print(f"[CatSegEvaCLIPViT] Gradient checkpointing enabled: {self.with_cp}")
        super().__init__(**kwargs)
        
        self.model_name = model_name
        self.pretrained = pretrained
        self.vit_layers = out_indices
        self.cat_seg_indices = cat_seg_indices

        # decoder guidance 上采样层
        self.upsample1 = nn.ConvTranspose2d(768, 256, kernel_size=2, stride=2)  # 2倍上采样
        self.upsample2 = nn.ConvTranspose2d(768, 128, kernel_size=4, stride=4)  # 4倍上采样

        # 加载clip模型
        self.clip_model = open_clip.create_model(model_name, pretrained="eva", cache_dir=pretrained)
        self.clip_model.float()

        # text level
        class_names = json.load(open(class_names, 'r'))
        text_promtps = PromptOrder().prompt(class_names)
        self.text_features = self.encode_text(text_promtps)
        
        # 自动获取 prompt 数量 (text_features 形状: T, P, C)
        prompt_channel = self.text_features.shape[1]

        # # pixel level
        # self.image_size = image_size
        # self.patch_size = self.clip_model.visual.patch_embed.patch_size[0]
        # self.num_patches = (image_size[0] // self.patch_size) * (image_size[1] // self.patch_size)
        # self.grid_size = (image_size[0] // self.patch_size, image_size[1] // self.patch_size)

        # cat-seg
        self.seg_decoder = Aggregator(
            text_guidance_dim=text_guidance_dim,
            text_guidance_proj_dim=text_guidance_proj_dim,
            appearance_guidance_dim=appearance_guidance_dim,
            appearance_guidance_proj_dim=appearance_guidance_proj_dim,
            decoder_dims=decoder_dims,
            decoder_guidance_dims=decoder_guidance_dims,
            decoder_guidance_proj_dims=decoder_guidance_proj_dims,
            num_layers=num_layers,
            nheads=nheads,
            hidden_dim=hidden_dim,
            pooling_size=pooling_size,
            feature_resolution=feature_resolution,
            window_size=window_size,
            attention_type=attention_type,
            prompt_channel=prompt_channel,  # 传入实际的 prompt 数量
            # pad_len=pad_len,
            # old_end=old_end,
            # k_old=k_old
        )

        self.proj_dim = self.clip_model.visual.embed_dim
        self.width = self.clip_model.visual.embed_dim
        
        # 多尺度
        # 上采样4倍
        self.interpolate1 = nn.Sequential(
            nn.ConvTranspose2d(self.width, self.width, kernel_size=2, stride=2),
            build_norm_layer(norm_cfg, self.width)[1] if norm_cfg else nn.Identity(),
            nn.GELU(),
            nn.ConvTranspose2d(self.width, self.width, kernel_size=2, stride=2),
        )
        # 上采样2倍
        self.interpolate2 = nn.ConvTranspose2d(self.width, self.width, kernel_size=2, stride=2)
        self.interpolate3 = nn.Identity()
        self.interpolate4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.visual = self.clip_model.visual

    def init_weights(self):
        clip_model = open_clip.create_model(self.model_name,
                                            pretrained="eva",
                                            cache_dir=self.pretrained,
                                            device="cpu")
        print_log(self.visual.load_state_dict(clip_model.visual.state_dict(), strict=True))
        for param in self.visual.parameters():  # only freeze the CLIP model
            param.requires_grad = False

    def train(self, mode=True):
        self.training = mode
        # 冻结clip编码器
        self.visual.train(False)
        self.interpolate1.train(mode)
        self.interpolate2.train(mode)
        self.interpolate3.train(mode)
        self.interpolate4.train(mode)
        self.seg_decoder.train(mode)

        return self
    
    def forward(self, x):
        visual = self.visual
        bs, _, h, w = x.shape
        h = h // visual.patch_embed.patch_size[0]
        w = w // visual.patch_embed.patch_size[1]

        # clip.visual 前向传播
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
            res = []
            for i, blk in enumerate(visual.blocks[:-1]):
                x = blk(x, rel_pos_bias=rel_pos_bias)
                if i in self.vit_layers:
                    outs.append(self._expand_x(x, h, w))
                if i in self.cat_seg_indices:
                    res.append(self._expand_x(x, h, w))
            x = visual.blocks[-1].forward_without_attn(x)
            if (len(visual.blocks) - 1) in self.vit_layers:
                outs.append(self._expand_x(x, h, w))
            if (len(visual.blocks) - 1) in self.cat_seg_indices:
                res.append(self._expand_x(x, h, w))
            
            # 处理 img_feats: 需要投影到输出维度并归一化
            img_feats = x[:, 1:]  # 去除 cls_token, (B, H*W, C)
            img_feats = visual.norm(img_feats)
            img_feats = visual.head(img_feats)  # 投影到 512 维
            assert visual.fc_norm is None
            img_feats = F.normalize(img_feats, dim=-1)  # 归一化
            img_feats = img_feats.view(bs, h, w, -1).permute(0, 3, 1, 2)  # (B, C, H, W)
        # cat-seg 前向传播
        # 扩展文本特征: (T, P, C) -> (bs, T, P, C)
        text_feats = self.text_features.unsqueeze(0).expand(bs, -1, -1, -1)
        text_feats = text_feats.to(x.device).to(x.dtype)
        
        # 调用分割解码器
        res3 = img_feats
        res4 = res[0]
        res5 = res[1]
        res4 = self.upsample1(res4)
        res5 = self.upsample2(res5)
        # [bsz, class_num, c, h, w]
        logits = self.seg_decoder(img_feats, text_feats, appearance_guidance=[res3, res4, res5])
        # feats = rearrange(feats, "B T C H W -> B (T C) H W")

        
        for idx, out in enumerate(outs):
            interpolate = getattr(self, f"interpolate{idx + 1}")
            outs[idx] = interpolate(out.detach())
        outs.append(logits)
        # outs.append(topk_indices)
        outs.append(img_feats) if not self.training else outs.append(None)
        return outs  # 4个多尺度特征 + 最终logits
    
    def forward_with_text(self, x, text_feats_override, force_indices=None):
        visual = self.visual
        bs, _, h, w = x.shape
        h = h // visual.patch_embed.patch_size[0]
        w = w // visual.patch_embed.patch_size[1]

        with torch.no_grad():
            x = visual.patch_embed(x)
            batch_size, seq_len, _ = x.size()
            cls_tokens = visual.cls_token.expand(batch_size, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)
            if visual.pos_embed is not None:
                x = x + visual.rescale_positional_embedding(out_size=(h, w))
            x = visual.pos_drop(x)
            x = visual.patch_dropout(x)
            
            rel_pos_bias = visual.rel_pos_bias() if visual.rel_pos_bias is not None else None

            res = []
            for i, blk in enumerate(visual.blocks[:-1]):
                x = blk(x, rel_pos_bias=rel_pos_bias)
                if i in self.cat_seg_indices:
                    res.append(self._expand_x(x, h, w))
            x = visual.blocks[-1].forward_without_attn(x)
            if (len(visual.blocks) - 1) in self.cat_seg_indices:
                res.append(self._expand_x(x, h, w))
            
            img_feats = x[:, 1:]
            img_feats = visual.norm(img_feats)
            img_feats = visual.head(img_feats)
            img_feats = F.normalize(img_feats, dim=-1)
            img_feats = img_feats.view(bs, h, w, -1).permute(0, 3, 1, 2)

        # B. 准备文本特征 (使用传入的 Teacher 文本)
        # text_feats_override: [T_teacher, P, C] -> [B, T_teacher, P, C]
        text_feats = text_feats_override.unsqueeze(0).expand(bs, -1, -1, -1)
        text_feats = text_feats.to(x.device).to(x.dtype)

        # C. 准备 Guidance
        res3 = img_feats
        res4 = res[0]
        res5 = res[1]
        res4 = self.upsample1(res4)
        res5 = self.upsample2(res5)

        # D. 调用 Aggregator (强制使用 Teacher 的 indices)
        logits, _ = self.seg_decoder(
            img_feats, 
            text_feats, 
            appearance_guidance=[res3, res4, res5],
            force_indices=force_indices # <--- 传进去
        )
        
        return logits

    def _expand_x(self, x, h, w):
        # x: bs q c
        x = x[:, 1:].permute(0, 2, 1).contiguous()
        x = x.view(-1, self.width, h, w)

        return x

    @torch.no_grad()
    def encode_text(self, text_prompts):
        self.clip_model = self.clip_model.cuda()
        tokenizer = open_clip.get_tokenizer(model_name=self.model_name)
        text_features = torch.empty(0).cuda()
        for templates in text_prompts:
            ensemble_text_features = torch.empty(0).cuda()
            for prompt in templates:
                texts = tokenizer(prompt)
                if hasattr(texts, 'input_ids'):
                    texts = texts.input_ids
                texts = texts.cuda()
                cur_embed = self.clip_model.encode_text(texts)
                cur_embed = cur_embed / cur_embed.norm(p=2, dim=-1, keepdim=True)
                ensemble_text_features = torch.cat(
                    [ensemble_text_features, cur_embed], dim=0)
            # ensemble_text_features.shape = [prompt_num, output_dim] = [P, C]
            ensemble_text_features = ensemble_text_features.unsqueeze(0)  # shape = [1, prompt_num, output_dim] = [1, P, C]
            text_features = torch.cat([text_features, ensemble_text_features], dim=0)
        # [class_num, prompt_num, output_dim] = [T, P, C]
        return text_features

@BACKBONES.register_module()
class CatSegFGCLIPViT(BaseModule):
    def __init__(self,
                    model_name,                    # clip模型名称
                    pretrained,                    # clip模型预训练权重路径
                    class_names,                   # 类别名称
                    text_guidance_dim,             # 文本guidance投影前维度
                    text_guidance_proj_dim,        # 文本guidance投影后维度
                    appearance_guidance_dim,       # 外观guidance投影前维度
                    appearance_guidance_proj_dim,  # 外观guidance投影后维度
                    decoder_dims,                  # 解码器维度
                    decoder_guidance_dims,         # 解码器guidance维度
                    decoder_guidance_proj_dims,    # 解码器guidance投影维度
                    num_layers,                    # 解码器层数
                    nheads,                        # 解码器注意力头数
                    hidden_dim,                    # 解码器隐藏维度
                    pooling_size,                  # 解码器池化大小
                    feature_resolution,            # 特征分辨率
                    window_size,                   # 窗口大小
                    attention_type,                # 注意力类型
                    out_indices,                   # appearance guidance 索引
                    norm_cfg,                      # 归一化配置
                    **kwargs                       # ⭐ 接受额外参数（包括with_cp）):
    ):
        super().__init__(**kwargs)
        self.vit_layers = out_indices
        self.model_name = model_name
        self.pretrained = pretrained

        if self.pretrained and os.path.exists(self.pretrained):
            print_log(f"[FG-CLIP] Loading model '{self.pretrained}' using transformers.AutoModel")
            clip_model = AutoModelForCausalLM.from_pretrained(self.pretrained, local_files_only=True, trust_remote_code=True)
        else:
            print_log(f"[FG-CLIP] Create model without pretrained (or path missing): {self.pretrained}")
            config_path = os.path.join("/hy-tmp/liuzihan/CLIPSelf/F-ViT/checkpoints/", self.model_name)
            model_config = AutoConfig.from_pretrained(config_path, local_files_only=True, trust_remote_code=True)
            clip_model = transformers.AutoModel.from_config(model_config)

        # self.visual = clip_model.vision_model
        # self.visual_projection = clip_model.visual_projection

        self.clip_model = clip_model
        self.visual = clip_model.vision_model
        self.visual_projection = clip_model.visual_projection
        self.patch_size = self.visual.config.patch_size
        self.width = self.visual.config.hidden_size

        # text level
        class_names = json.load(open(class_names, 'r'))
        # num_classes = len(class_names)
        text_promtps = PromptOrder().prompt(class_names)
        self.text_features = self.encode_text(text_promtps)
        
        # 自动获取 prompt 数量 (text_features 形状: T, P, C)
        prompt_channel = self.text_features.shape[1]

        self.upsample1 = nn.ConvTranspose2d(self.width, self.width, kernel_size=2, stride=2)  # 2倍上采样
        self.upsample2 = nn.Sequential(  # 4倍上采样 = 2倍 + 2倍
            nn.ConvTranspose2d(self.width, self.width, kernel_size=2, stride=2),
            nn.ConvTranspose2d(self.width, self.width, kernel_size=2, stride=2),
        )

        # self.width = self.visual.config.hidden_size
        # self.patch_size = self.visual.config.patch_size

        # interpolate 模块应该使用 decoder_dims[1]，因为 seg_decoder 的输出通道数是 decoder_dims[1]
        self.interpolate1 = nn.Sequential(
            nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2),
            build_norm_layer(norm_cfg, 512)[1] if norm_cfg else nn.Identity(),
            nn.GELU(),
            nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2),
        )
        self.interpolate2 = nn.Sequential(
            nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2),
        )
        self.interpolate3 = nn.Identity()
        self.interpolate4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # cat-seg
        self.seg_decoder = Aggregator(
            text_guidance_dim=text_guidance_dim,
            text_guidance_proj_dim=text_guidance_proj_dim,
            appearance_guidance_dim=appearance_guidance_dim,
            appearance_guidance_proj_dim=appearance_guidance_proj_dim,
            decoder_dims=decoder_dims,
            decoder_guidance_dims=decoder_guidance_dims,
            decoder_guidance_proj_dims=decoder_guidance_proj_dims,
            num_layers=num_layers,
            nheads=nheads,
            hidden_dim=hidden_dim,
            pooling_size=pooling_size,
            feature_resolution=feature_resolution,
            window_size=window_size,
            attention_type=attention_type,
            prompt_channel=prompt_channel,  # 传入实际的 prompt 数量
        )

    def init_weights(self):
        clip_model = transformers.AutoModel.from_pretrained(self.pretrained, local_files_only=True)
        print_log(self.visual.load_state_dict(clip_model.vision_model.state_dict(), strict=True))
        print_log(self.visual_projection.load_state_dict(clip_model.visual_projection.state_dict(), strict=True))
        for param in self.visual.parameters():  # only freeze the CLIP model
            param.requires_grad = False

        for param in self.visual_projection.parameters():
            param.requires_grad = False
        # for param in self.clip_model.parameters():
        #     param.requires_grad = False

    def train(self, mode=True):
        self.training = mode
        # 冻结clip编码器
        self.clip_model.train(False)
        self.visual.train(False)
        self.visual_projection.train(False)
        self.interpolate1.train(mode)
        self.interpolate2.train(mode)
        self.interpolate3.train(mode)
        self.interpolate4.train(mode)
        self.seg_decoder.train(mode)

        return self
    
    def forward(self, x):
        visual = self.visual
        bs, _, H, W = x.shape
        h = H // self.patch_size
        w = W // self.patch_size

        handles = []
        feats = {}

        post_ln_feats = {}
        post_ln_handles = []

        outs = []
        with torch.no_grad():
            # def make_hook(idx):
            #     def _hook(_m, _in, out):
            #         feats[idx] = out[0]
            #     return _hook

            # def post_ln_hook(_m, inputs, _output):
            #     x = inputs[0]
            #     if x.dim() == 3:
            #         post_ln_feats['tensors'] = x

            # post_ln_handles.append(self.clip_model.vision_model.post_layernorm.register_forward_hook(post_ln_hook))
            
            # for i in self.vit_layers:
            #     handles.append(self.clip_model.vision_model.encoder.layers[i].register_forward_hook(make_hook(i)))

            # dense_image_feature = self.clip_model.get_image_dense_features(x)
            # # feats[self.vit_layers[-1]] = post_ln_feats['tensors']

            # for i in self.vit_layers:
            #     tk = feats[i]
            #     outs.append(self._expand_x(tk, h, w))
            # outs.append(feats[self.vit_layers[-1]].view(-1, self.width, h, w))

            hidden_states = visual.embeddings(x)
            hidden_states = visual.pre_layrnorm(hidden_states)
            transformer_blocks = visual.encoder.layers
            for i, blk in enumerate(transformer_blocks[:-1]):
                layer_outputs = blk(hidden_states,
                                attention_mask=None,
                                causal_attention_mask=None)
                hidden_states = layer_outputs[0]
                if i in self.vit_layers:
                    outs.append(self._expand_x(hidden_states, h, w))
            hidden_states = self.forward_without_attn(hidden_states)
            if (len(transformer_blocks) - 1) in self.vit_layers:
                outs.append(self._expand_x(hidden_states, h, w))

            # if not self.training:
            patch_tokens = hidden_states[:, 1:]
            patch_tokens = visual.post_layernorm(patch_tokens)
            patch_tokens = self.visual_projection(patch_tokens)
            patch_tokens = F.normalize(patch_tokens, dim=-1)
            feature_map = patch_tokens.view(bs, h, w, -1).permute(0, 3, 1, 2)
            # else:
            #     feature_map = None

        text_feats = self.text_features.unsqueeze(0).expand(bs, -1, -1, -1)
        text_feats = text_feats.to(x.device).to(x.dtype)
        
        # 调用分割解码器
        # (B, C, H, W) -> (B, class_num, H, W)
        res3 = feature_map
        res4 = outs[0]
        res5 = outs[1]
        # res4 = cp.checkpoint(self.upsample1, res4)
        # res5 = cp.checkpoint(self.upsample2, res5)
        # res4 = self.upsample1(res4)
        # res5 = self.upsample2(res5)
        # logits = self.seg_decoder(feature_map, text_feats, [res3, res4, res5])
        logits = self.seg_decoder(feature_map, text_feats, [res3, res4, res5])
        #[bsz class_num channel h w]

        # 对不同类的特征图进行softmax归一化，然后与原特征图相乘
        logits_softmax = F.softmax(logits, dim=1)
        # [bsz channel h w]
        feature_map_expanded = feature_map.unsqueeze(1).expand(-1, logits.size(1), -1, -1, -1)
        logits = feature_map_expanded * logits_softmax
        
        # bsz = logits.size(0)
        # logits = rearrange(logits, 'B T C H W -> (B T) C H W')
        # logits = logits.mean(dim=1)  # [B, T, C, H, W] -> [B, C, H, W]

        # for idx, out in enumerate(outs):
        #     interpolate = getattr(self, f"interpolate{idx + 1}")
        #     outs[idx] = interpolate(out.detach())
        # # 由于在T维度取平均，输出是 [B, C, H', W']，不需要reshape

        bsz = logits.size(0)
        logits = rearrange(logits, 'B T C H W -> (B T) C H W')
        # 多尺度特征图
        outs = []
        for idx in range(0, 4):
            interpolate = getattr(self, f"interpolate{idx + 1}")
            outs.append(interpolate(logits))
        outs = [rearrange(out, '(B T) C H W -> B T C H W', B=bsz) for out in outs]

        # if self.training:
        #     outs.append(None)
        # else:
        #     outs.append(feature_map)

        # outs.append(feature_map)

        return tuple(outs)

    def _expand_x(self, x, h, w):
        # x: bs q c
        x = x[:, 1:].permute(0, 2, 1).contiguous()
        x = x.view(-1, self.width, h, w)

        return x

    def forward_without_attn(self, x):
        # get last layer 
        residual = x
        x = self.visual.encoder.layers[-1].layer_norm1(x)
        x = F.linear(input=x, weight=self.visual.encoder.layers[-1].self_attn.v_proj.weight, bias=self.visual.encoder.layers[-1].self_attn.v_proj.bias)
        x = self.visual.encoder.layers[-1].self_attn.out_proj(x)
        x = residual + x

        residual = x
        x = self.visual.encoder.layers[-1].layer_norm2(x)
        x = self.visual.encoder.layers[-1].mlp(x)
        x = residual + x

        return x

    @torch.no_grad()
    def encode_text(self, text_prompts):
        self.clip_model = self.clip_model.cuda()
        tokenizer = AutoTokenizer.from_pretrained(self.pretrained, local_files_only=True)
        text_features = torch.empty(0).cuda()
        for templates in text_prompts:
            ensemble_text_features = torch.empty(0).cuda()
            for prompt in templates:
                texts = tokenizer(prompt, return_tensors='pt', padding=True, truncation=True)
                texts = texts.input_ids
                texts = texts.cuda()
                cur_embed = self.clip_model.get_text_features(input_ids=texts)
                cur_embed = cur_embed / cur_embed.norm(p=2, dim=-1, keepdim=True)
                ensemble_text_features = torch.cat(
                    [ensemble_text_features, cur_embed], dim=0)
            # ensemble_text_features.shape = [prompt_num, output_dim] = [P, C]
            ensemble_text_features = ensemble_text_features.unsqueeze(0)  # shape = [1, prompt_num, output_dim] = [1, P, C]
            text_features = torch.cat([text_features, ensemble_text_features], dim=0)
        # [class_num, prompt_num, output_dim] = [T, P, C]
        return text_features
