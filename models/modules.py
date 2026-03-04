import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import torch
from einops import rearrange ,repeat
from einops.layers.torch import Rearrange
from timm.layers import trunc_normal_, DropPath
import numpy as np
import math
import time
import os
import matplotlib.pyplot as plt
from compressai.layers import (
    AttentionBlock,
    ResidualBlock,
    ResidualBlockUpsample,
    ResidualBlockWithStride,
    conv3x3,
    subpel_conv3x3,
)

SCALES_MIN = 0.11
SCALES_MAX = 256
SCALES_LEVELS = 64

def conv1x1(in_ch: int, out_ch: int, stride: int = 1) -> nn.Module:
    """1x1 convolution."""
    return nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride)

def conv(in_channels, out_channels, kernel_size=5, stride=2):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=kernel_size // 2,
    )

def deconv(in_channels, out_channels, kernel_size=5, stride=2):
    return nn.ConvTranspose2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        output_padding=stride - 1,
        padding=kernel_size // 2,
    )

def get_scale_table(min=SCALES_MIN, max=SCALES_MAX, levels=SCALES_LEVELS):
    return torch.exp(torch.linspace(math.log(min), math.log(max), levels)) 

def ste_round(x: Tensor) -> Tensor:
    return torch.round(x) - x.detach() + x 

def find_named_buffer(module, query):
    return next((b for n, b in module.named_buffers() if n == query), None) 

def _update_registered_buffer(
    module,
    buffer_name,
    state_dict_key,
    state_dict,
    policy="resize_if_empty",
    dtype=torch.int,
):
    new_size = state_dict[state_dict_key].size()
    registered_buf = find_named_buffer(module, buffer_name)
    if policy in ("resize_if_empty", "resize"): #resize
        if registered_buf is None:
            raise RuntimeError(f'buffer "{buffer_name}" was not registered')
        if policy == "resize" or registered_buf.numel() == 0:
            registered_buf.resize_(new_size)
    elif policy == "register":
        if registered_buf is not None:
            raise RuntimeError(f'buffer "{buffer_name}" was already registered')
        module.register_buffer(buffer_name, torch.empty(new_size, dtype=dtype).fill_(0))
    else:
        raise ValueError(f'Invalid policy "{policy}"')

def update_registered_buffers(
    module,
    module_name,
    buffer_names,
    state_dict,
    policy="resize_if_empty",
    dtype=torch.int,
):
    if not module:
        return
    valid_buffer_names = [n for n, _ in module.named_buffers()] 
    for buffer_name in buffer_names:
        if buffer_name not in valid_buffer_names:
            raise ValueError(f'Invalid buffer name "{buffer_name}"')
    for buffer_name in buffer_names: 
        _update_registered_buffer(
            module,
            buffer_name,
            f"{module_name}.{buffer_name}", 
            state_dict,
            policy,
            dtype,
        ) 

class ResidualBottleneckBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        mid_ch = min(in_ch, out_ch) // 2
        self.conv1 = conv1x1(in_ch, mid_ch)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(mid_ch, mid_ch)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = conv1x1(mid_ch, out_ch)
        self.skip = conv1x1(in_ch, out_ch) if in_ch != out_ch else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        identity = self.skip(x)
        out = x
        out = self.conv1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.conv3(out)
        return out + identity

class ResidualBottleneckBlockWithStride(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv =  conv(in_ch, out_ch, kernel_size=5, stride=2)
        self.res1 = ResidualBottleneckBlock(out_ch, out_ch)
        self.res2 = ResidualBottleneckBlock(out_ch, out_ch)
        self.res3 = ResidualBottleneckBlock(out_ch, out_ch)

    def forward(self, x: Tensor) -> Tensor:
        out = self.conv(x)
        out = self.res1(out)
        out = self.res2(out)
        out = self.res3(out)
        return out

class ResidualBottleneckBlockWithUpsample(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.res1 = ResidualBottleneckBlock(in_ch, in_ch)
        self.res2 = ResidualBottleneckBlock(in_ch, in_ch)
        self.res3 = ResidualBottleneckBlock(in_ch, in_ch)
        self.conv = deconv(in_ch, out_ch, kernel_size=5, stride=2)

    def forward(self, x: Tensor) -> Tensor:
        out = self.res1(x)
        out = self.res2(out)
        out = self.res3(out)
        out = self.conv(out)

        return out

class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=True, groups=dim)

    def forward(self, x):
        x = rearrange(x, 'b h w c -> b c h w')
        x = self.dwconv(x)
        x = rearrange(x, 'b c h w -> b h w c')
        return x

class Scale(nn.Module):
    def __init__(self, dim, init_value=1.0, trainable=True):
        super().__init__()
        self.scale = nn.Parameter(init_value * torch.ones(dim), requires_grad=trainable)

    def forward(self, x):
        return x * self.scale

class ConvolutionalGLU(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        hidden_features = hidden_features//2
        self.fc1 = nn.Linear(in_features, hidden_features * 2)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        x, v = self.fc1(x).chunk(2, dim=-1)
        x = self.act(self.dwconv(x)) * v
        x = self.fc2(x)
        return x

class SpatialAttentionModule(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttentionModule, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class ConvWithDW(nn.Module):
    def __init__(self, input_dim=320, output_dim=320):
        super(ConvWithDW, self).__init__()
        self.in_trans = nn.Conv2d(input_dim, output_dim, kernel_size=1, padding=0, stride=1, bias=True)
        self.act1 = nn.GELU()
        self.dw_conv = nn.Conv2d(output_dim, output_dim, kernel_size=3, padding=1, stride=1, groups=output_dim, bias=True)
        self.act2 = nn.GELU()
        self.out_trans = nn.Conv2d(output_dim, output_dim, kernel_size=1, padding=0, stride=1, bias=True)

    def forward(self, x):
        x = self.in_trans(x)
        x = self.act1(x)
        x = self.dw_conv(x)
        x = self.act2(x)
        x = self.out_trans(x)
        return x

class DenseBlock(nn.Module):
    def __init__(self, dim=320):
        super(DenseBlock, self).__init__()
        self.layer_num = 3      # m=3
        self.conv_layers = nn.ModuleList([
            nn.Sequential(
                nn.GELU(),
                ConvWithDW(dim, dim),
            ) for i in range(self.layer_num)  
        ])
        self.proj = nn.Conv2d(dim*(self.layer_num+1), dim, kernel_size=1, padding=0, stride=1, bias=True)

    def forward(self, x):
        outputs = [x]
        for i in range(self.layer_num):
            outputs.append(self.conv_layers[i](outputs[-1]))
        x = self.proj(torch.cat(outputs, dim=1))
        return x

class MultiScaleAggregation(nn.Module):
    def __init__(self, dim):
        super(MultiScaleAggregation, self).__init__()
        self.s = nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1, bias=True)
        self.spatial_atte = SpatialAttentionModule()
        self.dense = DenseBlock(dim)
        
    def forward(self, x):
        x = rearrange(x, 'b h w c -> b c h w')
        s = self.s(x)
        s_out = self.dense(s)
        x = s_out * self.spatial_atte(s_out)
        x = rearrange(x, 'b c h w -> b h w c')
        return x

class MutiScaleDictionaryCrossAttentionGLU(nn.Module):
    def __init__(self, input_dim, output_dim, mlp_rate=4, head_num=20, qkv_bias=True):
        super().__init__()
        dict_dim = 32 * head_num    
        self.head_num = head_num    
        self.scale = nn.Parameter(torch.ones(head_num, 1, 1))
        self.x_trans = nn.Linear(input_dim, dict_dim, bias=qkv_bias)
        self.ln_scale = nn.LayerNorm(dict_dim)
        self.msa = MultiScaleAggregation(dict_dim)
        self.lnx = nn.LayerNorm(dict_dim)
        self.q_trans = nn.Linear(dict_dim, dict_dim, bias=qkv_bias)
        self.dict_ln = nn.LayerNorm(dict_dim)
        self.k = nn.Linear(dict_dim,dict_dim, bias=qkv_bias)
        self.linear = nn.Linear(dict_dim, dict_dim, bias=qkv_bias)
        self.ln_mlp = nn.LayerNorm(dict_dim)
        self.mlp = ConvolutionalGLU(dict_dim, mlp_rate * dict_dim)
        self.output_trans = nn.Sequential(nn.Linear(dict_dim, output_dim))
        self.softmax = torch.nn.Softmax(dim=-1)
        self.res_scale_1 = Scale(dict_dim, init_value=1.0)
        self.res_scale_2 = Scale(dict_dim, init_value=1.0)
        self.res_scale_3 = Scale(dict_dim, init_value=1.0)

    def forward(self, x, dt):
        B, C, H, W = x.size() 
        x = rearrange(x, 'b c h w -> b h w c')
        x = self.x_trans(x)
        x = self.msa(self.ln_scale(x)) + self.res_scale_1(x)
        shortcut = x
        x = self.lnx(x)
        x = self.q_trans(x)
        x = rearrange(x, 'b h w c -> b c h w')
        q = rearrange(x, 'b (e c) h w -> b e (h w) c', e=self.head_num)
        dt = self.dict_ln(dt)
        k = self.k(dt)
        k = rearrange(k, 'b n (e c) -> b e n c', e=self.head_num)
        dt = rearrange(dt, 'b n (e c) -> b e n c', e=self.head_num)
        self.scale = self.scale.to(q.device)
        sim = torch.einsum('benc,bedc->bend', q, k)
        sim = sim * self.scale
        probs = self.softmax(sim)
        output = torch.einsum('bend,bedc->benc', probs, dt)
        output = rearrange(output, 'b e (h w) c -> b h w (e c) ', h = H, w = W)
        output = self.linear(output) + self.res_scale_2(shortcut)
        output = self.mlp(self.ln_mlp(output)) + self.res_scale_3(output)
        output = self.output_trans(output)
        output = rearrange(output, 'b h w c -> b c h w', )
        return output

class CrossSparseWindowAttention(nn.Module):
    def __init__(self, input_dim, output_dim, head_dim, window_size, num_global_tokens=2):
        super(CrossSparseWindowAttention, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.head_dim = head_dim 
        self.scale = self.head_dim ** -0.5 
        self.n_heads = input_dim // head_dim
        self.window_size = window_size
        self.embedding_layer = nn.Linear(self.input_dim, 3 * self.input_dim, bias=True)
        self.num_global_tokens = num_global_tokens
        self.global_tokens = nn.Parameter(torch.zeros(1, self.num_global_tokens, input_dim))
        trunc_normal_(self.global_tokens, std=.02)
        self.global_kv = nn.Linear(input_dim, input_dim * 2, bias=False)
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), self.n_heads)
        )
        trunc_normal_(self.relative_position_bias_table, std=.02)
        coords_h = torch.arange(window_size)
        coords_w = torch.arange(window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)
        self.linear = nn.Linear(self.input_dim, self.output_dim)
        self.register_buffer('global_alpha', torch.tensor(0.25))
        
    def forward(self, x):
        B, H, W, C = x.shape
        M = self.window_size
        nH = H // M
        nW = W // M
        x = x.view(B, nH, M, nW, M, C)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.view(B * nH * nW, M * M, C)
        num_windows = nH * nW
        qkv = self.embedding_layer(x)
        qkv = qkv.reshape(B * num_windows, M * M, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        if hasattr(F, 'scaled_dot_product_attention'):
            relative_position_bias = self.relative_position_bias_table[
                self.relative_position_index.view(-1)
            ].view(M * M, M * M, -1)
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
            relative_position_bias = relative_position_bias.unsqueeze(0)
            output_local = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=relative_position_bias.expand(B * num_windows, -1, -1, -1),
                dropout_p=0.0,
                scale=self.scale
            )
        else:
            q = q * self.scale
            sim_local = torch.einsum('bhpc,bhqc->bhpq', q, k)
            relative_position_bias = self.relative_position_bias_table[
                self.relative_position_index.view(-1)
            ].view(M * M, M * M, -1)
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
            sim_local = sim_local + relative_position_bias.unsqueeze(0)
            probs_local = sim_local.softmax(dim=-1)
            output_local = torch.einsum('bhij,bhjc->bhic', probs_local, v)
        x_mean = x.mean(dim=1, keepdim=True)
        global_tokens = self.global_tokens.expand(B * num_windows, -1, -1)
        global_tokens = global_tokens + x_mean
        global_kv = self.global_kv(global_tokens)
        global_kv = global_kv.reshape(B * num_windows, self.num_global_tokens, 2, self.n_heads, self.head_dim)
        global_kv = global_kv.permute(2, 0, 3, 1, 4)
        k_global, v_global = global_kv[0], global_kv[1]
        q = q * self.scale if not hasattr(F, 'scaled_dot_product_attention') else q
        sim_global = torch.einsum('bhpc,bhgc->bhpg', q, k_global)
        probs_global = sim_global.softmax(dim=-1)
        output_global = torch.einsum('bhpg,bhgc->bhpc', probs_global, v_global)
        output = (1 - self.global_alpha) * output_local + self.global_alpha * output_global
        output = output.transpose(1, 2).reshape(B * num_windows, M * M, C)
        output = self.linear(output)
        output = output.view(B, nH, nW, M, M, C)
        output = output.permute(0, 1, 3, 2, 4, 5).contiguous()
        output = output.view(B, H, W, C)
        return output
    
    def relative_embedding(self):
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(self.window_size * self.window_size, self.window_size * self.window_size, -1)
        return relative_position_bias.permute(2, 0, 1).contiguous()

class SpatialAttentionLayer(nn.Module):
    def __init__(self, input_dim, output_dim, head_dim, window_size, drop_path, input_resolution=None):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.ln1 = nn.LayerNorm(input_dim)
        self.msa = CrossSparseWindowAttention(input_dim, input_dim, head_dim, window_size)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.ln2 = nn.LayerNorm(input_dim)
        self.mlp = ConvolutionalGLU(input_dim, input_dim * 4)
        self.res_scale_1 = Scale(input_dim, init_value=1.0)
        self.res_scale_2 = Scale(input_dim, init_value=1.0)

    def forward(self, x):
        x = self.res_scale_1(x) + self.drop_path(self.msa(self.ln1(x)))
        x = self.res_scale_2(x) + self.drop_path(self.mlp(self.ln2(x)))
        return x


class SpatialAttentionBlock(nn.Module):
    def __init__(self, input_dim, output_dim, head_dim, window_size, drop_path, block=SpatialAttentionLayer, block_num=2, **kwargs) -> None:
        super().__init__()
        self.layers = nn.ModuleList()
        self.block_num = block_num
        for i in range(block_num):
            self.layers.append(block(input_dim, input_dim, head_dim, window_size, drop_path))
        self.conv = conv(input_dim, output_dim, 3, 1)
        self.window_size = window_size

    def forward(self, x):
        resize = False
        padding_row = 0
        padding_col = 0
        if (x.size(-1) <= self.window_size) or (x.size(-2) <= self.window_size):
            resize = True
            padding_row = (self.window_size - x.size(-2)) // 2
            padding_col = (self.window_size - x.size(-1)) // 2
            x = F.pad(x, (padding_col, padding_col + 1, padding_row, padding_row + 1))
        trans_x = Rearrange('b c h w -> b h w c')(x)
        for i in range(self.block_num):
            trans_x = self.layers[i](trans_x)
        trans_x = Rearrange('b h w c -> b c h w')(trans_x)
        trans_x = self.conv(trans_x)
        out = trans_x + x
        if resize:
            out = F.pad(out, (-padding_col, -padding_col - 1, -padding_row, -padding_row - 1))
        return out

