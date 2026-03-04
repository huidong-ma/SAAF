from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from compressai.ans import BufferedRansEncoder, RansDecoder
from compressai.models import CompressionModel
from compressai.layers import (
    AttentionBlock,
    ResidualBlock,
    ResidualBlockUpsample,
    ResidualBlockWithStride,
    conv3x3,
    subpel_conv3x3,
)

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
import pywt
from torch.autograd import Function

from models.modules import *

class OLP(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True):
        super(OLP, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=bias)
        self.in_dim = in_dim
        self.out_dim = out_dim
        eye_dim = min(in_dim, out_dim)
        self.register_buffer('identity_matrix', torch.eye(eye_dim))

    def loss(self):
        kernel_matrix = self.linear.weight
        if self.in_dim > self.out_dim:
            gram_matrix = torch.mm(kernel_matrix, kernel_matrix.t())
        else:
            gram_matrix = torch.mm(kernel_matrix.t(), kernel_matrix)
        loss_ortho = F.mse_loss(gram_matrix, self.identity_matrix)
        
        return loss_ortho
    
    def forward(self, x):
        return self.linear(x)

class AdaptiveFrequencyBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.olp = OLP(in_dim, out_dim)
        mid_dim = max(in_dim // 4, 4)
        self.freq_attn = nn.Sequential(
            nn.Conv2d(in_dim, mid_dim, 1),
            nn.GELU(),
            nn.Conv2d(mid_dim, 4, 1),
            nn.Softmax(dim=1)
        )
        self.freq_weights = nn.Parameter(torch.tensor([
            1.0,  # LL
            0.8,  # LH
            0.8,  # HL
            0.6,  # HH
        ]))
        
    def forward(self, x):
        b, _, h, w = x.shape
        freq_attn = self.freq_attn(x)
        freq_w = torch.exp(self.freq_weights).view(1, 4, 1, 1)
        x_freq = x.unsqueeze(1) * freq_attn.unsqueeze(2) * freq_w.unsqueeze(2)
        x_freq = x_freq.sum(dim=1)
        x_flat = x_freq.flatten(2).permute(0, 2, 1)
        x_out = self.olp(x_flat)
        return x_out.permute(0, 2, 1).view(b, -1, h, w)

class InverseAdaptiveFrequencyBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.olp = OLP(in_dim, out_dim)
        mid_dim = max(in_dim // 4, 4)
        self.freq_attn = nn.Sequential(
            nn.Conv2d(in_dim, mid_dim, 1),
            nn.GELU(),
            nn.Conv2d(mid_dim, 4, 1),
            nn.Softmax(dim=1)
        )
    
    def forward(self, x):
        b, _, h, w = x.shape
        freq_weights = self.freq_attn(x)
        x_flat = x.flatten(2).permute(0, 2, 1)
        x_out = self.olp(x_flat)
        x_out = x_out.permute(0, 2, 1).view(b, -1, h, w)
        freq_enhanced = x_out * freq_weights.mean(dim=1, keepdim=True)
        return x_out + 0.1 * freq_enhanced

class DenoisingAsRegularizer(nn.Module):
    def __init__(self, latent_dim=320):
        super().__init__()
        self.latent_dim = latent_dim
        self.time_embed = nn.Sequential(
            nn.Linear(1, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, latent_dim)
        )
        self.noise_predictor = nn.Sequential(
            nn.Conv2d(latent_dim, latent_dim, 3, padding=1),
            nn.GroupNorm(8, latent_dim),
            nn.SiLU(),
            ResidualBottleneckBlock(latent_dim, latent_dim),
            ResidualBottleneckBlock(latent_dim, latent_dim),
            nn.Conv2d(latent_dim, latent_dim, 3, padding=1),
            nn.GroupNorm(8, latent_dim),
            nn.SiLU(),
            nn.Conv2d(latent_dim, latent_dim, 1)
        )
        self.condition_encoder = nn.Sequential(
            nn.Conv2d(192, 256, 1),
            nn.GroupNorm(8, 256),
            nn.GELU(),
            nn.Conv2d(256, 320, 3, padding=1),
            nn.Dropout(0.1),
            nn.GELU()
        )
        
    def forward(self, y, z_hat):
        B, C, H, W = y.size()
        condition = self.condition_encoder(z_hat)
        condition = F.interpolate(condition, size=(H, W), mode='bilinear', align_corners=False)
        t = torch.rand(B, 1, device=y.device)
        noise = torch.randn_like(y)
        y_noisy = y + noise * t.view(B, 1, 1, 1)
        t_emb = self.time_embed(t).view(B, C, 1, 1)
        pred_noise = self.noise_predictor(y_noisy + t_emb + condition)
        loss = F.mse_loss(pred_noise, noise)
        return loss

class SLIC(CompressionModel):
    def __init__(self, head_dim=[8, 16, 32, 32, 16, 8], drop_path_rate=0, N=192, M=320, num_slices=5, max_support_slices=5, **kwargs):
        super().__init__() 
        self.head_dim = head_dim
        self.window_size = 8
        self.num_slices = num_slices
        self.max_support_slices = max_support_slices
        dim = N 
        self.M = M
    
        input_image_channel = 3
        output_image_channel = 3
        feature_dim = [96, 144, 256]
        basic_block = [SpatialAttentionLayer for _ in range(3)]
        swin_block = [SpatialAttentionBlock for _ in range(3)]
        hyperprior_block = SpatialAttentionLayer
        hyperprior_stage = SpatialAttentionBlock
        block_num = [1, 2, 12]
        dict_num = 128
        dict_head_num = 20
        dict_dim = 32 * dict_head_num
        self.dt = nn.Parameter(torch.randn([dict_num, dict_dim]), requires_grad=True)
        prior_dim = M
        mlp_rate = 4
        qkv_bias = True
        self.dt_cross_attention = nn.ModuleList(
            MutiScaleDictionaryCrossAttentionGLU(
                input_dim=M*2+(M//self.num_slices)*min(i, 5), 
                output_dim=M, 
                head_num=dict_head_num, 
                mlp_rate=mlp_rate, 
                qkv_bias=qkv_bias
            ) for i in range(num_slices)
        )
        self.diffusion_prior = DenoisingAsRegularizer(latent_dim=M)
        self.m_down1 = [
            swin_block[0](
                input_dim=feature_dim[0], 
                output_dim=feature_dim[0], 
                head_dim=self.head_dim[0], 
                window_size=self.window_size, 
                drop_path=0,
                block_num=block_num[0],
                block=basic_block[0]
            ),
            ResidualBottleneckBlockWithStride(feature_dim[0], feature_dim[1])
        ]
        self.m_down2 = [
            swin_block[1](
                input_dim=feature_dim[1], 
                output_dim=feature_dim[1], 
                head_dim=self.head_dim[1], 
                window_size=self.window_size, 
                drop_path=0,
                block_num=block_num[1],
                block=basic_block[1]
            ),
            ResidualBottleneckBlockWithStride(feature_dim[1], feature_dim[2])
        ]
        self.m_down3 = [
            swin_block[2](
                input_dim=feature_dim[2], 
                output_dim=feature_dim[2], 
                head_dim=self.head_dim[2], 
                window_size=self.window_size, 
                drop_path=0,
                block_num=block_num[2],
                block=basic_block[2]
            ),
            conv(feature_dim[2], M, kernel_size=5, stride=2)
        ]
        self.g_a = nn.Sequential(
            ResidualBottleneckBlockWithStride(input_image_channel, feature_dim[0]),
            *self.m_down1,
            *self.m_down2,
            *self.m_down3
        )
        self.aux_enc = nn.ModuleList([
            AdaptiveFrequencyBlock(input_image_channel, feature_dim[0]),
            AdaptiveFrequencyBlock(feature_dim[0], feature_dim[1]),
            AdaptiveFrequencyBlock(feature_dim[1], feature_dim[2]),
            AdaptiveFrequencyBlock(feature_dim[2], M),
        ])
        self.m_up1 = [
            swin_block[2](
                input_dim=feature_dim[2], 
                output_dim=feature_dim[2], 
                head_dim=self.head_dim[3], 
                window_size=self.window_size, 
                drop_path=0,
                block_num=block_num[2],
                block=basic_block[2]
            ),
            ResidualBottleneckBlockWithUpsample(feature_dim[2], feature_dim[1])
        ]
        self.m_up2 = [
            swin_block[1](
                input_dim=feature_dim[1], 
                output_dim=feature_dim[1], 
                head_dim=self.head_dim[4], 
                window_size=self.window_size, 
                drop_path=0,
                block_num=block_num[1],
                block=basic_block[1]
            ),
            ResidualBottleneckBlockWithUpsample(feature_dim[1], feature_dim[0])
        ]
        self.m_up3 = [
            swin_block[0](
                input_dim=feature_dim[0], 
                output_dim=feature_dim[0], 
                head_dim=self.head_dim[5], 
                window_size=self.window_size, 
                drop_path=0,
                block_num=block_num[0],
                block=basic_block[0]
            ),
            ResidualBottleneckBlockWithUpsample(feature_dim[0], output_image_channel)
        ]
        self.g_s = nn.Sequential(
            deconv(M, feature_dim[2], kernel_size=5, stride=2),
            *self.m_up1,
            *self.m_up2,
            *self.m_up3
        )
        self.aux_dec = nn.ModuleList([
            InverseAdaptiveFrequencyBlock(M, feature_dim[2]),
            InverseAdaptiveFrequencyBlock(feature_dim[2], feature_dim[1]),
            InverseAdaptiveFrequencyBlock(feature_dim[1], feature_dim[0]),
            InverseAdaptiveFrequencyBlock(feature_dim[0], output_image_channel),
        ])
        self.ha_down = [
            hyperprior_stage(
                input_dim=N, 
                output_dim=N, 
                head_dim=32, 
                window_size=4, 
                drop_path=0,
                block_num=1,
                block=hyperprior_block
            ),
            conv(N, 192, kernel_size=3, stride=2)
        ]
        self.h_a = nn.Sequential(
            ResidualBottleneckBlockWithStride(M, N),
            *self.ha_down 
        )
        self.hs_up1 = [
            hyperprior_stage(
                input_dim=N, 
                output_dim=N, 
                head_dim=32, 
                window_size=4, 
                drop_path=0,
                block_num=1,
                block=hyperprior_block
            ),
            ResidualBottleneckBlockWithUpsample(N, M)
        ] 
        self.h_z_s1 = nn.Sequential(
            deconv(192, N, kernel_size=3, stride=2),
            *self.hs_up1
        )
        self.hs_up2 = [
            hyperprior_stage(
                input_dim=N, 
                output_dim=N, 
                head_dim=32, 
                window_size=4, 
                drop_path=0,
                block_num=1,
                block=hyperprior_block
            ),
            ResidualBottleneckBlockWithUpsample(N, M)
        ]
        self.h_z_s2 = nn.Sequential(
            deconv(192, N, kernel_size=3, stride=2),
            *self.hs_up2
        )
        self.cc_mean_transforms = nn.ModuleList(
            nn.Sequential(
                conv(M*2 + (M//self.num_slices)*min(i, 5) + prior_dim, 224, stride=1, kernel_size=3),
                nn.GELU(),
                conv(224, 128, stride=1, kernel_size=3),
                nn.GELU(),
                conv(128, M//self.num_slices, stride=1, kernel_size=3),
            ) for i in range(self.num_slices) 
        )
        self.cc_scale_transforms = nn.ModuleList(
            nn.Sequential(
                conv(M*2 + (M//self.num_slices)*min(i, 5) + prior_dim, 224, stride=1, kernel_size=3),
                nn.GELU(),
                conv(224, 128, stride=1, kernel_size=3),
                nn.GELU(),
                conv(128, M//self.num_slices, stride=1, kernel_size=3),
            ) for i in range(self.num_slices)
        )
        self.lrp_transforms = nn.ModuleList(
            nn.Sequential(
                conv(M*2 + (M//self.num_slices)*min(i+1, 6) + prior_dim, 224, stride=1, kernel_size=3),
                nn.GELU(),
                conv(224, 128, stride=1, kernel_size=3),
                nn.GELU(),
                conv(128, M//self.num_slices, stride=1, kernel_size=3),
            ) for i in range(self.num_slices)
        )
        self.entropy_bottleneck = EntropyBottleneck(192)
        self.gaussian_conditional = GaussianConditional(None)

    def forward(self, x):
        b = x.size(0)
        dt = self.dt.repeat([b, 1, 1])
        y_main = x
        y_aux = x
        y_main = self.g_a[0](y_main)
        y_aux = self.aux_enc[0](y_aux)
        if y_aux.shape[2:] != y_main.shape[2:]:
            y_aux = F.interpolate(y_aux, size=y_main.shape[2:], mode='bilinear', align_corners=False)
        y_main = y_main + y_aux
        for layer in self.m_down1:
            y_main = layer(y_main)
        y_aux = self.aux_enc[1](y_aux)
        if y_aux.shape[2:] != y_main.shape[2:]:
            y_aux = F.interpolate(y_aux, size=y_main.shape[2:], mode='bilinear', align_corners=False)
        y_main = y_main + y_aux
        for layer in self.m_down2:
            y_main = layer(y_main)
        y_aux = self.aux_enc[2](y_aux)
        if y_aux.shape[2:] != y_main.shape[2:]:
            y_aux = F.interpolate(y_aux, size=y_main.shape[2:], mode='bilinear', align_corners=False)
        y_main = y_main + y_aux
        for layer in self.m_down3:
            y_main = layer(y_main)
        y_aux = self.aux_enc[3](y_aux)
        if y_aux.shape[2:] != y_main.shape[2:]:
            y_aux = F.interpolate(y_aux, size=y_main.shape[2:], mode='bilinear', align_corners=False)
        y = y_main + y_aux
        y_shape = y.shape[2:]
        z = self.h_a(y)
        _, z_likelihoods = self.entropy_bottleneck(z)
        z_offset = self.entropy_bottleneck._get_medians()
        z_tmp = z - z_offset
        z_hat = ste_round(z_tmp) + z_offset
        latent_scales = self.h_z_s1(z_hat)
        latent_means = self.h_z_s2(z_hat)
        diff_loss = torch.tensor(0.0, device=x.device)
        if self.training:
            diff_loss = self.diffusion_prior(y, z_hat)
    
        y_slices = y.chunk(self.num_slices, 1)
        y_hat_slices = []
        y_likelihood = []
        mu_list = []
        scale_list = []
        
        for slice_index, y_slice in enumerate(y_slices):
            support_slices = (y_hat_slices if self.max_support_slices < 0 else y_hat_slices[:self.max_support_slices])
            query = torch.cat([latent_scales, latent_means] + support_slices, dim=1)
            dict_info = self.dt_cross_attention[slice_index](query, dt)
            support = torch.cat([query, dict_info], dim=1)
            mu = self.cc_mean_transforms[slice_index](support)
            mu = mu[:, :, :y_shape[0], :y_shape[1]]
            mu_list.append(mu)
            scale = self.cc_scale_transforms[slice_index](support)
            scale = scale[:, :, :y_shape[0], :y_shape[1]]
            scale_list.append(scale)
            _, y_slice_likelihood = self.gaussian_conditional(y_slice, scale, mu)
            y_likelihood.append(y_slice_likelihood)
            y_hat_slice = ste_round(y_slice - mu) + mu
            lrp_support = torch.cat([support, y_hat_slice], dim=1)
            lrp = self.lrp_transforms[slice_index](lrp_support)
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice += lrp
            y_hat_slices.append(y_hat_slice)
        y_hat = torch.cat(y_hat_slices, dim=1)
        means = torch.cat(mu_list, dim=1)
        scales = torch.cat(scale_list, dim=1)
        y_likelihoods = torch.cat(y_likelihood, dim=1)
        x_hat_main = y_hat
        x_hat_aux = y_hat
        x_hat_main = self.g_s[0](x_hat_main)
        x_hat_aux = self.aux_dec[0](x_hat_aux)
        if x_hat_aux.shape[2:] != x_hat_main.shape[2:]:
            x_hat_aux = F.interpolate(x_hat_aux, size=x_hat_main.shape[2:], mode='bilinear', align_corners=False)
        x_hat_main = x_hat_main + x_hat_aux
        for layer in self.m_up1:
            x_hat_main = layer(x_hat_main)
        x_hat_aux = self.aux_dec[1](x_hat_aux)
        if x_hat_aux.shape[2:] != x_hat_main.shape[2:]:
            x_hat_aux = F.interpolate(x_hat_aux, size=x_hat_main.shape[2:], mode='bilinear', align_corners=False)
        x_hat_main = x_hat_main + x_hat_aux
        for layer in self.m_up2:
            x_hat_main = layer(x_hat_main)
        x_hat_aux = self.aux_dec[2](x_hat_aux)
        if x_hat_aux.shape[2:] != x_hat_main.shape[2:]:
            x_hat_aux = F.interpolate(x_hat_aux, size=x_hat_main.shape[2:], mode='bilinear', align_corners=False)
        x_hat_main = x_hat_main + x_hat_aux
        for layer in self.m_up3:
            x_hat_main = layer(x_hat_main)
        x_hat_aux = self.aux_dec[3](x_hat_aux)
        if x_hat_aux.shape[2:] != x_hat_main.shape[2:]:
            x_hat_aux = F.interpolate(x_hat_aux, size=x_hat_main.shape[2:], mode='bilinear', align_corners=False)
        x_hat = x_hat_main + x_hat_aux
        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
            "para": {"means": means, "scales": scales, "y": y},
            "diffusion_loss": diff_loss
        }

    def ortho_loss(self):
        if not self.training:
            return torch.tensor(0.0)
        ortho_loss = sum(
            m.loss() for m in self.modules() if isinstance(m, OLP)
        )
        return ortho_loss

    def update(self, scale_table=None, force=False):
        if scale_table is None:
            scale_table = get_scale_table()
        updated = self.gaussian_conditional.update_scale_table(scale_table, force=force)
        updated |= super().update(force=force)
        return updated

    def load_state_dict(self, state_dict, strict=True):
        update_registered_buffers(
            self.gaussian_conditional,
            "gaussian_conditional",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
            state_dict,
        )
        super().load_state_dict(state_dict, strict=strict)

    @classmethod
    def from_state_dict(cls, state_dict):
        N = state_dict["g_a.0.weight"].size(0)
        M = state_dict["g_a.6.weight"].size(0)
        net = cls(N=N, M=M)
        net.load_state_dict(state_dict)
        return net

    def compress(self, x):
        b = x.size(0)
        dt = self.dt.repeat([b, 1, 1])
        y = self._encode(x)
        y_shape = y.shape[2:]
        z = self.h_a(y)
        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])
        latent_scales = self.h_z_s1(z_hat)
        latent_means = self.h_z_s2(z_hat)
        y_slices = y.chunk(self.num_slices, 1)
        y_hat_slices = []
        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        offsets = self.gaussian_conditional.offset.reshape(-1).int().tolist()
        encoder = BufferedRansEncoder()
        symbols_list = []
        indexes_list = []
        y_strings = []

        for slice_index, y_slice in enumerate(y_slices):
            support_slices = (y_hat_slices if self.max_support_slices < 0 else y_hat_slices[:self.max_support_slices])
            query = torch.cat([latent_scales, latent_means] + support_slices, dim=1)
            dict_info = self.dt_cross_attention[slice_index](query, dt)
            support = torch.cat([query, dict_info], dim=1)
            mu = self.cc_mean_transforms[slice_index](support)
            mu = mu[:, :, :y_shape[0], :y_shape[1]]
            scale = self.cc_scale_transforms[slice_index](support)
            scale = scale[:, :, :y_shape[0], :y_shape[1]]
            index = self.gaussian_conditional.build_indexes(scale)
            y_q_slice = self.gaussian_conditional.quantize(y_slice, "symbols", mu)
            y_hat_slice = y_q_slice + mu
            symbols_list.extend(y_q_slice.reshape(-1).tolist())
            indexes_list.extend(index.reshape(-1).tolist())
            lrp_support = torch.cat([support, y_hat_slice], dim=1)
            lrp = self.lrp_transforms[slice_index](lrp_support)
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice += lrp
            y_hat_slices.append(y_hat_slice)
        encoder.encode_with_indexes(symbols_list, indexes_list, cdf, cdf_lengths, offsets)
        y_string = encoder.flush()
        y_strings.append(y_string)
        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}

    def decompress(self, strings, shape):
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        latent_scales = self.h_z_s1(z_hat)
        latent_means = self.h_z_s2(z_hat)
        b = z_hat.size(0)
        dt = self.dt.repeat([b, 1, 1])
        y_shape = [z_hat.shape[2] * 4, z_hat.shape[3] * 4]
        y_string = strings[0][0]
        y_hat_slices = []
        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        offsets = self.gaussian_conditional.offset.reshape(-1).int().tolist()
        decoder = RansDecoder()
        decoder.set_stream(y_string)

        for slice_index in range(self.num_slices):
            support_slices = (y_hat_slices if self.max_support_slices < 0 else y_hat_slices[:self.max_support_slices])
            query = torch.cat([latent_scales, latent_means] + support_slices, dim=1)
            dict_info = self.dt_cross_attention[slice_index](query, dt)
            support = torch.cat([query, dict_info], dim=1)
            mu = self.cc_mean_transforms[slice_index](support)
            mu = mu[:, :, :y_shape[0], :y_shape[1]]
            scale = self.cc_scale_transforms[slice_index](support)
            scale = scale[:, :, :y_shape[0], :y_shape[1]]
            index = self.gaussian_conditional.build_indexes(scale)
            rv = decoder.decode_stream(index.reshape(-1).tolist(), cdf, cdf_lengths, offsets)
            rv = torch.Tensor(rv).reshape(1, -1, y_shape[0], y_shape[1])
            y_hat_slice = self.gaussian_conditional.dequantize(rv, mu)
            lrp_support = torch.cat([support, y_hat_slice], dim=1)
            lrp = self.lrp_transforms[slice_index](lrp_support)
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice += lrp
            y_hat_slices.append(y_hat_slice)
        y_hat = torch.cat(y_hat_slices, dim=1)
        x_hat = self._decode(y_hat).clamp_(0, 1)
        return {"x_hat": x_hat}

    def _encode(self, x):
        y_main = x
        y_aux = x
        y_main = self.g_a[0](y_main)
        y_aux = self.aux_enc[0](y_aux)
        if y_aux.shape[2:] != y_main.shape[2:]:
            y_aux = F.interpolate(y_aux, size=y_main.shape[2:], mode='bilinear', align_corners=False)
        y_main = y_main + y_aux
        for layer in self.m_down1:
            y_main = layer(y_main)
        y_aux = self.aux_enc[1](y_aux)
        if y_aux.shape[2:] != y_main.shape[2:]:
            y_aux = F.interpolate(y_aux, size=y_main.shape[2:], mode='bilinear', align_corners=False)
        y_main = y_main + y_aux
        for layer in self.m_down2:
            y_main = layer(y_main)
        y_aux = self.aux_enc[2](y_aux)
        if y_aux.shape[2:] != y_main.shape[2:]:
            y_aux = F.interpolate(y_aux, size=y_main.shape[2:], mode='bilinear', align_corners=False)
        y_main = y_main + y_aux
        for layer in self.m_down3:
            y_main = layer(y_main)
        y_aux = self.aux_enc[3](y_aux)
        if y_aux.shape[2:] != y_main.shape[2:]:
            y_aux = F.interpolate(y_aux, size=y_main.shape[2:], mode='bilinear', align_corners=False)
        y = y_main + y_aux
        return y

    def _decode(self, y_hat):
        x_hat_main = y_hat
        x_hat_aux = y_hat
        x_hat_main = self.g_s[0](x_hat_main)
        x_hat_aux = self.aux_dec[0](x_hat_aux)
        if x_hat_aux.shape[2:] != x_hat_main.shape[2:]:
            x_hat_aux = F.interpolate(x_hat_aux, size=x_hat_main.shape[2:], mode='bilinear', align_corners=False)
        x_hat_main = x_hat_main + x_hat_aux
        for layer in self.m_up1:
            x_hat_main = layer(x_hat_main)
        x_hat_aux = self.aux_dec[1](x_hat_aux)
        if x_hat_aux.shape[2:] != x_hat_main.shape[2:]:
            x_hat_aux = F.interpolate(x_hat_aux, size=x_hat_main.shape[2:], mode='bilinear', align_corners=False)
        x_hat_main = x_hat_main + x_hat_aux
        for layer in self.m_up2:
            x_hat_main = layer(x_hat_main)
        x_hat_aux = self.aux_dec[2](x_hat_aux)
        if x_hat_aux.shape[2:] != x_hat_main.shape[2:]:
            x_hat_aux = F.interpolate(x_hat_aux, size=x_hat_main.shape[2:], mode='bilinear', align_corners=False)
        x_hat_main = x_hat_main + x_hat_aux
        for layer in self.m_up3:
            x_hat_main = layer(x_hat_main)
        x_hat_aux = self.aux_dec[3](x_hat_aux)
        if x_hat_aux.shape[2:] != x_hat_main.shape[2:]:
            x_hat_aux = F.interpolate(x_hat_aux, size=x_hat_main.shape[2:], mode='bilinear', align_corners=False)
        x_hat = x_hat_main + x_hat_aux
        return x_hat

    def _likelihood(self, inputs, scales, means=None):
        half = float(0.5)
        if means is not None:
            values = inputs - means
        else:
            values = inputs
        scales = torch.max(scales, torch.tensor(0.11))
        values = torch.abs(values)
        upper = self._standardized_cumulative((half - values) / scales)
        lower = self._standardized_cumulative((-half - values) / scales)
        likelihood = upper - lower
        return likelihood

    def _standardized_cumulative(self, inputs):
        half = float(0.5)
        const = float(-(2 ** -0.5))
        return half * torch.erfc(const * inputs)