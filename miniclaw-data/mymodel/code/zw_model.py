"""
===========================================================
STD-Net
Semantic-Topological Decoupling Network

Main Modifications:
-----------------------------------------------------------
1. Remove TAB (Token Aggregation Block)
2. Replace encoder with MSDRB
3. Keep ETA in shallow-to-middle encoder stages
4. Adopt asymmetric dual decoders
- Mask Decoder: skip-based semantic recovery
- Boundary Decoder: topology-aware progressive fusion
5. Replace single GMRM with multi-stage TAGM
(Topology-Aware Guidance Module)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import DeformConv2d


# =========================================================
# Basic Conv
# =========================================================
class Conv3BN(nn.Module):
    def __init__(self, in_, out, bn=True):
        super().__init__()

        self.conv = nn.Conv2d(in_, out, 3, padding=1)
        self.bn = nn.BatchNorm2d(out) if bn else nn.Identity()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return self.relu(x)


# =========================================================
# Multi-Scale Deformable Convolution
# =========================================================
class MSDConv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        if in_channels <= 32:
            reduction = 2
        elif in_channels <= 128:
            reduction = 4
        else:
            reduction = 8

        reduced_channels = max(1, in_channels // reduction)

        self.conv1 = nn.Conv2d(in_channels, out_channels, 1)

        self.conv3 = nn.Conv2d(
            in_channels,
            out_channels,
            3,
            padding=1
        )

        self.offset_conv = nn.Conv2d(
            in_channels,
            18,
            kernel_size=3,
            padding=1
        )

        self.dcn = DeformConv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1
        )

        self.global_pool = nn.AdaptiveAvgPool2d(1)

        self.w_gen = nn.Sequential(
            nn.Conv2d(
                in_channels,
                reduced_channels,
                1,
                bias=False
            ),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                reduced_channels,
                3,
                1,
                bias=False
            ),
            nn.Softmax(dim=1)
        )

    def forward(self, x):

        B, _, H, W = x.size()

        weights = self.global_pool(x)

        weights = self.w_gen(weights).view(
            B, 3, 1, 1, 1
        )

        out1 = self.conv1(x).unsqueeze(1)
        out2 = self.conv3(x).unsqueeze(1)

        offset = self.offset_conv(x)
        out3 = self.dcn(x, offset).unsqueeze(1)

        branches = torch.cat(
            [out1, out2, out3],
            dim=1
        )

        fused = (weights * branches).sum(dim=1)

        return fused


# =========================================================
# MSDConv + BN
# =========================================================
class MSDConvBN(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.msd = MSDConv(in_channels, out_channels)

        self.bn = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):

        x = self.msd(x)

        x = self.bn(x)

        return self.relu(x)


# =========================================================
# MSD Residual Block
# =========================================================
class MSDResBlock(nn.Module):

    def __init__(self, in_, out, dropout_rate=0.2):
        super().__init__()

        self.l1 = MSDConvBN(in_, out)

        self.l2 = MSDConvBN(out, out)

        self.dropout = nn.Dropout2d(dropout_rate)

        self.shortcut = (
            nn.Conv2d(in_, out, 1)
            if in_ != out
            else nn.Identity()
        )

    def forward(self, x):

        residual = self.shortcut(x)

        x = self.l1(x)

        x = self.dropout(x)

        x = self.l2(x)

        return x + residual


# =========================================================
# ETA
# =========================================================
class BasicConv(nn.Module):

    def __init__(
            self,
            in_,
            out_,
            kernel_size,
            stride=1,
            padding=0,
            relu=True,
            bn=True,
            bias=False
    ):
        super().__init__()

        self.conv = nn.Conv2d(
            in_,
            out_,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=bias
        )

        self.bn = (
            nn.BatchNorm2d(out_)
            if bn else None
        )

        self.relu = (
            nn.ReLU(inplace=True)
            if relu else None
        )

    def forward(self, x):

        x = self.conv(x)

        if self.bn is not None:
            x = self.bn(x)

        if self.relu is not None:
            x = self.relu(x)

        return x


class ChannelPool(nn.Module):

    def forward(self, x):

        return torch.cat(
            (
                torch.max(x, 1)[0].unsqueeze(1),
                torch.mean(x, 1).unsqueeze(1)
            ),
            dim=1
        )


class SpatialGate(nn.Module):

    def __init__(self):
        super().__init__()

        self.compress = ChannelPool()

        self.spatial = BasicConv(
            2,
            1,
            kernel_size=7,
            padding=3,
            relu=False
        )

    def forward(self, x):

        x_compress = self.compress(x)

        x_out = self.spatial(x_compress)

        scale = torch.sigmoid_(x_out)

        return x * scale


class EnhancedTripletAttention(nn.Module):

    def __init__(self, in_channels, reduction_ratio=16):
        super().__init__()

        self.ChannelGateH = SpatialGate()

        self.ChannelGateW = SpatialGate()

        self.SpatialGate = SpatialGate()

        self.adaptive_pool = nn.AdaptiveAvgPool2d(1)

        self.channel_fc = nn.Sequential(
            nn.Conv2d(
                in_channels,
                in_channels // reduction_ratio,
                1
            ),

            nn.ReLU(inplace=True),

            nn.Conv2d(
                in_channels // reduction_ratio,
                in_channels,
                1
            ),

            nn.Sigmoid()
        )

    def forward(self, x):

        x_perm1 = x.permute(0, 2, 1, 3).contiguous()

        x_out1 = self.ChannelGateH(x_perm1)

        x_out11 = x_out1.permute(
            0, 2, 1, 3
        ).contiguous()

        x_perm2 = x.permute(
            0, 3, 2, 1
        ).contiguous()

        x_out2 = self.ChannelGateW(x_perm2)

        x_out21 = x_out2.permute(
            0, 3, 2, 1
        ).contiguous()

        pooled = self.adaptive_pool(x)

        channel = self.channel_fc(pooled)

        x_out = self.SpatialGate(x)

        x_out = (
                x_out +
                x_out11 +
                x_out21
        ) / 3

        out = (channel + x_out) / 2

        return out


# =========================================================
# Mask Decoder Block
# =========================================================
class MaskDecoderBlock(nn.Module):

    def __init__(self, in_, out):
        super().__init__()

        self.l1 = Conv3BN(in_, out)

        self.l2 = Conv3BN(out, out)

        self.shortcut = (
            nn.Conv2d(in_, out, 1)
            if in_ != out
            else nn.Identity()
        )

    def forward(self, x):

        residual = self.shortcut(x)

        x = self.l1(x)

        x = self.l2(x)

        return x + residual


# =========================================================
# Boundary Decoder Block
# =========================================================
class BoundaryDecoderBlock(nn.Module):

    def __init__(
            self,
            low_channels,
            high_channels,
            mid_channels
    ):
        super().__init__()

        self.low_conv = nn.Conv2d(
            low_channels,
            mid_channels,
            1
        )

        self.high_conv = nn.Conv2d(
            high_channels,
            mid_channels,
            1
        )

        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),

            nn.Conv2d(
                mid_channels * 2,
                mid_channels * 2,
                1
            ),

            nn.ReLU(inplace=True),

            nn.Conv2d(
                mid_channels * 2,
                mid_channels * 2,
                1
            ),

            nn.Sigmoid()
        )

        self.fuse_conv = nn.Conv2d(
            mid_channels * 2,
            mid_channels,
            3,
            padding=1
        )

    def forward(self, low_feat, high_feat):

        low_feat = self.low_conv(low_feat)

        high_feat = self.high_conv(high_feat)

        low_up = F.interpolate(
            low_feat,
            size=high_feat.shape[2:],
            mode='bilinear',
            align_corners=False
        )

        x = torch.cat(
            [low_up, high_feat],
            dim=1
        )

        attn = self.attention(x)

        x = x * attn

        x = F.relu(self.fuse_conv(x))

        return x


# =========================================================
# TAGM
# =========================================================
class TAGM(nn.Module):
    """
    Topology-Aware Guidance Module

    Progressive topology interaction module.
    Boundary features guide semantic mask refinement.
    """

    def __init__(self, in_channels):
        super().__init__()

        self.boundary_attention = nn.Sequential(
            nn.Conv2d(
                in_channels,
                in_channels,
                1
            ),
            nn.Sigmoid()
        )

        self.mask_enhancer = nn.Conv2d(
            in_channels,
            in_channels,
            3,
            padding=1
        )

        self.spatial_attention = nn.Sequential(
            nn.Conv2d(
                1,
                1,
                7,
                padding=3
            ),
            nn.Sigmoid()
        )

        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),

            nn.Conv2d(
                in_channels,
                in_channels // 4,
                1
            ),

            nn.ReLU(inplace=True),

            nn.Conv2d(
                in_channels // 4,
                in_channels,
                1
            ),

            nn.Sigmoid()
        )

        self.fusion = nn.Sequential(
            nn.Conv2d(
                in_channels,
                in_channels,
                3,
                padding=1
            ),

            nn.BatchNorm2d(in_channels),

            nn.ReLU(inplace=True)
        )

    def forward(self, mask_feat, boundary_feat):

        attention_map = self.boundary_attention(boundary_feat)

        mask_feat = F.interpolate(
            mask_feat,
            size=attention_map.shape[2:],
            mode='bilinear',
            align_corners=False
        )

        guided_mask = mask_feat * attention_map

        enhanced_mask = self.mask_enhancer(guided_mask)

        spatial_att = self.spatial_attention(
            torch.mean(
                boundary_feat,
                dim=1,
                keepdim=True
            )
        )

        boundary_spatial = boundary_feat * spatial_att

        channel_att = self.channel_attention(boundary_feat)

        boundary_channel = boundary_feat * channel_att

        boundary_refined = (
                boundary_spatial +
                boundary_channel
        )

        combined = (
                enhanced_mask +
                boundary_refined
        )

        return self.fusion(combined)


# =========================================================
# STD-Net
# =========================================================
class STDNet(nn.Module):

    def __init__(
            self,
            input_channels=3,
            num_classes=1
    ):
        super().__init__()

        # =========================
        # Encoder
        # =========================
        self.conv1 = MSDResBlock(input_channels, 32)

        self.conv2 = MSDResBlock(32, 64)

        self.conv3 = MSDResBlock(64, 128)

        self.conv4 = MSDResBlock(128, 256)

        self.conv5 = MSDResBlock(256, 512)

        # =========================
        # ETA
        # =========================
        self.triplet1 = EnhancedTripletAttention(32)

        self.triplet2 = EnhancedTripletAttention(64)

        self.triplet3 = EnhancedTripletAttention(128)

        self.triplet4 = EnhancedTripletAttention(256)

        # =========================
        # Mask Decoder
        # =========================
        self.m_decoder1 = MaskDecoderBlock(768, 256)

        self.m_decoder2 = MaskDecoderBlock(384, 128)

        self.m_decoder3 = MaskDecoderBlock(192, 64)

        self.m_decoder4 = MaskDecoderBlock(96, 32)

        # =========================
        # Boundary Decoder
        # =========================
        self.b_decoder1 = BoundaryDecoderBlock(
            256, 512, 256
        )

        self.b_decoder2 = BoundaryDecoderBlock(
            128, 256, 128
        )

        self.b_decoder3 = BoundaryDecoderBlock(
            64, 128, 64
        )

        self.b_decoder4 = BoundaryDecoderBlock(
            32, 64, 32
        )

        # =========================
        # Progressive TAGM
        # =========================
        self.tagm1 = TAGM(256)

        self.tagm2 = TAGM(128)

        self.tagm3 = TAGM(64)

        self.tagm4 = TAGM(32)

        # =========================
        # Upsample
        # =========================
        self.pool = nn.MaxPool2d(2, 2)

        self.upsample = nn.Upsample(
            scale_factor=2,
            mode='bilinear',
            align_corners=True
        )

        # =========================
        # Heads
        # =========================
        self.mask_head = nn.Sequential(
            Conv3BN(32, 16),
            nn.Conv2d(16, num_classes, 1)
        )

        self.boundary_head = nn.Sequential(
            Conv3BN(32, 16),
            nn.Conv2d(16, num_classes, 1)
        )

    def forward(self, x):

        # =====================================================
        # Encoder
        # =====================================================
        x1 = self.conv1(x)

        x2 = self.pool(self.conv2(x1))

        x3 = self.pool(self.conv3(x2))

        x4 = self.pool(self.conv4(x3))

        x5 = self.pool(self.conv5(x4))

        # =====================================================
        # ETA
        # =====================================================
        x1 = self.triplet1(x1)

        x2 = self.triplet2(x2)

        x3 = self.triplet3(x3)

        x4 = self.triplet4(x4)

        # =====================================================
        # No TAB
        # =====================================================
        x5_up = self.upsample(x5)

        # =====================================================
        # Boundary Decoder
        # =====================================================
        b1 = self.b_decoder1(
            low_feat=x4,
            high_feat=x5_up
        )

        b2 = self.upsample(b1)

        b2 = self.b_decoder2(
            low_feat=x3,
            high_feat=b2
        )

        b3 = self.upsample(b2)

        b3 = self.b_decoder3(
            low_feat=x2,
            high_feat=b3
        )

        b4 = self.upsample(b3)

        b4 = self.b_decoder4(
            low_feat=x1,
            high_feat=b4
        )

        # =====================================================
        # Mask Decoder + TAGM
        # =====================================================
        m1 = self.m_decoder1(
            torch.cat([x5_up, x4], dim=1)
        )

        m1 = self.tagm1(m1, b1)

        m2 = self.upsample(m1)

        m2 = self.m_decoder2(
            torch.cat([m2, x3], dim=1)
        )

        m2 = self.tagm2(m2, b2)

        m3 = self.upsample(m2)

        m3 = self.m_decoder3(
            torch.cat([m3, x2], dim=1)
        )

        m3 = self.tagm3(m3, b3)

        m4 = self.upsample(m3)

        m4 = self.m_decoder4(
            torch.cat([m4, x1], dim=1)
        )

        m4 = self.tagm4(m4, b4)

        # =====================================================
        # Output
        # =====================================================
        mask = self.mask_head(m4)

        boundary = self.boundary_head(b4)

        return mask, boundary


# =========================================================
# Test
# =========================================================
# if __name__ == "__main__":
#
#     model = STDNet(
#         input_channels=3,
#         num_classes=1
#     )
#
#     x = torch.randn(1, 3, 512, 512)
#
#     mask, boundary = model(x)
#
#     print("Mask:", mask.shape)
#
#     print("Boundary:", boundary.shape)