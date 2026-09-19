import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import DeformConv2d


class Conv3BN(nn.Module):
    def __init__(self, in_: int, out: int, bn=True):
        super().__init__()
        self.conv = nn.Conv2d(in_, out, 3, padding=1)
        self.bn = nn.BatchNorm2d(out) if bn else None
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        x = self.relu(x)
        return x

class MSDConv(nn.Module):
    """
    Multi-Scale Deformable Convolution
    """

    def __init__(self, in_channels, out_channels):
        super(MSDConv, self).__init__()

        # Reduction factor for weight generator
        if in_channels <= 32:
            reduction = 2
        elif in_channels <= 128:
            reduction = 4
        else:
            reduction = 8
        reduced_channels = max(1, in_channels // reduction)

        # Multi-scale convolutions
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.offset_conv = nn.Conv2d(in_channels, 18, kernel_size=3, padding=1)  # offset for deformable
        self.dcn = DeformConv2d(in_channels, out_channels, kernel_size=3, padding=1)

        # Dynamic weight generator
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.w_gen = nn.Sequential(
            nn.Conv2d(in_channels, reduced_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_channels, 3, kernel_size=1, bias=False),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        B, _, H, W = x.size()

        # Generate fusion weights
        weights = self.global_pool(x)  # [B, C, 1, 1]
        weights = self.w_gen(weights).view(B, 3, 1, 1, 1)  # [B, 3, 1, 1, 1]

        # Multi-scale branches
        out1 = self.conv1(x).unsqueeze(1)  # [B, 1, C, H, W]
        out2 = self.conv3(x).unsqueeze(1)
        offset = self.offset_conv(x)
        out3 = self.dcn(x, offset).unsqueeze(1)

        # Weighted fusion
        branches = torch.cat([out1, out2, out3], dim=1)  # [B, 3, C, H, W]
        fused = (weights * branches).sum(dim=1)  # [B, C, H, W]

        return fused


class MSDConvBN(nn.Module):
    def __init__(self, in_channels, out_channels, bn=True):
        super().__init__()
        self.dcd = MSDConv(in_channels, out_channels)
        self.bn = nn.BatchNorm2d(out_channels) if bn else nn.Identity()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.dcd(x)
        x = self.bn(x)
        return self.relu(x)


class MSDResBlock(nn.Module):
    def __init__(self, in_: int, out: int, dropout_rate=0.2):
        super().__init__()
        self.l1 = MSDConvBN(in_, out, bn=True)
        self.l2 = MSDConvBN(out, out, bn=True)
        self.dropout = nn.Dropout2d(dropout_rate)
        self.shortcut = nn.Conv2d(in_, out, 1) if in_ != out else nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)
        x = self.l1(x)
        x = self.dropout(x)
        x = self.l2(x)
        # return x
        return x + residual  # 添加残差连接


class MaskNetModule(nn.Module):
    def __init__(self, in_: int, out: int):
        super().__init__()
        self.l1 = Conv3BN(in_, out, bn=True)
        self.l2 = Conv3BN(out, out, bn=True)
        self.shortcut = nn.Conv2d(in_, out, 1) if in_ != out else nn.Identity()


    def forward(self, x):
        residual = self.shortcut(x)
        x = self.l1(x)
        x = self.l2(x)
        return  x + residual


class BoundaryNetModule(nn.Module):
    def __init__(self, low_channels, high_channels, mid_channels):
        super(BoundaryNetModule, self).__init__()
        self.low_conv = nn.Conv2d(low_channels, mid_channels, kernel_size=1)
        self.high_conv = nn.Conv2d(high_channels, mid_channels, kernel_size=1)
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(mid_channels * 2, mid_channels * 2, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels * 2, mid_channels * 2, kernel_size=1),
            nn.Sigmoid()
        )
        self.fuse_conv = nn.Conv2d(mid_channels * 2, mid_channels, kernel_size=3, padding=1)

    def forward(self, low_feat, high_feat):
        low_feat = self.low_conv(low_feat)
        high_feat = self.high_conv(high_feat)
        low_up = F.interpolate(low_feat, size=high_feat.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([low_up, high_feat], dim=1)
        attn = self.attention(x)
        x = x * attn
        x = F.relu(self.fuse_conv(x))
        return x

# Triple
class BasicConv(nn.Module):
    def __init__(
            self, in_, out_, kernel_size, stride=1, padding=0,
            dilation=1, groups=1, relu=True, bn=True, bias=False, ):
        super(BasicConv, self).__init__()
        self.conv = nn.Conv2d(
            in_, out_, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias,)
        self.bn = (nn.BatchNorm2d(out_, eps=1e-5, momentum=0.01, affine=True) if bn else None)
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)

class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False)

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid_(x_out)
        return x * scale

class EnhancedTripletAttention(nn.Module):
    def __init__(
            self,
            in_channels,
            reduction_ratio=16,
    ):
        super(EnhancedTripletAttention, self).__init__()
        self.ChannelGateH = SpatialGate()
        self.ChannelGateW = SpatialGate()
        self.SpatialGate = SpatialGate()

        # 新增自适应池化
        self.adaptive_pool = nn.AdaptiveAvgPool2d(1)
        # 改进后的通道处理模块
        self.channel_fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, 1),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 空间通道分离
        x_perm1 = x.permute(0, 2, 1, 3).contiguous()
        x_out1 = self.ChannelGateH(x_perm1)
        x_out11 = x_out1.permute(0, 2, 1, 3).contiguous()

        x_perm2 = x.permute(0, 3, 2, 1).contiguous()
        x_out2 = self.ChannelGateW(x_perm2)
        x_out21 = x_out2.permute(0, 3, 2, 1).contiguous()

        # 自适应池化特征
        pooled = self.adaptive_pool(x)
        Channel = self.channel_fc(pooled)

        # 合并各通道输出和自适应池化的输出
        x_out = self.SpatialGate(x)
        x_out = (1 / 3) * (x_out + x_out11 + x_out21)
        out = (1 / 2) * (Channel+x_out)

        return out


class LocalTokenAttention(nn.Module):
    def __init__(self, dim, num_heads, window_size=7, attn_drop=0.1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.scale = (dim // num_heads) ** -0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(attn_drop)

    def forward(self, x):
        B, H, W, C = x.shape

        pad_h = (self.window_size - H % self.window_size) % self.window_size
        pad_w = (self.window_size - W % self.window_size) % self.window_size
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))

        _, Hp, Wp, _ = x.shape
        x = x.view(B, Hp // self.window_size, self.window_size,
                   Wp // self.window_size, self.window_size, C)
        x = x.permute(0, 1, 3, 2, 4, 5).reshape(-1, self.window_size * self.window_size, C)

        qkv = self.qkv(x).reshape(-1, self.window_size * self.window_size, 3,
                                  self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(-1, self.window_size * self.window_size, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        x = x.view(B, Hp // self.window_size, Wp // self.window_size,
                   self.window_size, self.window_size, C)
        x = x.permute(0, 1, 3, 2, 4, 5).reshape(B, Hp, Wp, C)
        if pad_h > 0 or pad_w > 0:
            x = x[:, :H, :W, :].contiguous()

        return x


class GlobalTokenAttention(nn.Module):
    def __init__(self, dim, num_heads, attn_drop=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(attn_drop)

    def forward(self, x):
        B, H, W, C = x.shape
        N = H * W
        x = x.view(B, N, C)

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2,0,3,1,4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        out = out.view(B, H, W, C)
        return out


class TokenAggregationBlock(nn.Module):
    """
    TokenAggregationBlock
    """

    def __init__(self, dim, num_heads, window_size=7, global_weight=0.5, attn_drop=0.1, mlp_drop=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.local_attn = LocalTokenAttention(dim, num_heads, window_size, attn_drop)
        self.global_attn = GlobalTokenAttention(dim, num_heads, attn_drop)
        self.global_weight = global_weight

        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(mlp_drop),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, x):
        x_ = x.permute(0, 2, 3, 1)  # [B, C, H, W] -> [B, H, W, C]
        x_norm = self.norm1(x_)

        local_out = self.local_attn(x_norm)
        global_out = self.global_attn(x_norm)
        attn_out = (1 - self.global_weight) * local_out + self.global_weight * global_out

        x_ = x_ + attn_out
        x_ = x_ + self.mlp(self.norm2(x_))
        x_ = x_.permute(0, 3, 1, 2)  # [B, H, W, C] -> [B, C, H, W]
        return x_


class MaskGuidedByBoundaryModule(nn.Module):
    """
        Guided Mask Refinement Module (GMRM)
        引导式掩码细化模块
        This module fuses auxiliary decoder features (e.g., detail-enhancing features)
        with the main mask decoder features to refine mask predictions.

        The auxiliary features are not restricted to boundary; they can be any high-frequency or fine-detail features.

        """

    def __init__(self, in_channels):
        super(MaskGuidedByBoundaryModule, self).__init__()
        # 边界引导注意力生成（原始注意力）
        self.boundary_attention = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.Sigmoid()
        )

        # 掩码特征增强
        self.mask_enhancer = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)

        # 边界特征的空间注意力
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )

        # 边界特征的通道注意力
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // 4, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels // 4, in_channels, kernel_size=1),
            nn.Sigmoid()
        )

        # 最终融合层
        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, mask_feat, boundary_feat):
        # 1. 原始边界注意力调制
        attention_map = self.boundary_attention(boundary_feat)
        mask_feat = F.interpolate(mask_feat, size=attention_map.shape[2:], mode='bilinear', align_corners=False)
        guided_mask = mask_feat * attention_map

        # 2. 掩码特征增强
        enhanced_mask = self.mask_enhancer(guided_mask)

        # 3. 边界特征的双重注意力处理
        # 空间注意力
        spatial_att = self.spatial_attention(
            torch.mean(boundary_feat, dim=1, keepdim=True))  # [B,1,H,W]
        boundary_spatial = boundary_feat * spatial_att

        # 通道注意力
        channel_att = self.channel_attention(boundary_feat)  # [B,C,1,1]
        boundary_channel = boundary_feat * channel_att

        # 4. 注意力加权融合
        boundary_att = (boundary_spatial + boundary_channel) / 2
        combined = enhanced_mask * boundary_att + enhanced_mask  # 残差连接

        return self.fusion(combined)





class Tripmodel(nn.Module):
    def __init__(self, input_channels=3, num_classes=1):
        super().__init__()
        self.n_classes = num_classes

        # 编码器保持不变
        self.conv1 = MSDResBlock(input_channels, 32)
        self.conv2 = MSDResBlock(32, 64)
        self.conv3 = MSDResBlock(64, 128)
        self.conv4 = MSDResBlock(128, 256)
        self.conv5 = MSDResBlock(256, 512)

        # 注意力模块保持不变
        # self.triplet1 = EnhancedTripletAttention(32)
        # self.triplet2 = EnhancedTripletAttention(64)
        # self.triplet3 = EnhancedTripletAttention(128)
        # self.triplet4 = EnhancedTripletAttention(256)

        # self.bottleneck = nn.Sequential(
        #     TokenAggregationBlock(512, 8, 7),
        #     TokenAggregationBlock(512, 16, 7),
        # )

        # 掩码解码器保持不变
        self.m_decoder1 = MaskNetModule(768, 256)
        self.m_decoder2 = MaskNetModule(384, 128)
        self.m_decoder3 = MaskNetModule(192, 64)
        self.m_decoder4 = MaskNetModule(96, 32)

        # self.b_decoder1 = BoundaryNetModule(256, 512, 256)  # x4 + x5_up
        # self.b_decoder2 = BoundaryNetModule(128, 256, 128)  # x3 + 上一层
        # self.b_decoder3 = BoundaryNetModule(64, 128, 64)  # x2 + 上一层
        # self.b_decoder4 = BoundaryNetModule(32, 64, 32)  # x1 + 上一层


        self.mask_fuse = MaskGuidedByBoundaryModule(32)

        self.pool = nn.MaxPool2d(2, 2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.boundary_conv = nn.Sequential(
            Conv3BN(32, 32),
        )

        # 输出头保持不变
        self.boundary_head = nn.Sequential(
            Conv3BN(32, 16),
            nn.Conv2d(16, num_classes, 1)
        )

        self.mask_head = nn.Sequential(
            Conv3BN(32, 16),
            nn.Conv2d(16, num_classes, 1)
        )

    def forward(self, x):
        # 编码器
        x1 = self.conv1(x)  # [B,32,H,W]
        x2 = self.pool(self.conv2(x1))  # [B,64,H/2,W/2]
        x3 = self.pool(self.conv3(x2))  # [B,128,H/4,W/4]
        x4 = self.pool(self.conv4(x3))  # [B,256,H/8,W/8]
        x5 = self.pool(self.conv5(x4))  # [B,512,H/16,W/16]

        # x1 = self.triplet1(x1)
        # x2 = self.triplet2(x2)
        # x3 = self.triplet3(x3)
        # x4 = self.triplet4(x4)

        # x5 = self.bottleneck(x5)  # [B,512,H/16,W/16]
        x5_up = self.upsample(x5)  # [B,512,H/8,W/8]

        # # 边界解码器
        # b1 = self.b_decoder1(low_feat=x4, high_feat=x5_up)  # 输出 128, H/8
        #
        # b2 = self.upsample(b1)  # H/4
        # b2 = self.b_decoder2(low_feat=x3, high_feat=b2)  # 输出 64, H/4
        #
        # b3 = self.upsample(b2)  # H/2
        # b3 = self.b_decoder3(low_feat=x2, high_feat=b3)  # 输出 32, H/2
        #
        # b4 = self.upsample(b3)  # H
        # b4 = self.b_decoder4(low_feat=x1, high_feat=b4)  # 输出 32, H

        # 掩码解码器
        m1 = self.m_decoder1(torch.cat([x5_up, x4], dim=1))  # [B,768→256,H/8,W/8]

        m2 = self.upsample(m1)  # [B,256,H/4,W/4]
        m2 = self.m_decoder2(torch.cat([m2, x3], dim=1))  # [B,384→128,H/4,W/4]

        m3 = self.upsample(m2)  # [B,128,H/2,W/2]
        m3 = self.m_decoder3(torch.cat([m3, x2], dim=1))  # [B,192→64,H/2,W/2]

        m4 = self.upsample(m3)  # [B,64,H,W]
        m4 = self.m_decoder4(torch.cat([m4, x1], dim=1))  # 主掩码特征

        # 边界引导掩码

        boundary_feat = self.boundary_conv(m4)
        mask_feat = self.mask_fuse(m4, boundary_feat)


        mask = self.mask_head(mask_feat)
        boundary = self.boundary_head(boundary_feat)

        return mask, boundary

# 创建模型实例
# model = Tripmodel(input_channels=3, num_classes=1)
#
# # 测试输入 (batch_size=1, channels=3, height=512, width=512)
# x = torch.randn(1, 3, 512, 512)
# model(x)
# print(x.shape)
#
