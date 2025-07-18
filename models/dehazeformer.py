import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from thop import profile
from torch.nn.init import _calculate_fan_in_and_fan_out
from timm.models.layers import to_2tuple, trunc_normal_


class DilatedConvExample(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super(DilatedConvExample, self).__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            dilation=dilation,  # 设置空洞卷积
            padding=(kernel_size - 1) * dilation // 2  # 适应空洞卷积的填充
        )

    def forward(self, x):
        return self.conv(x)



class Depth_Separable_Conv(nn.Module):
    def __init__(self, in_channels, depthwise_out_channels, pointwise_out_channels):
        super(Depth_Separable_Conv, self).__init__()
        # 深度可分卷积（深度卷积 + 逐点卷积）
        self.depthwise_conv = nn.Conv2d(in_channels=in_channels, out_channels=depthwise_out_channels, kernel_size=3, stride=1, padding=1,groups=in_channels)  # 深度卷积
        self.pointwise_conv = nn.Conv2d(in_channels=depthwise_out_channels, out_channels=pointwise_out_channels, kernel_size=1)  # 逐点卷积
    def forward(self, x):
        # 深度卷积
        x_depthwise = self.depthwise_conv(x)
        # 逐点卷积
        x = self.pointwise_conv(x_depthwise)
        return x

import torch.nn.functional as F
import matplotlib.pyplot as plt

def RgbBlock(input_tensor):
    # 获取输入张量的高度和宽度
    height, width = input_tensor.shape[2], input_tensor.shape[3]

    # 使用最大池化（max pooling）来实现最小池化
    # 通过负值池化来模拟最小池化，因为最大池化会取最大值
    # 将输入张量乘以 -1 后使用最大池化，然后再乘以 -1 得到最小值
    x1_tensor = -F.max_pool2d(-input_tensor, kernel_size=2, stride=2)  # out[3, 128, 128]

    # 计算切割位置，使用 0.5 的比例来划分张量
    h_split = int(height * 0.25)  # 高度的0.25
    w_split = int(width * 0.25)  # 宽度的0.25

# draw picture test-----------------------------------------------------------
    # 将张量转换为 NumPy 数组，因为 Matplotlib 需要 NumPy 数组或者类似的对象来显示图像
    # photo_tensor = x1_tensor.permute(1, 2, 0).numpy()  # 转换为形状为 (128, 128, 3) 的 NumPy 数组
    # photo_input_tensor = input_tensor.permute(1, 2, 0).numpy()
    # # 创建一个包含 1 行 2 列的子图
    # plt.figure(figsize=(10, 5))
    #
    # # 第一个子图：显示 photo_tensor
    # plt.subplot(1, 2, 1)  # 1 行 2 列，第 1 个位置
    # plt.imshow(photo_tensor)
    # plt.axis('off')  # 关闭坐标轴
    # plt.title('Photo Tensor')  # 给图像加标题
    #
    # # 第二个子图：显示 photo_input_tensor
    # plt.subplot(1, 2, 2)  # 1 行 2 列，第 2 个位置
    # plt.imshow(photo_input_tensor)
    # plt.axis('off')  # 关闭坐标轴
    # plt.title('Input Tensor')  # 给图像加标题
    # # 显示图像
    # plt.tight_layout()  # 自动调整子图之间的间距
    # plt.show()
#---------------------------------------------------------------------------
    # 通过切片将张量切分为四个 (3, 64, 64) 的块
    # 假设切割成上下两个块和左右两个块
    # 通过切片将张量切分为四个 (3, height*0.5, width*0.5) 的块
    top_left = x1_tensor[:, :, :h_split, :w_split]  # 上左块
    top_right = x1_tensor[:, :, :h_split, w_split:]  # 上右块
    bottom_left = x1_tensor[:, :, h_split:, :w_split]  # 下左块
    bottom_right = x1_tensor[:, :, h_split:, w_split:]  # 下右块
    # 将四个块拼接成一个 (12, 64, 64) 的张量
    # 首先按列拼接前两个块，然后再按列拼接后两个块，最后合并
    top = torch.cat((top_left, top_right), dim=1)  # 拼接 top_left 和 top_right，形状为 (6, 64, 64)
    bottom = torch.cat((bottom_left, bottom_right), dim=1)  # 拼接 bottom_left 和 bottom_right，形状为 (6, 64, 64)
    # 最后将 top 和 bottom 合并，得到形状为 (12, 64, 64)
    output_tensor = torch.cat((top, bottom), dim=1)
    return output_tensor

class SoftReLU(nn.Module):
    def __init__(self, alpha=1.0):
        """
        构造函数，定义改造后的 SoftReLU 激活函数
        :param alpha: 控制激活函数的平滑度
        """
        super(SoftReLU, self).__init__()
        self.alpha = alpha

    def forward(self, x):
        """
        计算改造后的 SoftReLU 激活函数
        :param x: 输入张量
        :return: 激活后的输出
        """
        return (x / 2) + (torch.sqrt(x ** 2 + self.alpha ** 2) / 2) - (self.alpha / 2)


# dim：该层的输入特征的维度（通常是通道数）。
# eps：用于防止除零错误的小常数，默认为 1e-5。
# detach_grad：一个布尔值，控制是否在计算归一化时“断开”梯度计算
class RLN(nn.Module):  # RescaleNorm
    r"""Revised LayerNorm"""

    def __init__(self, dim, eps=1e-5, detach_grad=False):
        super(RLN, self).__init__()
        self.eps = eps
        self.detach_grad = detach_grad

        self.weight = nn.Parameter(torch.ones((1, dim, 1, 1)))  # r
        self.bias = nn.Parameter(torch.zeros((1, dim, 1, 1)))  # β

        self.meta1 = nn.Conv2d(1, dim, 1)  # 用来变换标准差的卷积层
        self.meta2 = nn.Conv2d(1, dim, 1)  # 用来变换均值的卷积层
        # 卷积核权重进行初始化，使用了截断正态分布（trunc_normal_），标准差为 0.02。这种初始化方式通常有助于避免梯度消失或爆炸的问题
        trunc_normal_(self.meta1.weight, std=.02)  # 权重Wr
        nn.init.constant_(self.meta1.bias, 1)  # 偏置进行初始化，设置为常数值 1 Br

        trunc_normal_(self.meta2.weight, std=.02)  # 权重Wβ
        nn.init.constant_(self.meta2.bias, 0)  # 偏置Bb

    def forward(self, input):
        # 计算输入张量 input 的均值，沿着通道、高度和宽度维度进行求均值,keepdim=True 保持维度一致，结果的形状为 (batch_size, 1, 1, 1)
        mean = torch.mean(input, dim=(1, 2, 3), keepdim=True)  # μ
        # 计算输入的标准差。首先计算每个元素与均值的差的平方，然后沿着 dim=(1, 2, 3) 求均值，最后取平方根得到标准差。+ self.eps 是为了避免除零错误，确保稳定性
        std = torch.sqrt((input - mean).pow(2).mean(dim=(1, 2, 3), keepdim=True) + self.eps)  # σ
        # 进行标准化操作，将输入数据减去均值并除以标准差，得到归一化的输入 normalized_input
        normalized_input = (input - mean) / std  # x=（x−μ）/σ

        if self.detach_grad:
            rescale, rebias = self.meta1(std.detach()), self.meta2(mean.detach())  # 计算σWr+Br，μWβ+Bβ
        else:  # 直接使用std和mean进行计算，允许计算图中的梯度传播
            rescale, rebias = self.meta1(std), self.meta2(mean)

        out = normalized_input * self.weight + self.bias
        return out, rescale, rebias  # 返回三个参，为F函数计算做准备


# --------------------------Transformer-------------------------
# 1.多层感知机（MLP），但它并不是一个传统的全连接网络，而是由两个卷积层构成的网络
class Mlp(nn.Module):  # 这一层网络结构它有所更改
    def __init__(self, network_depth, in_features, hidden_features=None, out_features=None):
        super().__init__()
        # 如果未传入 out_features 或 hidden_features，则默认它们等于 in_features
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.network_depth = network_depth
        self.soft_relu = SoftReLU(alpha=1.0)  # 使用改造后的 SoftReLU 激活函数
        self.mlp = nn.Sequential(
            nn.Conv2d(in_features, hidden_features, 1),
            # nn.ReLU(True),
            SoftReLU(alpha=1.0),
            nn.Conv2d(hidden_features, out_features, 1)
        )

        self.apply(self._init_weights)  # 进入_init_weights函数

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            gain = (8 * self.network_depth) ** (-1 / 4)
            fan_in, fan_out = _calculate_fan_in_and_fan_out(m.weight)
            std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
            trunc_normal_(m.weight, std=std)  # 截断正态分布来初始化卷积层的权重
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.mlp(x)


# 2.该函数将输入的张量（例如图像特征图）按照给定的窗口大小进行分块。常用于 Vision Transformer（ViT）等网络模型中
def window_partition(x, window_size):  # x是一个形状为 (B, H, W, C) 的张量
    B, H, W, C = x.shape
    # H // window_size 和 W // window_size 是图像高度和宽度分别按窗口大小 window_size 分割后，得到的块的数量
    # 如果输入图像的大小是 (B, 8, 8, 3)，且窗口大小 window_size=2，则 x 将被重塑为形状 (B, 4, 2, 4, 2, 3)，
    # 即每个窗口为 2x2，且图像被分割为 4 个这样的窗口
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    # 重新排列维度，将 x 中的空间维度（高度、宽度）按窗口的顺序交换，使得窗口的数据排布变得连续
    # 1是高度方向上分块后的部分，3是宽度方向上分块后的部分，2和4是窗口大小，5是通道数
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size ** 2, C)
    return windows


# 3.将分割的窗口恢复为原始图像或特征图 swin-transformer
def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


# 用于计算图像中相邻窗口之间的相对位置
# 4.相对位置是一个常见的操作，尤其是在视觉 Transformer（如 Swin Transformer）等模型中，用来捕捉局部窗口之间的相对位置信息
def get_relative_positions(window_size):
    coords_h = torch.arange(window_size)  # 表示窗口在高度方向的坐标
    coords_w = torch.arange(window_size)

    coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww 两个二维张量沿着新的维度堆叠起来
    coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
    relative_positions = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww

    relative_positions = relative_positions.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
    #relative_positions_log = torch.sign(relative_positions) * torch.log(1. + relative_positions.abs())
    relative_positions = torch.sign(relative_positions) * torch.log(1. + relative_positions.abs())
    #return relative_positions_log
    return relative_positions

from torch import Tensor
class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads):
        super().__init__()
        # 添加可学习的缩放参数 τ（每个头和层独立）
        self.tau = nn.Parameter(torch.ones(num_heads) * 0.01)  # 初始值需>0.01
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = nn.Parameter(torch.ones(1))  # 可学习的缩放参数

        # 计算窗口的相对位置
        relative_positions = get_relative_positions(self.window_size)  # 获取窗口位置
        self.register_buffer("relative_positions", relative_positions)  # 将这个张量作为一个模块的 buffer 存储在模型中

        # 替换为连续位置偏置，通过MLP生成
        # 连续位置偏置的生成使用 MLP 来处理
        self.meta = nn.Sequential(
            nn.Linear(2, 256),  # 输入是log坐标的2维
            SoftReLU(alpha=1.0),
            nn.Linear(256, num_heads)
        )
        self.softmax = nn.Softmax(dim=-1)

    # 在 forward 方法中进行修改
    def forward(self, qkv: Tensor) -> Tensor:
        B_, N, _ = qkv.shape

        # 将 qkv tensor 拆分为 q, k, v，并调整维度
        qkv = qkv.reshape(B_, N, 3, self.num_heads, self.dim // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # 使用余弦相似度计算 q 和 k 之间的相似度
        q = F.normalize(q, dim=-1)  # L2归一化
        k = F.normalize(k, dim=-1)
        # 计算余弦相似度
        attn = (q @ k.transpose(-2, -1))  # 余弦相似度

        # 可学习的缩放参数，直接乘以 scale
        attn = attn * self.scale

        # 计算相对位置偏置，使用MLP生成
        relative_position_bias = self.meta(self.relative_positions)  # 通过 MLP 获取偏置
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww

        # 加入相对位置偏置
        # attn = attn + relative_position_bias.unsqueeze(0)
        # 确保 tau 的维度是 [B_, num_heads, 1, 1]
        attn = attn / self.tau.view(1, self.num_heads, 1, 1)  # 这里将 tau 的维度调整为 [1, num_heads, 1, 1]

        # 使用 softmax 进行归一化
        attn = self.softmax(attn)

        # 计算最终的输出
        x = torch.matmul(attn, v).transpose(1, 2).reshape(B_, N, self.dim)

        return x


# 6.Transformer 整体网络
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn.init import trunc_normal_


class Attention(nn.Module):
    def __init__(self, network_depth, dim, num_heads, window_size, shift_size, use_attn=False, conv_type=None):
        super().__init__()
        self.dim = dim
        self.head_dim = int(dim // num_heads)
        self.num_heads = num_heads

        self.window_size = window_size
        self.shift_size = shift_size

        self.network_depth = network_depth
        self.use_attn = use_attn
        self.conv_type = conv_type

        # 卷积层类型
        if self.conv_type == 'Conv':
            self.conv = nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size=3, padding=1, padding_mode='reflect'),
                SoftReLU(alpha=1.0),
                nn.Conv2d(dim, dim, kernel_size=3, padding=1, padding_mode='reflect')
            )

        if self.conv_type == 'DWConv':
            self.conv = nn.Conv2d(dim, dim, kernel_size=5, padding=2, groups=dim, padding_mode='reflect')

        # 注意力相关的卷积层
        if self.conv_type == 'DWConv' or self.use_attn:
            self.V = nn.Conv2d(dim, dim, 1)
            self.proj = nn.Conv2d(dim, dim, 1)

        # 注意力机制设置
        if self.use_attn:
            self.QK = nn.Conv2d(dim, dim * 2, 1)
            self.attn = WindowAttention(dim, window_size, num_heads)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            w_shape = m.weight.shape

            if w_shape[0] == self.dim * 2:  # QK
                fan_in, fan_out = _calculate_fan_in_and_fan_out(m.weight)
                std = math.sqrt(2.0 / float(fan_in + fan_out))
                trunc_normal_(m.weight, std=std)
            else:
                gain = (8 * self.network_depth) ** (-1 / 4)
                fan_in, fan_out = _calculate_fan_in_and_fan_out(m.weight)
                std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
                trunc_normal_(m.weight, std=std)

            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def check_size(self, x, shift=False):
        _, _, h, w = x.size()
        mod_pad_h = (self.window_size - h % self.window_size) % self.window_size
        mod_pad_w = (self.window_size - w % self.window_size) % self.window_size

        # 在Swin V2中，我们不再使用循环移位，因此只需要进行填充
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x

    def forward(self, X):
        B, C, H, W = X.shape
        if self.conv_type == 'DWConv' or self.use_attn:
            V = self.V(X)

        if self.use_attn:
            QK = self.QK(X)
            QKV = torch.cat([QK, V], dim=1)

            # 直接填充，不进行循环移位
            padded_QKV = self.check_size(QKV, shift=False)
            Ht, Wt = padded_QKV.shape[2:]

            # 划分成窗口
            padded_QKV = padded_QKV.permute(0, 2, 3, 1)
            qkv = window_partition(padded_QKV, self.window_size)  # nW*B, window_size**2, C

            # 在窗口上执行注意力操作
            attn_windows = self.attn(qkv)

            # 将窗口结果合并回来
            merged_out = window_reverse(attn_windows, self.window_size, Ht, Wt)  # B H' W' C

            # 不需要进行循环移位反转
            out = merged_out[:, :H, :W, :]
            attn_out = out.permute(0, 3, 1, 2)

            # 如果需要卷积，则应用卷积
            if self.conv_type in ['Conv', 'DWConv']:
                conv_out = self.conv(V)
                out = self.proj(conv_out + attn_out)
            else:
                out = self.proj(attn_out)

        else:
            # 如果没有注意力，只进行卷积操作
            if self.conv_type == 'Conv':
                out = self.conv(X)  # 无注意力且使用卷积，无投影
            elif self.conv_type == 'DWConv':
                out = self.proj(self.conv(V))

        return out


# -----------------------------------------------------------------------------------
# 标准的 SwinTransformerV2Block 编码器块，包含了自注意力机制（self-attention）和前馈网络（MLP）
class SwinTransformerV2Block(nn.Module):
    def __init__(self, network_depth, dim, num_heads, mlp_ratio=4.,
                 norm_layer=nn.LayerNorm, mlp_norm=False,
                 window_size=8, shift_size=0, use_attn=True, conv_type=None):
        super().__init__()
        self.use_attn = use_attn
        self.mlp_norm = mlp_norm

        self.norm1 = norm_layer(dim) if use_attn else nn.Identity()
        self.attn = Attention(network_depth, dim, num_heads=num_heads, window_size=window_size,
                              shift_size=shift_size, use_attn=use_attn, conv_type=conv_type)

        self.norm2 = norm_layer(dim) if use_attn and mlp_norm else nn.Identity()
        self.mlp = Mlp(network_depth, dim, hidden_features=int(dim * mlp_ratio))

    def forward(self, x):
        identity = x
        x = self.attn(x)  # 注意力计算
        x = identity + x  # 先残差连接
        if self.use_attn:
            x, rescale, rebias = self.norm1(x)  # 后归一化

        identity = x
        x = self.mlp(x)
        if self.use_attn and self.mlp_norm: x, rescale, rebias = self.norm2(x)
        x = identity + x
        return x


class DehazeFormer_Block(nn.Module):  # backbone  DehazeFormer_Block
    def __init__(self, network_depth, dim, depth, num_heads, mlp_ratio=4.,
                 norm_layer=nn.LayerNorm, window_size=8,
                 attn_ratio=0., attn_loc='last', conv_type=None):

        super().__init__()
        self.dim = dim
        self.depth = depth

        attn_depth = attn_ratio * depth

        if attn_loc == 'last':
            use_attns = [i >= depth - attn_depth for i in range(depth)]
        elif attn_loc == 'first':
            use_attns = [i < attn_depth for i in range(depth)]
        elif attn_loc == 'middle':
            use_attns = [i >= (depth - attn_depth) // 2 and i < (depth + attn_depth) // 2 for i in range(depth)]

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerV2Block(network_depth=network_depth,
                             dim=dim,
                             num_heads=num_heads,
                             mlp_ratio=mlp_ratio,
                             norm_layer=norm_layer,
                             window_size=window_size,
                             shift_size=0 if (i % 2 == 0) else window_size // 2,
                             use_attn=use_attns[i], conv_type=conv_type)
            # SwinTransformerV2(input_resolution=(256, 256), window_size=8, in_channels=24,  # 这里改为 24 通道
            # 				  sequential_self_attention=False, use_checkpoint=False, embedding_channels=24,
            # 				  depths=[2], number_of_heads=[3])
            for i in range(depth)])

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x


class PatchEmbed(nn.Module):  # 既可以实现down-sample下采样，也可以实现3*3卷积
    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, kernel_size=None):
        super().__init__()
        self.in_chans = in_chans
        self.embed_dim = embed_dim

        if kernel_size is None:
            kernel_size = patch_size

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=kernel_size, stride=patch_size,
                              padding=(kernel_size - patch_size + 1) // 2, padding_mode='reflect')

    def forward(self, x):
        x = self.proj(x)
        return x


class PatchUnEmbed(nn.Module):
    def __init__(self, patch_size=4, out_chans=3, embed_dim=96, kernel_size=None):
        super().__init__()
        self.out_chans = out_chans
        self.embed_dim = embed_dim

        if kernel_size is None:
            kernel_size = 1

        self.proj = nn.Sequential(
            nn.Conv2d(embed_dim, out_chans * patch_size ** 2, kernel_size=kernel_size,
                      padding=kernel_size // 2, padding_mode='reflect'),
            nn.PixelShuffle(patch_size)
        )

    def forward(self, x):
        x = self.proj(x)
        return x


# 公式4
class SKFusion(nn.Module):  # 叠加计算在这个函数中进行，可以优化它
    def __init__(self, dim, height=2, reduction=8):
        super(SKFusion, self).__init__()

        self.height = height
        d = max(int(dim / reduction), 4)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 使用全局平均池化GAP（·）
        self.mlp = nn.Sequential(  # MLP (Linear-ReLU-Linear)
            nn.Conv2d(dim, d, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(d, dim * height, 1, bias=False)
        )

        self.softmax = nn.Softmax(dim=1)

    def forward(self, in_feats):
        B, C, H, W = in_feats[0].shape

        in_feats = torch.cat(in_feats, dim=1)
        in_feats = in_feats.view(B, self.height, C, H, W)

        feats_sum = torch.sum(in_feats, dim=1)
        attn = self.mlp(self.avg_pool(feats_sum))
        attn = self.softmax(attn.view(B, self.height, C, 1, 1))

        out = torch.sum(in_feats * attn, dim=1)
        return out

    # 特征卷积提取网络 4层卷积


class FeatureBlock(nn.Module):
    def __init__(self):
        super(FeatureBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=7, stride=1, padding=3)  # out[1, 6, 256, 256]
        self.conv2 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=1)  # out[1, 6, 256, 256]
        self.conv3 = nn.Conv2d(in_channels=12, out_channels=24, kernel_size=5, stride=2,
                               padding=2)  # out[1, 24, 128, 128]
        self.conv4 = nn.Conv2d(in_channels=12, out_channels=24, kernel_size=3, stride=2,
                               padding=1)  # out[1, 24, 128, 128]
        self.flatten = nn.Flatten()

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x = torch.cat((x1, x2), dim=1)  # out[1, 12, 256, 256]
        x1 = self.conv3(x)
        x2 = self.conv3(x)
        x = torch.cat((x1, x2), dim=1)  # out[1, 48, 128, 128]
        return x


class DehazeFormer(nn.Module):
    def __init__(self, in_chans=3, out_chans=4, window_size=8,
                 embed_dims=[24, 48, 96, 48, 24],
                 mlp_ratios=[2., 4., 4., 2., 2.],
                 depths=[16, 16, 16, 8, 8],
                 num_heads=[2, 4, 6, 1, 1],
                 attn_ratio=[1 / 4, 1 / 2, 3 / 4, 0, 0],
                 conv_type=['DWConv', 'DWConv', 'DWConv', 'DWConv', 'DWConv'],
                 norm_layer=[RLN, RLN, RLN, RLN, RLN]):
        super(DehazeFormer, self).__init__()
        # 在 __init__ 中实例化 FeatureBlock
        self.feature_block = FeatureBlock()  # 如果需要参数，可以传递给 FeatureBlock
        # setting
        self.patch_size = 4
        self.window_size = window_size
        self.mlp_ratios = mlp_ratios

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            patch_size=1, in_chans=in_chans, embed_dim=embed_dims[0], kernel_size=3)  # kernel_size=3：卷积核的大小为 3x3

        # backbone  DehazeFormer_Block1
        self.layer1 = DehazeFormer_Block(network_depth=sum(depths), dim=embed_dims[0], depth=depths[0],
        			   			 num_heads=num_heads[0], mlp_ratio=mlp_ratios[0],
        			   			 norm_layer=norm_layer[0], window_size=window_size,
        			   			 attn_ratio=attn_ratio[0], attn_loc='last', conv_type=conv_type[0])

        self.patch_merge1 = PatchEmbed(  # 2x2进行卷积，并且步幅也为2。因此，每次卷积操作会将输入图像尺寸缩小 2 倍，完成下采样
            patch_size=2, in_chans=embed_dims[0], embed_dim=embed_dims[1])

        self.skip1 = nn.Conv2d(embed_dims[0], embed_dims[0], 1)
        # self.convf = nn.Conv2d(in_channels=96, out_channels=48, kernel_size=3, stride=1, padding=1)  # out[1, 48, 128, 128]
        self.depth_separable_convf = Depth_Separable_Conv(in_channels=96, depthwise_out_channels=96,
                                                          pointwise_out_channels=48)
        self.layer2 = DehazeFormer_Block(network_depth=sum(depths), dim=embed_dims[1], depth=depths[1],
        						 num_heads=num_heads[1], mlp_ratio=mlp_ratios[1],
        						 norm_layer=norm_layer[1], window_size=window_size,
        						 attn_ratio=attn_ratio[1], attn_loc='last', conv_type=conv_type[1])

        self.patch_merge2 = PatchEmbed(  # 第二次下采样
            patch_size=2, in_chans=embed_dims[1], embed_dim=embed_dims[2])

        self.skip2 = nn.Conv2d(embed_dims[1], embed_dims[1], 1)
        # self.convr = nn.Conv2d(in_channels=96, out_channels=84,kernel_size=1)
        self.depth_separable_convr = Depth_Separable_Conv(in_channels=108, depthwise_out_channels=108,
                                                          pointwise_out_channels=96)
        self.layer3 = DehazeFormer_Block(network_depth=sum(depths), dim=embed_dims[2], depth=depths[2],
        						 num_heads=num_heads[2], mlp_ratio=mlp_ratios[2],
        						 norm_layer=norm_layer[2], window_size=window_size,
        						 attn_ratio=attn_ratio[2], attn_loc='last', conv_type=conv_type[2])

        self.patch_split1 = PatchUnEmbed(  # 上采样，反卷积
            patch_size=2, out_chans=embed_dims[3], embed_dim=embed_dims[2])

        assert embed_dims[1] == embed_dims[3]
        self.fusion1 = SKFusion(embed_dims[3])

        self.layer4 = DehazeFormer_Block(network_depth=sum(depths), dim=embed_dims[3], depth=depths[3],
        						 num_heads=num_heads[3], mlp_ratio=mlp_ratios[3],
        						 norm_layer=norm_layer[3], window_size=window_size,
        						 attn_ratio=attn_ratio[3], attn_loc='last', conv_type=conv_type[3])

        self.patch_split2 = PatchUnEmbed(
            patch_size=2, out_chans=embed_dims[4], embed_dim=embed_dims[3])

        assert embed_dims[0] == embed_dims[4]
        self.fusion2 = SKFusion(embed_dims[4])

        self.layer5 = DehazeFormer_Block(network_depth=sum(depths), dim=embed_dims[4], depth=depths[4],
        			   			 num_heads=num_heads[4], mlp_ratio=mlp_ratios[4],
        			   			 norm_layer=norm_layer[4], window_size=window_size,
        			   			 attn_ratio=attn_ratio[4], attn_loc='last', conv_type=conv_type[4])

        # merge non-overlapping patches into image
        self.patch_unembed = PatchUnEmbed(  # 最后3*3卷积
            patch_size=1, out_chans=out_chans, embed_dim=embed_dims[4], kernel_size=3)

    def check_image_size(self, x):
        # NOTE: for I2I test
        _, _, h, w = x.size()
        mod_pad_h = (self.patch_size - h % self.patch_size) % self.patch_size
        mod_pad_w = (self.patch_size - w % self.patch_size) % self.patch_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x

    def forward_features(self, x):
        input_tensor = x
        FeatureBlock_tensor = x
        _, _, h, w = x.size()  # 获取当前输入的高度和宽度
        resolution = (h, w)  # 动态设置 input_resolution
        self.layer1.input_resolution = (h, w)
        self.layer2.input_resolution = (0.5 * h, 0.5 * w)
        self.layer3.input_resolution = (0.25 * h, 0.25 * w)
        self.layer4.input_resolution = (0.5 * h, 0.5 * w)
        self.layer5.input_resolution = (h, w)
        x = self.patch_embed(x)  # 3*3卷积
        x = self.layer1(x)  # in（x，24，256，256）  out(x，24，256，256)
        skip1 = x

        x = self.patch_merge1(x)  # 下采样 in（x，24，256，256） out(x，48，128，128)
        x1 = self.feature_block(FeatureBlock_tensor)  # 调用 FeatureBlock out(x，48，128，128)
        x = torch.cat((x, x1), dim=1)  # out(x，96，128，128)
        # x = self.convf(x)  # out(x，48，128，128)
        x = self.depth_separable_convf(x)  # out(x，48，128，128)
        x = self.layer2(x)
        skip2 = x

        x = self.patch_merge2(x)  # out(x，96，64，64)
        # x =  self.convr(x)     # out(x，84，64，64)
        x2 = RgbBlock(input_tensor)
        x = torch.cat((x, x2), dim=1)  # out(x，108，64，64)
        x = self.depth_separable_convr(x)  # out(x，96，64，64)

        x = self.layer3(x)  # out(x，96，64，64)
        x = self.patch_split1(x)

        x = self.fusion1([x, self.skip2(skip2)]) + x
        x = self.layer4(x)  # in(16,48,128,128)  out(16,48,128,128)
        x = self.patch_split2(x)

        x = self.fusion2([x, self.skip1(skip1)]) + x
        x = self.layer5(x)  # （，24，256，256）
        x = self.patch_unembed(x)  # （，4，256，256）
        return x

    def forward(self, x):
        H, W = x.shape[2:]
        x = self.check_image_size(x)  # # （，3，256，256）

        feat = self.forward_features(x)  # （，4，256，256）
        K, B = torch.split(feat, (1, 3), dim=1)

        x = K * x - B + x
        x = x[:, :, :H, :W]
        return x


def dehazeformer_t():
    return DehazeFormer(
        embed_dims=[24, 48, 96, 48, 24],
        mlp_ratios=[2., 4., 4., 2., 2.],
        depths=[4, 4, 4, 2, 2],
        num_heads=[2, 4, 6, 1, 1],
        attn_ratio=[0, 1 / 2, 1, 0, 0],
        conv_type=['DWConv', 'DWConv', 'DWConv', 'DWConv', 'DWConv'])


# def dehazeformer_s():
#     return DehazeFormer(
#         embed_dims=[24, 48, 96, 48, 24],
#         mlp_ratios=[2., 4., 4., 2., 2.],
#         depths=[8, 8, 8, 4, 4],
#         num_heads=[2, 4, 6, 1, 1],
#         attn_ratio=[1 / 4, 1 / 2, 3 / 4, 0, 0],
#         conv_type=['DWConv', 'DWConv', 'DWConv', 'DWConv', 'DWConv'])

def dehazeformer_s():
    return DehazeFormer(
        embed_dims=[24, 48, 96, 48, 24],
        mlp_ratios=[2., 4., 4., 2., 2.],
        depths=[16, 16, 16, 8, 8],        # 更深的块数
        num_heads=[2, 4, 6, 1, 1],
        attn_ratio=[1/4, 1/2, 3/4, 0, 0],
        conv_type=['DWConv', 'DWConv', 'DWConv', 'DWConv', 'DWConv'],
        norm_layer=[RLN, RLN, RLN, RLN, RLN]  # 使用后归一化
    )

def dehazeformer_b():
    return DehazeFormer(
        embed_dims=[24, 48, 96, 48, 24],
        mlp_ratios=[2., 4., 4., 2., 2.],
        depths=[16, 16, 16, 8, 8],
        num_heads=[2, 4, 6, 1, 1],
        attn_ratio=[1 / 4, 1 / 2, 3 / 4, 0, 0],
        conv_type=['DWConv', 'DWConv', 'DWConv', 'DWConv', 'DWConv'])


def dehazeformer_d():
    return DehazeFormer(
        embed_dims=[24, 48, 96, 48, 24],
        mlp_ratios=[2., 4., 4., 2., 2.],
        depths=[32, 32, 32, 16, 16],
        num_heads=[2, 4, 6, 1, 1],
        attn_ratio=[1 / 4, 1 / 2, 3 / 4, 0, 0],
        conv_type=['DWConv', 'DWConv', 'DWConv', 'DWConv', 'DWConv'])


def dehazeformer_w():
    return DehazeFormer(
        embed_dims=[48, 96, 192, 96, 48],
        mlp_ratios=[2., 4., 4., 2., 2.],
        depths=[16, 16, 16, 8, 8],
        num_heads=[2, 4, 6, 1, 1],
        attn_ratio=[1 / 4, 1 / 2, 3 / 4, 0, 0],
        conv_type=['DWConv', 'DWConv', 'DWConv', 'DWConv', 'DWConv'])


def dehazeformer_m():
    return DehazeFormer(
        embed_dims=[24, 48, 96, 48, 24],
        mlp_ratios=[2., 4., 4., 2., 2.],
        depths=[12, 12, 12, 6, 6],
        num_heads=[2, 4, 6, 1, 1],
        attn_ratio=[1 / 4, 1 / 2, 3 / 4, 0, 0],
        conv_type=['Conv', 'Conv', 'Conv', 'Conv', 'Conv'])


def dehazeformer_l():
    return DehazeFormer(
        embed_dims=[48, 96, 192, 96, 48],
        mlp_ratios=[2., 4., 4., 2., 2.],
        depths=[16, 16, 16, 12, 12],
        num_heads=[2, 4, 6, 1, 1],
        attn_ratio=[1 / 4, 1 / 2, 3 / 4, 0, 0],
        conv_type=['Conv', 'Conv', 'Conv', 'Conv', 'Conv'])



if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # 强制检查GPU可用性
    if not torch.cuda.is_available():
        raise RuntimeError("该程序需要GPU支持，但未检测到可用CUDA设备！")

    # 初始化模型并立即转移到GPU
    model = DehazeFormer().cuda()  # 等效于.to('cuda')

    # # 创建输入张量时直接在GPU上生成
    # input_tensor = torch.randn(1, 3, 460, 80).cuda()  # 形状为(1,3,720,720)
    #
    # # 验证设备一致性
    # print(f"模型参数设备: {next(model.parameters()).device}")  # 应显示cuda:0
    # print(f"输入张量设备: {input_tensor.device}")  # 应显示cuda:0
    #
    # # 进行前向传播
    # with torch.no_grad():
    #     output_tensor = model(input_tensor)
    #
    # print(output_tensor.shape)

    input = torch.randn(1, 3, 224, 224)
    input = input.to(device)

    # 计算Params和MACs
    macs, params = profile(model, inputs=(input,))
    print(f"Params: {params / 1e6:.3f}M")
    print(f"MACs: {macs / 1e9:.3f}G")