import torch
import torch.nn as nn
import torch.nn.functional as F

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


#特征卷积提取网络 4层卷积
class FeatureBlock(nn.Module):
    def __init__(self):
        super(FeatureBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=7, stride=1, padding=3)   # out[1, 6, 256, 256]
        #self.conv1 = DilatedConvExample(in_channels=3, out_channels=6, kernel_size=3, dilation=2)
        self.conv2 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=1)   # out[1, 6, 256, 256]
        self.conv3 = nn.Conv2d(in_channels=12, out_channels=24, kernel_size=5, stride=2, padding=2) # out[1, 24, 128, 128]
        self.conv4 = nn.Conv2d(in_channels=12, out_channels=24, kernel_size=3, stride=2,padding=1)  # out[1, 24, 128, 128]
        self.flatten = nn.Flatten()

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x = torch.cat((x1, x2), dim=1)   # out[1, 12, 256, 256]
        x1 = self.conv3(x)
        x2 = self.conv3(x)
        x = torch.cat((x1, x2), dim=1)  # out[1, 48, 128, 128]
        return x

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


if __name__ == '__main__':
    # 初始化模型
    model = FeatureBlock()

    # 输入 tensor 的形状为 (batch_size, 3, 256, 256)
    input_tensor = torch.randn(1, 3, 720, 1080)

    # 进行前向传播
    output_tensor = model(input_tensor)
    print(output_tensor.shape)
    rgb_tensor=RgbBlock(input_tensor)
    print(rgb_tensor.shape)
