import torch
import torch.nn as nn
import torch.nn.functional as F


class EnhancedImg(nn.Module):
    def __init__(self):
        super(EnhancedImg, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 1, kernel_size=3, padding=1)
        # self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x1 = F.silu(self.conv1(x))
        x1 = x1 * x
        x1 = F.silu(self.conv2(x1))
        x1 = self.conv3(x1)
        return x1

# class EnhancedSwiGLU(nn.Module):
#     def __init__(self, input_channels, input_height, input_width):
#         super(EnhancedSwiGLU, self).__init__()
#         self.flatten = nn.Flatten()  # 用于展平输入张量
#         flattened_dim = input_channels * input_height * input_width  # 展平后的维度
#         # hidden_dim = flattened_dim // 2  # 隐藏层维度为展平后的维度的一半
#
#         # 定义线性层和 SwigLU 核心计算
#         self.linear1 = nn.Linear(flattened_dim, 2 * flattened_dim)
#         self.linear2 = nn.Linear(flattened_dim, flattened_dim)
#         # self.w3 = nn.Parameter(torch.randn(flattened_dim))
#
#     def forward(self, x):
#         batch_size, _, _, _ = x.shape
#         # 展平输入
#         x = self.flatten(x)  # 从 (batch_size, 1, H, W) 转换为 (batch_size, H*W)
#         # SwigLU 核心计算
#         x = self.linear1(x)
#         x1, x2 = torch.chunk(x, 2, dim=-1)
#         x1 = F.silu(x1)
#         output = x1 * x2
#         output = self.linear2(output)
#         # output = output + self.w3  # 自定义权重操作
#         # 恢复到原始形状 (batch_size, 1, H, W)
#         output = output.view(batch_size, 1, int(output.shape[-1] ** 0.5), int(output.shape[-1] ** 0.5))
#         return output