import torch
from torch import nn


class Conv1x1(nn.Module):
    # 卷积+ReLU函数
    def __init__(self, in_channels, out_channels, kernel_sizes, paddings, dilations):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=kernel_sizes, padding=paddings, dilation=dilations,
                      bias=False),  ###, bias=False
            # nn.BatchNorm2d(out_channels),
            # nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class Shift_Module(nn.Module):
    # 初始化方法，接收shift_size（滚动步长）作为参数，默认值为1
    def __init__(self, shift_size=1):
        # 调用父类nn.Module的初始化方法（必须操作）
        super(Shift_Module, self).__init__()
        # 将传入的滚动步长保存为实例变量
        self.shift_size = shift_size

    def forward(self, x):
        #######################################################
        # 因为切片差异大,所以仅位移w/h
        #####################################################

        # 将输入张量x沿通道维度（dim=1）均分为4个部分
        # 假设输入通道数为4的倍数（例如32通道会被分为4个8通道的子张量）
        x1, x2, x3, x4 = x.chunk(4, dim=1)

        # 对x1张量沿高度维度（dim=2）正向滚动shift_size步
        # 例如：如果高度是50，向上滚动1步，第一行变为最后一行
        x1 = torch.roll(x1, self.shift_size, dims=3)

        # 对x2张量沿高度维度（dim=2）反向滚动shift_size步
        # 例如：向下滚动1步，最后一行变为第一行
        x2 = torch.roll(x2, -self.shift_size, dims=3)

        # 对x3张量沿宽度维度（dim=3）正向滚动shift_size步
        # 例如：如果宽度是50，向左滚动1步，第一列变为最后一列
        x3 = torch.roll(x3, self.shift_size, dims=4)

        # 对x4张量沿宽度维度（dim=3）反向滚动shift_size步
        # 例如：向右滚动1步，最后一列变为第一列
        x4 = torch.roll(x4, -self.shift_size, dims=4)

        # 将处理后的4个子张量沿通道维度（dim=1）重新拼接成完整张量
        x = torch.cat([x1, x2, x3, x4], 1)
        return x

class Fuseblock(nn.Module):
    def __init__(self, in_channels):
        super(Fuseblock, self).__init__()
        out_channels = in_channels
        self.project = nn.Sequential(
            nn.Conv3d(out_channels, out_channels, 1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU())
        self.gap = nn.AdaptiveAvgPool3d(1)
        self.softmax = nn.Softmax(dim=2)
        self.softmax_1 = nn.Sigmoid()
        self.gate1 = Conv1x1(in_channels, in_channels, 1, 0, 1)
        self.gate2 = Conv1x1(in_channels, in_channels, 1, 0, 1)
        self.gate3 = Conv1x1(in_channels, in_channels, 1, 0, 1)
        self.gate4 = Conv1x1(in_channels, in_channels, 1, 0, 1)
        self.shift = Shift_Module(shift_size=1)

    def forward(self, x0, x1, x2, x3):
        # res = torch.cat([y0,y1,y2,y3], dim=1)
        x0_weight = self.gate1(self.gap(x0))
        x1_weight = self.gate2(self.gap(x1))
        x2_weight = self.gate3(self.gap(x2))
        x3_weight = self.gate4(self.gap(x3))
        weight = torch.cat([x0_weight, x1_weight, x2_weight, x3_weight], 2)
        weight = self.softmax(self.softmax_1(weight))
        x0_weight = torch.unsqueeze(weight[:, :, 0], 2)
        x1_weight = torch.unsqueeze(weight[:, :, 1], 2)
        x2_weight = torch.unsqueeze(weight[:, :, 2], 2)
        x3_weight = torch.unsqueeze(weight[:, :, 3], 2)
        x_att = x0_weight * x0 + x1_weight * x1 + x2_weight * x2 + x3_weight * x3

        # 进行位移
        x_att = self.shift(x_att)
        # 这里的归一化等下一个3x3卷积后比较好
        return self.project(x_att)


if __name__ == '__main__':
    # 创建输入数据
    x1 = torch.randn(1, 32, 32, 224, 256)
    x2 = torch.randn(1, 32, 32, 224, 256)
    x3 = torch.randn(1, 32, 32, 224, 256)
    x4 = torch.randn(1, 32, 32, 224, 256)
    block = Fuseblock(in_channels=32)
    out = block(x1, x2, x3, x4)
    print(out.shape)
