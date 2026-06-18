import torch
import torch.nn as nn
from dynamic_network_architectures.my_nnunet.blocks.FFTmix.fft import Three_dimensional_Fast_Fourier_Transform
from dynamic_network_architectures.my_nnunet.blocks.FFTmix.mix_shift import Fuseblock


class FFTMixConv(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(FFTMixConv, self).__init__()
        self.fft = Three_dimensional_Fast_Fourier_Transform()
        # 这里已经包含激活了,然后再过一个3x3卷积
        self.fuse = Fuseblock(in_channel)

        # self.norm = nn.BatchNorm3d(out_channel)
        # self.activation = nn.ReLU()
        # self.fc = nn.Conv3d(in_channel, out_channel, 1, bias=False)

    def forward(self, x):
        # 通过快速傅里叶分解得到4个不同频率的空域图
        low, mid1, mid2, high = self.fft(x)
        out = self.fuse(low, mid1, mid2, high)

        # 还是需要的啊
        out = out + x
        return out

if __name__ == '__main__':
    # 创建输入数据
    input_3d = torch.randn(1, 32, 32, 224, 256)  # 3D 输入数据，大小为 (batch_size, channels, depth, height, width)

    # 定义模型
    model_3d = FFTMixConv(in_channel=32, out_channel=32)

    # 前向传递
    output_3d = model_3d(input_3d)

    # 打印输入和输出的形状
    print("3D 输入形状:", input_3d.shape)
    print("3D 输出形状:", output_3d.shape)