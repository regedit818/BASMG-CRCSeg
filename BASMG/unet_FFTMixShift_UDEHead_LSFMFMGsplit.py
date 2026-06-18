from typing import List

from nnunetv2.training.nnUNetTrainer.BASMG.blocks.DWTmix.level_set_torch_3d_lap import CVModel
from nnunetv2.training.nnUNetTrainer.BASMG.blocks.DWTmix.wt import Dwt_split1x3x3, Dwt_split3x3x3
from torch import nn
import torch
from dynamic_network_architectures.building_blocks.simple_conv_blocks import StackedConvBlocks
from dynamic_network_architectures.initialization.weight_init import InitWeights_He
from nnunetv2.training.nnUNetTrainer.BASMG.blocks.UD_EdgeHead_new import UD_EdgeHead
from nnunetv2.training.nnUNetTrainer.BASMG.blocks.FFTmix.FFTMixConv_shift import FFTMixConv
from nnunetv2.training.nnUNetTrainer.BASMG.blocks.MFMLSFGsplit import MFMLSFGroupsplit


class StarReLU(nn.Module):
    """
    StarReLU激活函数：s * relu(x) ** 2 + b
    其中s和b是可学习的参数。
    """

    def __init__(self, scale_value=1.0, bias_value=0.0,
                 scale_learnable=True, bias_learnable=True,
                 mode=None, inplace=False):
        super().__init__()  # 调用父类nn.Module的构造函数
        self.inplace = inplace  # 是否进行原地操作
        self.relu = nn.ReLU(inplace=inplace)  # 定义ReLU激活层
        # 定义可学习的缩放参数s，默认值为scale_value，是否需要梯度更新由scale_learnable决定
        self.scale = nn.Parameter(scale_value * torch.ones(1),
                                  requires_grad=scale_learnable)
        # 定义可学习的偏置参数b，默认值为bias_value，是否需要梯度更新由bias_learnable决定
        self.bias = nn.Parameter(bias_value * torch.ones(1),
                                 requires_grad=bias_learnable)

    def forward(self, x):
        """
        前向传播函数，计算StarReLU激活后的输出。
        """
        return self.scale * self.relu(x) ** 2 + self.bias  # 应用StarReLU公式


def multiply_tuples(t1, t2):
    return tuple(map(lambda x, y: x * y, t1, t2))


class Decoder:
    def __init__(self, deep=True):
        self.deep_supervision = deep


class M_UNet(nn.Module):
    def __init__(self,
                 in_channels: int = 1,
                 num_classes: int = 2,
                 feature: List[int] = [],
                 kernel_size: List[int] = [],
                 strides: List[int] = [],
                 input_shape: List[int] = [2, 32, 224, 256],
                 use_k=0.1,
                 direction=True,
                 spacing=(3.2999961376190186, 0.5077999830245972, 0.5077999830245972),
                 deep_supervision: bool = False):
        super(M_UNet, self).__init__()

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.decoder = Decoder(deep_supervision)

        # 设置第一层卷积,都是双层卷积
        new_shape = [feature[0]] + [x // y for x, y in zip(input_shape[1:], strides[0])]
        self.in_conv = nn.Sequential(
            StackedConvBlocks(2, nn.Conv3d, 1, 32, kernel_size[0], strides[0],
                              norm_op=nn.BatchNorm3d, nonlin=nn.ReLU),
        )

        # spacing = (1.0, 1.0, 1.0)
        self.level_set = CVModel(epochs=20, spacing=spacing)
        self.level_set.use_k = use_k
        self.level_set.direction = direction

        self.down_1x3x3 = Dwt_split1x3x3()
        self.down_3x3x3 = Dwt_split3x3x3()

        # 获取网络一起有多少层
        s = len(kernel_size)

        self.strides = [i[0] for i in strides][1:-1]

        encoder_stages = []
        # 大概少一半？同时这个函数如果out通道不是列表(就是第二个),就扩展为2个一样的out通道
        for i in range(1, s):
            encoder_stages.append(nn.Sequential(
                StackedConvBlocks(1, nn.Conv3d, feature[i - 1], feature[i], kernel_size[i], strides[i],
                                  norm_op=nn.BatchNorm3d, nonlin=nn.ReLU),
                FFTMixConv(feature[i], feature[i]),
                StackedConvBlocks(1, nn.Conv3d, feature[i], feature[i], kernel_size[i], 1,
                                  norm_op=nn.BatchNorm3d, nonlin=nn.ReLU),
            ))

        # 不注册则无法使用cuda()
        self.encoder_stages = nn.ModuleList(encoder_stages)

        down_sample = []
        # 这里设置使用反卷积, 上采样应该少一个,因为是直接从encoder拿过来
        for i in range(s - 1, 0, -1):
            down_sample.append(
                nn.ConvTranspose3d(feature[i], feature[i - 1], strides[i], strides[i])
            )
        self.down_sample = nn.ModuleList(down_sample)

        # decoder这里是接上面反卷积的
        decoder_stages = []
        seg_out = []
        skip_stages = []
        for i in range(s - 1, 0, -1):
            decoder_stages.append(nn.Sequential(
                StackedConvBlocks(2, nn.Conv3d, feature[i - 1], feature[i - 1], kernel_size[i], 1,
                                  norm_op=nn.BatchNorm3d, nonlin=nn.ReLU),
            ))
            k = 3 + 2 * ((5 - i) // 2)
            # k = 3 + 2 * (5 - i)
            # print(k)
            skip_stages.append(
                MFMLSFGroupsplit(feature[i - 1], kernel_size=(1, k, k))
            )
            # 预设深监督
            seg_out.append(nn.Conv3d(feature[i - 1], num_classes, 1, 1, 0, bias=True))
        self.decoder_stages = nn.ModuleList(decoder_stages)
        self.seg_out = nn.ModuleList(seg_out)
        self.skip_stages = nn.ModuleList(skip_stages)
        self.ud_edge_head = UD_EdgeHead(feature[0], num_classes)
        # self.edge_head = nn.Conv3d(feature[0], num_classes, 1, 1, 0, bias=True)

        self.auto_wt = False
        if self.auto_wt:
            wt_weight = []
            for i in self.strides:
                if i == 1:
                    wt_weight.append(nn.Parameter(torch.ones(4)))
                else:
                    wt_weight.append(nn.Parameter(torch.ones(8)))
            self.wt_weight = nn.ParameterList(wt_weight)

    def get_lsf(self, x):
        B, _, _, _, _ = x.shape
        # 第一层的32x224x256直接从原图水平集分割
        lsf, _ = self.level_set(x, if_split=True)
        lsf_list = [lsf]

        if self.auto_wt:
            for i, w in zip(self.strides, self.wt_weight):
                if i == 1:
                    all_wt = self.down_1x3x3(lsf)
                    all_wt_tensor = torch.stack(list(all_wt), dim=0)
                    now_weight = w.view(4, 1, 1, 1, 1, 1).expand(4, B, 1, 1, 1, 1)
                    lsf = (all_wt_tensor * now_weight).mean(dim=0)
                    lsf_list.append(lsf)
                elif i == 2:
                    all_wt = self.down_3x3x3(lsf)
                    all_wt_tensor = torch.stack(list(all_wt), dim=0)
                    now_weight = w.view(8, 1, 1, 1, 1, 1).expand(8, B, 1, 1, 1, 1)
                    lsf = (all_wt_tensor * now_weight).mean(dim=0)
                    lsf_list.append(lsf)
        else:
            for i in self.strides:
                if i == 1:
                    LL, LH, HL, HH = self.down_1x3x3(lsf)
                    lsf = (LL + LH + HL) / 3
                    # lsf = (LL + LH + HL + HH) / 4
                    lsf_list.append(lsf)
                elif i == 2:
                    LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH = self.down_3x3x3(lsf)
                    lsf = (LLL + LLH + LHL + LHH + HLL + HLH + HHL) / 7
                    # lsf = (LLL + LLH + LHL + LHH + HLL + HLH + HHL + HHH) / 8
                    lsf_list.append(lsf)

        return lsf_list

    def forward(self, x, tr=False, level=None):
        # 如果不是训练,那用于生成水平集的来源就是自身
        if level is None:
            level = x

        with torch.no_grad():
            lsf_list = self.get_lsf(level)

        #######################################################################################
        # encoder部分
        x = self.in_conv(x)
        encoder_list = [x]
        for idx, stage in enumerate(self.encoder_stages):
            x = stage(x)
            encoder_list.append(x)

        #######################################################################################
        # decoder部分

        # 深监督记录
        seg_output = []

        # encoder去除最后一个然后取反
        for idx, (stage, skip, encoder_stage, decoder_stage, out_conv, now_lsf) in enumerate(zip(
                self.down_sample, self.skip_stages, encoder_list[:-1][::-1], self.decoder_stages, self.seg_out,
                lsf_list[::-1])):
            # 经过反卷积以后,已经恢复shape
            x = stage(x)
            x = skip(x, encoder_stage, now_lsf)
            x = decoder_stage(x)

            if (idx + 1) == len(self.decoder_stages):
                # 第一个进去的特征就是
                x, edge = self.ud_edge_head(x, seg_output[0])

            # 这里因为要回传
            # if self.decoder.deep_supervision:
            seg_output.append(out_conv(x))

            # 这里需要判断是否为最后一层

        # 验证的时候也要增强特征啊,一个卷积的事,不返回就行了
        if tr:
            if self.decoder.deep_supervision:
                return seg_output[::-1], edge

            # 如果没有深监督直接返回
            return self.seg_out[-1](x), edge
        else:
            if self.decoder.deep_supervision:
                return seg_output[::-1]

            # 如果没有深监督直接返回
            return self.seg_out[-1](x)

    @staticmethod
    def initialize(module):
        InitWeights_He(1e-2)(module)


if __name__ == '__main__':
    len_num = 13
    model = M_UNet(1,
                   2,
                   (32, 64, 128, 256, 320, 320, 320)[:len_num],  # (32, 48, 96, 168, 224, 224)
                   ([1, 3, 3], [1, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3])[:len_num],
                   ([1, 1, 1], [1, 2, 2], [1, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2][:len_num]),
                   deep_supervision=True,
                   ).to('cuda')

    data = torch.rand(1, 1, 8, 64, 64).to('cuda')
    with torch.no_grad():
        mask, medge = model(data, True)
        print(1)
        for ii in mask:
            print(ii.shape)
