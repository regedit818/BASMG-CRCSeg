import torch
import torch.nn as nn


class MFMLSFGroupsplit(nn.Module):
    def __init__(self, dim, height=2, reduction=8, kernel_size=(1, 7, 7), nonlin=nn.ReLU):
        super(MFMLSFGroupsplit, self).__init__()

        # 保存 height 参数，height 表示特征图的分组数
        self.height = height
        # 计算中间层的维度 d，取 dim 除以 reduction 的结果和 4 中的最大值
        d = max(int(dim / reduction), 3)

        # 定义自适应平均池化层，将输入特征图池化到 1x1 大小
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        # 定义一个顺序容器，包含一系列的卷积层和激活函数
        self.mlp = nn.Sequential(
            # 第一个卷积层，输入通道数为 dim，输出通道数为 d，卷积核大小为 1
            nn.Conv3d(dim, d, 1, bias=False),
            # ReLU 激活函数，增加模型的非线性
            nonlin(),
            # 第二个卷积层，输入通道数为 d，输出通道数为 dim * height，卷积核大小为 1
            nn.Conv3d(d, dim * height, 1, bias=False)
        )

        # 定义 Softmax 激活函数，用于在维度 1 上进行归一化
        self.softmax = nn.Softmax(dim=1)

        lsf_attention = []
        for _ in range(4):
            lsf_attention.append(
                nn.Sequential(
                    nn.Conv3d(2, 1, kernel_size=kernel_size, padding='same'),
                    # nn.BatchNorm3d(1),
                    nn.Sigmoid()
                )
            )
        self.lsf_attention = nn.ModuleList(lsf_attention)

        self.project = nn.Sequential(
            nn.Conv3d(dim, dim, 1, bias=False),
            nn.BatchNorm3d(dim),
            nonlin()
        )
        # self.gamma = nn.Parameter(torch.zeros(1))

    # 前向传播方法，定义了模块的前向计算逻辑
    def forward(self, in_feats1, in_feats2, lsf):
        # 将输入的两个特征图存储在一个列表中
        # in_feats1, in_feats2 = in_feats
        # 获取输入特征图的批次大小 B、通道数 C、高度 H 和宽度 W
        B, C, D, H, W = in_feats1.shape

        # 沿着通道维度将两个输入特征图拼接在一起
        in_feats = torch.cat((in_feats1, in_feats2), dim=1)
        # in_feats = torch.cat((in_feats, lsf), dim=1)
        # 调整输入特征图的形状，将其按照 height 进行分组
        in_feats = in_feats.view(B, self.height, C, D, H, W)

        # 沿着 height 维度对输入特征图进行求和
        feats_sum = torch.sum(in_feats, dim=1)

        # 对求和后的特征图进行自适应平均池化，然后通过 MLP 网络
        attn = self.mlp(self.avg_pool(feats_sum))
        # 调整注意力图的形状，并通过 Softmax 函数进行归一化
        attn = self.softmax(attn.view(B, self.height, C, 1, 1, 1))

        # 将输入特征图与注意力图逐元素相乘，然后沿着 height 维度求和
        out = torch.sum(in_feats * attn, dim=1)

        out_group = out.chunk(4, dim=1)
        ########################################################
        # pos_lsf = torch.clamp(lsf, min=0)
        # neg_lsf = torch.clamp(-lsf, min=0)
        edge_lsf = torch.exp(-lsf ** 2)
        pos_lsf = torch.sigmoid(lsf) + 0.5
        neg_lsf = torch.sigmoid(-lsf) + 0.5

        # input_lsf = torch.cat((pos_lsf, neg_lsf), dim=1)
        # lsf_feat = self.lsf_project(input_lsf)

        # 分组相乘
        result = []
        # 原始特征组
        max_result_0, _ = torch.max(out_group[0], dim=1, keepdim=True)
        avg_result_0 = torch.mean(out_group[0], dim=1, keepdim=True)
        group_split_feat_0 = torch.cat([max_result_0, avg_result_0], dim=1)
        result.append(self.lsf_attention[0](group_split_feat_0) * out_group[0])

        # 正向特征组
        max_result_1, _ = torch.max(out_group[1] * pos_lsf, dim=1, keepdim=True)
        avg_result_1 = torch.mean(out_group[1] * pos_lsf, dim=1, keepdim=True)
        group_split_feat_1 = torch.cat([max_result_1, avg_result_1], dim=1)
        result.append(self.lsf_attention[1](group_split_feat_1) * out_group[1])

        # 负向特征组
        max_result_2, _ = torch.max(out_group[2] * neg_lsf, dim=1, keepdim=True)
        avg_result_2 = torch.mean(out_group[2] * neg_lsf, dim=1, keepdim=True)
        group_split_feat_2 = torch.cat([max_result_2, avg_result_2], dim=1)
        result.append(self.lsf_attention[2](group_split_feat_2) * out_group[2])

        # 边缘特组
        max_result_3, _ = torch.max(out_group[3] * edge_lsf, dim=1, keepdim=True)
        avg_result_3 = torch.mean(out_group[3] * edge_lsf, dim=1, keepdim=True)
        group_split_feat_3 = torch.cat([max_result_3, avg_result_3], dim=1)
        result.append(self.lsf_attention[3](group_split_feat_3) * out_group[3])

        lsf_result = self.project(torch.cat(result, dim=1)) + out

        # 返回最终的输出特征图
        return lsf_result


if __name__ == "__main__":
    input1 = torch.randn(1, 64, 32, 64, 64)
    input2 = torch.randn(1, 64, 32, 64, 64)
    lsf_input = torch.randn(1, 1, 32, 64, 64)
    model = MFMLSFGroupsplit(64)
    output = model(input1, input2, lsf_input)
    print(f"Input1 Shape: {input1.shape}")
    print(f"Input2 Shape: {input2.shape}")
    print(f"Output Shape: {output.shape}")
