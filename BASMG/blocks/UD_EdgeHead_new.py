import torch
import torch.nn as nn
import torch.nn.functional as F


class EdgeHead(nn.Module):
    def __init__(self, in_c=96, feat_c=32, n_class=2):
        super(EdgeHead, self).__init__()
        self.last_feat = nn.Sequential(
            nn.Conv3d(in_c, feat_c, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(feat_c),
            nn.ReLU(inplace=True),
        )
        self.edge = nn.Sequential(
            nn.Conv3d(in_c, n_class, kernel_size=1, stride=1, padding=0), )

    def forward(self, x):
        last_feat = self.last_feat(x)
        edge = self.edge(x)
        return last_feat, edge


# 因为高维回来的会错位,所以使用膨胀卷积
class MultDilatedConv(nn.Module):
    def __init__(self, in_c, out_c):
        super(MultDilatedConv, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv3d(in_c, out_c, 1, 1),
            nn.BatchNorm3d(out_c),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(in_c, out_c // 4, 3, 1, padding=3, dilation=3),
            nn.BatchNorm3d(out_c // 4),
            nn.ReLU(inplace=True),
        )

        self.conv3 = nn.Sequential(
            nn.Conv3d(in_c, out_c // 8, 3, 1, padding=5, dilation=5),
            nn.BatchNorm3d(out_c // 8),
            nn.ReLU(inplace=True),
        )

        self.conv4 = nn.Sequential(
            nn.Conv3d(in_c, out_c // 16, 3, 1, padding=7, dilation=7),
            nn.BatchNorm3d(out_c // 16),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x4 = self.conv4(x)
        x = torch.cat((x1, x2, x3, x4), dim=1)
        return x


class UD_EdgeHead(nn.Module):
    def __init__(self, feat_c, class_c):
        super(UD_EdgeHead, self).__init__()
        self.fg_conv = MultDilatedConv(feat_c * 2, feat_c)
        self.bg_conv = MultDilatedConv(feat_c * 2, feat_c)

        self.seg_edge = EdgeHead(2 * (feat_c + feat_c // 4 + feat_c // 8 + feat_c // 16),
                                 feat_c, class_c)

    # 输入头和尾
    def forward(self, feat_last, seg_deep):
        _, _, D, H, W = feat_last.size()
        # 放缩成和预测一样的尺寸,但是保留通道维度，高维信息指导
        # 深监督处只是做了卷积没有01化
        last_feature = torch.sigmoid(
            F.interpolate(seg_deep,
                          size=(D, H, W),
                          mode='trilinear',
                          align_corners=True))
        # 1. 计算前景 [B, C, M, D, H, W]
        foreground = torch.einsum('bcdhw,bmdhw->bcmdhw', feat_last, last_feature)
        background = torch.einsum('bcdhw,bmdhw->bcmdhw', feat_last, 1 - last_feature)
        # # 2. 对齐 feat_last (同样是 [B, C, M, D, H, W])
        # feat_last_aligned = feat_last.unsqueeze(2).expand(-1, -1, foreground.shape[2], -1, -1, -1)
        # background = feat_last_aligned - foreground

        # 3. 调整顺序：将 M 调到 C 前面，变成 [B, M, C, D, H, W]
        # permute(0, 2, 1, 3, 4, 5) 交换 1 和 2 维
        foreground_ordered = foreground.permute(0, 2, 1, 3, 4, 5).flatten(1, 2)
        background_ordered = background.permute(0, 2, 1, 3, 4, 5).flatten(1, 2)

        foreground_feat = self.fg_conv(foreground_ordered)
        background_feat = self.bg_conv(background_ordered)

        UD_feat = torch.cat((foreground_feat, background_feat), 1)

        last_feat, edge = self.seg_edge(UD_feat)
        return last_feat, edge


# p4为mask前的也就是深监督出来前
if __name__ == "__main__":
    # 定义输入张量的尺寸 (batch_size, channels, height, width)
    feat_last = torch.randn(1, 32, 32, 128, 128).cuda()
    seg_deep = torch.randn(1, 2, 4, 16, 16).cuda()

    model = UD_EdgeHead(feat_c=32, class_c=2).cuda()

    feat, edge = model(feat_last, seg_deep)

    print(1)


