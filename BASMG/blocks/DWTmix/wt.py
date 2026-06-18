import torch
import torch.nn as nn
from dynamic_network_architectures.my_nnunet.blocks.DWTmix.dwt3 import dwt3x3x3, dwt1x3x3, dwt3x1x1
from pytorch_wavelets import DWTForward
from dynamic_network_architectures.my_nnunet.blocks.DWTmix.level_set_torch_3d_lap import CVModel
import numpy as np
import SimpleITK as sitk
import torch.nn.functional as F


def load_nifti(file_path):
    sitk_image = sitk.ReadImage(file_path)  # 读取 NIfTI
    data = sitk.GetArrayFromImage(sitk_image)  # (D, H, W) numpy 格式
    affine = np.array(sitk_image.GetDirection()).reshape(3, 3)  # 获取仿射信息

    tensor = torch.tensor(data, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # (1, 1, D, H, W)
    return tensor, affine, sitk_image.GetSpacing(), sitk_image.GetOrigin()


def save_nifti(tensor, affine, spacing, origin, save_path):
    array = tensor.squeeze().cpu().numpy()  # (D, H, W) 转换回 numpy
    sitk_image = sitk.GetImageFromArray(array)  # 转换为 SimpleITK 影像
    sitk_image.SetSpacing(spacing)
    sitk_image.SetOrigin(origin)
    sitk.WriteImage(sitk_image, save_path)
    print(f"Saved: {save_path}")


class Dwt_split3x3x3(nn.Module):
    def __init__(self):
        super(Dwt_split3x3x3, self).__init__()
        # self.wt = dwt3(J=1, mode='zero', wave='haar')

    def forward(self, x):
        dwt_x = dwt3x3x3(x.cuda(), "haar")
        LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH = torch.unbind(dwt_x, dim=1)

        return LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH


class Dwt_split3x1x1(nn.Module):
    def __init__(self):
        super(Dwt_split3x1x1, self).__init__()
        # self.wt = dwt3(J=1, mode='zero', wave='haar')

    def forward(self, x):
        dwt_x = dwt3x1x1(x.cuda(), "haar")
        L, H = torch.unbind(dwt_x, dim=1)

        return L, H


class Dwt_split1x3x3(nn.Module):
    def __init__(self):
        super(Dwt_split1x3x3, self).__init__()
        self.wt = DWTForward(J=1, mode='zero', wave='haar')

    def forward(self, x):
        B, C, D, H, W = x.shape
        x_reshaped = x.reshape(B * D, C, H, W)
        dwt_x = dwt1x3x3(x_reshaped.cuda(), "haar")
        LL, LH, HL, HH = torch.unbind(dwt_x, dim=1)

        LL = LL.reshape(B, D, C, H // 2, W // 2).permute(0, 2, 1, 3, 4)
        LH = LH.reshape(B, D, C, H // 2, W // 2).permute(0, 2, 1, 3, 4)
        HL = HL.reshape(B, D, C, H // 2, W // 2).permute(0, 2, 1, 3, 4)
        HH = HH.reshape(B, D, C, H // 2, W // 2).permute(0, 2, 1, 3, 4)

        return LL, LH, HL, HH


class SobelEdge2D(nn.Module):
    def __init__(self):
        super(SobelEdge2D, self).__init__()

        # 定义 4 个方向的 Sobel 核，形状：(4, 1, 3, 3)
        sobel_kernels = torch.stack([
            torch.tensor([[-1, 0, 1],
                          [-2, 0, 2],
                          [-1, 0, 1]], dtype=torch.float32),  # x

            torch.tensor([[-1, -2, -1],
                          [0, 0, 0],
                          [1, 2, 1]], dtype=torch.float32),  # y

            torch.tensor([[0, 1, 2],
                          [-1, 0, 1],
                          [-2, -1, 0]], dtype=torch.float32),  # diag1 ↘

            torch.tensor([[2, 1, 0],
                          [1, 0, -1],
                          [0, -1, -2]], dtype=torch.float32),  # diag2 ↙
        ])

        self.register_buffer("weight", sobel_kernels[:, None, :, :])  # shape: (4, 1, 3, 3)

    def forward(self, x, direction=True):
        # 输入 x: (B, 1, D, H, W)
        b, c, d, h, w = x.shape
        assert c == 1, "Only single-channel input supported."

        x2d = x.view(b * d, 1, h, w)  # (B*D, 1, H, W)

        # 卷积操作，输出 shape: (B*D, 4, H, W)
        edge = F.conv2d(x2d, self.weight.to(x.device), padding=1)

        # 逐方向平方后求和，再开根号
        edge_mag = torch.sqrt(torch.sum(edge ** 2, dim=1, keepdim=True))  # (B*D, 1, H, W)

        if not direction:
            return edge_mag.view(b, 1, d, h, w)
        else:
            return edge_mag.view(b, 1, d, h, w), edge.view(b, 4, d, h, w)


class Dwt_for_DoubleDown(nn.Module):
    def __init__(self):
        super(Dwt_for_DoubleDown, self).__init__()
        self.dwt_down = Dwt_split1x3x3()
        self.sobel_ex = SobelEdge2D()

    def forward(self, x):
        LL1, LH1, HL1, HH1 = self.dwt_down(x)
        output_x2 = (LL1 + LH1 + HL1) / 3
        output_x2_edge = self.sobel_ex(output_x2)

        LL2, LH2, HL2, HH2 = self.dwt_down(output_x2)
        output_x4 = (LL2 + LH2 + HL2) / 3
        output_x4_edge = self.sobel_ex(output_x4)

        return torch.cat((output_x2, output_x2_edge), dim=1), torch.cat((output_x4, output_x4_edge), dim=1)


class Dwt_for_OneDown(nn.Module):
    def __init__(self):
        super(Dwt_for_OneDown, self).__init__()
        self.dwt_down = Dwt_split1x3x3()
        self.sobel_ex = SobelEdge2D()
        self.level_set = CVModel(epochs=10)

    def forward(self, x, if_tiny=False):
        LL1, LH1, HL1, HH1 = self.dwt_down(x)
        output_x2 = (LL1 + LH1 + HL1) / 3
        output_x2_sobel, sobel_direction = self.sobel_ex(output_x2, direction=True)

        if not if_tiny:
            level_set_feature = self.level_set(output_x2, if_split=True)
            return torch.cat((output_x2, LL1, LH1, HL1, HH1, output_x2_sobel, sobel_direction, level_set_feature),
                             dim=1)
        else:
            output_x2_fh = (LH1 + HL1 + HH1) / 3
            level_set_lsf1, level_set_geo1, level_set_seg1 = self.level_set(output_x2, if_split=True)

            return torch.cat((output_x2, output_x2_fh, output_x2_sobel, level_set_lsf1[:, [0], :, :, :],
                              level_set_geo1[:, [0], :, :, :]), dim=1)


class Dwt_for_OneSplitDown(nn.Module):
    def __init__(self):
        super(Dwt_for_OneSplitDown, self).__init__()
        self.dwt_down = Dwt_split1x3x3()
        self.sobel_ex = SobelEdge2D()
        self.level_set = CVModel(epochs=10)

    def forward(self, x):
        LL1, LH1, HL1, HH1 = self.dwt_down(x)
        output_x2 = (LL1 + LH1 + HL1) / 3
        output_x2_fh = (LH1 + HL1 + HH1) / 3
        output_x2_sobel, sobel_direction = self.sobel_ex(output_x2, direction=True)
        lsf1, cur1 = self.level_set(output_x2, if_split=True)

        return (torch.cat((output_x2, output_x2_fh), dim=1),
                output_x2_sobel,
                torch.cat((lsf1, cur1), dim=1))


class Dwt_Only_LSF(nn.Module):
    def __init__(self, spacing=(3.2999961376190186, 0.5077999830245972, 0.5077999830245972)):
        super(Dwt_Only_LSF, self).__init__()
        self.dwt_down = Dwt_split1x3x3()
        self.sobel_ex = SobelEdge2D()
        self.level_set = CVModel(epochs=10, spacing=spacing)

    def forward(self, x):
        LL, LH, HL, HH = self.dwt_down(x)
        output_x2 = (LL + LH + HL) / 3
        # output_x2_fh = (LH1 + HL1 + HH1) / 3
        # output_x2_sobel, sobel_direction = self.sobel_ex(output_x2, direction=True)
        lsf1, cur1 = self.level_set(output_x2, if_split=True)

        return output_x2, lsf1


class Dwt_Only_LSF_3D(nn.Module):
    def __init__(self, spacing=(3.2999961376190186, 0.5077999830245972, 0.5077999830245972)):
        super(Dwt_Only_LSF_3D, self).__init__()
        self.dwt_down = Dwt_split3x3x3()
        self.sobel_ex = SobelEdge2D()
        self.level_set = CVModel(epochs=10, spacing=spacing)

    def forward(self, x):
        LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH = self.dwt_down(x)
        output_x2 = (LLL + LLH + LHL + HLL) / 4

        lsf1, cur1 = self.level_set(output_x2, if_split=True)

        return output_x2, lsf1


class Dwt_for_TwoDown(nn.Module):
    def __init__(self):
        super(Dwt_for_TwoDown, self).__init__()
        self.dwt_down = Dwt_split1x3x3()
        self.sobel_ex = SobelEdge2D()
        self.level_set = CVModel(epochs=10)

    def forward(self, x):
        # LL1, LH1, HL1, HH1 = self.dwt_down(x)
        # output_x2 = (LL1 + LH1 + HL1) / 3
        # output_x2_fh = (LH1 + HL1 + HH1) / 3
        # output_x2_sobel1, sobel_direction1 = self.sobel_ex(output_x2, direction=True)
        # lsf1, cur1 = self.level_set(output_x2, if_split=True)
        #
        # output1 = torch.cat((output_x2, output_x2_fh, output_x2_sobel1, lsf1, cur1), dim=1)
        # ############################################################
        # LL2, LH2, HL2, HH2 = self.dwt_down(output_x2)
        # output_x4 = (LL2 + LH2 + HL2) / 3
        # output_x4_fh = (LH2 + HL2 + HH2) / 3
        # output_x4_sobel2, sobel_direction2 = self.sobel_ex(output_x4, direction=True)
        # level_set_feature2 = self.level_set(output_x4)
        #
        # output2 = torch.cat((output_x4, output_x4_fh, LL2, LH2, HL2, HH2, output_x4_sobel2, sobel_direction2, level_set_feature2),
        #                     dim=1)

        LL1, LH1, HL1, HH1 = self.dwt_down(x)
        output_x2 = (LL1 + LH1 + HL1) / 3
        output_x2_fh = (LH1 + HL1 + HH1) / 3
        output_x2_sobel1, sobel_direction1 = self.sobel_ex(output_x2, direction=True)
        lsf1, cur1 = self.level_set(output_x2, if_split=True)

        output1 = torch.cat((output_x2, output_x2_fh, output_x2_sobel1, lsf1, cur1), dim=1)
        ############################################################
        LL2, LH2, HL2, HH2 = self.dwt_down(output_x2)
        output_x4 = (LL2 + LH2 + HL2) / 3
        output_x4_fh = (LH2 + HL2 + HH2) / 3
        output_x4_sobel2, sobel_direction2 = self.sobel_ex(output_x4, direction=True)
        lsf2, cur2 = self.level_set(output_x4, if_split=True)

        output2 = torch.cat((output_x4, output_x4_fh, output_x4_sobel2, lsf2, cur2), dim=1)
        return output1, output2


class Dwt_for_FourDown(nn.Module):
    def __init__(self):
        super(Dwt_for_FourDown, self).__init__()
        self.dwt_down = Dwt_split1x3x3()
        self.sobel_ex = SobelEdge2D()
        self.level_set = CVModel(epochs=10)

    def forward(self, x):
        LL1, LH1, HL1, HH1 = self.dwt_down(x)
        output_x2 = (LL1 + LH1 + HL1) / 3
        output_x2_sobel1, sobel_direction1 = self.sobel_ex(output_x2, direction=True)
        level_set_feature1 = self.level_set(output_x2)

        output1 = torch.cat((output_x2, LL1, LH1, HL1, HH1, output_x2_sobel1, sobel_direction1, level_set_feature1),
                            dim=1)
        ############################################################
        LL2, LH2, HL2, HH2 = self.dwt_down(output_x2)
        output_x4 = (LL2 + LH2 + HL2) / 3
        output_x4_sobel2, sobel_direction2 = self.sobel_ex(output_x4, direction=True)
        level_set_feature2 = self.level_set(output_x4)

        output2 = torch.cat((output_x4, LL2, LH2, HL2, HH2, output_x4_sobel2, sobel_direction2, level_set_feature2),
                            dim=1)

        return output1, output2


class Dwt_for_UpDown(nn.Module):
    def __init__(self):
        super(Dwt_for_UpDown, self).__init__()
        self.dwt_down = Dwt_split1x3x3()
        self.sobel_ex = SobelEdge2D()
        self.level_set = CVModel(epochs=10)

    def forward(self, x):
        LL1, LH1, HL1, HH1 = self.dwt_down(x)
        output_x2 = (LL1 + LH1 + HL1) / 3
        dwt_feat1 = torch.cat((output_x2, LL1, LH1, HL1, HH1), dim=1)  # 5个

        output_x2_sobel1, sobel_direction1 = self.sobel_ex(output_x2, direction=True)
        sobel_feat1 = torch.cat((output_x2_sobel1, sobel_direction1), dim=1)  # 5个

        level_set_lsf1, level_set_geo1, level_set_seg1 = self.level_set(output_x2, if_split=True)  # 3个 3个 2个

        ############################################################
        LL2, LH2, HL2, HH2 = self.dwt_down(output_x2)
        output_x4 = (LL2 + LH2 + HL2) / 3
        dwt_feat2 = torch.cat((output_x4, LL2, LH2, HL2, HH2), dim=1)  # 5个

        output_x4_sobel2, sobel_direction2 = self.sobel_ex(output_x4, direction=True)
        sobel_feat2 = torch.cat((output_x4_sobel2, sobel_direction2), dim=1)  # 5个

        level_set_lsf2, level_set_geo2, level_set_seg2 = self.level_set(output_x4, if_split=True)  # 3个 3个 2个

        return (dwt_feat1, sobel_feat1, level_set_lsf1, level_set_geo1, level_set_seg1), (
            dwt_feat2, sobel_feat2, level_set_lsf2, level_set_geo2, level_set_seg2)


if __name__ == '__main__':
    # blocks = Dwt_split1x3x3().cuda()
    # # input = torch.rand(2, 3, 32, 256, 256).cuda()
    # input, affine, spacing, origin = load_nifti(r'./img.nii.gz')
    # B, C, D, H, W = input.shape
    #
    # output = blocks(input)
    # out_all = (output[0] + output[1] + output[2]) / 3
    #
    # save_nifti(out_all, affine, spacing, origin, f"img_down2.nii.gz")
    # print(out_all.size())
    #
    # output = blocks(out_all)
    # out_all = (output[0] + output[1] + output[2]) / 3
    #
    # save_nifti(out_all, affine, spacing, origin, f"img_down4.nii.gz")
    # print(out_all.size())
    # #######################################################
    # pool = nn.MaxPool2d(kernel_size=2, stride=2)
    # max_pool_out = pool(input.view(B * D, C, H, W)).view(B, C, D, H // 2, W // 2)
    # save_nifti(max_pool_out, affine, spacing, origin, f"img_pool2.nii.gz")
    # print(max_pool_out.size())
    #
    # max_pool_out = pool(max_pool_out.view(B * D, C, H // 2, W // 2)).view(B, C, D, H // 4, W // 4)
    # save_nifti(max_pool_out, affine, spacing, origin, f"img_pool4.nii.gz")
    # print(max_pool_out.size())

    block = Dwt_Only_LSF_3D(spacing=(3.2999961376190186, 0.5077999830245972, 0.5077999830245972)).cuda()
    input, affine, spacing, origin = load_nifti(r'./img.nii.gz')
    # input = torch.rand(2, 1, 32, 256, 256).cuda()
    down, lsf = block(input)

    save_nifti(down, affine, spacing, origin, f"edge_3D.nii.gz")
