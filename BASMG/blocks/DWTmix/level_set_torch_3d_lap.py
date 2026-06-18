import math

import torch
import torch.nn as nn
import numpy as np
import SimpleITK as sitk
import torch.nn.functional as F
from matplotlib import pyplot as plt

from dynamic_network_architectures.my_nnunet.blocks.FFTmix.fft import Three_dimensional_Fast_Fourier_Transform


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


class CVModel(nn.Module):
    def __init__(self, epochs, mu=1.0, nu=1.0, epsilon=1.0, step=0.05, a=20,
                 spacing=(3.2999961376190186, 0.5077999830245972, 0.5077999830245972)):
        super(CVModel, self).__init__()
        self.mu = mu
        self.nu = nu
        self.epsilon = epsilon
        self.step = step
        self.epochs = epochs
        self.spacing = spacing
        self.a = a

        self.use_k = 0.1
        self.direction = True

        # 注册3D卷积核
        self.register_buffer('laplacian_kernel', self.create_3d_laplacian(spacing))
        self.register_buffer('grad_kernel_x', self.create_3d_gradient_kernel(axis='x'))
        self.register_buffer('grad_kernel_y', self.create_3d_gradient_kernel(axis='y'))
        self.register_buffer('grad_kernel_z', self.create_3d_gradient_kernel(axis='z'))

    def create_3d_laplacian(self, spacing):
        """
        构建带 z 轴 spacing 缩放的 3D 拉普拉斯核。
        x, y spacing 视为 1，z spacing = spacing_z。
        返回形状 (1, 1, 3, 3, 3)，可用于 3D 卷积。
        """
        sz2_inv = 1.0 / ((spacing[0] / spacing[1]) ** 2)

        kernel = torch.zeros((3, 3, 3), dtype=torch.float32)

        # x 方向
        kernel[1, 1, 0] = 1
        kernel[1, 1, 2] = 1

        # y 方向
        kernel[1, 0, 1] = 1
        kernel[1, 2, 1] = 1

        # z 方向（按 spacing 缩放）
        kernel[0, 1, 1] = sz2_inv
        kernel[2, 1, 1] = sz2_inv

        # 中心 voxel
        kernel[1, 1, 1] = -4.0 - 2.0 * sz2_inv

        # 加 shape 和 device
        return kernel.view(1, 1, 3, 3, 3)

    def create_3d_gradient_kernel(self, axis):
        k = np.zeros((3, 3, 3), dtype=np.float32)
        if axis == 'x':
            k[1, 1, 0] = -0.5
            k[1, 1, 2] = 0.5
        elif axis == 'y':
            k[1, 0, 1] = -0.5
            k[1, 2, 1] = 0.5
        elif axis == 'z':
            k[0, 1, 1] = -0.5
            k[2, 1, 1] = 0.5
        return torch.tensor(k).view(1, 1, 3, 3, 3)

    def init_lsf(self, img):
        b, c, d, h, w = img.shape
        device = self.laplacian_kernel.device
        b_init = []
        for idx in range(b):
            IniLSF = torch.ones((1, d, h, w), dtype=torch.float32, device=device)

            flat_img = img[idx].view(-1)  # 展平为1D张量
            flat_img, _ = torch.sort(flat_img)
            k = max(1, int(self.use_k * flat_img.numel()))  # 计算10%位置
            threshold = torch.kthvalue(flat_img, k).values

            if self.direction:
                IniLSF[img[idx] <= threshold] = -1
            else:
                IniLSF[img[idx] >= threshold] = -1

            IniLSF = -IniLSF
            b_init.append(IniLSF)
        out_init = torch.stack(b_init, 0)
        return out_init

    def compute_gradient_3d(self, x, spacing):
        sz, sy, sx = spacing
        dx = F.conv3d(x, self.grad_kernel_x, padding=1) / sx
        dy = F.conv3d(x, self.grad_kernel_y, padding=1) / sy
        dz = F.conv3d(x, self.grad_kernel_z, padding=1) / sz
        return dz, dy, dx  # 注意顺序 (z, y, x) 与 torch.gradient 一致

    def norm(self, x):
        return (x - x.mean()) / (x.std() + 1e-8)

    def fix_lsf_sign(self, lsf_slice, img_slice):
        mask = lsf_slice >= 0  # 分割区域
        fg_mean = img_slice[mask].mean()
        bg_mean = img_slice[~mask].mean()
        # 如果前景亮度小于背景亮度，认为是反色了，翻转符号
        if fg_mean < bg_mean:
            return -lsf_slice
        return lsf_slice

    def forward(self, img, if_split=False):
        img = img * self.a  # 放大到标准灰度范围
        # 初始化水平集函数
        n_LSF = self.init_lsf(img)

        for i in range(self.epochs + 1):
            Drc = (self.epsilon / math.pi) / (self.epsilon ** 2 + n_LSF ** 2)
            Hea = 0.5 * (1 + (2 / math.pi) * torch.atan(n_LSF / self.epsilon))

            # 计算梯度
            dz, dy, dx = self.compute_gradient_3d(n_LSF, self.spacing)
            s = torch.sqrt(dx ** 2 + dy ** 2 + dz ** 2 + 1e-8) * self.a
            Nx = dx / s
            Ny = dy / s
            Nz = dz / s

            # 计算曲率项（散度 div(N)）
            dNx = self.compute_gradient_3d(Nx, self.spacing)
            dNy = self.compute_gradient_3d(Ny, self.spacing)
            dNz = self.compute_gradient_3d(Nz, self.spacing)
            cur = dNx[2] + dNy[1] + dNz[0]  # dNx/dx + dNy/dy + dNz/dz

            # 拉普拉斯项
            Lap = F.conv3d(n_LSF, self.laplacian_kernel, padding=1)

            # 惩罚项
            Penalty = self.mu * (Lap - cur)

            # CV项
            s1 = Hea * img
            s2 = (1 - Hea) * img
            s3 = 1 - Hea
            C1 = torch.sum(s1, dim=[2, 3, 4], keepdim=True) / (torch.sum(Hea, dim=[2, 3, 4], keepdim=True) + 1e-8)
            C2 = torch.sum(s2, dim=[2, 3, 4], keepdim=True) / (torch.sum(s3, dim=[2, 3, 4], keepdim=True) + 1e-8)
            CVterm = Drc * (-(img - C1) ** 2 + (img - C2) ** 2)

            # Length项
            Length = self.nu * Drc * cur * (self.a ** 2)

            if i == self.epochs:
                # 如果为当前epoch则跳过,因为只是为了计算
                pass
            else:
                # 更新水平集函数
                n_LSF = n_LSF + self.step * (Length + Penalty + CVterm)

        # return n_LSF / 100  # 缩放便于观察
        # 这个Lap是对LSF的拉普拉斯
        if not if_split:
            return torch.cat((
                self.norm(n_LSF),
                self.norm(Drc),
                self.norm(Hea),

                self.norm(cur),
                self.norm(Lap),
                self.norm(Penalty),

                self.norm(Length),
                self.norm(CVterm)
            ), dim=1)
        else:
            # 分成若干组
            # return (torch.cat((self.norm(n_LSF), self.norm(Drc), self.norm(Hea)), dim=1),  # 基础能力派生项
            #         torch.cat((self.norm(cur), self.norm(Lap), self.norm(Penalty)), dim=1),  # 几何形态项
            #         torch.cat((self.norm(Length), self.norm(CVterm)), dim=1))  # 区域对比项目(区域对比、边界演化)
            return self.norm(n_LSF), self.norm(s)


# 示例使用
if __name__ == "__main__":
    input, affine, spacing, origin = load_nifti(r'./img.nii.gz')

    # 这里是反的
    block = CVModel(epochs=20, spacing=spacing[::-1]).cuda()
    lsf, _ = block(input.cuda())

    save_nifti(lsf, affine, spacing, origin, f"lsf_org.nii.gz")





