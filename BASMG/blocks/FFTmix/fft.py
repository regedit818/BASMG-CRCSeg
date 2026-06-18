import torch
import torch.nn as nn
import numpy as np
import SimpleITK as sitk
import torch.nn.functional as F


class Three_dimensional_Fast_Fourier_Transform(nn.Module):
    def __init__(self):
        super(Three_dimensional_Fast_Fourier_Transform, self).__init__()

    def forward(self, tensor):
        device = tensor.device
        # 将输入张量的数据类型转换为float

        tensor = tensor.float()
        # 调用reshape_to_square方法将输入张量转换为方形张量，并获取相关信息

        # 获取方形张量的批量大小B和通道数C
        B, C, D, W, H = tensor.shape

        fft_tensor = torch.fft.fftn(tensor, dim=(-3, -2, -1))  # 计算三维傅里叶变换
        fft_shifted = torch.fft.fftshift(fft_tensor, dim=(-3, -2, -1))  # 频谱中心化

        d_range = ((torch.arange(D, device=device) - D // 2) / D).float()  # 归一化 D 方向
        w_range = ((torch.arange(W, device=device) - W // 2) / W).float()  # 归一化 W 方向
        h_range = ((torch.arange(H, device=device) - H // 2) / H).float()  # 归一化 H 方向

        d_grid, w_grid, h_grid = torch.meshgrid(d_range, w_range, h_range, indexing='ij')
        radius = torch.sqrt(d_grid ** 2 + w_grid ** 2 + h_grid ** 2).to(tensor.device)  # 计算归一化频率半径

        R_max = radius.max()
        low_thresh, mid1_thresh, mid2_thresh = R_max * 0.25, R_max * 0.5, R_max * 0.75

        low_mask = radius <= low_thresh
        mid1_mask = (radius > low_thresh) & (radius <= mid1_thresh)
        mid2_mask = (radius > mid1_thresh) & (radius <= mid2_thresh)
        high_mask = radius > mid2_thresh

        # 低频成分
        low_freq = fft_shifted * low_mask
        # 中频成分
        mid1_freq = fft_shifted * mid1_mask
        mid2_freq = fft_shifted * mid2_mask
        # 高频成分
        high_freq = fft_shifted * high_mask

        # 低频还原
        low_reconstructed = torch.fft.ifftn(torch.fft.ifftshift(low_freq, dim=(-3, -2, -1)), dim=(-3, -2, -1)).real
        # 中频还原
        mid1_reconstructed = torch.fft.ifftn(torch.fft.ifftshift(mid1_freq, dim=(-3, -2, -1)), dim=(-3, -2, -1)).real
        mid2_reconstructed = torch.fft.ifftn(torch.fft.ifftshift(mid2_freq, dim=(-3, -2, -1)), dim=(-3, -2, -1)).real
        # 高频还原
        high_reconstructed = torch.fft.ifftn(torch.fft.ifftshift(high_freq, dim=(-3, -2, -1)), dim=(-3, -2, -1)).real

        return low_reconstructed, mid1_reconstructed, mid2_reconstructed, high_reconstructed


def load_nifti(file_path):
    sitk_image = sitk.ReadImage(file_path)  # 读取 NIfTI
    data = sitk.GetArrayFromImage(sitk_image)  # (D, H, W) numpy 格式
    affine = np.array(sitk_image.GetDirection()).reshape(3, 3)  # 获取仿射信息

    tensor = torch.tensor(data, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # (1, 1, D, H, W)
    return tensor, affine, sitk_image.GetSpacing(), sitk_image.GetOrigin()


def save_nifti(tensor, affine, spacing, origin, save_path):
    array = tensor.squeeze().numpy()  # (D, H, W) 转换回 numpy
    sitk_image = sitk.GetImageFromArray(array)  # 转换为 SimpleITK 影像
    sitk_image.SetSpacing(spacing)
    sitk_image.SetOrigin(origin)
    sitk.WriteImage(sitk_image, save_path)
    print(f"Saved: {save_path}")


if __name__ == '__main__':
    output_prefix = '1'

    data = torch.randn(1, 3, 32, 224, 256)
    print("input1.shape:", data.shape)
    block = Three_dimensional_Fast_Fourier_Transform()
    data, affine, spacing, origin = load_nifti(r'D:\nnunet\DATASET\nnUNet_raw\Dataset350_fullCRC\imagesTr\fullCRC_13.08884381_0000.nii.gz')
    data = data

    low_freq_tensor, mid1_freq_tensor, mid2_freq_tensor, high_freq_tensor = block(data)

    save_nifti(low_freq_tensor, affine, spacing, origin, f"{output_prefix}_low.nii.gz")
    save_nifti(mid1_freq_tensor, affine, spacing, origin, f"{output_prefix}_mid1.nii.gz")
    save_nifti(mid2_freq_tensor, affine, spacing, origin, f"{output_prefix}_mid2.nii.gz")
    save_nifti(high_freq_tensor, affine, spacing, origin, f"{output_prefix}_high.nii.gz")

    save_nifti(low_freq_tensor + mid1_freq_tensor + mid2_freq_tensor + high_freq_tensor
               , affine, spacing, origin, f"{output_prefix}_reshape.nii.gz")

    print("low_freq_tensor.shape:", low_freq_tensor.shape)
    print("mid1_freq_tensor.shape:", mid1_freq_tensor.shape)
    print("mid2_freq_tensor.shape:", mid2_freq_tensor.shape)
    print("high_freq_tensor.shape:", high_freq_tensor.shape)
