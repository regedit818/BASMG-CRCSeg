import torch
from pywt import Wavelet
import torch.nn.functional as F


def _to_wavelet_coefs(wavelet: str | torch.Tensor | Wavelet) -> torch.Tensor:
    match wavelet:
        case str():
            return torch.tensor(Wavelet(wavelet).filter_bank)[2:]
        case torch.Tensor():
            return wavelet
        case Wavelet():
            return torch.tensor(wavelet.filter_bank)[2:]
        case _:
            raise Exception("")


@torch.jit.script
def _dwt1d(x: torch.Tensor, lo_hi: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """performs a 1d dwt on the defined dimension

    Args:
        x (torch.Tensor): 4d tensor of shape [N,C,D,H,W]
        lo_hi (torch.Tensor): low and highpass filter (shape [2,K])
        dim (int, optional): dimension to apply the dwt to . Defaults to -1.

    Returns:
        torch.Tensor: dwt coefs of shape [N,2,C,D_out,H_out,W_out]. The average and detail coefs are concatenated in the channels
    """
    dim = dim % 5
    groups = x.shape[1]
    # repeat filter to match number of channels
    filter_c = lo_hi[:, None, None, None, :].repeat(groups, 1, 1, 1, 1).swapaxes(4, dim)

    if x.shape[dim] % 2 != 0:
        # pad dwt dimension to multiple of two
        pad = [0] * 6
        pad[(4 - dim) * 2 + 1] = 1
        x = F.pad(x, pad)

    # stride of 2 for dwt dim
    stride = [1, 1, 1]
    stride[dim - 2] = 2

    padding = [0, 0, 0]
    padding[dim - 2] = lo_hi.shape[-1] - 2

    filtered = F.conv3d(x, filter_c, stride=stride, padding=padding, groups=groups)
    return filtered.reshape(
        filtered.shape[0],
        groups,
        2,
        filtered.shape[2],
        filtered.shape[3],
        filtered.shape[4],
    ).swapaxes(1, 2)


@torch.jit.script
def _dwt3(x: torch.Tensor, lohi: torch.Tensor) -> torch.Tensor:
    x_c = _dwt1d(x, lohi, -1)
    y_c = _dwt1d(x_c.flatten(1, 2), lohi, -2)
    z_c = _dwt1d(y_c.flatten(1, 2), lohi, -3)
    return z_c.reshape(
        z_c.shape[0], 8, x.shape[1], z_c.shape[3], z_c.shape[4], z_c.shape[5]
    )


@torch.jit.script
def _dwt2(x: torch.Tensor, lohi: torch.Tensor) -> torch.Tensor:
    lh = _dwt1d(x[:, :, None, :, :], lohi, -1)
    y = _dwt1d(lh.flatten(1, 2), lohi, -2).squeeze(-3)

    # reorder coefs to match pywt ordering
    return y.reshape(x.shape[0], 4, x.shape[1], y.shape[-2], y.shape[-1])


def dwt3x3x3(x: torch.Tensor, wavelet: str | torch.Tensor | Wavelet) -> torch.Tensor:
    filter = _to_wavelet_coefs(wavelet).to(x.device)
    return _dwt3(x, filter)


def dwt1x3x3(x: torch.Tensor, wavelet: str | torch.Tensor | Wavelet) -> torch.Tensor:
    """performs the 2D discrete wavelet transform

    Args:
        x (torch.Tensor): [N,C,H,W] data
        wavelet (str | torch.Tensor | Wavelet): wavelet

    Returns:
        torch.Tensor: dwt coefs of shape [M,4,C,H_out,W_out] \\
            (coef oder: cA,cV,cW,cD)
    """
    filter = _to_wavelet_coefs(wavelet).to(x.device)
    return _dwt2(x, filter)


def dwt3x1x1(x: torch.Tensor, wavelet: str | torch.Tensor | Wavelet) -> torch.Tensor:
    """performs the 2D discrete wavelet transform

    Args:
        x (torch.Tensor): [N,C,H,W] data
        wavelet (str | torch.Tensor | Wavelet): wavelet

    Returns:
        torch.Tensor: dwt coefs of shape [M,4,C,H_out,W_out] \\
            (coef oder: cA,cV,cW,cD)
    """
    filter = _to_wavelet_coefs(wavelet).to(x.device)
    return _dwt1d(x, filter, dim=2)


if __name__ == '__main__':
    data = torch.randn((1, 1, 16, 56, 64))

    y = dwt3x1x1(data, "haar")
    L, H = torch.unbind(y, dim=1)
    print(L.shape, H.shape)


