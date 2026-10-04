import torch


def _rms(x, num_channels, num_groups, gamma, eps):
    shp = x.shape
    xf = x.float().reshape(*shp[:-1], num_groups, num_channels // num_groups)
    rstd = torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    y = xf * rstd
    y = y.reshape(shp)
    if gamma is not None:
        y = y * gamma.float()
    return y.to(x.dtype), rstd.squeeze(-1)


def group_rms_norm_fwd_affine(x, num_channels, num_groups, gamma, eps):
    return _rms(x, num_channels, num_groups, gamma, eps)


def group_rms_norm_fwd(x, num_channels, num_groups, eps):
    return _rms(x, num_channels, num_groups, None, eps)


def __getattr__(name):
    def _missing(*a, **k):
        raise NotImplementedError(f"xllm_extension.ops.{name} is not available in the inference stub")
    return _missing
