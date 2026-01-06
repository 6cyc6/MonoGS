#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import random
import sys
from datetime import datetime

import numpy as np
import torch

_STRIP_IDX_CPU = torch.tensor([0, 1, 2, 4, 5, 8], dtype=torch.long)

# @torch.jit.script  # Cannot use with multiprocessing (PickleError)
def inverse_sigmoid(x) -> torch.Tensor:
    return torch.log(x / (1 - x))


def PILtoTorch(pil_image, resolution):
    resized_image_PIL = pil_image.resize(resolution)
    resized_image = torch.from_numpy(np.array(resized_image_PIL)) / 255.0
    if len(resized_image.shape) == 3:
        return resized_image.permute(2, 0, 1)
    else:
        return resized_image.unsqueeze(dim=-1).permute(2, 0, 1)


def PILtoTorch2(pil_image):
    # resized_image_PIL = pil_image.resize(resolution)
    resized_image = torch.from_numpy(np.array(pil_image)) / 255.0
    if len(resized_image.shape) == 3:
        return resized_image.permute(2, 0, 1)
    else:
        return resized_image.unsqueeze(dim=-1).permute(2, 0, 1)


def get_expon_lr_func(
    lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000
):
    """
    Copied from Plenoxels

    Continuous learning rate decay function. Adapted from JaxNeRF
    The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
    is log-linearly interpolated elsewhere (equivalent to exponential decay).
    If lr_delay_steps>0 then the learning rate will be scaled by some smooth
    function of lr_delay_mult, such that the initial learning rate is
    lr_init*lr_delay_mult at the beginning of optimization but will be eased back
    to the normal learning rate when steps>lr_delay_steps.
    :param conf: config subtree 'lr' or similar
    :param max_steps: int, the number of steps during optimization.
    :return HoF which takes step as input
    """
    # def helper(step):
    #     if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
    #         # Disable this parameter
    #         return 0.0
    #     if lr_delay_steps > 0:
    #         # A kind of reverse cosine decay.
    #         delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
    #             0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
    #         )
    #     else:
    #         delay_rate = 1.0
    #     t = np.clip(step / max_steps, 0, 1)
    #     log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
    #     return delay_rate * log_lerp

    return helper
    # return helper(lr_init=lr_init, lr_final=lr_final,
    #               lr_delay_steps=lr_delay_steps, lr_delay_mult=lr_delay_mult, max_steps=max_steps)


def helper(
    step, lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000
):
    if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
        # Disable this parameter
        return 0.0
    if lr_delay_steps > 0:
        # A kind of reverse cosine decay.
        delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
            0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
        )
    else:
        delay_rate = 1.0
    t = np.clip(step / max_steps, 0, 1)
    log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
    return delay_rate * log_lerp


# def strip_lowerdiag(L):
#     """ Take lower-diagonal matrix to 6-vector """
#     uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device="cuda")

#     uncertainty[:, 0] = L[:, 0, 0]
#     uncertainty[:, 1] = L[:, 0, 1]
#     uncertainty[:, 2] = L[:, 0, 2]
#     uncertainty[:, 3] = L[:, 1, 1]
#     uncertainty[:, 4] = L[:, 1, 2]
#     uncertainty[:, 5] = L[:, 2, 2]
#     return uncertainty

def strip_lowerdiag(L):
    """ Take lower-diagonal matrix to 6-vector """
    idx = _STRIP_IDX_CPU.to(device=L.device)
    return L.reshape(L.shape[0], 9).index_select(1, idx)


def strip_symmetric(sym):
    """ Strip symmetric matrix to 6-vector """
    # take the lower-diagonal part
    return strip_lowerdiag(sym)


# def build_rotation(r):
#     """ Quaternion to rotation matrix """   
#     # formula (10) in appendix.
#     norm = torch.sqrt(
#         r[:, 0] * r[:, 0] + r[:, 1] * r[:, 1] + r[:, 2] * r[:, 2] + r[:, 3] * r[:, 3]
#     )

#     q = r / norm[:, None]

#     R = torch.zeros((q.size(0), 3, 3), device="cuda")

#     r = q[:, 0]
#     x = q[:, 1]
#     y = q[:, 2]
#     z = q[:, 3]

#     R[:, 0, 0] = 1 - 2 * (y * y + z * z)
#     R[:, 0, 1] = 2 * (x * y - r * z)
#     R[:, 0, 2] = 2 * (x * z + r * y)
#     R[:, 1, 0] = 2 * (x * y + r * z)
#     R[:, 1, 1] = 1 - 2 * (x * x + z * z)
#     R[:, 1, 2] = 2 * (y * z - r * x)
#     R[:, 2, 0] = 2 * (x * z - r * y)
#     R[:, 2, 1] = 2 * (y * z + r * x)
#     R[:, 2, 2] = 1 - 2 * (x * x + y * y)
#     return R


# @torch.jit.script  # Cannot use with multiprocessing (PickleError)
def normalize(x: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """Normalizes a given input tensor to unit length.

    Args:
        x: Input tensor of shape (N, dims).
        eps: A small value to avoid division by zero. Defaults to 1e-9.

    Returns:
        Normalized tensor of shape (N, dims).
    """
    return x / x.norm(p=2, dim=-1).clamp(min=eps, max=None).unsqueeze(-1)


# @torch.jit.script  # Cannot use with multiprocessing (PickleError)
def matrix_from_quat(quaternions: torch.Tensor) -> torch.Tensor:
    """Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: The quaternion orientation in (w, x, y, z). Shape is (..., 4).

    Returns:
        Rotation matrices. The shape is (..., 3, 3).

    Reference:
        https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/transforms/rotation_conversions.py#L41-L70
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


def build_rotation(r):
    """ Quaternion to rotation matrix """
    R = matrix_from_quat(normalize(r))
    return R


# def build_scaling_rotation(s, r):
#     """ Build Covariance matrix from scaling and rotation """
#     # formula (6): Sigma = RS S^T R^T
#     S = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda")
#     R = build_rotation(r)

#     S[:, 0, 0] = s[:, 0]
#     S[:, 1, 1] = s[:, 1]
#     S[:, 2, 2] = s[:, 2]

#     return R @ S

def build_scaling_rotation(s, r):
    """ Build Covariance matrix from scaling and rotation """
    # formula (6): Sigma = RS S^T R^T
    R = build_rotation(r)

    return R * s[:, None, :]


def safe_state(silent):
    old_f = sys.stdout

    class F:
        def __init__(self, silent):
            self.silent = silent

        def write(self, x):
            if not self.silent:
                if x.endswith("\n"):
                    old_f.write(
                        x.replace(
                            "\n",
                            " [{}]\n".format(
                                str(datetime.now().strftime("%d/%m %H:%M:%S"))
                            ),
                        )
                    )
                else:
                    old_f.write(x)

        def flush(self):
            old_f.flush()

    # sys.stdout = F(silent)

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.set_device(torch.device("cuda:0"))
