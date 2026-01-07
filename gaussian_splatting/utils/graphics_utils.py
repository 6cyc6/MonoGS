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

import math
from typing import NamedTuple

import numpy as np
import torch


class BasicPointCloud(NamedTuple):
    points: np.array
    colors: np.array
    normals: np.array


def getWorld2View(R, t):
    ''' Get world to view matrix from rotation and translation '''
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0
    return np.float32(Rt)

# def getWorld2View(R, t):
#     ''' Get world to view matrix from rotation and translation '''
#     Rt = np.eye(4, dtype=np.float32)
#     Rt[:3, :3] = R.T.astype(np.float32, copy=False)
#     Rt[:3, 3] = t.astype(np.float32, copy=False)
#     return Rt


# def getWorld2View2(R, t, translate=torch.tensor([0.0, 0.0, 0.0]), scale=1.0):
#     """ Modify camera pose with translation and scaling """
#     # First invert the pose to get the camera center in world space
#     # Then apply the translation and scaling
#     # Finally invert back to get the new Rt
#     translate = translate.to(R.device)
#     Rt = torch.zeros((4, 4), device=R.device)
#     # Rt[:3, :3] = R.transpose()
#     Rt[:3, :3] = R
#     Rt[:3, 3] = t
#     Rt[3, 3] = 1.0

#     C2W = torch.linalg.inv(Rt)
#     cam_center = C2W[:3, 3]
#     cam_center = (cam_center + translate) * scale
#     C2W[:3, 3] = cam_center
#     Rt = torch.linalg.inv(C2W)
#     return Rt

def getWorld2View2(R, t, translate=None, scale=1.0):
    """ Modify camera pose with translation and scaling """
    # R: (3,3) world->cam rotation
    # t: (3,)  world->cam translation
    # translate: (3,) world-space offset to apply to camera center
    if translate is None:
        translate = t.new_zeros(3)
    else:
        translate = translate.to(device=R.device, dtype=torch.float32)

    # camera center in world: c = -R^T t  (if R is a rotation matrix)
    cam_center = -(R.transpose(0, 1) @ t)

    # modify in world space
    cam_center = (cam_center + translate) * scale

    # new translation: t' = -R c'
    t_new = -(R @ cam_center)

    Rt = torch.eye(4, device=R.device, dtype=torch.float32)
    Rt[:3, :3] = R
    Rt[:3, 3] = t_new
    return Rt


def getProjectionMatrix(znear, zfar, fovX, fovY):
    # Not sure about this one
    # Mentioned in https://github.com/graphdeco-inria/gaussian-splatting/issues/399
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = -(zfar + znear) / (zfar - znear)
    P[2, 3] = -2 * (zfar * znear) / (zfar - znear)
    return P


# def getProjectionMatrix2(znear, zfar, cx, cy, fx, fy, W, H):
#     """ Build projection matrix from intrinsics """
#     # X-axis: Right, Y-axis: Top, Z-axis: Front
#     # This one is correct, check my notion notes
#     left = ((2 * cx - W) / W - 1.0) * W / 2.0
#     right = ((2 * cx - W) / W + 1.0) * W / 2.0
#     top = ((2 * cy - H) / H + 1.0) * H / 2.0
#     bottom = ((2 * cy - H) / H - 1.0) * H / 2.0
#     left = znear / fx * left
#     right = znear / fx * right
#     top = znear / fy * top
#     bottom = znear / fy * bottom
#     P = torch.zeros(4, 4)

#     z_sign = 1.0

#     P[0, 0] = 2.0 * znear / (right - left)
#     P[1, 1] = 2.0 * znear / (top - bottom)
#     P[0, 2] = (right + left) / (right - left)
#     P[1, 2] = (top + bottom) / (top - bottom)
#     P[3, 2] = z_sign
#     P[2, 2] = z_sign * zfar / (zfar - znear)
#     P[2, 3] = -(zfar * znear) / (zfar - znear)

#     return P


def getProjectionMatrix2(znear, zfar, cx, cy, fx, fy, W, H):
    """ Build projection matrix from intrinsics """
    # simplified version
    inv_d = 1.0 / (zfar - znear)

    P = torch.zeros((4, 4), dtype=torch.float32)
    P[0, 0] = 2.0 * fx / W
    P[1, 1] = 2.0 * fy / H

    # principal point offsets
    P[0, 2] = 2.0 * cx / W - 1.0
    P[1, 2] = 2.0 * cy / H - 1.0   # if your NDC y is up, this may need a sign flip

    # depth: [0,1], w=z, +Z forward
    P[3, 2] = 1.0
    P[2, 2] = zfar * inv_d
    P[2, 3] = -(zfar * znear) * inv_d
    return P


def fov2focal(fov, pixels):
    """
    tan(fov / 2) = half image size / focal = pixels / (2 * focal)
    => focal = pixels / (2 * tan(fov / 2))
    """
    return pixels / (2 * math.tan(fov / 2))


def focal2fov(focal, pixels):
    """
    tan(fov / 2) = half image size / focal = pixels / (2 * focal)
    => fov = 2 * atan(pixels / (2 * focal))
    """
    return 2 * math.atan(pixels / (2 * focal))
