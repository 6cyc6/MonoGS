import torch

def getProjectionMatrix1(znear, zfar, cx, cy, fx, fy, W, H):
    """ Build projection matrix from intrinsics """
    # X-axis: Right, Y-axis: Top, Z-axis: Front
    # This one is correct, check my notion notes
    left = ((2 * cx - W) / W - 1.0) * W / 2.0
    right = ((2 * cx - W) / W + 1.0) * W / 2.0
    top = ((2 * cy - H) / H + 1.0) * H / 2.0
    bottom = ((2 * cy - H) / H - 1.0) * H / 2.0
    left = znear / fx * left
    right = znear / fx * right
    top = znear / fy * top
    bottom = znear / fy * bottom
    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)

    return P


def getProjectionMatrix2(znear, zfar, cx, cy, fx, fy, W, H):
    """ Build projection matrix from intrinsics """
    # simplified version
    inv_d = 1.0 / (zfar - znear)

    P = torch.zeros((4, 4))
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

znear = 0.01
zfar = 100.0
W = 480
H = 640
cx = 320.1
cy = 247.6
fx = 535.4
fy = 539.2

P1 = getProjectionMatrix1(znear, zfar, cx, cy, fx, fy, W, H)
print(P1)
P2 = getProjectionMatrix2(znear, zfar, cx, cy, fx, fy, W, H)
print(P2)

print(torch.allclose(P1, P2))
print(P1.dtype, P2.dtype)