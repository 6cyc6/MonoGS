import torch
import lietorch
import numpy as np


def rt2mat(R, T):
    mat = np.eye(4)
    mat[0:3, 0:3] = R
    mat[0:3, 3] = T
    return mat


# def skew_sym_mat(x):
#     device = x.device
#     dtype = x.dtype
#     ssm = torch.zeros(3, 3, device=device, dtype=dtype)
#     ssm[0, 1] = -x[2]
#     ssm[0, 2] = x[1]
#     ssm[1, 0] = x[2]
#     ssm[1, 2] = -x[0]
#     ssm[2, 0] = -x[1]
#     ssm[2, 1] = x[0]
#     return ssm

# def V(theta: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
#     """ Compute the left Jacobian of SO(3) """
#     # theta: (3,)
#     I = torch.eye(3, device=theta.device, dtype=theta.dtype)

#     th2 = theta.dot(theta)
#     W = skew_sym_mat(theta)
#     W2 = torch.outer(theta, theta) - th2 * I
#     angle = torch.sqrt(th2 + 1e-12)

#     if angle.item() < eps:
#         # V ≈ I + 1/2 W + 1/6 W^2
#         return I + 0.5 * W + (1.0 / 6.0) * W2

#     a = (1.0 - torch.cos(angle)) / (angle * angle)
#     b = (angle - torch.sin(angle)) / (angle * angle * angle)
#     return I + a * W + b * W2


# def SO3_exp(theta: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
#     # theta: (3,)
#     I = torch.eye(3, device=theta.device, dtype=theta.dtype)

#     th2 = theta.dot(theta)                 # ||theta||^2
#     W = skew_sym_mat(theta)
#     W2 = torch.outer(theta, theta) - th2 * I

#     # angle = sqrt(th2)
#     angle = torch.sqrt(th2 + 1e-12)

#     if angle.item() < eps:
#         # R ≈ I + W + 1/2 W^2
#         return I + W + 0.5 * W2

#     a = torch.sin(angle) / angle
#     b = (1.0 - torch.cos(angle)) / (angle * angle)
#     return I + a * W + b * W2


def skew_sym_mat(w: torch.Tensor) -> torch.Tensor:
    """ Create skew-symmetric matrix from vector """
    wx, wy, wz = w.unbind()
    O = torch.zeros((), device=w.device, dtype=w.dtype)
    return torch.stack([
        torch.stack([ O, -wz,  wy]),
        torch.stack([ wz,  O, -wx]),
        torch.stack([-wy, wx,  O]),
    ])


def SO3_exp(theta):
    """ Exponential map for SO(3) """
    # Rodrigues' rotation formula
    device = theta.device
    dtype = theta.dtype

    W = skew_sym_mat(theta)
    W2 = W @ W
    angle = torch.norm(theta)
    I = torch.eye(3, device=device, dtype=dtype)
    if angle < 1e-5:
        return I + W + 0.5 * W2
    else:
        return (
            I
            + (torch.sin(angle) / angle) * W
            + ((1 - torch.cos(angle)) / (angle**2)) * W2
        )


def V(theta):
    """ Compute the left Jacobian of SO(3) """
    dtype = theta.dtype
    device = theta.device
    I = torch.eye(3, device=device, dtype=dtype)
    W = skew_sym_mat(theta)
    W2 = W @ W
    angle = torch.norm(theta)
    if angle < 1e-5:
        V = I + 0.5 * W + (1.0 / 6.0) * W2
    else:
        V = (
            I
            + W * ((1.0 - torch.cos(angle)) / (angle**2))
            + W2 * ((angle - torch.sin(angle)) / (angle**3))
        )
    return V


def SE3_exp(tau):
    dtype = tau.dtype
    device = tau.device

    rho = tau[:3]
    theta = tau[3:]
    R = SO3_exp(theta)
    t = V(theta) @ rho

    T = torch.eye(4, device=device, dtype=dtype)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def SE3_exp_Rt(tau: torch.Tensor):
    rho = tau[:3]
    theta = tau[3:]
    R = SO3_exp(theta)
    t = V(theta) @ rho
    return R, t


# def update_pose(camera, converged_threshold=1e-4):
#     tau = torch.cat([camera.cam_trans_delta, camera.cam_rot_delta], axis=0)

#     T_w2c = torch.eye(4, device=tau.device)
#     T_w2c[0:3, 0:3] = camera.R
#     T_w2c[0:3, 3] = camera.T

#     new_w2c = SE3_exp(tau) @ T_w2c

#     new_R = new_w2c[0:3, 0:3]
#     new_T = new_w2c[0:3, 3]

#     converged = tau.norm() < converged_threshold
#     camera.update_RT(new_R, new_T)

#     camera.cam_rot_delta.data.fill_(0)
#     camera.cam_trans_delta.data.fill_(0)
#     return converged


@torch.no_grad()
def update_pose(camera, converged_threshold=1e-4):
    tau = torch.cat((camera.cam_trans_delta, camera.cam_rot_delta), dim=0)

    dR, dt = SE3_exp_Rt(tau)

    # Apply left-multiplication update: new = exp(tau) @ old
    R = camera.R.float()
    T = camera.T.float()
    new_R = dR @ R
    new_T = dR @ T + dt

    converged = tau.norm() < converged_threshold
    camera.update_RT(new_R, new_T)

    camera.cam_rot_delta.zero_()
    camera.cam_trans_delta.zero_()
    
    return converged
