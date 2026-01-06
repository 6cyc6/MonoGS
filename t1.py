import torch
import time

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

# def getWorld2View2_fast(R, t, translate=None, scale=1.0):
#     """ Modify camera pose with translation and scaling """
#     # R: (3,3) world->cam rotation
#     # t: (3,)  world->cam translation
#     # translate: (3,) world-space offset to apply to camera center
#     if translate is None:
#         translate = t.new_zeros(3)
#     else:
#         translate = translate.to(device=R.device, dtype=R.dtype)

#     # camera center in world: c = -R^T t  (if R is a rotation matrix)
#     cam_center = -(R.transpose(0, 1) @ t)

#     # modify in world space
#     cam_center = (cam_center + translate) * scale

#     # new translation: t' = -R c'
#     t_new = -(R @ cam_center)

#     Rt = torch.eye(4, device=R.device, dtype=R.dtype)
#     Rt[:3, :3] = R
#     Rt[:3, 3] = t_new
#     return Rt

# # Generate a random rotation matrix using QR decomposition
# def random_rotation_matrix():
#     # Generate random 3x3 matrix
#     A = torch.randn(3, 3)
#     # QR decomposition gives us an orthogonal matrix Q
#     Q, R_qr = torch.linalg.qr(A)
#     # Ensure det(Q) = 1 (proper rotation, not reflection)
#     if torch.det(Q) < 0:
#         Q[:, 0] *= -1
#     return Q

# R = random_rotation_matrix()
# t = torch.randn(3)
# translate = torch.tensor([1.0, 2.0, 3.0])
# scale = 2.0

# # Verify both methods give same result
# R_trans_fast = getWorld2View2_fast(R, t, translate=translate, scale=scale)
# R_trans_orig = getWorld2View2(R, t, translate=translate, scale=scale)

# print("Verifying both methods produce same result:")
# print(f"Results match: {torch.allclose(R_trans_fast, R_trans_orig, atol=1e-6)}\n")

# # Performance comparison over 100 iterations
# num_iterations = 1000

# # Warm up
# for _ in range(10):
#     _ = getWorld2View2(R, t, translate=translate, scale=scale)
#     _ = getWorld2View2_fast(R, t, translate=translate, scale=scale)

# # Benchmark original method
# start = time.time()
# for _ in range(num_iterations):
#     R = random_rotation_matrix()
#     t = torch.randn(3)
#     _ = getWorld2View2(R, t, translate=translate, scale=scale)
# time_original = time.time() - start

# # Benchmark fast method
# start = time.time()
# for _ in range(num_iterations):
#     R = random_rotation_matrix()
#     t = torch.randn(3)
#     _ = getWorld2View2_fast(R, t, translate=translate, scale=scale)
# time_fast = time.time() - start

# print(f"Performance comparison over {num_iterations} iterations:")
# print(f"{'Method':<20} {'Time (ms)':<15} {'Speedup':<10}")
# print("-" * 45)
# print(f"{'Original':<20} {time_original*1000:<15.3f} {'1.00x':<10}")
# print(f"{'Fast':<20} {time_fast*1000:<15.3f} {f'{time_original/time_fast:.2f}x':<10}")
# print(f"\nFast method is {time_original/time_fast:.2f}x faster")


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


def SO3_exp_new(theta: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """ Exponential map for SO(3) """
    # Rodrigues' rotation formula
    # theta: (3,)
    I = torch.eye(3, device=theta.device, dtype=theta.dtype)

    th2 = theta.dot(theta)  # ||theta||^2
    W = skew_sym_mat(theta) # [w]x
    W2 = torch.outer(theta, theta) - th2 * I  # [w]x^2 = ww^T - ||w||^2 I

    # angle = sqrt(th2)
    angle = torch.sqrt(th2 + 1e-12)

    if angle.item() < eps:
        # R ≈ I + W + 1/2 W^2
        return I + W + 0.5 * W2

    a = torch.sin(angle) / angle
    b = (1.0 - torch.cos(angle)) / (angle * angle)
    return I + a * W + b * W2


def V_new(theta: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """ Compute the left Jacobian of SO(3) """
    # theta: (3,)
    I = torch.eye(3, device=theta.device, dtype=theta.dtype)

    th2 = theta.dot(theta)
    W = skew_sym_mat(theta)
    W2 = torch.outer(theta, theta) - th2 * I
    angle = torch.sqrt(th2 + 1e-12)

    if angle.item() < eps:
        # V ≈ I + 1/2 W + 1/6 W^2
        return I + 0.5 * W + (1.0 / 6.0) * W2

    a = (1.0 - torch.cos(angle)) / (angle * angle)
    b = (angle - torch.sin(angle)) / (angle * angle * angle)
    return I + a * W + b * W2


# ===== TESTS =====
print("=" * 60)
print("Testing SO3_exp vs SO3_exp_new")
print("=" * 60)

# Test cases for SO3
test_thetas = [
    torch.tensor([0.1, 0.2, 0.3]),
    torch.tensor([1.0, 0.5, -0.3]),
    torch.tensor([0.01, 0.02, 0.03]),  # Small angle
    torch.tensor([1e-6, 1e-6, 1e-6]),  # Very small angle
    torch.randn(3),
    torch.randn(3),
]

print("\n--- Correctness Test for SO3_exp ---")
all_match = True
for i, theta in enumerate(test_thetas):
    R_old = SO3_exp(theta)
    R_new = SO3_exp_new(theta)
    match = torch.allclose(R_old, R_new, atol=1e-6)
    all_match = all_match and match
    
    # Verify it's a valid rotation matrix
    is_orthogonal = torch.allclose(R_new @ R_new.T, torch.eye(3), atol=1e-5)
    det_is_one = torch.allclose(torch.det(R_new), torch.tensor(1.0), atol=1e-5)
    
    status = "✓ PASSED" if match and is_orthogonal and det_is_one else "✗ FAILED"
    print(f"Test {i+1}: {status} (allclose: {match}, orthogonal: {is_orthogonal}, det=1: {det_is_one})")

print(f"\nAll SO3 tests passed: {all_match}\n")

print("--- Correctness Test for V (Left Jacobian) ---")
all_match = True
for i, theta in enumerate(test_thetas):
    V_old = V(theta)
    V_new_result = V_new(theta)
    match = torch.allclose(V_old, V_new_result, atol=1e-6)
    all_match = all_match and match
    
    status = "✓ PASSED" if match else "✗ FAILED"
    print(f"Test {i+1}: {status} (allclose: {match})")

print(f"\nAll V tests passed: {all_match}\n")

# Speed tests
print("=" * 60)
print("Speed Comparison (1000 iterations)")
print("=" * 60)

num_iterations = 1000
test_thetas_perf = [torch.randn(3) for _ in range(num_iterations)]

# Warm up
for _ in range(10):
    theta = torch.randn(3)
    _ = SO3_exp(theta)
    _ = SO3_exp_new(theta)
    _ = V(theta)
    _ = V_new(theta)

# Benchmark SO3_exp
start = time.time()
for theta in test_thetas_perf:
    _ = SO3_exp(theta)
time_so3_old = time.time() - start

start = time.time()
for theta in test_thetas_perf:
    _ = SO3_exp_new(theta)
time_so3_new = time.time() - start

# Benchmark V
start = time.time()
for theta in test_thetas_perf:
    _ = V(theta)
time_v_old = time.time() - start

start = time.time()
for theta in test_thetas_perf:
    _ = V_new(theta)
time_v_new = time.time() - start

print(f"\n--- SO3_exp Performance ---")
print(f"{'Method':<20} {'Time (ms)':<15} {'Speedup':<10}")
print("-" * 45)
print(f"{'SO3_exp (old)':<20} {time_so3_old*1000:<15.3f} {'1.00x':<10}")
print(f"{'SO3_exp_new':<20} {time_so3_new*1000:<15.3f} {f'{time_so3_old/time_so3_new:.2f}x':<10}")

if time_so3_new < time_so3_old:
    print(f"\n✓ SO3_exp_new is {time_so3_old/time_so3_new:.2f}x faster")
else:
    print(f"\n✗ SO3_exp_new is {time_so3_new/time_so3_old:.2f}x slower")

print(f"\n--- V (Left Jacobian) Performance ---")
print(f"{'Method':<20} {'Time (ms)':<15} {'Speedup':<10}")
print("-" * 45)
print(f"{'V (old)':<20} {time_v_old*1000:<15.3f} {'1.00x':<10}")
print(f"{'V_new':<20} {time_v_new*1000:<15.3f} {f'{time_v_old/time_v_new:.2f}x':<10}")

if time_v_new < time_v_old:
    print(f"\n✓ V_new is {time_v_old/time_v_new:.2f}x faster")
else:
    print(f"\n✗ V_new is {time_v_new/time_v_old:.2f}x slower")
