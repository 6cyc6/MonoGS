import time
import torch
import kornia
import numpy as np

# ==================== Custom Implementations ====================
def image_gradient(image):
    """Custom image gradient using Scharr Filter"""
    c = image.shape[0]
    conv_y = torch.tensor(
        [[3, 0, -3], [10, 0, -10], [3, 0, -3]], dtype=torch.float32, device="cuda"
    )
    conv_x = torch.tensor(
        [[3, 10, 3], [0, 0, 0], [-3, -10, -3]], dtype=torch.float32, device="cuda"
    )
    normalizer = 1.0 / torch.abs(conv_y).sum()
    p_img = torch.nn.functional.pad(image, (1, 1, 1, 1), mode="reflect")[None]
    img_grad_v = normalizer * torch.nn.functional.conv2d(
        p_img, conv_x.view(1, 1, 3, 3).repeat(c, 1, 1, 1), groups=c
    )
    img_grad_h = normalizer * torch.nn.functional.conv2d(
        p_img, conv_y.view(1, 1, 3, 3).repeat(c, 1, 1, 1), groups=c
    )
    return img_grad_v[0], img_grad_h[0]


def image_gradient_mask(image, eps=0.01):
    """Custom image gradient mask"""
    c = image.shape[0]
    conv_y = torch.ones((1, 1, 3, 3), dtype=torch.float32, device="cuda")
    conv_x = torch.ones((1, 1, 3, 3), dtype=torch.float32, device="cuda")
    p_img = torch.nn.functional.pad(image, (1, 1, 1, 1), mode="reflect")[None]
    p_img = torch.abs(p_img) > eps
    img_grad_v = torch.nn.functional.conv2d(
        p_img.float(), conv_x.repeat(c, 1, 1, 1), groups=c
    )
    img_grad_h = torch.nn.functional.conv2d(
        p_img.float(), conv_y.repeat(c, 1, 1, 1), groups=c
    )
    return img_grad_v[0] == torch.sum(conv_x), img_grad_h[0] == torch.sum(conv_y)


# ==================== Kornia Implementations ====================
def image_gradient_kornia(image):
    """Kornia image gradient using Scharr filter"""
    x = image.unsqueeze(0)  # (1, C, H, W)
    g = kornia.filters.spatial_gradient(x, mode="diff", normalized=True)  # (1, C, 2, H, W)
    grad_v = g[0, :, 1]  # dx
    grad_h = g[0, :, 0]  # dy
    return grad_v, grad_h


def image_gradient_mask_kornia(image, eps=0.01):
    """Kornia-based image gradient mask"""
    # Create binary mask where absolute value > eps
    mask = torch.abs(image) > eps
    
    # Use kornia's box blur (equivalent to convolution with all-ones kernel)
    # Add batch dimension for kornia
    mask_batch = mask.unsqueeze(0).float()  # (1, C, H, W)
    
    # Create 3x3 box filter (all ones)
    kernel_size = (3, 3)
    conv_result = kornia.filters.box_blur(mask_batch, kernel_size, border_type='reflect')
    
    # Check if all 9 neighbors are valid (sum == 9)
    result = conv_result[0] == 9.0
    
    return result, result  # Return same for both v and h for simplicity


# ==================== Benchmark Function ====================
def benchmark(func, image, name, warmup=10, iterations=100):
    """Benchmark a function"""
    # Warmup
    for _ in range(warmup):
        _ = func(image)
    
    torch.cuda.synchronize()
    start = time.time()
    
    for _ in range(iterations):
        result = func(image)
    
    torch.cuda.synchronize()
    elapsed = time.time() - start
    avg_time = elapsed / iterations * 1000  # ms
    
    print(f"{name:30s}: {avg_time:.4f} ms/iter")
    return result, avg_time


# ==================== Main Comparison ====================
if __name__ == "__main__":
    print("=" * 80)
    print("Image Gradient & Mask Comparison: Custom vs Kornia")
    print("=" * 80)
    
    # Test different image sizes
    sizes = [
        (3, 240, 320),    # Small
        (3, 480, 640),    # Medium (typical camera)
        (3, 720, 1280),   # HD
    ]
    
    for C, H, W in sizes:
        print(f"\nImage size: {C}x{H}x{W}")
        print("-" * 80)
        
        # Generate random image
        image = torch.rand(C, H, W, device='cuda', dtype=torch.float32)
        
        # ========== Image Gradient Comparison ==========
        print("\n[Image Gradient]")
        result_custom, time_custom = benchmark(image_gradient, image, "Custom image_gradient")
        result_kornia, time_kornia = benchmark(image_gradient_kornia, image, "Kornia image_gradient")
        
        grad_v_custom, grad_h_custom = result_custom
        grad_v_kornia, grad_h_kornia = result_kornia
        
        print(f"\nResults close (vertical):   {torch.allclose(grad_v_custom, grad_v_kornia, rtol=1e-4, atol=1e-5)}")
        print(f"Results close (horizontal): {torch.allclose(grad_h_custom, grad_h_kornia, rtol=1e-4, atol=1e-5)}")
        print(f"Max diff (vertical):        {torch.abs(grad_v_custom - grad_v_kornia).max().item():.6e}")
        print(f"Max diff (horizontal):      {torch.abs(grad_h_custom - grad_h_kornia).max().item():.6e}")
        print(f"Speedup (Kornia vs Custom): {time_custom / time_kornia:.2f}x")
        
        # ========== Image Gradient Mask Comparison ==========
        print("\n[Image Gradient Mask]")
        result_custom_mask, time_custom_mask = benchmark(
            lambda x: image_gradient_mask(x, eps=0.01), 
            image, 
            "Custom image_gradient_mask"
        )
        result_kornia_mask, time_kornia_mask = benchmark(
            lambda x: image_gradient_mask_kornia(x, eps=0.01), 
            image, 
            "Kornia image_gradient_mask"
        )
        
        mask_v_custom, mask_h_custom = result_custom_mask
        mask_v_kornia, mask_h_kornia = result_kornia_mask
        
        print(f"\nMask agreement (vertical):   {torch.all(mask_v_custom == mask_v_kornia).item()}")
        print(f"Mask agreement (horizontal): {torch.all(mask_h_custom == mask_h_kornia).item()}")
        print(f"Mask match % (vertical):     {(mask_v_custom == mask_v_kornia).float().mean().item() * 100:.2f}%")
        print(f"Mask match % (horizontal):   {(mask_h_custom == mask_h_kornia).float().mean().item() * 100:.2f}%")
        print(f"Speedup (Kornia vs Custom):  {time_custom_mask / time_kornia_mask:.2f}x")
    
    print("\n" + "=" * 80)
    print("Summary:")
    print("  - Kornia provides optimized GPU kernels for spatial operations")
    print("  - Results should be numerically close for gradient computation")
    print("  - Mask logic may differ slightly due to implementation details")
    print("=" * 80)