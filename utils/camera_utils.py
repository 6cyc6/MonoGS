import torch
from torch import nn

from gaussian_splatting.utils.graphics_utils import getProjectionMatrix2
from utils.slam_utils import image_gradient, image_gradient_mask
import torch.nn.functional as F


class Camera(nn.Module):
    """ Camera Model Class """
    def __init__(
        self,
        uid,
        color,
        depth,
        gt_T,
        projection_matrix,
        fx,
        fy,
        cx,
        cy,
        fovx,
        fovy,
        image_height,
        image_width,
        device="cuda:0",
        mask=None,
    ):
        super(Camera, self).__init__()
        self.uid = uid
        self.device = device

        # camera init pose and ground truth pose (transforms from world to camera)
        self.T = torch.eye(4, device=device).to(torch.float32)
        self.T_gt = gt_T.to(device=device).to(torch.float32).clone()
        
        # input info
        self.original_image = color
        self.depth = depth
        self.grad_mask = None
        self.mask = mask

        # camera intrinsics
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.FoVx = fovx
        self.FoVy = fovy
        self.image_height = image_height
        self.image_width = image_width

        # camera pose deltas
        self.cam_rot_delta = nn.Parameter(
            torch.zeros(3, requires_grad=True, device=device)
        )
        self.cam_trans_delta = nn.Parameter(
            torch.zeros(3, requires_grad=True, device=device)
        )

        # exposure parameters
        # a and b for exposure compensation: I_corrected = I * exp(a) + b
        self.exposure_a = nn.Parameter(
            torch.tensor([0.0], requires_grad=True, device=device)
        )
        self.exposure_b = nn.Parameter(
            torch.tensor([0.0], requires_grad=True, device=device)
        )

        # projection matrix (camera frame to NDC frame)
        self.projection_matrix = projection_matrix.to(device=device)
        

    @staticmethod
    def init_from_dataset(dataset, idx, projection_matrix, custom=False):
        """ Initialize Camera from dataset """
        if custom:
            gt_color, gt_depth, gt_pose, gt_mask = dataset[idx]
            return Camera(
                idx,
                gt_color,
                gt_depth,
                gt_pose,
                projection_matrix,
                dataset.fx,
                dataset.fy,
                dataset.cx,
                dataset.cy,
                dataset.fovx,
                dataset.fovy,
                dataset.height,
                dataset.width,
                device=dataset.device,
                mask=gt_mask,
            )
            
        gt_color, gt_depth, gt_pose = dataset[idx]
        return Camera(
            idx,
            gt_color,
            gt_depth,
            gt_pose,
            projection_matrix,
            dataset.fx,
            dataset.fy,
            dataset.cx,
            dataset.cy,
            dataset.fovx,
            dataset.fovy,
            dataset.height,
            dataset.width,
            device=dataset.device,
        )

    @staticmethod
    def init_from_gui(uid, T, FoVx, FoVy, fx, fy, cx, cy, H, W):
        """ Initialize Camera from GUI """
        projection_matrix = getProjectionMatrix2(
            znear=0.01, zfar=100.0, fx=fx, fy=fy, cx=cx, cy=cy, W=W, H=H
        ).transpose(0, 1)
        return Camera(
            uid, None, None, T, projection_matrix, fx, fy, cx, cy, FoVx, FoVy, H, W
        )

    @property
    def world_view_transform(self):
        return self.T.transpose(0, 1).to(device=self.device)

    @property
    def full_proj_transform(self):
        """ Projection Matrix from world to NDC """
        return (
            self.world_view_transform.unsqueeze(0).bmm(
                self.projection_matrix.unsqueeze(0)
            )
        ).squeeze(0)

    @property
    def camera_center(self):
        return self.world_view_transform #TODO: Need to invert for high order SHs by inverse_t(self.world_view_transform).
    
    
    def get_inv_K(self):
        """ Get camera intrinsics """
        intrinsics = torch.tensor(
            [
                [self.fx, 0.0, self.cx],
                [0.0, self.fy, self.cy],
                [0.0, 0.0, 1.0],
            ],
            device=self.device,
        )
        
        return torch.linalg.inv(intrinsics)
    
        
    def compute_grad_mask(self, config):
        """ Compute gradient mask and rgb pixel mask """
        # Use Scharr filter to extract edge pixels for tracking
        edge_threshold = config["Training"]["edge_threshold"]

        gray_img = self.original_image.mean(dim=0, keepdim=True)
        gray_grad_v, gray_grad_h = image_gradient(gray_img)
        mask_v, mask_h = image_gradient_mask(gray_img)
        gray_grad_v = gray_grad_v * mask_v
        gray_grad_h = gray_grad_h * mask_h
        img_grad_intensity = torch.sqrt(gray_grad_v**2 + gray_grad_h**2)
        
        if config["Dataset"]["type"] == "replica":
            size = 32
            multiplier = edge_threshold
            _, h, w = self.original_image.shape
            I = img_grad_intensity.unsqueeze(0)
            I_unf = F.unfold(I, size, stride=size)
            median_patch, _ = torch.median(I_unf, dim=1,keepdim=True)
            mask = (I_unf > (median_patch * multiplier)).float()
            I_f = F.fold(mask, I.shape[-2:],size,stride=size).squeeze(0)
            self.grad_mask = I_f
        else:
            median_img_grad_intensity = img_grad_intensity.median()
            self.grad_mask = (
                img_grad_intensity > median_img_grad_intensity * edge_threshold
            )

        gt_image = self.original_image.cuda()
        rgb_boundary_threshold = config["Training"]["rgb_boundary_threshold"]
        rgb_pixel_mask = (gt_image.sum(dim=0, keepdim=True) > rgb_boundary_threshold)
        self.rgb_pixel_mask = rgb_pixel_mask & self.grad_mask
        self.rgb_pixel_mask_mapping = rgb_pixel_mask
        
        # Visualize grad_mask (optional, enable for debugging)
        if config.get("visualize_grad_mask", False):
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            
            # Original image
            axes[0, 0].imshow(self.original_image.cpu().permute(1, 2, 0))
            axes[0, 0].set_title("Original RGB")
            axes[0, 0].axis('off')
            
            # Grayscale
            axes[0, 1].imshow(gray_img.squeeze().cpu(), cmap='gray')
            axes[0, 1].set_title("Grayscale")
            axes[0, 1].axis('off')
            
            # Gradient intensity
            im = axes[0, 2].imshow(img_grad_intensity.squeeze().cpu(), cmap='hot')
            axes[0, 2].set_title(f"Gradient Intensity\nMedian={img_grad_intensity.median():.4f}")
            axes[0, 2].axis('off')
            
            # RGB boundary mask
            axes[1, 0].imshow(rgb_pixel_mask.squeeze().cpu(), cmap='gray')
            pct1 = rgb_pixel_mask.float().mean().item() * 100
            axes[1, 0].set_title(f"RGB Boundary Mask\n{pct1:.1f}% pixels")
            axes[1, 0].axis('off')
            
            # Gradient mask
            axes[1, 1].imshow(self.grad_mask.squeeze().cpu(), cmap='gray')
            pct2 = self.grad_mask.float().mean().item() * 100
            axes[1, 1].set_title(f"Gradient Mask\n{pct2:.1f}% pixels (edges)")
            axes[1, 1].axis('off')
            
            # Final tracking mask
            axes[1, 2].imshow(self.rgb_pixel_mask.squeeze().cpu(), cmap='gray')
            pct3 = self.rgb_pixel_mask.float().mean().item() * 100
            axes[1, 2].set_title(f"Tracking Mask (AND)\n{pct3:.1f}% pixels", color='red', weight='bold')
            axes[1, 2].axis('off')
            
            plt.suptitle(f"Grad Mask - Frame {self.uid}\nEdge Threshold: {edge_threshold}", fontsize=14)
            plt.tight_layout()
            plt.savefig(f"grad_mask_frame_{self.uid:04d}.png", dpi=100, bbox_inches='tight')
            plt.close()
            print(f"Saved grad_mask_frame_{self.uid:04d}.png - Tracking uses {pct3:.1f}% pixels (edges only)")
        
        if self.depth is not None:
            self.gt_depth = torch.from_numpy(self.depth).to(dtype=torch.float32, device=self.device).unsqueeze(0)

    
    def clean(self):
        """ Clean buffer to save memory """
        self.original_image = None
        self.depth = None
        self.grad_mask = None

        self.cam_rot_delta = None
        self.cam_trans_delta = None

        self.exposure_a = None
        self.exposure_b = None
        
        self.rgb_pixel_mask = None
        self.rgb_pixel_mask_mapping = None
        self.gt_depth = None


class CameraMsg():
    """ Camera Message Class """
    # save the ID, pose, and ground truth pose 
    def __init__(self, Camera: Camera):
        self.uid = Camera.uid
        self.T = Camera.T
        self.T_gt = Camera.T_gt