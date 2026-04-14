import os
import time
import torch
from gaussian_renderer import render
import sys
from scene import GaussianModel
from utils.general_utils import safe_state
import uuid
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
import math
from cam_utils import OrbitCamera
import numpy as np
import dearpygui.dearpygui as dpg
import datetime
from PIL import Image
from train_gui_utils import DeformKeypoints
from scipy.spatial.transform import Rotation as R
import pytorch3d.ops
import cv2

# SAM imports
try:
    from segment_anything import sam_model_registry, SamPredictor
    SAM_AVAILABLE = True
    print("SAM is available!")
except ImportError:
    SAM_AVAILABLE = False
    print("Warning: SAM not available. Please install segment-anything.")

try:
    from torch_batch_svd import svd
    print('Using speed up torch_batch_svd!')
except:
    svd = torch.svd
    print('Use original torch svd!')

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

# ... (keep all the existing utility functions: getProjectionMatrix, getWorld2View2, etc.)

def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 1 / tanHalfFovX
    P[1, 1] = 1 / tanHalfFovY
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P

def getWorld2View2(R, t, translate=np.array([.0, .0, .0]), scale=1.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)

# ... (keep all matrix/quaternion utility functions)
def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    """
    Returns torch.sqrt(torch.max(0, x))
    but with a zero subgradient where x is 0.
    """
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    ret[positive_mask] = torch.sqrt(x[positive_mask])
    return ret


def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to quaternions.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(batch_dim + (9,)), dim=-1
    )

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )

    # we produce the desired quaternion multiplied by each of r, i, j, k
    quat_by_rijk = torch.stack(
        [
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    # We floor here at 0.1 but the exact level is not important; if q_abs is small,
    # the candidate won't be picked.
    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

    # if not for numerical problems, quat_candidates[i] should be same (up to a sign),
    # forall i; we pick the best-conditioned one (with the largest denominator)

    return quat_candidates[
        torch.nn.functional.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :
    ].reshape(batch_dim + (4,))


def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    r, i, j, k = torch.unbind(quaternions, -1)
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

class SAMProcessor:
    """Integrated SAM processor for the GUI"""
    def __init__(self, model_type="vit_h", checkpoint_path=None):
        self.sam = None
        self.predictor = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if SAM_AVAILABLE and checkpoint_path and os.path.exists(checkpoint_path):
            try:
                self.sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
                self.sam.to(device=self.device)
                self.predictor = SamPredictor(self.sam)
                print(f"SAM model loaded successfully on {self.device}")
            except Exception as e:
                print(f"Failed to load SAM model: {e}")
                self.sam = None
                self.predictor = None
        
        # SAM state
        self.current_image = None
        self.current_mask = None
        self.sam_points = []
        self.sam_labels = []
        self.is_sam_mode = False
        
    def set_image(self, image_rgb):
        """Set image for SAM processing"""
        if self.predictor is None:
            return False
        try:
            self.predictor.set_image(image_rgb)
            self.current_image = image_rgb
            return True
        except Exception as e:
            print(f"Error setting SAM image: {e}")
            return False
    
    def add_point(self, x, y, is_positive=True):
        """Add a point for SAM segmentation"""
        if self.predictor is None:
            return None
        
        self.sam_points.append([x, y])
        self.sam_labels.append(1 if is_positive else 0)
        
        try:
            points = np.array(self.sam_points)
            labels = np.array(self.sam_labels)
            
            masks, scores, _ = self.predictor.predict(
                point_coords=points,
                point_labels=labels,
                multimask_output=True,
            )
            
            # Select best mask
            best_mask = masks[np.argmax(scores)]
            self.current_mask = best_mask
            return best_mask
        except Exception as e:
            print(f"Error in SAM prediction: {e}")
            return None
    
    def reset_points(self):
        """Reset SAM points and mask"""
        self.sam_points = []
        self.sam_labels = []
        self.current_mask = None
    
    def get_mask_overlay(self, image, mask):
        """Create overlay visualization of mask on image"""
        if mask is None:
            return image
        
        overlay = image.copy()
        mask_colored = np.zeros_like(image)
        mask_colored[mask] = [0, 255, 0]  # Green for positive mask
        
        # Blend with original image
        overlay = cv2.addWeighted(overlay, 0.7, mask_colored, 0.3, 0)
        
        # Draw points
        for point, label in zip(self.sam_points, self.sam_labels):
            color = (0, 255, 0) if label == 1 else (0, 0, 255)
            cv2.circle(overlay, (int(point[0]), int(point[1])), 5, color, -1)
            cv2.circle(overlay, (int(point[0]), int(point[1])), 5, (255, 255, 255), 2)
        
        return overlay

# ... (keep existing NodeDriver and MiniCam classes)

class NodeDriver:
    def __init__(self):
        pass
    
    def p2dR(self, p, p0, K=8, as_quat=True):
        p = p.detach()
        nn_weight, nn_dist, nn_idx = self.cal_nn_weight(p0, p0, K=K, XisNode=True, cache_target='node')
        edges = torch.gather(p0[:, None].expand([p0.shape[0], K, p0.shape[-1]]), dim=0, index=nn_idx[..., None].expand([p0.shape[0], K, p0.shape[-1]])) - p0[:, None]
        t0_deform = None
        edges_t = torch.gather(p[:, None].expand([p.shape[0], K, p.shape[-1]]), dim=0, index=nn_idx[..., None].expand([p.shape[0], K, p.shape[-1]])) - p[:, None]
        edges, edges_t = edges / (edges.norm(dim=-1, keepdim=True) + 1e-5), edges_t / (edges_t.norm(dim=-1, keepdim=True) + 1e-5)
        W = torch.zeros([edges.shape[0], K, K], dtype=torch.float32, device=edges.device)
        W[:, range(K), range(K)] = nn_weight
        S = torch.einsum('nka,nkg,ngb->nab', edges, W, edges_t)
        U, _, V = svd(S)
        dR = torch.matmul(V, U.permute(0, 2, 1))
        if as_quat:
            dR = matrix_to_quaternion(dR)
        return dR, t0_deform
    
    def geodesic_distance_floyd(self, cur_node, K=3):
        node_num = cur_node.shape[0]
        nn_dist, nn_idx, _ = pytorch3d.ops.knn_points(cur_node[None], cur_node[None], None, None, K=K+1)
        nn_dist, nn_idx = nn_dist[0]**.5, nn_idx[0]
        dist_mat = torch.inf * torch.ones([node_num, node_num], dtype=torch.float32, device=cur_node.device)
        dist_mat.scatter_(dim=1, index=nn_idx, src=nn_dist)
        dist_mat = torch.minimum(dist_mat, dist_mat.T)
        for i in range(nn_dist.shape[0]):
            dist_mat = torch.minimum((dist_mat[:, i, None] + dist_mat[None, i, :]), dist_mat)
        return dist_mat
        
    def cal_nn_weight(self, x:torch.Tensor, nodes, K=None, method='floyd', XisNode=False, node_radius=1., cache_target=None, force=False):
        if force or cache_target is None or not hasattr(self, f'cached_{cache_target}') or not getattr(self, f'cached_{cache_target}'):
            if method == 'floyd':
                print(f'Use floyd distance, which is better for topological change!: {cache_target}')
                node_dist_mat = self.geodesic_distance_floyd(cur_node=nodes, K=2)
                floyd_nn_dist, floyd_nn_idx = node_dist_mat.sort(dim=1)
                offset = 1 if XisNode else 0
                node_nn_dist = floyd_nn_dist[:, offset:K+offset]
                node_nn_idxs = floyd_nn_idx[:, offset:K+offset]
                nn1_dist, nn1_idxs, _ = pytorch3d.ops.knn_points(x[None], nodes[None], None, None, K=1)  # N, 1
                nn1_dist, nn1_idxs = nn1_dist[0, :, 0], nn1_idxs[0, :, 0]  # N
                nn_idxs = node_nn_idxs[nn1_idxs]  # N, K
                nn_dist = node_nn_dist[nn1_idxs] + nn1_dist[:, None]  # N, K
            else:
                print(f'Use euclidean distance to calculate the nearest neighbor weight!')
                K = self.K if K is None else K
                K = K + 1 if XisNode else K  # +1 for the node itself
                # Weights of control nodes
                nn_dist, nn_idxs, _ = pytorch3d.ops.knn_points(x[None], nodes[None], None, None, K=K)  # N, K
                nn_dist, nn_idxs = nn_dist[0], nn_idxs[0]  # N, K'
                if XisNode:
                    nn_dist, nn_idxs = nn_dist[:, 1:], nn_idxs[:, 1:]  # N, K
            nn_weight = torch.exp(- nn_dist / (2 * node_radius ** 2))  # N, K
            nn_weight = nn_weight + 1e-7
            nn_weight = nn_weight / nn_weight.sum(dim=-1, keepdim=True)  # N, K
            if cache_target is not None:
                setattr(self, f'cached_{cache_target}', True)
                setattr(self, f'{cache_target}_nn_weights_dist_idxs', [nn_weight, nn_dist, nn_idxs])
        else:
            nn_weight, nn_dist, nn_idxs = getattr(self, f'{cache_target}_nn_weights_dist_idxs')
        return nn_weight, nn_dist, nn_idxs

    @torch.no_grad()
    def __call__(self, x, nodes, node_trans_bias, node_radius=1.):
        
        x = x.detach()
        rot_bias = torch.tensor([1., 0, 0, 0]).float().to(x.device)
        # Animation
        return_dict = {'d_xyz': torch.zeros_like(x), 'd_rotation': 0., 'd_scaling': 0.}
        # Initial nodes and gs
        init_node = nodes
        init_gs = x
        init_nn_weight, _, init_nn_idx = self.cal_nn_weight(x=init_gs, nodes=init_node, K=4, node_radius=node_radius, XisNode=False, cache_target='gs')
        # New nodes and gs
        nodes_t = init_node + node_trans_bias
        node_rot_bias, _ = self.p2dR(p=nodes_t, p0=init_node, K=4, as_quat=True)
        d_nn_node_rot_R = quaternion_to_matrix(node_rot_bias)[init_nn_idx]
        # Aligh the relative distance considering the rotation
        gs_t = nodes_t[init_nn_idx] + torch.einsum('gkab,gkb->gka', d_nn_node_rot_R, (init_gs[:, None] - init_node[init_nn_idx]))
        gs_t_avg = (gs_t * init_nn_weight[..., None]).sum(dim=1)
        translate = gs_t_avg - x
        return_dict['d_xyz'] = translate
        return_dict['d_rotation_bias'] = ((node_rot_bias[init_nn_idx] * init_nn_weight[..., None]).sum(dim=1) - rot_bias) + rot_bias
        return_dict['d_opacity'] = None
        return_dict['d_color'] = None
        return return_dict


class MiniCam:
    def __init__(self, c2w, width, height, fovy, fovx, znear, zfar, fid):
        self.image_width = width
        self.image_height = height
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.fid = fid
        self.c2w = c2w

        w2c = np.linalg.inv(c2w)

        # rectify...
        w2c[1:3, :3] *= -1
        w2c[:3, 3] *= -1

        self.world_view_transform = torch.tensor(w2c).transpose(0, 1).cuda().float()
        self.projection_matrix = (
            getProjectionMatrix(
                znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy
            )
            .transpose(0, 1)
            .cuda().float()
        )
        self.full_proj_transform = self.world_view_transform @ self.projection_matrix
        self.camera_center = -torch.tensor(c2w[:3, 3]).cuda()

    def reset_extrinsic(self, R, T):
        self.world_view_transform = torch.tensor(getWorld2View2(R, T)).transpose(0, 1).cuda()
        self.full_proj_transform = (
            self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

class GUI:
    def __init__(self, args, pipe) -> None:
        self.args = args
        self.pipe = pipe
        self.gui = True

        self.gaussians = GaussianModel(0)
        # Only load Gaussian model if path exists
        if hasattr(args, 'gs_path') and args.gs_path and os.path.exists(args.gs_path):
            self.gaussians.load_ply(args.gs_path)
            print(f"Loaded Gaussian model from {args.gs_path}")
        else:
            print("No Gaussian model loaded - will be created after SAM segmentation")

        bg_color = [1, 1, 1] if args.white_background else [0, 0, 0]
        self.background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        # For UI
        self.visualization_mode = 'RGB'

        self.W = args.W
        self.H = args.H
        self.cam = OrbitCamera(args.W, args.H, r=args.radius, fovy=args.fovy)
        self.vis_scale_const = None
        self.mode = "render"
        self.seed = "random"
        self.buffer_image = np.ones((self.W, self.H, 3), dtype=np.float32)
        self.video_speed = 1.

        # SAM Integration
        self.sam_processor = SAMProcessor(
            checkpoint_path=getattr(args, 'sam_checkpoint', None)
        )
        self.sam_mode = False
        self.original_image_for_sam = None
        self.sam_overlay_buffer = None

        # For Animation (existing code)
        self.animation_time = 0.
        self.is_animation = False
        self.need_update_overlay = False
        self.buffer_overlay = None
        # ... (keep all existing animation variables)
        self.animation_trans_bias = None
        self.animation_ = None

        if self.gui:
            dpg.create_context()
            self.register_dpg()
            if hasattr(args, 'gs_path') and args.gs_path and os.path.exists(args.gs_path):
                self.test_step()

    def load_image_for_sam(self, image_path):
        """Load image for SAM processing"""
        try:
            # Load image
            image_bgr = cv2.imread(image_path)
            if image_bgr is None:
                print(f"Failed to load image: {image_path}")
                return False
            
            # Convert BGR to RGB
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            
            # Set image in SAM processor
            if not self.sam_processor.set_image(image_rgb):
                print("Failed to set image in SAM processor")
                return False
            
            # Store for display
            self.original_image_for_sam = image_rgb
            
            # Convert for display buffer
            display_image = cv2.resize(image_rgb, (self.W, self.H))
            self.buffer_image = display_image.astype(np.float32) / 255.0
            
            print(f"Image loaded for SAM: {image_path}")
            return True
            
        except Exception as e:
            print(f"Error loading image: {e}")
            return False

    def handle_sam_click(self, x, y, is_positive=True):
        """Handle SAM segmentation click"""
        if not self.sam_mode or self.original_image_for_sam is None:
            return
        
        # Convert display coordinates to original image coordinates
        orig_h, orig_w = self.original_image_for_sam.shape[:2]
        orig_x = int(x * orig_w / self.W)
        orig_y = int(y * orig_h / self.H)
        
        # Add point to SAM
        mask = self.sam_processor.add_point(orig_x, orig_y, is_positive)
        
        if mask is not None:
            # Create overlay
            overlay = self.sam_processor.get_mask_overlay(
                self.original_image_for_sam, mask
            )
            
            # Resize overlay for display
            display_overlay = cv2.resize(overlay, (self.W, self.H))
            self.buffer_image = display_overlay.astype(np.float32) / 255.0
            
            print(f"SAM point added at ({orig_x}, {orig_y}), positive: {is_positive}")

    def save_sam_segmentation(self):
        """Save current SAM segmentation"""
        if self.sam_processor.current_mask is None:
            print("No segmentation available!")
            return
        
        try:
            # Create output directory
            output_dir = os.path.join(self.args.model_path, 'sam_output')
            os.makedirs(output_dir, exist_ok=True)
            
            # Save mask
            mask_path = os.path.join(output_dir, 'mask.png')
            mask_uint8 = (self.sam_processor.current_mask * 255).astype(np.uint8)
            cv2.imwrite(mask_path, mask_uint8)
            
            # Save cropped object (transparent background)
            if self.original_image_for_sam is not None:
                mask = self.sam_processor.current_mask
                image = self.original_image_for_sam
                
                # Find bounding box
                y_indices, x_indices = np.where(mask)
                if len(x_indices) > 0:
                    x_min, x_max = x_indices.min(), x_indices.max()
                    y_min, y_max = y_indices.min(), y_indices.max()
                    
                    # Crop
                    cropped_image = image[y_min:y_max+1, x_min:x_max+1]
                    cropped_mask = mask[y_min:y_max+1, x_min:x_max+1]
                    
                    # Create RGBA
                    h, w = cropped_image.shape[:2]
                    result = np.zeros((h, w, 4), dtype=np.uint8)
                    result[:, :, :3] = cropped_image
                    result[:, :, 3] = (cropped_mask * 255).astype(np.uint8)
                    
                    # Save
                    object_path = os.path.join(output_dir, 'object.png')
                    pil_image = Image.fromarray(result, 'RGBA')
                    pil_image.save(object_path)
                    
                    print(f"SAM segmentation saved to {output_dir}")
                    return True
            
        except Exception as e:
            print(f"Error saving SAM segmentation: {e}")
            return False

    def register_dpg(self):
        ### register texture
        with dpg.texture_registry(show=False):
            dpg.add_raw_texture(
                self.W,
                self.H,
                self.buffer_image,
                format=dpg.mvFormat_Float_rgb,
                tag="_texture",
            )

        ### register window
        # the rendered image, as the primary window
        with dpg.window(
            tag="_primary_window",
            width=self.W,
            height=self.H,
            pos=[0, 0],
            no_move=True,
            no_title_bar=True,
            no_scrollbar=True,
        ):
            # add the texture
            dpg.add_image("_texture")

        # control window
        with dpg.window(
            label="Control",
            tag="_control_window",
            width=600,
            height=self.H,
            pos=[self.W, 0],
            no_move=True,
            no_title_bar=True,
        ):
            # button theme
            with dpg.theme() as theme_button:
                with dpg.theme_component(dpg.mvButton):
                    dpg.add_theme_color(dpg.mvThemeCol_Button, (23, 3, 18))
                    dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (51, 3, 47))
                    dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (83, 18, 83))
                    dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 5)
                    dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 3, 3)

            # timer stuff
            with dpg.group(horizontal=True):
                dpg.add_text("Infer time: ")
                dpg.add_text("no data", tag="_log_infer_time")

            def callback_setattr(sender, app_data, user_data):
                setattr(self, user_data, app_data)

            # SAM Section - NEW!
            with dpg.collapsing_header(label="SAM Segmentation", default_open=True):
                
                # Load image for SAM
                def callback_load_image_sam(sender, app_data):
                    for k, v in app_data["selections"].items():
                        if self.load_image_for_sam(v):
                            dpg.set_value("_log_sam_image", k)
                            self.sam_mode = True
                            dpg.configure_item("_button_sam_mode", label="Exit SAM Mode")
                        break

                with dpg.file_dialog(
                    directory_selector=False,
                    show=False,
                    callback=callback_load_image_sam,
                    file_count=1,
                    tag="file_dialog_sam",
                    width=700,
                    height=400,
                ):
                    dpg.add_file_extension("Images{.jpg,.jpeg,.png}")

                with dpg.group(horizontal=True):
                    dpg.add_button(
                        label="Load Image",
                        callback=lambda: dpg.show_item("file_dialog_sam"),
                    )
                    dpg.add_text("", tag="_log_sam_image")

                # SAM mode toggle
                def callback_sam_mode_toggle(sender, app_data):
                    if self.sam_mode:
                        self.sam_mode = False
                        self.sam_processor.reset_points()
                        dpg.configure_item("_button_sam_mode", label="Enter SAM Mode")
                        # Return to normal rendering if Gaussian model exists
                        if hasattr(self.gaussians, '_xyz') and self.gaussians._xyz is not None:
                            self.test_step()
                    else:
                        if self.original_image_for_sam is not None:
                            self.sam_mode = True
                            dpg.configure_item("_button_sam_mode", label="Exit SAM Mode")

                dpg.add_button(
                    label="Enter SAM Mode",
                    tag="_button_sam_mode",
                    callback=callback_sam_mode_toggle,
                )
                dpg.bind_item_theme("_button_sam_mode", theme_button)

                with dpg.group(horizontal=True):
                    def callback_reset_sam(sender, app_data):
                        self.sam_processor.reset_points()
                        if self.original_image_for_sam is not None:
                            display_image = cv2.resize(self.original_image_for_sam, (self.W, self.H))
                            self.buffer_image = display_image.astype(np.float32) / 255.0
                    
                    dpg.add_button(
                        label="Reset Points",
                        callback=callback_reset_sam,
                    )
                    dpg.bind_item_theme("Reset Points", theme_button)

                    def callback_save_sam(sender, app_data):
                        self.save_sam_segmentation()
                    
                    dpg.add_button(
                        label="Save Segmentation",
                        callback=callback_save_sam,
                    )
                    dpg.bind_item_theme("Save Segmentation", theme_button)

                # SAM instructions
                dpg.add_text("SAM Instructions:")
                dpg.add_text("• Left click: Add positive point")
                dpg.add_text("• Right click: Add negative point")
                dpg.add_text("• SHIFT+Left: Generate 3D from mask")

            # ... (keep all existing GUI sections: Initialize, Motion Editing, etc.)
            
            # init stuff
            with dpg.collapsing_header(label="Initialize", default_open=True):
                # ... (existing initialization code)
                pass

        ### register mouse handlers
        def callback_mouse_click_unified(sender, app_data):
            if not dpg.is_item_focused("_primary_window"):
                return
            
            x, y = app_data
            
            if self.sam_mode:
                # SAM mode - handle segmentation
                is_positive = sender == dpg.mvMouseButton_Left
                self.handle_sam_click(x, y, is_positive)
            else:
                # Normal mode - existing animation functionality
                if hasattr(self, 'callback_keypoint_add'):
                    self.callback_keypoint_add(sender, app_data)

        def callback_camera_drag_rotate(sender, app_data):
            if not dpg.is_item_focused("_primary_window"):
                return
            
            if self.sam_mode:
                return  # Disable camera movement in SAM mode
            
            dx = app_data[1]
            dy = app_data[2]
            self.cam.orbit(dx, dy)
            self.need_update = True

        def callback_camera_drag_pan(sender, app_data):
            if not dpg.is_item_focused("_primary_window"):
                return
            
            if self.sam_mode:
                return  # Disable camera movement in SAM mode
            
            dx = app_data[1]
            dy = app_data[2]
            self.cam.pan(dx, dy)
            self.need_update = True

        def callback_camera_wheel_scale(sender, app_data):
            if not dpg.is_item_focused("_primary_window"):
                return
            
            if self.sam_mode:
                return  # Disable camera movement in SAM mode
            
            delta = app_data
            self.cam.scale(delta)
            self.need_update = True

        with dpg.handler_registry():
            # Mouse handlers
            dpg.add_mouse_click_handler(
                button=dpg.mvMouseButton_Left, 
                callback=callback_mouse_click_unified
            )
            dpg.add_mouse_click_handler(
                button=dpg.mvMouseButton_Right, 
                callback=callback_mouse_click_unified
            )
            
            # Camera movement (disabled in SAM mode)
            dpg.add_mouse_drag_handler(
                button=dpg.mvMouseButton_Left,
                callback=callback_camera_drag_rotate,
            )
            dpg.add_mouse_wheel_handler(callback=callback_camera_wheel_scale)
            dpg.add_mouse_drag_handler(
                button=dpg.mvMouseButton_Middle, 
                callback=callback_camera_drag_pan
            )

        dpg.create_viewport(
            title="SC-GS with SAM",
            width=self.W + 600,
            height=self.H + (45 if os.name == "nt" else 0),
            resizable=False,
        )

        ### global theme
        with dpg.theme() as theme_no_padding:
            with dpg.theme_component(dpg.mvAll):
                dpg.add_theme_style(
                    dpg.mvStyleVar_WindowPadding, 0, 0, category=dpg.mvThemeCat_Core
                )
                dpg.add_theme_style(
                    dpg.mvStyleVar_FramePadding, 0, 0, category=dpg.mvThemeCat_Core
                )
                dpg.add_theme_style(
                    dpg.mvStyleVar_CellPadding, 0, 0, category=dpg.mvThemeCat_Core
                )

        dpg.bind_item_theme("_primary_window", theme_no_padding)
        dpg.setup_dearpygui()
        dpg.show_viewport()

    @torch.no_grad()
    def test_step(self, specified_cam=None):
        if self.sam_mode:
            # In SAM mode, just update the texture with current buffer
            if self.gui:
                dpg.set_value("_texture", self.buffer_image)
            return self.buffer_image

        # Original rendering code for Gaussian model
        # ... (keep existing test_step implementation)
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        starter.record()

        # Check if we have a valid Gaussian model
        if not hasattr(self.gaussians, '_xyz') or self.gaussians._xyz is None:
            # No Gaussian model loaded, show placeholder
            placeholder = np.ones((self.H, self.W, 3), dtype=np.float32) * 0.5
            if self.gui:
                dpg.set_value("_texture", placeholder)
            return placeholder

        # ... (rest of existing test_step code)
        
        ender.record()
        torch.cuda.synchronize()
        t = starter.elapsed_time(ender)

        if self.gui:
            dpg.set_value("_log_infer_time", f"{t:.4f}ms ({int(1000/t)} FPS)")
            dpg.set_value("_texture", self.buffer_image)
        
        return self.buffer_image

    # ... (keep all other existing methods)

    def render(self):
        assert self.gui
        while dpg.is_dearpygui_running():
            if not self.sam_mode and hasattr(self, 'should_render_customized_trajectory') and self.should_render_customized_trajectory:
                self.render_customized_trajectory(use_spiral=self.should_render_customized_trajectory_spiral)
            self.test_step()
            dpg.render_dearpygui_frame()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    pp = PipelineParams(parser)
    
    parser.add_argument('--gs_path', type=str, help="path to the Gaussian Splatting model")
    parser.add_argument('--sam_checkpoint', type=str, help="path to SAM checkpoint")
    parser.add_argument('--gui', action='store_true', help="start a GUI")
    parser.add_argument('--W', type=int, default=800, help="GUI width")
    parser.add_argument('--H', type=int, default=800, help="GUI height")
    parser.add_argument('--elevation', type=float, default=0, help="default GUI camera elevation")
    parser.add_argument('--radius', type=float, default=5, help="default GUI camera radius from center")
    parser.add_argument('--fovy', type=float, default=50, help="default GUI camera fovy")
    parser.add_argument('--white_background', action='store_true', default=False, help="use white background in GUI")
    parser.add_argument('--model_path', type=str, default='./', help="path to save the model and logs")

    # ... (other existing arguments)

    args = parser.parse_args(sys.argv[1:])

    print("Starting GUI with SAM integration...")
    safe_state(args.quiet if hasattr(args, 'quiet') else False)

    gui = GUI(args=args, pipe=pp.extract(args))
    gui.render()
    
    print("\nSession complete.")