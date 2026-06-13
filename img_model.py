import numpy as np

from utils.lighting import *

import torch
import torch.nn.functional as F
import yaml
import os

class ImagingModel:
    def __init__(self, device, psf_mask=None):
        self.read_config()

        diameter = set_diameter(self.NA, self.layers, self.z_res)
        self.ray_num, ray_check = set_incidental_light(diameter, self.apa_size)
        self.intensity = 1 / self.ray_num

        self.ray_mat = range_matrix_generation(self.ray_num, ray_check, self.layers,
                                               diameter, self.apa_size, self.xy_res, self.z_res)

        ray_check = np.array(ray_check).reshape((self.apa_size, self.apa_size))
        self.ray_sel = ray_check != -1
        self.rescale = 1
        if psf_mask is not None:
            mask = np.load(psf_mask, allow_pickle=True)
            self.rescale = mask.mean()
            self.ray_sel = np.logical_and(self.ray_sel, mask)
            self.ray_mat = self.ray_mat[ray_check[self.ray_sel]]

        self.ray_mat = torch.from_numpy(self.ray_mat).clone()  # (ray_num, 2*layer-1, , )
        self.ray_mat = self.ray_mat.to(torch.float32).to(device)

        self.padding_size = int((self.ray_mat.size()[2] - 1) / 2)

    def read_config(self):
        config_path = os.path.join(os.getcwd(), "config.yaml")
        with open(config_path) as f:
            params = yaml.load(f, Loader=yaml.SafeLoader)
        img_params = params["ImgModelParams"]

        self.apa_size = img_params["apa_size"]
        self.xy_res = img_params["xy_res"]
        self.z_res = img_params["z_res"]
        self.NA = img_params["NA"]
        self.layers = img_params["layers"]

    def __call__(self, omega):
        img_list = []
        for s in range(self.layers):
            out = F.conv2d(omega[0],
                           self.ray_mat[:, self.layers - 1 - s:2 * self.layers - 1 - s, :, :],
                           padding=self.padding_size)
            out = self.intensity * torch.sum(torch.exp(out), dim=1).squeeze()
            out = torch.clamp(out / self.rescale, 0, 1)
            img_list.append(out)
        return img_list


class MaskedImagingModel:
    """
    Hybrid imaging model combining physical PSF with snapshot masks.
    
    This model:
    1. Applies PSF (light propagation) to 3D volume (physical optics)
    2. Applies coded masks to each depth slice (MAUF19 hardware)
    3. Returns masked depth slices
    
    This represents the actual MAUF19 system where:
    - PSF represents the optical blur at each focal depth
    - Masks represent the 2×2 coded aperture pattern
    
    The forward model is: m_s = Σ_xy PSF_s(x,y) * ω(x,y,z_s) * mask_s(x,y)
    """
    
    def __init__(self, device, psf_mask=None, snapshot_masks_tensor=None):
        """
        Args:
            device: torch device
            psf_mask: path to PSF mask .npy (optional)
            snapshot_masks_tensor: (1, N, H, W) tensor of binary masks (required)
        """
        self.device = device
        
        if snapshot_masks_tensor is None:
            raise ValueError("snapshot_masks_tensor is required for MaskedImagingModel")
        
        # Initialize PSF components (from ImagingModel)
        self.read_config()
        
        diameter = set_diameter(self.NA, self.layers, self.z_res)
        self.ray_num, ray_check = set_incidental_light(diameter, self.apa_size)
        self.intensity = 1 / self.ray_num

        self.ray_mat = range_matrix_generation(self.ray_num, ray_check, self.layers,
                                               diameter, self.apa_size, self.xy_res, self.z_res)

        ray_check = np.array(ray_check).reshape((self.apa_size, self.apa_size))
        self.ray_sel = ray_check != -1
        self.rescale = 1
        if psf_mask is not None:
            mask = np.load(psf_mask, allow_pickle=True)
            self.rescale = mask.mean()
            self.ray_sel = np.logical_and(self.ray_sel, mask)
            self.ray_mat = self.ray_mat[ray_check[self.ray_sel]]

        self.ray_mat = torch.from_numpy(self.ray_mat).clone()
        self.ray_mat = self.ray_mat.to(torch.float32).to(device)
        self.padding_size = int((self.ray_mat.size()[2] - 1) / 2)
        
        # Store snapshot masks
        self.masks = snapshot_masks_tensor.to(device)  # (1, N, H, W)
        self.n_masks = snapshot_masks_tensor.shape[1]
        
        print(f"[MaskedImagingModel] Initialized with PSF + {self.n_masks} snapshot masks")
        print(f"  - PSF layers: {self.layers}")
        print(f"  - Snapshot masks: {self.n_masks}")
    
    def read_config(self):
        config_path = os.path.join(os.getcwd(), "config.yaml")
        with open(config_path) as f:
            params = yaml.load(f, Loader=yaml.SafeLoader)
        img_params = params["ImgModelParams"]

        self.apa_size = img_params["apa_size"]
        self.xy_res = img_params["xy_res"]
        self.z_res = img_params["z_res"]
        self.NA = img_params["NA"]
        self.layers = img_params["layers"]
    
    def __call__(self, omega):
        """
        Apply PSF convolution to get focal images, then apply masks and sum.
        
        Forward model: snapshot = clamp(Σ_s (PSF_s ⊗ ω) × mask_s)
        
        This simulates the actual MAUF19 process:
        1. Each depth s produces a focal image (with PSF blur)
        2. Each focal image is masked by mask_s
        3. All masked focal images are summed to create the snapshot
        
        Args:
            omega: (1, 1, D, H, W) volume tensor
        
        Returns:
            img_list: List containing [snapshot_image]
        """
        # Accumulate masked focal images from all depth slices
        snapshot = 0
        
        for s in range(self.layers):
            # PSF convolution for this depth slice → focal image
            focal_img = F.conv2d(omega[0],
                                 self.ray_mat[:, self.layers - 1 - s:2 * self.layers - 1 - s, :, :],
                                 padding=self.padding_size)
            focal_img = self.intensity * torch.sum(torch.exp(focal_img), dim=1).squeeze()
            # Removed: focal_img = focal_img / self.rescale
            # Reason: Avoid double normalization (rescale + mask_sum)
            # Only mask_sum normalization is used for consistency with SnapshotImagingModel
            
            # Apply mask for this depth slice
            mask_s = self.masks[0, s]  # (H, W)
            masked_focal = focal_img * mask_s
            
            # Accumulate
            snapshot = snapshot + masked_focal
        
        # Normalize by n_slices (constant) to simulate real camera accumulation
        snapshot = snapshot / self.n_masks
        
        # Clamp final snapshot
        snapshot = torch.clamp(snapshot, 0, 1)
        
        return [snapshot]


## class SnapshotImagingModel:
##     """
##     Simplified imaging model for snapshot reconstruction.
##     
##     Instead of physically modeling light propagation, this model:
##     1. Takes a 3D volume
##     2. Extracts slices at different depths
##     3. Applies coded masks to each slice
##     4. Sums them to create a snapshot image
##     
##     This is the inverse problem of: snapshot = sum(slice_i * mask_i)
##     """
##     
##     def __init__(self, device, masks_tensor):
##         """
##         Args:
##             device: torch device
##             masks_tensor: (1, N, H, W) tensor of binary masks
##         """
##         self.device = device
##         self.masks = masks_tensor.to(device)  # (1, N, H, W)
##         self.layers = masks_tensor.shape[1]
##         
##         # Create a dummy ray_sel for compatibility with trainer
##         # Assume square aperture
##         apa_size = 128  # Default aperture size
##         self.ray_sel = np.ones((apa_size, apa_size), dtype=bool)
##         
##         print(f"[SnapshotImagingModel] Initialized with {self.layers} depth layers")
##     
##     def __call__(self, omega):
##         """
##         Simulate snapshot measurement from 3D volume.
##         
##         Forward model: snapshot = clamp(Σ_s (exp(omega_s) × mask_s) / mask_sum)
##         
##         Args:
##             omega: (1, 1, D, H, W) volume tensor (log-transformed, unified with other models)
##         
##         Returns:
##             img_list: List containing [snapshot_image]
##         """
##         # Convert from log-space to linear-space (unified with MaskedImagingModel)
##         volume = torch.exp(omega[0, 0])  # (D, H, W)
##         
##         # Accumulate masked slices
##         snapshot = 0
##         for s in range(self.layers):
##             # Get slice at depth s
##             slice_s = volume[s]  # (H, W)
##             
##             # Apply mask (simulate coded aperture)
##             mask_s = self.masks[0, s]  # (H, W)
##             snapshot = snapshot + (slice_s * mask_s)
##         
##         # Normalize by mask sum
##         mask_sum = torch.sum(self.masks[0], dim=0)  # (H, W)
##         mask_sum_safe = torch.where(mask_sum > 0, mask_sum, torch.ones_like(mask_sum))
##         snapshot = snapshot / mask_sum_safe
##         
##         # Clamp to valid range
##         snapshot = torch.clamp(snapshot, 0, 1)
##         
##         return [snapshot]