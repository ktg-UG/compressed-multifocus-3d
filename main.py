import argparse
import yaml
import numpy as np
from collections import defaultdict

from trainers.iter_trainer import IterTrainer
from trainers.dip_trainer import DIPTrainer
from trainers.hqs_trainer import HQSTrainer
from trainers.nf_trainer import NFTrainer
from img_model import ImagingModel
from utils.loading import *


if __name__ == "__main__":
    # ------------------ ARGUMENT PARSING ------------------
    parser = argparse.ArgumentParser(description="Run reconstruction", allow_abbrev=False)

    parser.add_argument("--model", "-m", required=True, choices=["iter", "dip", "hqs", "nf"], help="model selection")
    parser.add_argument("--input", "-i", help="observed images path (multi-focus images)")
    parser.add_argument("--snapshot", "-s", help="snapshot image path (single coded image)")
    parser.add_argument("--snapshot-masks", "-sm", help="path to snapshot masks .mat or .npz file")

    parser.add_argument("--ground_truth", "-gt", help="GT vol path")
    parser.add_argument("--pretraining", "-p", choices=["sc", "v2", "v3", "const"], help="type of pretraining")
    parser.add_argument("--version", "-v", type=str, help="version name for Tensorboard")
    parser.add_argument("--weights", "-w", type=str, help="path to load model state dict")
    parser.add_argument("--denoiser_weights", "-dn", type=str, help="path to denoiser state dict")
    parser.add_argument("--noise_level", "-n", type=float, default=0, help="variance of noise to apply to observations")
    parser.add_argument("--psf_mask", "-psf", type=str, help="path to psf mask .npy")
    parser.add_argument("--save_every_iter", action="store_true", help="when using dip, whether to save inter. voxels")
    parser.add_argument("--output", "-o", type=str, default="outputs", help="output directory for saving results")
    parser.add_argument("--save_interval", type=int, help="save intermediate results every N iterations (e.g., 50, 100)")
    parser.add_argument("--n_iter", type=int, default=None, help="override number of training iterations (dip only)")
    parser.add_argument("--image_format", type=str, choices=["png", "jpg"], default="png", help="image format for saving results (png or jpg)")
    parser.add_argument("--auto_upload", action="store_true", help="automatically upload results to Google Drive after training")
    parser.add_argument("--gdrive_base", type=str, default="gdrive:multifocus-3d-reconstruction", help="base path for Google Drive (default: gdrive:multifocus-3d-reconstruction)")
    parser.add_argument("--early_stop", action="store_true", help="enable early stopping based on windowed variance")
    parser.add_argument("--early_stop_window", type=int, default=50, help="window size for early stopping variance (default: 15)")
    parser.add_argument("--early_stop_patience", type=int, default=100, help="patience for early stopping (default: 20)")
    parser.add_argument("--tv-weight", dest="tv_weight", type=float, default=0.0,
                        help="TV regularization weight for DIP (L1 TV applied to alpha). 0 = OFF (default), >0 = ON")
    parser.add_argument("--tv-axes", dest="tv_axes", choices=["3d", "z"], default="3d",
                        help="TV regularization axes: '3d' (x+y+z) or 'z' (depth only). Only applies when --tv-weight > 0")

    args = parser.parse_args()

    # Validate input arguments
    if args.input is None and args.snapshot is None:
        parser.error("Either --input or --snapshot must be provided")
    if args.input is not None and args.snapshot is not None:
        parser.error("Cannot use both --input and --snapshot at the same time")
    if args.snapshot is not None and args.snapshot_masks is None:
        parser.error("--snapshot-masks is required when using --snapshot")

    if args.version is not None:
        print(args.version)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------ GET OBSERVED IMAGES TENSOR ------------------
    snapshot_image = None  # Store actual snapshot for loss calculation
    
    if args.snapshot is not None:
        # Snapshot mode: load single snapshot image
        print(f"[Snapshot Mode] Loading snapshot from: {args.snapshot}")
        
        # Load snapshot image
        from PIL import Image
        snapshot_img = Image.open(args.snapshot).convert('L')
        snapshot_arr = np.array(snapshot_img, dtype=np.float32) / 255.0
        
        # Load masks
        print(f"[Snapshot Mode] Loading masks from: {args.snapshot_masks}")
        if args.snapshot_masks.endswith('.mat'):
            from scipy import io as scio
            masks_data = scio.loadmat(args.snapshot_masks)
            masks = masks_data['ExpPtn']  # (H, W, N)
        elif args.snapshot_masks.endswith('.npz'):
            masks_data = np.load(args.snapshot_masks)
            masks = masks_data['ExpPtn']  # (H, W, N)
        else:
            raise ValueError("--snapshot-masks must be .mat or .npz file")
        
        # Convert masks from (H, W, N) to (N, H, W)
        masks = np.transpose(masks, (2, 0, 1))  # (N, H, W)
        n_images = masks.shape[0]
        
        print(f"[Snapshot Mode] Snapshot shape: {snapshot_arr.shape}, Masks shape: {masks.shape}")
        print(f"[Snapshot Mode] Number of virtual images: {n_images}")
        
        # Store the actual snapshot image for loss calculation
        snapshot_image = torch.from_numpy(snapshot_arr).float()  # (H, W)
        
        # For compatibility, create a dummy multi-focus observation
        # (the actual loss will be computed against snapshot_image)
        imgs = snapshot_image.unsqueeze(0).unsqueeze(0).repeat(1, n_images, 1, 1)  # (1, N, H, W)
        
        # Store masks for imaging model
        snapshot_masks_tensor = torch.from_numpy(masks).unsqueeze(0).float()  # (1, N, H, W)
        
    else:
        # Original multi-focus mode
        print(f"[Multi-focus Mode] Loading images from: {args.input}")
        in_paths = get_image_paths(args.input)
        img_list = imreads_uint(in_paths, 1)
        imgs = imglist2tensor(img_list)  # tensor(C, N, H, W)
        snapshot_masks_tensor = None

    if args.noise_level != 0:
        x = imgs.to('cpu').detach().numpy().copy()

        np.random.seed(seed=0)  # for reproducibility
        x += np.random.normal(0, args.noise_level, x.shape)  # add AWGN

        imgs = torch.from_numpy(x.astype(np.float32)).clone()
    imgs = torch.clip(imgs, min=0, max=1)

    # ------------------ IF GT PATH, GET GT SLICES ------------------
    slices = None
    if args.ground_truth is not None:
        val_paths = get_image_paths(args.ground_truth)
        slice_list = imreads_uint(val_paths, 1)
        slices = imglist2tensor(slice_list)

    # ------------------ ASSIGN MODELS AND TRAINER ------------------
    
    # Snapshot mode uses a simplified imaging model by default
    if args.snapshot is not None:
        extra_kwargs = {'is_snapshot_mode': True, 'snapshot_target': snapshot_image}
        print("[Snapshot Mode] Using MaskedImagingModel (PSF + masks) with snapshot inputs")
        from img_model import MaskedImagingModel
        img_model = MaskedImagingModel(device, args.psf_mask, snapshot_masks_tensor)
    else:
        print("[Multi-focus Mode] Using multi-focus imaging model")
        img_model = ImagingModel(device, args.psf_mask)
        extra_kwargs = {}

    if args.n_iter is not None and args.model == "dip":
        extra_kwargs["n_iter_override"] = args.n_iter

    model_dict = defaultdict()
    model_dict["iter"] = IterTrainer
    model_dict["dip"] = DIPTrainer
    model_dict["hqs"] = HQSTrainer
    model_dict["nf"] = NFTrainer

    trainer = model_dict[args.model](img_model,
                                     imgs,
                                     device,
                                     gt_slices=slices,
                                     version=args.version,
                                     weights=args.weights,
                                     denoiser_weights=args.denoiser_weights,
                                     noise_level=args.noise_level,
                                     save_every_iter=args.save_every_iter,
                                     output_dir=args.output,
                                     save_interval=args.save_interval,
                                     image_format=args.image_format,
                                     auto_upload=args.auto_upload,
                                     gdrive_base=args.gdrive_base,
                                     early_stop=args.early_stop,
                                     early_stop_window=args.early_stop_window,
                                     early_stop_patience=args.early_stop_patience,
                                     tv_weight=args.tv_weight,
                                     tv_axes=args.tv_axes,
                                     **extra_kwargs)

    # ------------------ TRAIN ------------------
    if args.pretraining is not None:
        trainer.pretrain(args.pretraining)
    trainer.train()
