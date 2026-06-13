import os
import subprocess
from collections import deque
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

from utils.plotting import plot_slices, plot_figure


class Trainer:
    def __init__(self, img_model, input_imgs, device, gt_slices=None, version=None, output_dir="outputs",
                 save_interval=None, image_format="png", auto_upload=False,
                 gdrive_base="gdrive:multifocus-3d-reconstruction",
                 early_stop=False, early_stop_window=15, early_stop_patience=20):
        self.read_config()

        self.obs = input_imgs[0].to(device)
        self.device = device
        self.gt = gt_slices
        if self.gt is not None:
            self.gt = self.gt.to(device)
        self.fwd = img_model
        self.layers = input_imgs.shape[-3]
        self.version = version
        self.output_dir = output_dir
        self.save_interval = save_interval
        self.image_format = image_format.lower() if image_format else "png"
        self.auto_upload = auto_upload
        self.gdrive_base = gdrive_base

        # Early stopping settings (windowed variance)
        self.early_stop = early_stop
        self.early_stop_window = max(1, int(early_stop_window))
        self.early_stop_patience = max(1, int(early_stop_patience))
        self._es_queue = deque(maxlen=self.early_stop_window) if self.early_stop else None
        self._es_best_var = float("inf")
        self._es_wait = 0
        self._es_best_iter = None
        self._es_best_volume = None
        self._early_stopped = False

        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Create intermediate results directory
        self.intermediate_dir = os.path.join(output_dir, "intermediate")
        if save_interval is not None and save_interval > 0:
            if not os.path.exists(self.intermediate_dir):
                os.makedirs(self.intermediate_dir)

        if version is not None:
            self.writer = SummaryWriter(log_dir=f"tb_logs/{version}")
        fig = plot_figure(img_model.ray_sel)
        self.writer.add_figure("aperture", fig, global_step=0)
        plt.close(fig)

    def read_config(self):
        raise NotImplementedError

    def pretrain(self, type):
        raise NotImplementedError

    def train(self):
        raise NotImplementedError

    # ------------------ EARLY STOPPING ------------------
    def _prepare_volume_for_es(self, volume):
        import torch

        if isinstance(volume, torch.Tensor):
            volume_np = volume.detach().cpu().numpy()
        else:
            volume_np = volume

        if len(volume_np.shape) == 5:
            volume_np = volume_np[0, 0]
        elif len(volume_np.shape) == 4:
            volume_np = volume_np[0]

        return volume_np

    def _queue_variance(self, queue):
        avg = np.mean(queue, axis=0)
        norms = [np.linalg.norm(vol - avg) ** 2 for vol in queue]
        return float(np.mean(norms))

    def update_early_stopping(self, volume, iteration):
        if not self.early_stop:
            return False

        vol_np = self._prepare_volume_for_es(volume)
        self._es_queue.append(vol_np)

        if len(self._es_queue) < self.early_stop_window:
            return False

        var = self._queue_variance(self._es_queue)
        if var < self._es_best_var:
            self._es_best_var = var
            self._es_best_iter = iteration
            self._es_best_volume = vol_np
            self._es_wait = 0
        else:
            self._es_wait += 1

        if self._es_wait >= self.early_stop_patience:
            self._early_stopped = True
            print(
                f"Early stopping triggered at iter {iteration}. "
                f"Best iter: {self._es_best_iter}, var: {self._es_best_var:.6e}"
            )
            return True

        return False

    def get_best_volume_numpy(self):
        return self._es_best_volume

    def has_early_stopped(self):
        return self._early_stopped and self._es_best_volume is not None

    # --------------------- LOGGING ---------------------
    def log_metrics(self, step, loss, psnr=None, ssim=None, rmse=None, dice_score=None):
        if hasattr(self, "writer"):
            self.writer.add_scalar("loss", loss, global_step=step)
            if psnr is not None:
                self.writer.add_scalar("psnr", psnr, global_step=step)
            if ssim is not None:
                self.writer.add_scalar("ssim", ssim, global_step=step)
            if rmse is not None:
                self.writer.add_scalar("rmse", rmse, global_step=step)
            if dice_score is not None:
                self.writer.add_scalar("dice_score", dice_score, global_step=step)

    def log_figs(self, step, *vols):
        vol_list = list(vols)
        if hasattr(self, "writer") and step % 20 == 0:
            for i in range(len(vol_list)):
                if len(vol_list[i].shape) == 5:
                    vol_list[i] = vol_list[i][0, 0]
                elif len(vol_list[i].shape) == 4:
                    vol_list[i] = vol_list[i][0]
            
            # Check if we have 3D slices or a single 2D image (snapshot mode)
            if vol_list[0].ndim == 3:
                # Multi-focus mode: plot multiple slices
                fig = plot_slices(*vol_list)
            else:
                # Snapshot mode: plot single image
                from utils.plotting import plot_figure
                fig = plot_figure(vol_list[0])
            self.writer.add_figure("training/img", fig, global_step=step)

    def log_hparams(self):
        trainer_params = {key: vars(self)[key] for key in vars(self)
                          if (type(vars(self)[key]) == int or type(vars(self)[key]) == float)}
        self.writer.add_text("trainer_hparams", str(trainer_params), 0)

        img_model_hparams = {key: vars(self.fwd)[key] for key in vars(self.fwd)
                             if (type(vars(self.fwd)[key]) == int or type(vars(self.fwd)[key]) == float)}
        self.writer.add_text("img_model_hparams", str(img_model_hparams), 0)

        if hasattr(self, "net"):
            net_params = {key: vars(self.net)[key] for key in vars(self.net)
                          if (type(vars(self.net)[key]) == int or type(vars(self.net)[key]) == float)}
            self.writer.add_text("trainer_hparams", str(net_params), 0)

        self.writer.flush()
        self.writer.close()

    def save_intermediate(self, volume, iteration):
        """
        Save intermediate reconstruction results during training.

        Args:
            volume: The reconstructed 3D volume (tensor or numpy array)
            iteration: Current iteration number
        """
        if self.save_interval is None or self.save_interval <= 0:
            return

        if iteration % self.save_interval != 0:
            return

        import torch

        # Convert to numpy if tensor
        if isinstance(volume, torch.Tensor):
            volume_np = volume.cpu().detach().numpy()
        else:
            volume_np = volume

        # Remove batch and channel dimensions if present
        if len(volume_np.shape) == 5:
            volume_np = volume_np[0, 0]
        elif len(volume_np.shape) == 4:
            volume_np = volume_np[0]

        # Save volume as .npy
        volume_path = os.path.join(self.intermediate_dir, f"volume_iter_{iteration:05d}.npy")
        np.save(volume_path, volume_np)

        # Save each slice as individual image file
        img_ext = "jpg" if self.image_format == "jpg" else "png"
        n_slices = volume_np.shape[0]

        for slice_idx in range(n_slices):
            slice_img = volume_np[slice_idx]

            # Create figure for this slice
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.imshow(slice_img, cmap='gray', vmin=0, vmax=1)
            ax.axis('off')

            slice_path = os.path.join(self.intermediate_dir,
                                     f"iter_{iteration:05d}_slice_{slice_idx:02d}.{img_ext}")

            if self.image_format == "jpg":
                fig.savefig(slice_path, dpi=150, bbox_inches='tight', format='jpg',
                           quality=95, pad_inches=0)
            else:
                fig.savefig(slice_path, dpi=150, bbox_inches='tight', format='png',
                           pad_inches=0)

            plt.close(fig)

    def save_results(self, volume, output_imgs=None, prefix="result"):
        """
        Save reconstruction results to output directory.

        Args:
            volume: The reconstructed 3D volume (tensor or numpy array)
            output_imgs: Optional output images from forward model
            prefix: Prefix for output filenames
        """
        import torch

        # Convert to numpy if tensor
        if isinstance(volume, torch.Tensor):
            volume_np = volume.cpu().detach().numpy()
        else:
            volume_np = volume

        # Remove batch and channel dimensions if present
        if len(volume_np.shape) == 5:
            volume_np = volume_np[0, 0]
        elif len(volume_np.shape) == 4:
            volume_np = volume_np[0]

        # Save volume as .npy
        volume_path = os.path.join(self.output_dir, f"{prefix}_volume.npy")
        np.save(volume_path, volume_np)
        print(f"Saved 3D volume to: {volume_path}")

        # Save each volume slice as individual image file
        img_ext = "jpg" if self.image_format == "jpg" else "png"
        n_slices = volume_np.shape[0]

        print(f"Saving {n_slices} volume slices...")
        for slice_idx in range(n_slices):
            slice_img = volume_np[slice_idx]

            # Create figure for this slice
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.imshow(slice_img, cmap='gray', vmin=0, vmax=1)
            ax.axis('off')

            slice_path = os.path.join(self.output_dir,
                                     f"{prefix}_slice_{slice_idx:02d}.{img_ext}")

            if self.image_format == "jpg":
                fig.savefig(slice_path, dpi=150, bbox_inches='tight', format='jpg',
                           quality=95, pad_inches=0)
            else:
                fig.savefig(slice_path, dpi=150, bbox_inches='tight', format='png',
                           pad_inches=0)

            plt.close(fig)

        print(f"Saved {n_slices} volume slices to: {self.output_dir}")

        # Save output images if provided
        if output_imgs is not None:
            if isinstance(output_imgs, torch.Tensor):
                output_imgs_np = output_imgs.cpu().detach().numpy()
            else:
                output_imgs_np = output_imgs

            if len(output_imgs_np.shape) == 4:
                output_imgs_np = output_imgs_np[0]
            elif len(output_imgs_np.shape) == 3:
                pass  # Already correct shape

            n_outputs = output_imgs_np.shape[0]
            print(f"Saving {n_outputs} output images...")

            for img_idx in range(n_outputs):
                output_img = output_imgs_np[img_idx]

                # Create figure for this output image
                fig, ax = plt.subplots(figsize=(6, 6))
                ax.imshow(output_img, cmap='gray', vmin=0, vmax=1)
                ax.axis('off')

                output_path = os.path.join(self.output_dir,
                                          f"{prefix}_output_{img_idx:02d}.{img_ext}")

                if self.image_format == "jpg":
                    fig.savefig(output_path, dpi=150, bbox_inches='tight', format='jpg',
                               quality=95, pad_inches=0)
                else:
                    fig.savefig(output_path, dpi=150, bbox_inches='tight', format='png',
                               pad_inches=0)

                plt.close(fig)

            print(f"Saved {n_outputs} output images to: {self.output_dir}")

    def upload_to_gdrive(self):
        """
        Upload results to Google Drive using rclone.
        """
        if not self.auto_upload:
            return

        print("\n" + "="*50)
        print("Uploading results to Google Drive...")
        print("="*50)

        # Check if rclone is available
        try:
            subprocess.run(["rclone", "version"], capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("ERROR: rclone is not installed or not configured.")
            print("Please install rclone and configure it with 'rclone config'")
            print("Skipping upload.")
            return

        experiment_name = self.version if self.version else "unnamed_experiment"

        # Upload output directory
        if os.path.exists(self.output_dir):
            gdrive_output_path = f"{self.gdrive_base}/outputs/{experiment_name}"
            print(f"\nUploading outputs: {self.output_dir} -> {gdrive_output_path}")

            try:
                # Upload without intermediate directory (large files)
                subprocess.run([
                    "rclone", "sync", self.output_dir, gdrive_output_path,
                    "--exclude", "intermediate/**",
                    "--progress",
                    "-v"
                ], check=True)
                print(f"✓ Output files uploaded successfully")
            except subprocess.CalledProcessError as e:
                print(f"✗ Failed to upload output files: {e}")

        # Upload TensorBoard logs
        if self.version:
            tb_log_dir = f"tb_logs/{self.version}"
            if os.path.exists(tb_log_dir):
                gdrive_tb_path = f"{self.gdrive_base}/tb_logs/{experiment_name}"
                print(f"\nUploading TensorBoard logs: {tb_log_dir} -> {gdrive_tb_path}")

                try:
                    subprocess.run([
                        "rclone", "sync", tb_log_dir, gdrive_tb_path,
                        "--progress",
                        "-v"
                    ], check=True)
                    print(f"✓ TensorBoard logs uploaded successfully")
                except subprocess.CalledProcessError as e:
                    print(f"✗ Failed to upload TensorBoard logs: {e}")

        # Upload intermediate results if they exist
        intermediate_dir = os.path.join(self.output_dir, "intermediate")
        if os.path.exists(intermediate_dir):
            gdrive_intermediate_path = f"{self.gdrive_base}/outputs/{experiment_name}/intermediate"
            print(f"\nUploading intermediate results: {intermediate_dir} -> {gdrive_intermediate_path}")
            print("(This may take a while...)")

            try:
                subprocess.run([
                    "rclone", "sync", intermediate_dir, gdrive_intermediate_path,
                    "--progress",
                    "-v"
                ], check=True)
                print(f"✓ Intermediate results uploaded successfully")
            except subprocess.CalledProcessError as e:
                print(f"✗ Failed to upload intermediate results: {e}")

        print("\n" + "="*50)
        print("Upload complete!")
        print("Access your results at: https://drive.google.com/")
        print("="*50 + "\n")
