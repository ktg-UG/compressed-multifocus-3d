import os
import yaml
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from models.skipnet3d import SkipNet3D
from trainer import Trainer
from utils.pretraining import pretraining_v3, pretraining_sc, pretraining_v2
from total_variation_3d import TotalVariationL1, TotalVariationL1Z


class DIPTrainer(Trainer):
    def __init__(self,
                 img_model,
                 input_imgs,
                 device,
                 gt_slices=None,
                 version=None,
                 weights=None,
                 save_every_iter=False,
                 output_dir="outputs",
                 save_interval=None,
                 image_format="png",
                 auto_upload=False,
                 gdrive_base="gdrive:multifocus-3d-reconstruction",
                 is_snapshot_mode=False,
                 snapshot_target=None,
             early_stop=False,
             early_stop_window=15,
             early_stop_patience=20,
                 tv_weight=0.0,
                 tv_axes="3d",
                 n_iter_override=None,
                 **kwargs):
        super().__init__(img_model, input_imgs, device, gt_slices, version, output_dir,
                 save_interval, image_format, auto_upload, gdrive_base,
                 early_stop=early_stop,
                 early_stop_window=early_stop_window,
                 early_stop_patience=early_stop_patience)

        # CLI override for iteration count (read_config sets the default from config.yaml)
        if n_iter_override is not None:
            self.n_iter = int(n_iter_override)
            print(f"[DIP] n_iter overridden to {self.n_iter}")

        # ------------- DIP SPECIFIC INIT -------------
        self.weights = weights
        self.net = SkipNet3D().to(device)

        # Snapshot mode settings
        self.is_snapshot_mode = is_snapshot_mode
        self.snapshot_target = snapshot_target.to(device) if snapshot_target is not None else None

        if weights is not None:
            self.net.load_state_dict(torch.load(weights + "/net.pt"))
            self.inp = torch.load(weights + "/inp.pt")
        else:
            cell_dim = tuple(input_imgs.shape[-3:])
            self.inp = torch.zeros((1, 1) + cell_dim, device=device, dtype=torch.float32)
            self.inp.uniform_()
            self.inp *= 1.0 / 10  # inp_noise_var
            self.inp = self.inp.detach().clone()

        self.optim = torch.optim.Adam(self.net.parameters(), lr=self.lr)

        self.loss_fn = nn.L1Loss()

        # Total Variation regularization (applied to alpha, not omega — see ResearchPath.md)
        self.tv_weight = float(tv_weight)
        self.tv_axes = tv_axes
        if self.tv_weight > 0:
            if self.tv_axes == "z":
                self.tv_loss = TotalVariationL1Z(is_mean_reduction=True)
            else:
                self.tv_loss = TotalVariationL1(is_mean_reduction=True)
            print(f"[DIP] TV regularization: ENABLED (weight={self.tv_weight}, axes={self.tv_axes}, target=alpha)")
        else:
            self.tv_loss = None
            print("[DIP] TV regularization: DISABLED")

        self.save_every_iter = save_every_iter

    def read_config(self):
        config_path = os.path.join(os.getcwd(), "config.yaml")
        with open(config_path) as f:
            params = yaml.load(f, Loader=yaml.SafeLoader)
        dip_params = params["DIPParams"]

        self.n_iter = dip_params["n_iter"]
        self.lr = dip_params["lr"]
        self.pretr_iter = dip_params["pretr_iter"]

    def pretrain(self, type):
        if self.weights is not None:
            print("Both weights and pretraining were given, skipping pretraining and loading weights.")
            return

        writer = self.writer if hasattr(self, "writer") else None
        if type == "v3":
            pretraining_v3(self.inp, self.net, self.pretr_iter, self.version, writer=writer)
        elif type == "v2":
            pretraining_v2(self.inp, self.net, self.version, writer=writer)
        elif type == "sc":
            pretraining_sc(self.inp, self.net, 500, writer=writer)
        else:
            raise Exception("Invalid pretraining version.")
    
    def compute_masked_l1_loss(self, pred, target):
        """
        Compute L1 loss ignoring pixels where target == 0 (masked regions).
        This ensures consistent loss calculation across all modes.
        
        Args:
            pred: Predicted tensor
            target: Ground truth tensor
        
        Returns:
            Scalar loss value
        """
        mask = (target != 0).to(dtype=pred.dtype, device=pred.device)
        abs_diff = torch.abs(pred - target)
        masked_sum = (abs_diff * mask).sum()
        num_unmasked = mask.sum()
        return masked_sum / (num_unmasked + 1e-8)  # Avoid division by zero

    def train(self):
        psnr_final = None
        ssim_final = None
        rmse_final = None
        dice_score_final = None
        loss_final = None
        try:
            with tqdm(range(self.n_iter), total=self.n_iter) as pbar:
                for i in pbar:
                    # -------- OPTIMIZATION --------
                    self.optim.zero_grad()

                    alpha = self.net(self.inp)
                    # 数値計算の安定性のために下限を設定（log(0)回避）
                    alpha = torch.clamp(alpha, min=1e-6)
                    
                    # Forward pass through imaging model
                    # All models now expect log-transformed input for consistency
                    out = self.fwd(torch.log(alpha))
                    out = torch.stack(out)

                    # Compute loss based on mode (unified masked L1 loss)
                    if self.is_snapshot_mode and self.snapshot_target is not None:
                        # Snapshot mode: imaging model already returns composed snapshot
                        snapshot_pred = out[0]  # out is already [snapshot_image]

                        # Unified masked L1 loss (ignores 0 pixels, consistent with multi-focus)
                        recon_loss = self.compute_masked_l1_loss(snapshot_pred, self.snapshot_target)

                        # Debug output for first few iterations
                        if i < 5 or i % 100 == 0:
                            print(f"\n[DEBUG] Iter {i}:")
                            print(f"  snapshot_pred shape: {snapshot_pred.shape}, range: [{snapshot_pred.min():.4f}, {snapshot_pred.max():.4f}]")
                            print(f"  snapshot_target shape: {self.snapshot_target.shape}, range: [{self.snapshot_target.min():.4f}, {self.snapshot_target.max():.4f}]")
                            print(f"  loss: {recon_loss.item():.6f}")
                    else:
                        # Multi-focus mode: unified masked L1 loss
                        recon_loss = self.compute_masked_l1_loss(out, self.obs)

                    # Add TV regularization on alpha (transmittance) when enabled
                    if self.tv_loss is not None:
                        tv_term = self.tv_weight * self.tv_loss(alpha)
                        loss = recon_loss + tv_term
                    else:
                        tv_term = None
                        loss = recon_loss

                    loss.backward()
                    self.optim.step()

                    # -------- ADD METRICS TO PBAR --------
                    psnr, ssim, rmse, dice_score = None, None, None, None
                    if self.gt is not None:
                        alpha_np = alpha[0, 0].detach().cpu().numpy()
                        gt_np = self.gt[0].cpu().detach().numpy()
                        psnr = peak_signal_noise_ratio(alpha_np, gt_np)
                        ssim = structural_similarity(alpha_np, gt_np, channel_axis=False, win_size=self.layers)
                        rmse = np.sqrt(np.mean((alpha_np - gt_np) ** 2))
                        dice_score = 2 * np.sum(alpha_np * gt_np) / (np.sum(alpha_np ** 2) + np.sum(gt_np ** 2) + 1e-8)
                        pbar.set_postfix({"psnr": psnr, "ssim": ssim, "rmse": rmse, "dice_score": dice_score, "loss": loss.item()})
                        # 最終値を保存
                        psnr_final = psnr
                        ssim_final = ssim
                        rmse_final = rmse
                        dice_score_final = dice_score
                        loss_final = loss.item()
                    else:
                        pbar.set_postfix({"loss": loss.item()})
                        loss_final = loss.item()

                    # -------- LOG TO TENSORBOARD --------
                    self.log_metrics(i, loss.item(), psnr, ssim, rmse, dice_score)
                    if hasattr(self, "writer"):
                        self.writer.add_scalar("loss/recon", recon_loss.item(), global_step=i)
                        self.writer.add_scalar("loss/total", loss.item(), global_step=i)
                        if tv_term is not None:
                            self.writer.add_scalar("loss/tv", tv_term.item(), global_step=i)
                    
                    if self.is_snapshot_mode:
                        # Snapshot mode: only log reconstructed volume
                        vol_list = [alpha]
                        if self.gt is not None:
                            vol_list.append(self.gt)
                    else:
                        # Multi-focus mode: log observations, output, GT, and volume
                        vol_list = [self.obs, out, self.gt, alpha] if self.gt is not None else [self.obs, out, alpha]
                    
                    vol_list = [vol.cpu().detach().numpy() for vol in vol_list]
                    self.log_figs(i, *vol_list)

                    # -------- SAVE INTERMEDIATE RESULTS --------
                    self.save_intermediate(alpha, i)

                    # -------- EARLY STOPPING --------
                    if self.update_early_stopping(alpha, i):
                        break

                    # Legacy support for save_every_iter flag
                    if self.save_every_iter and i % 50 == 0:
                        np.save(f"tb_logs/{self.version}/alpha_{i:05}.npy", alpha.cpu().detach().numpy())
        except KeyboardInterrupt:
            print("Training interrupted.")

        self.log_hparams()

        # Save final results
        print("\nSaving final reconstruction results...")
        if self.has_early_stopped():
            best_volume = self.get_best_volume_numpy()
            best_alpha = torch.from_numpy(best_volume).to(self.device)
            if best_alpha.ndim == 3:
                best_alpha = best_alpha.unsqueeze(0).unsqueeze(0)
            out_list = self.fwd(torch.log(best_alpha))
            out = torch.stack(out_list)
            self.save_results(best_volume, out, prefix="dip_final")
        else:
            self.save_results(alpha, out, prefix="dip_final")

        # Upload to Google Drive if auto_upload is enabled
        self.upload_to_gdrive()

        # --- 終了時に最終結果をresult.txtに保存 ---
        result_path = os.path.join(self.output_dir, 'result.txt')
        with open(result_path, 'w') as f:
            f.write('psnr\tssim\trmse\tdice_score\tloss\n')
            f.write(f'{psnr_final}\t{ssim_final}\t{rmse_final}\t{dice_score_final}\t{loss_final}\n')
            tv_status = f"weight={self.tv_weight}, axes={self.tv_axes}, target=alpha" if self.tv_loss is not None else "DISABLED"
            f.write(f'# tv_regularization: {tv_status}\n')
        print(f"Result saved to {result_path}")

