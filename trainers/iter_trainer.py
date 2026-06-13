import os
import yaml
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from total_variation_3d import TotalVariationL2
from utils.lighting import default_transmittance
from trainer import Trainer


class IterTrainer(Trainer):
    def __init__(self, img_model, input_imgs, device, gt_slices=None, version=None, output_dir="outputs",
                 save_interval=None, image_format="png", auto_upload=False,
                 gdrive_base="gdrive:multifocus-3d-reconstruction",
                 is_snapshot_mode=False,
                 snapshot_target=None,
                 early_stop=False, early_stop_window=15, early_stop_patience=20,
                 **kwargs):
        super().__init__(img_model, input_imgs, device, gt_slices, version, output_dir,
                         save_interval, image_format, auto_upload, gdrive_base,
                         early_stop=early_stop,
                         early_stop_window=early_stop_window,
                         early_stop_patience=early_stop_patience)

        # Snapshot mode settings
        self.is_snapshot_mode = is_snapshot_mode
        self.snapshot_target = snapshot_target.to(device) if snapshot_target is not None else None

        # ------------- ITER SPECIFIC INIT -------------
        # In snapshot mode self.obs is the snapshot repeated N times, so
        # default_transmittance(flag=0) gives snapshot^(1/N) per slice — uniform across depth,
        # which is a reasonable starting point. flag=1 (all 0.75) is also fine.
        alpha = default_transmittance(self.flag, self.obs)
        alpha = torch.clip(alpha, min=1./255, max=1)
        alpha = torch.log(alpha).unsqueeze(0).unsqueeze(0)  # (1, 1, layer, height, width)

        alpha = alpha.to(device)
        self.omega = alpha.detach().requires_grad_()

        self.optim = torch.optim.Adam([self.omega], lr=self.lr)

        self.loss_fn = nn.MSELoss(reduction="sum")
        self.tv_loss = TotalVariationL2(is_mean_reduction=False)

    def read_config(self):
        config_path = os.path.join(os.getcwd(), "config.yaml")
        with open(config_path) as f:
            params = yaml.load(f, Loader=yaml.SafeLoader)
        iter_params = params["IterParams"]

        self.mu = iter_params["mu"]
        self.lr = iter_params["lr"]
        self.n_iter = iter_params["n_iter"]
        self.flag = iter_params["flag"]

    def pretrain(self, type):
        # IterTrainer optimizes alpha directly (no neural network), so pretraining is a no-op.
        # Keeping this override allows users to pass -p flags consistently across trainers.
        print(f"[IterTrainer] pretrain('{type}') is a no-op (no network to pretrain).")

    def train(self):
        psnr_final, ssim_final, loss_final = None, None, None
        try:
            with tqdm(range(self.n_iter), total=self.n_iter) as pbar:
                for i in pbar:
                    # -------- OPTIMIZATION --------
                    self.optim.zero_grad()

                    img_list = self.fwd(self.omega)

                    if self.is_snapshot_mode and self.snapshot_target is not None:
                        # Snapshot mode: imaging model returns [snapshot]
                        loss = self.loss_fn(img_list[0], self.snapshot_target)
                        loss.backward()
                        loss_sum = loss.cpu().item()
                    else:
                        # Multi-focus mode: per-slice loss
                        loss_sum = 0
                        for s in range(len(img_list)):
                            out = img_list[s]
                            loss = self.loss_fn(out, self.obs[s])
                            loss.backward()
                            loss_sum = loss_sum + loss.cpu().item()

                    constraint = self.mu * self.tv_loss(self.omega)
                    constraint.backward()

                    loss_sum = loss_sum + constraint.cpu().item()

                    self.optim.step()

                    # -------- ADD METRICS TO PBAR --------
                    psnr, ssim = None, None
                    if self.gt is not None:
                        trans = torch.exp(self.omega.detach())
                        trans = trans.to('cpu')
                        trans = trans.data.squeeze().float().clamp_(0, 1)
                        psnr = peak_signal_noise_ratio(trans.cpu().detach().numpy(), self.gt[0].cpu().detach().numpy())
                        ssim = structural_similarity(trans.cpu().detach().numpy(), self.gt[0].cpu().detach().numpy(),
                                                     channel_axis=False, win_size=self.layers)
                        pbar.set_postfix({"psnr": psnr, "ssim": ssim, "loss": loss_sum})
                        psnr_final, ssim_final = psnr, ssim
                    else:
                        pbar.set_postfix({"loss": loss_sum})
                    loss_final = loss_sum

                    # -------- LOG TO TENSORBOARD --------
                    self.log_metrics(i, loss_sum, psnr, ssim)

                    if self.is_snapshot_mode:
                        # Snapshot mode: log only reconstructed volume (and GT if available)
                        vol_list = [torch.exp(self.omega)]
                        if self.gt is not None:
                            vol_list.append(self.gt)
                    else:
                        vol_list = [self.obs, torch.stack(img_list), self.gt, torch.exp(self.omega)] \
                            if self.gt is not None else \
                            [self.obs, torch.stack(img_list), torch.exp(self.omega)]
                    vol_list = [vol.cpu().detach().numpy() for vol in vol_list]
                    self.log_figs(i, *vol_list)

                    # -------- SAVE INTERMEDIATE RESULTS --------
                    trans = torch.exp(self.omega.detach())
                    self.save_intermediate(trans, i)

                    # Legacy support for .npy saving
                    if i % 5 == 0:
                        np.save(f"tb_logs/{self.version}/alpha_{i:05}.npy", trans.cpu().detach().numpy())
        except KeyboardInterrupt:
            print("Training interrupted.")

        self.log_hparams()
        np.save(f"tb_logs/{self.version}/alpha.npy", self.omega.cpu().detach().numpy())

        # Save final results
        print("\nSaving final reconstruction results...")
        alpha = torch.exp(self.omega.detach())
        img_list = self.fwd(self.omega)
        out = torch.stack(img_list)
        self.save_results(alpha, out, prefix="iter_final")

        # Upload to Google Drive if auto_upload is enabled
        self.upload_to_gdrive()

        # Save metrics summary (consistent with DIPTrainer format)
        result_path = os.path.join(self.output_dir, 'result.txt')
        with open(result_path, 'w') as f:
            f.write('psnr\tssim\tloss\n')
            f.write(f'{psnr_final}\t{ssim_final}\t{loss_final}\n')
        print(f"Result saved to {result_path}")
