"""Deep Sensing joint training（B-3） + M1 最小スパイク。

Encoder(学習可能パターン) と Decoder を end-to-end 共最適化する。
loss = MSE(focal_hat, focal_gt) + λ_open * aperture + λ_div * diversity

使用例:
  # M1: cell/0000 単体で動作確認（pattern が動く / loss が下がる）
  python3 phase_b/train_deep_sensing.py --sample-ids 0000 --n-iter 300

  # 通常学習: train 4 サンプル
  python3 phase_b/train_deep_sensing.py --sample-ids 0000 0001 0002 0003 --augment
"""

import os
import sys
import argparse

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from phase_b.dataset import CellFocalStackDataset
from phase_b.decoder import SnapshotDecoder
from phase_b.encoder import (
    LearnablePattern,
    simulate_snapshot,
    aperture_penalty,
    diversity_penalty,
)


def psnr(pred, target):
    mse = F.mse_loss(pred, target).item()
    if mse <= 0:
        return 99.0
    return 10.0 * torch.log10(torch.tensor(1.0 / mse)).item()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="inputs/multi_focus_data/cell")
    p.add_argument("--sample-ids", nargs="+", default=["0000"],
                   help="学習に使うサンプル id")
    p.add_argument("--depth", type=int, default=11)
    p.add_argument("--block", type=int, default=2)
    p.add_argument("--base", type=int, default=32, help="Decoder ベースチャネル")
    p.add_argument("--n-iter", type=int, default=300)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--lr-encoder", type=float, default=1e-2)
    p.add_argument("--lr-decoder", type=float, default=1e-3)
    p.add_argument("--lambda-open", type=float, default=1.0)
    p.add_argument("--lambda-div", type=float, default=1.0)
    p.add_argument("--augment", action="store_true")
    p.add_argument("--val-ids", nargs="+", default=None,
                   help="検証用サンプル id（best パターン選択に使用）")
    p.add_argument("--log-interval", type=int, default=20)
    p.add_argument("--logdir", default=None, help="TensorBoard ログ先（任意）")
    p.add_argument("--save", default=None, help="学習済み pattern logits の保存先 .pt")
    p.add_argument("--seed", type=int, default=None, help="再現性のための乱数シード")
    return p.parse_args()


@torch.no_grad()
def evaluate(encoder, decoder, loader, device):
    """val/test の平均 recon MSE と PSNR を返す。"""
    encoder.eval()
    decoder.eval()
    tot_mse, n = 0.0, 0
    for focal in loader:
        focal = focal.to(device)
        h, w = focal.shape[-2:]
        masks = encoder.tiled_masks(h, w)
        snapshot = simulate_snapshot(focal, masks)
        focal_hat = decoder(snapshot)
        tot_mse += F.mse_loss(focal_hat, focal).item() * focal.shape[0]
        n += focal.shape[0]
    encoder.train()
    decoder.train()
    mse = tot_mse / max(n, 1)
    p = 99.0 if mse <= 0 else 10.0 * torch.log10(torch.tensor(1.0 / mse)).item()
    return mse, p


def main():
    args = parse_args()
    if args.seed is not None:
        torch.manual_seed(args.seed)
        import numpy as np
        np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[phase_b] device={device}, samples={args.sample_ids}")

    dataset = CellFocalStackDataset(args.root, args.sample_ids, augment=args.augment)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)

    val_loader = None
    if args.val_ids:
        val_set = CellFocalStackDataset(args.root, args.val_ids, augment=False)
        val_loader = DataLoader(val_set, batch_size=1, shuffle=False)
        print(f"[phase_b] val samples={args.val_ids}")

    encoder = LearnablePattern(depth=args.depth, block=args.block).to(device)
    decoder = SnapshotDecoder(out_depth=args.depth, base=args.base).to(device)

    opt = torch.optim.Adam([
        {"params": encoder.parameters(), "lr": args.lr_encoder},
        {"params": decoder.parameters(), "lr": args.lr_decoder},
    ])

    writer = None
    if args.logdir:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(args.logdir)

    # M1 診断用に初期パターンを保存
    init_binary = encoder.binary_pattern().detach().cpu().clone()
    best_val = float("inf")
    best_binary = init_binary.clone()
    best_logits = encoder.logits.detach().cpu().clone()

    it = 0
    data_iter = iter(loader)
    while it < args.n_iter:
        try:
            focal = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            focal = next(data_iter)
        focal = focal.to(device)  # (B, D, H, W)
        h, w = focal.shape[-2:]

        masks = encoder.tiled_masks(h, w)  # (D, H, W)
        snapshot = simulate_snapshot(focal, masks)  # (B, 1, H, W)
        focal_hat = decoder(snapshot)  # (B, D, H, W)

        recon = F.mse_loss(focal_hat, focal)
        probs = encoder.probs()
        open_pen = aperture_penalty(probs)
        div_pen = diversity_penalty(probs)
        loss = recon + args.lambda_open * open_pen + args.lambda_div * div_pen

        opt.zero_grad()
        loss.backward()
        opt.step()

        if it % args.log_interval == 0 or it == args.n_iter - 1:
            ap = probs.mean().item()
            print(f"it={it:4d} loss={loss.item():.5f} recon={recon.item():.5f} "
                  f"psnr={psnr(focal_hat, focal):.2f}dB open={open_pen.item():.4f} "
                  f"div={div_pen.item():.4f} aperture={ap:.3f}")
            if writer:
                writer.add_scalar("loss/total", loss.item(), it)
                writer.add_scalar("loss/recon", recon.item(), it)
                writer.add_scalar("loss/open", open_pen.item(), it)
                writer.add_scalar("loss/div", div_pen.item(), it)
                writer.add_scalar("metric/aperture_ratio", ap, it)
                writer.add_scalar("metric/psnr", psnr(focal_hat, focal), it)

            if val_loader is not None:
                val_mse, val_psnr = evaluate(encoder, decoder, val_loader, device)
                print(f"          [val] mse={val_mse:.5f} psnr={val_psnr:.2f}dB")
                if writer:
                    writer.add_scalar("metric/psnr_val", val_psnr, it)
                if val_mse < best_val:
                    best_val = val_mse
                    best_binary = encoder.binary_pattern().detach().cpu().clone()
                    best_logits = encoder.logits.detach().cpu().clone()
        it += 1

    final_binary = encoder.binary_pattern().detach().cpu()
    changed = (final_binary != init_binary).float().mean().item()
    print(f"\n[診断] パターン変化率(初期→最終): {changed*100:.1f}% of cells flipped")

    # val があれば best パターン、なければ最終パターンを採用
    if val_loader is not None:
        save_binary, save_logits = best_binary, best_logits
        print(f"[診断] best val mse={best_val:.5f}")
    else:
        save_binary, save_logits = final_binary, encoder.logits.detach().cpu()
    print(f"[診断] 採用 binary pattern (D,b,b):\n{save_binary}")

    if args.save:
        os.makedirs(os.path.dirname(os.path.abspath(args.save)), exist_ok=True)
        torch.save({
            "logits": save_logits,
            "binary_pattern": save_binary,
            "init_binary": init_binary,
            "final_binary": final_binary,
            "args": vars(args),
        }, args.save)
        print(f"[phase_b] saved pattern to {args.save}")

    if writer:
        writer.close()


if __name__ == "__main__":
    main()
