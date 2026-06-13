"""cell 多焦点画像の DataLoader（B-0）。

各サンプル inputs/multi_focus_data/cell/XXXX/ に 00.jpg〜10.jpg（D=11, 128x128, grayscale）。
volume / GT は存在しないので、この 11 枚の焦点スタックをそのまま GT として扱う。
"""

import os
import glob

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


def _load_focal_stack(sample_dir):
    """サンプルディレクトリの jpg スタックを (D, H, W) float32 [0,1] で返す。"""
    paths = sorted(glob.glob(os.path.join(sample_dir, "*.jpg")))
    if not paths:
        raise FileNotFoundError(f"no jpg found in {sample_dir}")
    slices = []
    for p in paths:
        arr = np.asarray(Image.open(p).convert("L"), dtype=np.float32) / 255.0
        slices.append(arr)
    return np.stack(slices, axis=0)  # (D, H, W)


class CellFocalStackDataset(Dataset):
    """cell 多焦点スタックを供給する Dataset。

    Args:
        root: cell データのルート（例: inputs/multi_focus_data/cell）
        sample_ids: 使うサンプル id のリスト（例: ["0000", "0001"]）。None なら全て。
        augment: True で rotation(90/180/270) / flip(h/v) / intensity jitter(±10%) を適用
    """

    def __init__(self, root, sample_ids=None, augment=False):
        if sample_ids is None:
            sample_ids = sorted(
                d for d in os.listdir(root)
                if os.path.isdir(os.path.join(root, d))
            )
        self.root = root
        self.sample_ids = list(sample_ids)
        self.augment = augment
        # 少数サンプルなので起動時に全部メモリへ
        self.stacks = {
            sid: _load_focal_stack(os.path.join(root, sid))
            for sid in self.sample_ids
        }

    def __len__(self):
        return len(self.sample_ids)

    def _augment(self, stack):
        # stack: (D, H, W) numpy
        k = np.random.randint(0, 4)
        if k:
            stack = np.rot90(stack, k=k, axes=(1, 2))
        if np.random.rand() < 0.5:
            stack = stack[:, ::-1, :]
        if np.random.rand() < 0.5:
            stack = stack[:, :, ::-1]
        # intensity jitter ±10%
        gain = 1.0 + (np.random.rand() * 0.2 - 0.1)
        stack = np.clip(stack * gain, 0.0, 1.0)
        return np.ascontiguousarray(stack)

    def __getitem__(self, idx):
        sid = self.sample_ids[idx]
        stack = self.stacks[sid]
        if self.augment:
            stack = self._augment(stack)
        return torch.from_numpy(stack.copy()).float()  # (D, H, W)
