import os
import shutil
import numpy as np
from PIL import Image

from pathlib import Path
import json

# we read target_pngs and transform to numpy arrays (target_npys)
dataset_path = Path('output') / 'datasets' / 'mask_rcnn'
target_pngs = dataset_path / 'target_pngs'
target_npys = dataset_path / 'target_npys'
target_npys.mkdir(exist_ok=True)

n_trials = 1
n_timesteps = 120
n_max_masks = 20

# getting mask dimension
fname = f'{(1):03}_{(1):03}_{(1):03}.png'
mask = Image.open(target_pngs / fname)
mask_width, mask_height = mask.size

for trial in range(1, n_trials+1):
    for t in range(1, n_timesteps+1):
        # collect masks in N x W x H and save it as an npy
        masks = np.zeros((n_max_masks, mask_width, mask_height)).astype(np.uint8)
        for i in range(1, n_max_masks+1):
            fname = f'{(trial):03}_{(t):03}_{(i):03}.png'
            target_png_path = target_pngs / fname
            if not target_png_path.exists():
                break

            mask = Image.open(target_png_path)
            mask = np.array(mask).astype(np.uint8)

            mask[mask == 255] = i
            masks[i-1] = mask

        masks = masks[:i]
        fname = f'{(trial):03}_{(t):03}.npy'
        np.save(target_npys / fname, masks)

