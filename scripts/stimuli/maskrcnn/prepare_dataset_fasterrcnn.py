import os
import shutil
import numpy as np
from PIL import Image


root = 'render/'
#target = '../data/MOTDots/'
target = 'data/EvalDots/'
target_pngs = os.path.join(target, 'PNGImages')
target_masks = os.path.join(target, 'DotMasks')

# take files under render/*.png, collect masks and copy to data/MOTDots

#M = 10000 # number of examples. ids go from 2 to 1001
M = 100

for m in range(M):
    print(m)
    fname = f'{(m + 2):03}.png'
    fullfname = os.path.join(root, fname)
    shutil.copy(fullfname, os.path.join(target_pngs, f'{(m+1):03}.png'))

    # collect masks in N x W x H and save it as a png
    masks = np.zeros((20, 800, 800)).astype('int16')
    for i in range(20):
        fname = f'{(m+2):03}_{i+1}.png'
        fullfname = os.path.join(root, fname)
        if os.path.exists(fullfname) == False:
            break

        mask = Image.open(fullfname)
        mask = np.array(mask).astype('int16')[:, :, 0]
        mask[mask != 255] = 0
        mask = 255 - mask
        mask[mask == 255] = i + 1
        masks[i] = mask

    masks = masks[:i].astype('int16')

    np.save(os.path.join(target_masks, f'{(m+1):03}_mask.npy'), masks)


