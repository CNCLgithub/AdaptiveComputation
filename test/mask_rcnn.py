import mask_rcnn
import numpy as np
from PIL import Image


img = Image.open('render/001.png')
img = np.array(img)

get_masks(img)
