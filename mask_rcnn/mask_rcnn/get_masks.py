# partly adapted from
# Sample code from the TorchVision 0.3 Object Detection Finetuning Tutorial
# http://pytorch.org/tutorials/intermediate/torchvision_tutorial.html

from PIL import Image
import numpy as np

import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

def get_model_instance_segmentation(num_classes):
    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 128
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)

    return model


print('initializing mask_rcnn...')
checkpoint_path = 'output/checkpoints/mask_rcnn_weights_0.pth'

# train on the GPU or on the CPU, if a GPU is not available
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print("device:", device)

# background or dot
num_classes = 2

# get the model using helper function
model = get_model_instance_segmentation(num_classes)

# move model to the right device
model.to(device)

# loading weights
checkpoint = torch.load(checkpoint_path,
                        map_location=str(device))

model.load_state_dict(checkpoint['model'])
model.eval()

print('mask_rcnn initialized!')


def get_masks(img):
    imgs = torch.Tensor([img])

    if torch.cuda.is_available():
       imgs = imgs.cuda()

    with torch.no_grad():   
        outputs = model(imgs)
        outputs = [{k: v.to(device) for k, v in output.items()} for output in outputs]

    segments = outputs[0]['masks'].detach().cpu().numpy()
    w, h = segments[0].squeeze().shape
    masks = np.empty((segments.shape[0], w, h), dtype=bool) # TODO automate width and height

    for k in range(masks.shape[0]):
        segment = segments[k].squeeze()
        masks[k][segment >= 0.75] = True
        masks[k][segment < 0.75] = False
        
        # testing
        mask = Image.fromarray(segment).convert('L')
        mask.save('mask.png')

    return masks
