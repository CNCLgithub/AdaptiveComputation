# Modified from:
# Sample code from the TorchVision 0.3 Object Detection Finetuning Tutorial
# http://pytorch.org/tutorials/intermediate/torchvision_tutorial.html

import os
import numpy as np
import torch
from PIL import Image
import h5py

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from engine import train_one_epoch, evaluate
import utils as utils
import transforms as T

class MOTDataset(object):
    def __init__(self, root, transforms):
        self.root = root
        #self.mask_root = "../data/MOTDots"
        self.mask_root = "../data/EvalDots"
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, "PNGImages"))))
        #self.masks = list(sorted(os.listdir(os.path.join(root, "DotMasks"))))
        self.masks = list(sorted(os.listdir(os.path.join(self.mask_root, "DotMasks"))))

    def __getitem__(self, idx):
        # load images ad masks
        img_path = os.path.join(self.root, "PNGImages", self.imgs[idx])
        mask_path = os.path.join(self.mask_root, "DotMasks", self.masks[idx])
        img = Image.open(img_path).convert("RGB")
        # note that we haven't converted the mask to RGB,
        # because each color corresponds to a different instance
        # with 0 being background
        mask = np.load(mask_path)

        mask = np.array(mask)
        # instances are encoded as different colors
        obj_ids = np.unique(mask)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]

        # split the color-encoded mask into a set
        # of binary masks
        masks = mask == obj_ids[:, None, None]

        # get bounding box coordinates for each mask
        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)

def get_model_instance_segmentation(num_classes):
    # load an instance segmentation model pre-trained pre-trained on COCO
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


def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


def mix_match_score(boxes, target_boxes):
    N = boxes.shape[1]
    diff = np.zeros(N)
    for i in range(N):
        dists = [np.linalg.norm(boxes[:,i] - target_boxes[:,j]) for j in range(N)]
        diff[i] = np.array(dists).min()

    return diff

def main(args):
    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # our dataset has two classes only - background and person
    num_classes = 2
    # use our dataset and defined transformations
    dataset = MOTDataset(args.data_path, get_transform(train=True)) #this should give the folder of the dataset
    dataset_test = MOTDataset(args.data_path, get_transform(train=False))

    # split the dataset in train and test set
    #indices = torch.randperm(len(dataset)).tolist()
    indices = list(range(len(dataset)))
    dataset_test = torch.utils.data.Subset(dataset_test, indices)

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=2, shuffle=True, num_workers=2,
        collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=1,
        collate_fn=utils.collate_fn)

    # get the model using our helper function
    model = get_model_instance_segmentation(num_classes)

    # move model to the right device
    model.to(device)

    # load model
    checkpoint = torch.load(args.load_path)
    model.load_state_dict(checkpoint['model'])
    print("=> loaded checkpoint '{}' (epoch {})"
          .format(args.load_path, checkpoint['epoch']))
    model.eval()

    cpu_device = torch.device("cpu")

    counter = 0
    positions = []
    diffs = []
    f = h5py.File(os.path.join(args.output_dir, 'bottom-up-observations.hdf5'), 'w')
    for image, targets in data_loader_test:
        image = list(img.to(device) for img in image)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(image)

        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]

        """
        # visualize segments
        aggregate = np.zeros((800, 800))
        segments = outputs[0]['masks'].detach().cpu().numpy()

        for k in range(segments.shape[0]):
            segment = segments[k].squeeze()
            segment[segment < 0.75] = 0
            segment[segment != 0] = 1
            aggregate += segment
            
        aggregate[aggregate != 0] = 255
        aggregate_image = Image.fromarray(aggregate)
        aggregate_image = aggregate_image.convert('L')
        aggregate_image.save(os.path.join(args.output_dir, f'{counter:05}.png'))
        """

        boxes = outputs[0]['boxes'].detach().cpu().numpy()
        boxes_x = (boxes[:, 0] + boxes[:, 2]) / 2 - 400
        boxes_y = (boxes[:, 1] + boxes[:, 3]) / 2 - 400
        boxes = np.vstack((boxes_x, boxes_y))
        positions.append(boxes)

        target_boxes = targets[0]['boxes'].detach().cpu().numpy()
        target_boxes_x = (target_boxes[:, 0] + target_boxes[:,2]) / 2 - 400
        target_boxes_y = (target_boxes[:, 1] + target_boxes[:,3]) / 2 - 400
        target_boxes = np.vstack((target_boxes_x, target_boxes_y))
        
        if boxes.shape[1] == target_boxes.shape[1]:
            diff = mix_match_score(boxes, target_boxes)
            print(diff)
            diffs.append(diff)
        

        f.create_dataset(str(counter), data=positions[-1])

        counter += 1


    f.close()

    return positions, outputs, targets, diffs

    


    print("That's it!")
    
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="test faster-rcnn")

    parser.add_argument('--data-path', default='outputs/datasets/mask_rcnn', help='dataset')
    parser.add_argument('--load-path', default='outputs/checkpoints/model_9.pth', help='checkpoint file to load')
    parser.add_argument('--output-dir', default='outputs/segments/', help='where to save output segmentations')
    
    args = parser.parse_args()

    if args.output_dir:
        utils.mkdir(args.output_dir)

    if os.path.exists(args.load_path) == False:
        print("Checkpoint file can't be found.")
    else:
        a, b, c, d = main(args)

