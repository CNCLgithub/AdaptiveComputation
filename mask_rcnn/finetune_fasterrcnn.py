# Modified from:
# Sample code from the TorchVision 0.3 Object Detection Finetuning Tutorial
# http://pytorch.org/tutorials/intermediate/torchvision_tutorial.html

import os
import numpy as np
import torch
from PIL import Image, ImageDraw

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from engine import train_one_epoch, evaluate
import utils as utils
import transforms as T

import json
from pathlib import Path

from matplotlib import cm

class MOTDataset(object):
    def __init__(self, dataset_path, transforms):
        self.dataset_path = Path(dataset_path)
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.input_pngs = list(sorted(os.listdir(self.dataset_path / 'input_pngs')))
        self.target_npys = list(sorted(os.listdir(self.dataset_path / 'target_npys')))

    def __getitem__(self, idx):
        print(idx)

        # load images ad masks
        input_png_path = self.dataset_path / 'input_pngs' / self.input_pngs[idx]
        target_npys_path = self.dataset_path / 'target_npys' / self.target_npys[idx]

        img = Image.open(input_png_path).convert("RGB")
        # note that we haven't converted the mask to RGB,
        # because each color corresponds to a different instance
        # with 0 being background
        masks = np.load(target_npys_path)
        masks = np.array(masks)
        
        # we don't care about background
        masks = masks[1:]

        # instances are encoded as different colors
        obj_ids = np.unique(masks)
        # the first id is representing non-mask regions (0)
        obj_ids = obj_ids[1:]
    
        # split the color-encoded mask into a set
        # of binary masks
        masks = np.isin(masks, obj_ids)

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
        return len(self.input_pngs)

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


def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


def main(args):
    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print("device:", device)

    # our dataset has two classes only - background and person
    num_classes = 2
    # use our dataset and defined transformations
    dataset = MOTDataset(args.data_path, get_transform(train=True)) #this should give the folder of the dataset
    dataset_test = MOTDataset(args.data_path, get_transform(train=False))
    
    print("dataset length:", len(dataset))

    # split the dataset in train and test set
    torch.manual_seed(0)
    indices = torch.randperm(len(dataset)).tolist()
    dataset = torch.utils.data.Subset(dataset, indices[:-50])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])

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
    
    if args.checkpoint_file:
        # load model
        checkpoint_path = Path(args.checkpoints_dir) / args.checkpoint_file
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(checkpoint_path, checkpoint['epoch']))

    # if we're just testing
    if args.testing:
        utils.mkdir(args.segments_dir)

        model.eval()

        counter = 0
        positions = []
        diffs = []
        #f = h5py.File(os.path.join(args.output_dir, 'bottom-up-observations.hdf5'), 'w')
        for images, targets in data_loader_test:
            images = list(img.to(device) for img in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            outputs = model(images)
            outputs = [{k: v.to(device) for k, v in t.items()} for t in outputs]

            # visualize segments
            img = images[0].detach().cpu()
            img = np.transpose(img.numpy(), (1, 2, 0))
            img *= 255
            img = img.astype(np.uint8)
            img = Image.fromarray(img)

            segments = outputs[0]['masks'].detach().cpu().numpy()
            for k in range(segments.shape[0]):
                segment = segments[k].squeeze()
                segment[segment < 0.75] = 0
                segment[segment != 0] = 128
                
                color = np.random.rand(3) * 255
                color_array = np.broadcast_to(color, (800, 800, 3))
                color_img = Image.fromarray(color_array.astype('uint8')).convert('RGBA')
                mask = Image.fromarray(segment).convert('L')
                
                img = Image.composite(color_img, img, mask)

            for k in range(segments.shape[0]):
                segment = segments[k].squeeze()
                segment[segment < 0.75] = 0
                segment[segment != 0] = 128

                pos = np.where(segment)
                xmin = np.min(pos[0])
                ymin = np.min(pos[1])
                xmax = np.max(pos[0])
                ymax = np.max(pos[1])

                x = (xmin + xmax)/2
                y = (ymin + ymax)/2

                d = ImageDraw.Draw(img)
                d.text((y + np.random.uniform(-10,10), x + np.random.uniform(-10,10)), str(k+1), fill=(0,0,0))
                
            img_path = Path(args.segments_dir) / f'{counter:03}.png'
            img.save(img_path)
            
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

            """

            counter += 1

        # f.close()

        return positions, outputs, targets, diffs

    else: 

        # construct an optimizer
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=0.005,
                                    momentum=0.9, weight_decay=0.0005)
        # and a learning rate scheduler
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                       step_size=3,
                                                       gamma=0.1)

        # let's train it for 10 epochs
        num_epochs = 10

        #for epoch in range(num_epochs):
        for epoch in range(5, num_epochs):

            # train for one epoch, printing every 10 iterations
            train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
            # update the learning rate
            lr_scheduler.step()

            if args.checkpoints_dir:
                utils.save_on_master({
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'args': args,
                    'epoch': epoch},
                    os.path.join(args.checkpoints_dir, 'model_{}.pth'.format(epoch)))

            # evaluate on the test dataset
            evaluate(model, data_loader_test, device=device)

    
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="finetune faster-rcnn")

    parser.add_argument('--data-path', default='output/datasets/mask_rcnn', help='dataset')
    parser.add_argument('--checkpoints-dir', default='output/checkpoints/', help='where to save')
    parser.add_argument('--testing', action='store_true', help='testing the model')
    parser.add_argument('--checkpoint-file', default='model_9.pth', help='checkpoint file')
    parser.add_argument('--segments-dir', default='output/segments/',
                        help='directory for outputted segments')
    
    args = parser.parse_args()

    if args.checkpoints_dir:
        utils.mkdir(args.checkpoints_dir)
    main(args)

