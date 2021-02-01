import torch
from torch.utils.data import Dataset
from torchvision.datasets.voc import VOCDetection
from torchvision.transforms.functional import to_tensor

from utils import get_transform


class PascalVOCDataset(Dataset):
    def __init__(self, root, split, image_size=(300, 300), keep_difficult=False):
        """
        A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.

        :param root: path to stored voc data
        :param split: split, 'TRAIN' or 'TEST', train and val dataset from both 2007 and 2012 is used for 'TRAIN' split,
            test dataset from 2007 is used for 'TEST' split.
        :param image_size: (height, width) for network input size
        :param keep_difficult: keep ot discard objects considered diffcult to detect
        """
        split = split.upper()
        assert split in {'TRAIN', 'TEST'}

        self.keep_difficult = keep_difficult
        self.transform = get_transform(image_size, split)

        if split == 'TRAIN':
            self.datasets = [
                VOCDetection(root, year='2007', image_set='trainval'),
                VOCDetection(root, year='2012', image_set='trainval')
            ]
        else:
            self.datasets = [
                VOCDetection(root, year='2007', image_set='test')
            ]

    def __len__(self):
        return sum([len(ds) for ds in self.datasets])

    def __getitem__(self, i):
        for ds in self.datasets:
            if i < len(ds):
                break
            i -= len(ds)

        image, targets = ds[i]
        image = to_tensor(image)

        objects = targets['annotation']['object']
        # Discard difficult objects, if desired
        if not self.keep_difficult:
            objects = [obj for obj in objects if obj['difficult'] != '1']

        boxes = [obj['bndbox'] for obj in objects]  # (n_objects, 4)
        boxes = [[int(box['xmin']), int(box['ymin']), int(box['xmax']), int(box['ymax'])] for box in boxes]
        boxes = torch.FloatTensor(boxes)

        labels = [self.label_map()[obj['name']] for obj in objects]  # (n_objects)
        labels = torch.FloatTensor(labels)

        image, boxes, labels = self.transform(image, boxes, labels)
        return image, boxes, labels

    @staticmethod
    def collate_fn(batch):
        """
        Since each image may have a different number of objects, we need a collate function (to be passed to the DataLoader).

        This describes how to combine these tensors of different sizes. We use lists.

        Note: this need not be defined in this Class, can be standalone.

        :param batch: an iterable of N sets from __getitem__()
        :return: a tensor of images, lists of varying-size tensors of bounding boxes, labels, and difficulties
        """
        images = [i[0] for i in batch]
        boxes = [i[1] for i in batch]
        labels = [i[2] for i in batch]

        images = torch.stack(images, dim=0)

        return images, boxes, labels  # tensor (N, 3, height, width), 2 lists of N tensors eachs

    @staticmethod
    def label_map():
        return {
            'aeroplane': 1,
            'bicycle': 2,
            'bird': 3,
            'boat': 4,
            'bottle': 5,
            'bus': 6,
            'car': 7,
            'cat': 8,
            'chair': 9,
            'cow': 10,
            'diningtable': 11,
            'dog': 12,
            'horse': 13,
            'motorbike': 14,
            'person': 15,
            'pottedplant': 16,
            'sheep': 17,
            'sofa': 18,
            'train': 19,
            'tvmonitor': 20,
            'background': 0,
        }
