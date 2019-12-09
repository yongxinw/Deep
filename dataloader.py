# Created by yongxinwang at 2019-12-08 17:33
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torch
from PIL import Image

import json
import os
import os.path as osp
import numpy as np
import random

class VPDataset(Dataset):
    def __init__(self, image_dir, label_file, grid_resolution, val_size=10000, val=False):
        super(VPDataset, self).__init__()
        self.image_dir = image_dir
        np.random.seed(1)

        self.image_names = os.listdir(image_dir)
        random.shuffle(self.image_names)
        val_index = np.arange(len(self.image_names) - val_size, len(self.image_names))
        train_index = np.arange(0, len(self.image_names) - val_size)

        if val:
            self.image_names = self.image_names[val_index]
        else:
            self.image_names = self.image_names[train_index]

        self.labels = self.read_json(label_file)
        self.grid_resolution = grid_resolution

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    @staticmethod
    def read_json(label_file):
        data_dict = json.load(open(label_file, 'r'))
        return data_dict

    def label_to_grid(self, label):
        """
        Convert raw labels to grid labels
        :param label: vanishing point annotation with normalized coordinates, numpy array of shape (N, 2)
        :return:
        """
        pos_index = label * self.grid_resolution
        pos_index = pos_index.astype(np.int)
        grid_labels = np.zeros(self.grid_resolution, dtype=np.int)
        grid_labels[pos_index[:, 0], pos_index[:, 1]] = 1
        return grid_labels

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, index):

        # read image
        image_path = osp.join(self.image_dir, self.image_names[index])
        image = Image.open(image_path)
        image = self.transform(image)

        # construct the labels
        raw_label = np.array(self.labels["imgs"][self.image_names[index]])
        grid_label = self.label_to_grid(raw_label)
        grid_label = torch.from_numpy(grid_label).view(-1)

        return {"image": image, "grid_label": grid_label}


def get_dataloader(image_dir, label_file, grid_resolution, **kwargs):
    train_set = VPDataset(image_dir, label_file, grid_resolution, val=False)
    train_loader = DataLoader(train_set, **kwargs)

    val_set = VPDataset(image_dir, label_file, grid_resolution, val=True)
    val_loader = DataLoader(val_set, **kwargs)
    return train_loader, val_loader


if __name__ == "__main__":
    # dset = VPDataset(image_dir="./images", label_file="dummy.json", grid_resolution=[64, 64])
    # item = dset[0]
    loader = get_dataloader(image_dir="./images", label_file="dummy.json", grid_resolution=[64, 64],
                            batch_size=2, num_workers=2)

    for batch, data in enumerate(loader):
        import ipdb
        ipdb.set_trace()
