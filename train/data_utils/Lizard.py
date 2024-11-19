import os
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import torchvision.transforms.v2 as transforms
import scipy.io as sio

class SegmentationDataset(Dataset):
    def __init__(self, images, labels, split='train'):
        self.images = images
        self.labels = labels
        self.split = split
        self.train_transform = transforms.Compose([
            transforms.ToImage(),
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.ToDtype(torch.float32),
        ])

        self.val_transform = transforms.Compose([
            transforms.ToImage(),
            transforms.Resize((224, 224)),
            transforms.ToDtype(torch.float32),
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = np.expand_dims(self.labels[idx, :, :, 1].astype(np.int64), axis=2)
        
        # Convert image and label to PyTorch tensors and resize
        if self.split == 'train':
            image, label = self.train_transform(image, label)
        elif self.split=='val':
            image, label = self.val_transform(image, label)

        return image, torch.squeeze(label).long()

class Lizard():
    def __init__(self, dataset_path='/home/ec2-user/SageMaker/mnt/efs/Lizard', batch_size=16):
        images = np.load(f'{dataset_path}/data/images.npy')
        labels = np.load(f'{dataset_path}/data/labels.npy')

        # Split the data into train and test sets
        train_images, test_images, train_labels, test_labels = train_test_split(
            images, labels, test_size=0.2, random_state=42
        )

        # Create the train and test datasets
        train_dataset = SegmentationDataset(train_images, train_labels, split='train')
        test_dataset = SegmentationDataset(test_images, test_labels, split='val')

        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8)
        self.id2class = {
            0: 'Background',
            1: 'Neutrophil',
            2: 'Epithelial',
            3: 'Lymphocyte',
            4: 'Plasma',
            5: 'Neutrophil',
            6: 'Connective tissue'
        }
