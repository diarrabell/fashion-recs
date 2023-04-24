"""
This script contains classes to create training, validation, and test dataloaders. Returns dataloaders.
"""


import os
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


class TrainDataloader:
    def __init__(self, dir, batch_size) -> None:
        self.dir = dir
        self.batch_size = batch_size
        self.data_transforms = {
            'train': transforms.Compose([
                transforms.Resize((256,256)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.Resize((256,256)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),          
        }

        # Create datasets for training and validation sets
        self.train_dataset = datasets.ImageFolder(os.path.join(self.dir, 'train'), transform = self.data_transforms['train'])
        self.val_dataset = datasets.ImageFolder(os.path.join(self.dir, 'val'), transform=self.data_transforms['val'])

        # Create dataloaders for training and validation sets
        self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0)
        self.val_loader = torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0)

        # Set up dict for dataloaders
        self.dataloaders = {'train': self.train_loader, 'val': self.val_loader}
        self.dataset_sizes = {'train': len(self.train_dataset), 'val': len(self.val_dataset)}
        self.class_names = self.train_dataset.classes
        print("Classes: {0}".format(self.class_names))

        print('Dataset sizes: ', self.dataset_sizes['train'], self.dataset_sizes['val'])

        #set random seeds for reproducibility
        torch.manual_seed(0)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(0)
    
    def plot_images(self):
        images, labels = next(iter(self.train_loader))
        images = images.numpy()
        fig = plt.figure(figsize=(10, 6))
        for idx in np.arange(self.batch_size):
            ax = fig.add_subplot(2, self.batch_size//2, idx+1, xticks=[], yticks=[])
            image = images[idx]
            image = image.transpose((1, 2, 0))
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            image = std * image + mean
            image = np.clip(image, 0, 1)
            ax.imshow(image)
            ax.set_title("{}".format(self.class_names[labels[idx]]))
        plt.show()

class TestDataloader:
    def __init__(self, dir, batch_size) -> None:
        self.dir = dir
        self.batch_size = batch_size
        test_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        #create dataset for test dataset
        test_dataset = datasets.ImageFolder(self.dir, transform = test_transform)
        #create dataloader for test dataset
        self.test_dataloader = DataLoader(test_dataset, self.batch_size)
        #get class names
        self.class_names = test_dataset.classes

        #set random seeds for reproducibility
        torch.manual_seed(0)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(0)



