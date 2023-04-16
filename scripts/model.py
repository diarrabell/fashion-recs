import os
import torch
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import time
import numpy as np
import pandas as pd
import torch.nn.functional as F
import copy
import warnings
import matplotlib.pyplot as plt


warnings.filterwarnings('ignore')
MODELS_PATH = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'models'))

torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed(0)

class ResNet:
    ''' initialize resnet50 model '''
    def __init__(self, batch_size, dataset_sizes) -> None: 
      self.dataset_sizes = dataset_sizes
      self.batch_size = batch_size
      # instantiate pre-trained resnet
      self.model = torchvision.models.resnet50(pretrained=True)
      # shut off autograd for all layers to freeze model so layer weights are not trained
      for param in self.model.parameters():
          param.requires_grad = False       
      # get number of inputs to final linear layer
      self.num_ftrs = self.model.fc.in_features
      # replace final linear layer with new linear layer with 8 outputs
      self.model.fc = nn.Linear(self.num_ftrs, 8)
      # cost function: cross entropy loss
      self.criterion = nn.CrossEntropyLoss()
      # optimizer: SGD
      self.optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)
      # learning rate scheduler
      self.lr_scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=7, gamma=0.1)
      self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    ''' train model '''
    def train_model(self, dataloaders, num_epochs): #train model
        # self.dataloaders = dataloaders
        model = self.model.to(self.device)
        since = time.time()

        best_model_wts = copy.deepcopy(model.state_dict)
        best_acc = 0.0

        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)

            for phase in ['train', 'val']:
                if phase == 'train': #set model to training mode
                    model.train()
                else:
                    model.eval() # set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                # get input images and labels, and send to GPU if available

                for inputs, labels in dataloaders[phase]:
                  inputs = inputs.to(self.device)
                  labels = labels.to(self.device)

                  #zero the weight gradients
                  self.optimizer.zero_grad()

                  #forward pass to get outputs and calculate loss
                  #track gradient only for training data
                  with torch.set_grad_enabled(phase == 'train'):
                      outputs = model(inputs)
                      _, preds = torch.max(outputs, 1)
                      loss = self.criterion(outputs, labels)

                      #backpropagation to get the gradients with respect to each weight, only if in train
                      if phase == 'train':
                          loss.backward()
                          # update the weights
                          self.optimizer.step()

                  # convert loss into a scalar and add it to running loss
                  running_loss += loss.item() * inputs.size(0)
                  #track number of correct predictions
                  running_corrects += torch.sum(preds == labels.data)

                # step along learning rate scheduler when in train
                if phase == 'train':
                    self.lr_scheduler.step()
                
                # calculate and display average loss and accuracy for the epoch
                epoch_loss = running_loss / self.dataset_sizes[phase]
                epoch_acc = running_corrects.double() / self.dataset_sizes[phase]
                print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

                # if model performs better on val set, save weights as the best model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                print()
        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:3f}'.format(best_acc))

        #load weights from best model
        model.load_state_dict(best_model_wts)
        self.final_model = model

    def visualize_results(self, val_loader, class_names):
        model = self.final_model.to(self.device)
        with torch.no_grad():
            model.eval()
            # get a batch of validation images
            images, labels = next(iter(val_loader))
            images, labels = images.to(self.device), labels.to(self.device)
            # get predictions
            _, preds = torch.max(model(images), 1)
            preds = np.squeeze(preds.cpu().numpy())
            images = images.cpu().numpy()
    