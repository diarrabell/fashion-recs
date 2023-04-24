"""
This file contains a class that evaluates a test dataset on the trained model. It returns the most frequently predicted classes in the dataset."""

import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt


class ClassifyData:
    def __init__(self, model, dataloader, device, class_names) -> None:   
        self.dataloader = dataloader
        self.device = device
        self.class_names = class_names
        self.model = model.to(self.device)
        
    
    def test_model(self):
        #turn autograd off
        with torch.no_grad():
            #set the model to evaluation mode
            self.model.eval()
            #set up lists to store true and predicted values
            y_true = []
            test_preds = []
            test_probs = []
            #calculate the predictions on the test set and add to list
            for data in self.dataloader:
                inputs, labels = data[0].to(self.device), data[1].to(self.device)
                #feed inputs through model to get raw scores
                logits = self.model.forward(inputs)
                #convert raw scores to probabilities
                probs = F.softmax(logits, dim=1)
                #get discrete predictions using argmax
                preds = np.argmax(probs.detach().cpu().numpy(), axis=1)
                #add predictions and actuals to lists
                test_preds.extend(preds)
                test_probs.extend(probs)
                y_true.extend(labels.cpu())
        
        #calculate accuracy
        test_preds = np.array(test_preds)
        y_true = np.array(y_true)
        test_acc = np.sum(test_preds == y_true)/y_true.shape[0]

        self.test_preds = list(test_preds)
    
    def visualize_results(self):
        with torch.no_grad():
            self.model.eval()
            images, labels = next(iter(self.dataloader))
            images, labels = images.to(self.device), labels.to(self.device)

            #get predictions
            _, preds = torch.max(self.model(images), 1)
            preds = np.squeeze(preds.cpu().numpy())
            images = images.cpu().numpy()

            #plot the images with predictions
            fig = plt.figure(figsize = (15, 10))
            for idx in np.arange(len(preds)):
                ax = fig.add_subplot(2, len(preds)//2, idx+1, xsticks=[], yticks=[])
                image = images[idx]
                image = image.transpose((1, 2, 0))
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0,224, 0,225])
                image = std * image + mean
                image = np.clip(image, 0, 1)
                ax.imshow(image)
                ax.set_title("{}".format(self.class_names[preds[idx]], color="green"))

    def find_top_predictions(self):
        self.top_predictions = [x for x in self.test_preds if self.test_preds.count(x) > 1]
        self.top_predictions = list(set(self.top_predictions))
        self.top_predictions = [self.class_names[x] for x in self.top_predictions]
        return self.top_predictions



