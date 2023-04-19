import os
import urllib.request
import zipfile
import numpy as np
import pandas as pd
import model as md
import dataloader as dl


def main():
    #creating dataloaders
    data_dir = '/content/drive/MyDrive/data/data/'
    batch_size = 4
    d = dl.Dataloader(data_dir, 4)
    d.plot_images()

    #train model
    resnet = md.ResNet(batch_size, d.dataset_sizes)
    num_epochs = 24

    print("------------------------")
    print("Training model...")
    resnet.train_model(d.dataloaders, num_epochs)
    print("------------------------")
    print("Visualizing results...")
    resnet.visualize_results(d.val_loader, d.class_names)
    print("------------------------")
    print("Saving model")
    resnet.savemodel()

if __name__ == '__main__':
    main()
