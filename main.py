import streamlit as st
import os
import time
import numpy as np
import pandas as pd
import scripts.model as md
import scripts.dataloader as dl
import scripts.classifier as cl
import scripts.recommender as rc
import torch
from torchvision import datasets, transforms
from PIL import Image

os.environ['KMP_DUPLICATE_LIB_OK']='True'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#load trained model
def load_model():
    model = torch.load("models/model.pt", map_location=torch.device(device))
    model.eval()
    st.success("loading model...")
    return model

def load_image(image):
    return Image.open(image)

def test_images():
    #creating dataloaders
    data_dir = 'uploads'
    batch_size = 10
    test = dl.TestDataloader(data_dir, batch_size)
    test_set = test.test_dataloader
    class_names = test.class_names
    st.success("creating dataloaders...")

    #classify images in test set and find the most common predicted aesthetics
    model = load_model()
    test_set = cl.ClassifyData(model, test_set, device, class_names)
    test_set.test_model()
    predictions = test_set.find_top_predictions()
    st.success("classifying images...")

    #recommend 10 similar products from the top predicted classes
    rec = rc.Recommender(predictions, class_names)
    rec = rec.get_recs()
    st.success("generating recommendations...")

    #display results
    for r in rec:
        st.write(r)
    # st.dataframe(rec)

    
def main():
    st.title("Fashion Recommendations")
    st.subheader("by Diarra Bell and Sakura Anning Yoshihara")
    st.markdown("This application takes in a group of user-submitted images of clothing and classfies each image as one of the following aesthetics: 70s, 80s, 90s, boho, cottagecore, goth, kawaii, or y2k. Using the most frequent aesthetic labels for that group of images, it recommends similar products from Forever21.com.")

    #create director to store images
    if not os.path.exists("uploads/images"):
        os.makedirs("uploads/images")

    label = "upload 10-15 images"
    uploaded_files = st.file_uploader(label, accept_multiple_files=True, type=None)

    for uploaded_file in uploaded_files:
        #display images
        st.image(load_image(uploaded_file), caption="filename: {0}, file size: {1}".format(uploaded_file.name, uploaded_file.size))

        #save upload
        with open(os.path.join("uploads/images", uploaded_file.name), "wb") as f:
            f.write((uploaded_file).getbuffer())
        st.success("saved file!")

    # execute main function:
    if st.button("Go!"):
        if len(uploaded_files) >= 10:
            test_images()
            #remove images from file after testing:
            for uploaded_file in uploaded_files:
                os.remove(os.path.join("uploads/images", uploaded_file.name))
        else:
            st.error("please upload more images. images uploaded: {}/10".format(len(uploaded_files)))

if __name__ == '__main__':
    main()
