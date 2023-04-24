# Fashion Recommendations
# Diarra Bell, Sakura Anning Yoshihara

### Description 
#### Objective:
Given a set of input images, classify the images into predetermined aesthetics and recommend products that fit that aesthetic.
  
We trained our model on the following eight categories or "aesthetics":
- 70s
- 80s
- 90s
- y2k
- goth
- cottagecore
- kawaii


### Links to Datasets
Training and Validation Datasets
- https://drive.google.com/drive/folders/1dXOS4b3BTFcPT4nB09OAGNIkh_zZKOTN?usp=share_link


## Project Structure

### main.py
<insert description>

### models
- model.pt - ResNet 50 trained on 8 fashion aesthetics 

### data
- product_catalog.csv - Catalog of 249 women's fashion products on Forever21.com as of 4/24/2023

### scripts
- classifier.py - 
- dataloader.py - 
- model.py - 
- recommender.py - 

## Link to App Demo
```

```
## Instructions to run Streamlit app. 
### Upload images to Streamlit to generate recommendations to Forever21 products 
1. Install the requirements needed to use Streamlit
```
pip install -r requirements.txt
```
2. Start the Streamlit app
```
make run
```
3. Upload at least 10 photos depicting the aesthetic of the desired products
