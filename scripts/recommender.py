"""
This file contains a class for the recommendation system.
"""

import os
import urllib
import zipfile
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error


class Recommender():
    def __init__ (self, predictions, class_names)-> None:
        self.predictions = list(predictions)
        self.class_names = list(class_names)
        self.catalog_df = pd.read_csv("data/product_catalog.csv")

    def create_df(self):
        #convert list to string
        self.predictions = ' '.join(self.predictions)
        #insert string into dataframe
        self.test_df = pd.DataFrame(list(zip(["test"], [""], self.predictions)),
            columns =['img_name', 'links', "aesthetics"])
        
    def get_recs(self):
        ret = None
        vec = CountVectorizer()
        genres_vec = vec.fit_transform(self.catalog_df['aesthetics'])


        #calculate cosine similarity of items
        #build similarity matrix of movies based on similarity of genres
        csmatrix = cosine_similarity(genres_vec)
        csmatrix = pd.DataFrame(csmatrix, columns=self.catalog_df.img_name, index=self.catalog_df.img_name)

        #create dataframe for test data
        self.create_df()

        #append test row to training set
        self.full_df= pd.concat([self.catalog_df, self.test_df]).reset_index(drop=True)

        #rebuild cosine matrix
        genres_vec = vec.fit_transform(self.full_df['aesthetics'])
    
        # Build similarity marrix of movies based on similarity of genres
        csmatrix = cosine_similarity(genres_vec)
        csmatrix = pd.DataFrame(csmatrix,columns=self.full_df.img_name, index=self.full_df.img_name)
        img = self.test_df['img_name'][0]

        sims = csmatrix.loc[img,:]
        mostsimilar = sims.sort_values(ascending=False).index.values

        recs = np.delete(mostsimilar, np.where(mostsimilar == img))[:10]

        ret = self.catalog_df.loc[self.catalog_df['img_name'].isin(recs)]

        ret = list(ret['links'])

        return ret