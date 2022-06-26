#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 15:52:17 2022

@author: loganwoolfson
"""

import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from PIL import Image

# insert company logo
image = Image.open('pace_logo.jpeg')
st.image(image, width = 200)


st.write("""
# Prediction App

This app predicts whether a student will **Pass or Fail**

Adjust the parameters on the left, to see how they affect the probability of passing
""")

st.sidebar.header('User Input Parameters')

def user_input_features():
    feature1 = st.sidebar.slider('Sixltr', 5.0, 30.0, 10.0) # slider(text, min, max, default value)
    feature2 = st.sidebar.slider('posemo', 1.0, 15.0, 1.0)
    feature3 = st.sidebar.slider('negemo', 0.0, 4.0, 1.0)
    feature4 = st.sidebar.slider('achieve', 0.0, 6.0, 1.0)
    feature5 = st.sidebar.slider('reward', 0.0, 5.0, 1.0)
    feature6 = st.sidebar.slider('risk', 0.0, 2.0, 1.5)
    feature7 = st.sidebar.slider('focuspast', 0.0, 6.0, 2.0)
    feature8 = st.sidebar.slider('focuspresent', 5.0, 20.0, 10.0)
    feature9 = st.sidebar.slider('focusfuture', 0.0, 5.0, 2.0)
    data = {'Sixltr': feature1,
            'posemo': feature2,
            'negemo' : feature3,
            'achieve': feature4,
            'reward': feature5,
            'risk': feature6,
            'focuspast': feature7,
            'focuspresent': feature8,
            'focusfuture': feature9}
    features = pd.DataFrame(data, index=[0]) # best practice to pass data into dict then into dataframe
    return features

df = user_input_features() #assign df to features variable, not sure why made a function for this. maybe easier traceability

st.subheader('User Input parameters')
st.write(df) # displays the dataframe on webpage

#loading dataset into predictors and output
x = pd.read_excel('https://raw.githubusercontent.com/loganw621/pace_analytics/main/predictor_avron_data.xlsx')
#x = pd.read_excel('predictor_avron_data.xlsx')
first_9_columns  = x.iloc[: , :9]
first_9_columns = first_9_columns.to_numpy()

y = pd.read_excel('https://raw.githubusercontent.com/loganw621/pace_analytics/main/outcome_data.xlsx')
y = y.to_numpy()
y = y.ravel()

#fitting the model using all data
clf = RandomForestClassifier()
clf.fit(first_9_columns, y)

prediction = clf.predict(df) # predict the index of outcome (0,1)
prediction_proba = clf.predict_proba(df)

st.subheader('Prediction')
outcome_names = np.array(['Fail', 'Pass'])
st.write(outcome_names[prediction]) # this displays the name of plant for the index
#st.write(prediction) # this displays the index of prediction

st.subheader('Prediction Probability')
df_prediction_proba = pd.DataFrame(prediction_proba, columns = ['Fail','Pass'])
st.write(df_prediction_proba)






