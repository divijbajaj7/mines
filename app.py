import matplotlib.pyplot as plt
import pickle
import pandas
import numpy as np
import sklearn
import imblearn
import streamlit as st
from PIL import Image

st.title("Credit Card Fraud Detection Model")

st.image("images.jpg")



input_df = st.text_input("Please provide all the required feature details: ")
input_df_split = input_df.split(',')

submit = st.button("Submit")

if submit:
    model = pickle.load(open('MinevsRockLogisticRegression.pkl', 'rb'))
    features = np.asarray(input_df_split,dtype = np.float64)
    prediction = model.predict(features.reshape(1,-1))

if (prediction[0] == "R"):
    st.write("The object is a rock")
else:
    st.write("The object is a mine")

