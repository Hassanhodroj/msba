#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 15 00:17:53 2022

@author: hassan
"""

import streamlit as st
import pandas as pd
import requests
from io import StringIO
import plotly.express as px


#To load the dataset and read from google drive:

orig_url='https://drive.google.com/file/d/1DdKtQwVOUrE4Jyf293AQ_2CnkUXbV__7/view?usp=sharing'

file_id = orig_url.split('/')[-2]
dwn_url='https://drive.google.com/uc?export=download&id=' + file_id
url = requests.get(dwn_url).text
csv_raw = StringIO(url)
data = pd.read_csv(csv_raw)

st.title("Welcome!")
st.title("MSBA 325 - Assignment 3")
st.title("Dr. Fouad Zablith")
st.title("22-9-2022")
st.title("By Hassan Hodroj")


st.markdown("<h1 style='text-align: left; color: red;font-size:7'>House Rent</h1>", unsafe_allow_html=True)


#data=pd.read_csv("/Users/hassan/Desktop/AUB/MSBA 325/House_Rent_Dataset.csv")


if st.checkbox("Show Center Information data"):
    st.subheader("Center Information data")
    st.write(data.head())

options = st.multiselect(
    'In what dimension would you like to view the scatterplot?' ,
    ['2D', '3D'])

if options==['2D']: 
    fig=px.scatter(data, x= "Rent", y= "Size", color= "Size", title="Relation between Size and Rent")
    st.plotly_chart(fig,use_container_width=False)

if options==['3D']:
    fig12=px.scatter_3d(data, x= "Rent", y= "Size", z="BHK", color= "Size", title="Relation between Size and Rent 3D")
    st.plotly_chart(fig12,use_container_width=False)
    
if options==['2D', '3D']:
    fig=px.scatter(data, x= "Rent", y= "Size", color= "Size", title="Relation between Size and Rent")
    st.plotly_chart(fig,use_container_width=False)
    fig12=px.scatter_3d(data, x= "Rent", y= "Size", z="BHK", color= "Size", title="Relation between Size and Rent 3D")
    st.plotly_chart(fig12,use_container_width=False)

dfg=data.groupby(['Area Type']).count()
fig1=px.bar(dfg, x=dfg.index,y=[2,2298,2446], title='Area Type by Count', 
           labels=dict(y="Count"))
st.plotly_chart(fig1,use_container_width=False)

fig2=px.box(data, y= "BHK", color= "Area Type", title="BHK Distribution by Area Type")
st.plotly_chart(fig2,use_container_width=False)

fig3=px.pie(data, values= "Floor", names= "Floor", title="Number of Floors")
st.plotly_chart(fig3,use_container_width=False)


option = st.selectbox(
    'Which furnishing status would you like to view?',
    ('All','Furnished', 'Semi-Furnished', 'Unfurnished'))

if option=='All' :
    fig4=px.histogram(data, x= "Rent", color= "Furnishing Status", title="Distribution of Rent by Furnishing Status")
    st.plotly_chart(fig4,use_container_width=False)
    
if option=='Furnished' :
    fur=data[data["Furnishing Status"]=='Furnished' ]
    sub=px.histogram(fur, x= "Rent", title="Distribution of Rent by Furnished Residences", color_discrete_sequence=['green'])
    st.plotly_chart(sub,use_container_width=False)

if option=='Semi-Furnished' :
    fur=data[data["Furnishing Status"]=='Semi-Furnished' ]
    sub=px.histogram(fur, x= "Rent", title="Distribution of Rent by Semi-Furnished Residences", color_discrete_sequence=['red'])
    st.plotly_chart(sub,use_container_width=False)
    
if option=='Unfurnished' :
    fur=data[data["Furnishing Status"]=='Unfurnished' ]
    sub=px.histogram(fur, x= "Rent", title="Distribution of Rent by Unfurnished Residences", color_discrete_sequence=['blue'])
    st.plotly_chart(sub,use_container_width=False)
    
    
st.sidebar.title("Introduction")
st.sidebar.title("Relation between Size and Rent")
st.sidebar.title("Relation between Size and Rent 3D")
st.sidebar.title("Area Type by Count")
st.sidebar.title("BHK Distribution by Area Type")
st.sidebar.title("Number of Floors")
st.sidebar.title("Distribution of Rent by Furnishing Status")
option = st.selectbox(
    'How would you like to receive the data?',
    ('Email', 'Mobile phone'))

@st.cache
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf-8')

csv = convert_df(data)

st.download_button(
    label="Download data as CSV",
    data=csv,
    file_name='large_df.csv',
    mime='text/csv',
)

st.success('You Reached The End! Thanks for reading.')

st.snow()

st.title("By Hassan Hodroj")