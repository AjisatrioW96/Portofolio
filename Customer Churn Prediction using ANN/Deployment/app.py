import streamlit as st
import eda_p2m1
import prediction_p2m1

navigation = st.sidebar.selectbox('Pilih halaman :', ('EDA', 'Make a Prediction'))

if navigation == 'EDA' :
    eda_p2m1.run()
else :
    prediction_p2m1.run()