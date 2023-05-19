# type: ignore
import streamlit as st

from utils import get_answer_csv

st.header("Jungle Gym")
uploaded_file = st.file_uploader("Upload a csv file", type=["csv"])

if uploaded_file is not None:
    query = st.text_area("Ask any question related to the data :)")
    button = st.button("Submit")
    if button:
        get_answer_csv(uploaded_file, query)
        
