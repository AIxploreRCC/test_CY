import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
import joblib
from home import home

# URL des logos hébergés sur GitHub (lien brut)
logo1_url = "https://raw.githubusercontent.com/AIxploreRCC/Design/main/logo%203.png"
logo2_url = "https://raw.githubusercontent.com/AIxploreRCC/Design/main/images.png"

# Charger le CSS personnalisé
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

local_css("styles.css")

# Titre de l'application avec logos
st.markdown(f"""
    <div style="display: flex; align-items: center; justify-content: center;">
        <img src="{logo1_url}" alt="Logo 1" style="width: 150px; height: 100px; margin-right: 20px;">
        <img src="{logo2_url}" alt="Logo 2" style="width: 60px; height: 60px; margin-right: 20px;">
        <h1 style="margin: 0; text-align: center;">RenalCheck — RCC Clinical Radiomics Algorithm App</h1>
    </div>
    <hr style="border: 1px solid #ccc;">
""", unsafe_allow_html=True)

# Barre de navigation
menu = ["Home", "About", "Radiomics Score Generator", "Contact"]
choice = st.selectbox("Navigation", menu, key="main_navigation")

def about():
    st.header("About")
    st.write("""
    This application predicts survival using radiomics and clinical data.
    Adjust the input variables to see how the survival curve changes.
    """)

def radiomics_score_generator():
    st.header("Radiomics Score Generator")
    st.write("This feature is under development.")

def contact():
    st.header("Contact")
    st.write("""
    For any inquiries, please contact us at: support@radiomicsapp.com
    """)

if choice == "Home":
    home()
elif choice == "About":
    about()
elif choice == "Radiomics Score Generator":
    radiomics_score_generator()
elif choice == "Contact":
    contact()
