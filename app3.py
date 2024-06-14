import streamlit as st

st.title("Application de Segmentation Automatique")

menu = ["Étape 1: Préparation des fichiers", "Étape 2: Prédiction"]
choice = st.selectbox("Choisissez une étape", menu)

if choice == "Étape 1: Préparation des fichiers":
    st.write("Vous allez être redirigé vers l'étape 1.")
    st.experimental_rerun()
elif choice == "Étape 2: Prédiction":
    st.write("Vous allez être redirigé vers l'étape 2.")
    st.experimental_rerun()

# Inclure les sous-applications
if choice == "Étape 1: Préparation des fichiers":
    import app1
    app1.main()
elif choice == "Étape 2: Prédiction":
    import app2
    app2.main()
