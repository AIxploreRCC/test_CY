import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from joblib import load
from lifelines import KaplanMeierFitter
import os


# Configurer la page pour utiliser toute la largeur
st.set_page_config(layout="wide")

# Charger le modèle avec mise en cache
@st.cache(allow_output_mutation=True)
def load_model():
    model_path = 'coxph_model.joblib'
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    return load(model_path)

# Seuil optimal pour séparer les groupes de risque
optimal_threshold = 3.038141178443309

# Charger les données pour tracer la courbe de Kaplan-Meier
@st.cache
def load_km_data():
    file_path = "km_curve_data.csv"
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    df = pd.read_csv(file_path)
    return df

# Fonction pour tracer les courbes de Kaplan-Meier
def plot_kaplan_meier(data):
    kmf = KaplanMeierFitter()
    fig = go.Figure()
    for group in data['group'].unique():
        mask = data['group'] == group
        kmf.fit(data[mask]['TimeR'], event_observed=data[mask]['Rec'], label=group)
        survival_function = kmf.survival_function_
        fig.add_trace(go.Scatter(
            x=survival_function.index, 
            y=survival_function.iloc[:, 0],
            mode='lines',
            name=group
        ))
    fig.update_layout(
                      xaxis_title='Time (months)',
                      yaxis_title='Survival Probability',
                      width=500,  # Réduire la largeur de la figure
                      height=300)  # Réduire la hauteur de la figure
    
    return fig



def homee():
    st.write("""
    RCC Clinical Radiomics Algorithm App is an advanced AI algorithm designed to predict post-operative oncological outcomes 
    in patients with clear renal cell carcinoma. This tool is designed for patients at intermediate or high risk of recurrence, specifically 
    those meeting the eligibility criteria of the KEYNOTE 564 trial, including stages pT2 with G4 or sarcomatoid differentiation, pT3, pT4, or pN1.
    """)

    # Dictionnaires de mapping pour les sélecteurs
    
    N_mapping = {
        "N0": 0,
        "N1": 1,
        "Nx": 2
    }

    Thrombus_mapping = {
        "None": 0,
        "Segmental vein/arteriole invasion": 1,
        "Renal Vein invasion": 2,
        "Caval invasion": 3
    }

    col1, col2 = st.columns(2)

    with col1:
        hb = st.selectbox("Hemoglobin < lower limit of normal", options=[0, 1])
        N_label = st.selectbox("Pathological Lymph Node Involvement", options=list(N_mapping.keys()))
        rad = st.slider("Radiomics Signature", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
        Thrombus_label = st.selectbox("Vascular Invasion", options=list(Thrombus_mapping.keys()))

        # Bouton Predict Survival
        predict_button = st.button('Predict Survival')

        # Récupérer les valeurs numériques basées sur les labels sélectionnés
        N = N_mapping[N_label]
        Thrombus = Thrombus_mapping[Thrombus_label]

        input_df = pd.DataFrame({
            'HbN': [hb],
            'rad': [rad],
            'N': [N],
            'Thrombus': [Thrombus]
        })

        input_df['N'] = input_df['N'].astype('category')
        input_df['Thrombus'] = input_df['Thrombus'].astype('category')


    with col2:

        if predict_button:
            with st.spinner('Calculating... Please wait.'):
                try:
                    model_cox = load_model()
                    survival_function = model_cox.predict_survival_function(input_df)
                    time_points = survival_function[0].x
                    time_points = time_points[time_points <= 60]
                    survival_probabilities = [fn(time_points) for fn in survival_function]
                    survival_df = pd.DataFrame(survival_probabilities).transpose()
                    survival_df.columns = ['Survival Probability']
                    data = load_km_data()
                    fig = plot_kaplan_meier(data)
                    
                    fig.add_trace(go.Scatter(x=time_points, y=survival_df['Survival Probability'], mode='lines', name='Patient-specific prediction', line=dict(color='blue', dash='dot')))
                    fig.update_layout(xaxis_title='Time (months)', yaxis_title='Survival Probability')
                    st.plotly_chart(fig)

                    risk_score = model_cox.predict(input_df)[0]
                   
                    # Créer une ligne pour centrer le texte et l'icône entre les colonnes
                    st.markdown("<hr>", unsafe_allow_html=True)

                    # Déterminer le groupe de risque et ajouter l'icône de feu appropriée
                    risk_group = "High risk" if risk_score >= optimal_threshold else "Low risk"
                    st.write(f"Risk group: {risk_group}")


                except Exception as e:
                    st.error(f"Prediction failed: {e}")
