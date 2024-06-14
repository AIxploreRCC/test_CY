import streamlit as st
import os
from nnunet.inference.predict import predict_from_folder

st.title("Étape 2: Prédiction")

if 'new_file' in st.session_state and 'temp_dir' in st.session_state and 'model_folder' in st.session_state:
    new_file = st.session_state['new_file']
    temp_dir = st.session_state['temp_dir']
    model_folder = st.session_state['model_folder']

    input_folder = os.path.dirname(new_file)
    output_folder = os.path.join(input_folder, "output")
    os.makedirs(output_folder, exist_ok=True)

    st.write(f"Temporary directory for nnU-Net: {temp_dir}")
    st.write(f"Input file: {new_file}")
    st.write(f"Output folder: {output_folder}")

    if st.button("Start Prediction"):
        try:
            predict_from_folder(
                model_folder,
                input_folder,
                output_folder,
                folds=[0],
                save_npz=False,
                num_threads_preprocessing=1,
                num_threads_nifti_save=1,
                lowres_segmentations=None,
                part_id=0,
                num_parts=1,
                tta=False
            )
            st.write("La segmentation est terminée et les résultats sont enregistrés dans le dossier de sortie.")
        except Exception as e:
            st.write(f"Erreur lors de la prédiction : {str(e)}")
else:
    st.write("Veuillez d'abord passer par l'étape 1 pour préparer les fichiers.")
