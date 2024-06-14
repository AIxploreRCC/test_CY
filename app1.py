import streamlit as st
import os
import requests
import tempfile

# Fonction pour télécharger les fichiers du modèle
def download_model_files(base_url, model_folder):
    files = {
        "plans.pkl": "plans.pkl",
        "postprocessing.json": "postprocessing.json",
        "fold_0/model_final_checkpoint.model": "fold_0/model_final_checkpoint.model",
        "fold_0/model_final_checkpoint.model.pkl": "fold_0/model_final_checkpoint.model.pkl",
        "fold_0/debug.json": "fold_0/debug.json",
        "fold_1/model_final_checkpoint.model": "fold_1/model_final_checkpoint.model",
        "fold_1/model_final_checkpoint.model.pkl": "fold_1/model_final_checkpoint.model.pkl",
        "fold_1/debug.json": "fold_1/debug.json",
        "fold_2/model_final_checkpoint.model": "fold_2/model_final_checkpoint.model",
        "fold_2/model_final_checkpoint.model.pkl": "fold_2/model_final_checkpoint.model.pkl",
        "fold_2/debug.json": "fold_2/debug.json",
        "fold_3/model_final_checkpoint.model": "fold_3/model_final_checkpoint.model",
        "fold_3/model_final_checkpoint.model.pkl": "fold_3/model_final_checkpoint.model.pkl",
        "fold_3/debug.json": "fold_3/debug.json",
        "fold_4/model_final_checkpoint.model": "fold_4/model_final_checkpoint.model",
        "fold_4/model_final_checkpoint.model.pkl": "fold_4/model_final_checkpoint.model.pkl",
        "fold_4/debug.json": "fold_4/debug.json"
    }

    for local_file, remote_file in files.items():
        file_url = base_url + remote_file
        response = requests.get(file_url)
        file_path = os.path.join(model_folder, local_file)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        if response.status_code == 200:
            with open(file_path, 'wb') as f:
                f.write(response.content)

# Fonction pour configurer les chemins nnU-Net
def set_nnunet_paths():
    temp_dir = tempfile.mkdtemp()
    nnUNet_raw_data_base = os.path.join(temp_dir, 'nnUNet_raw_data_base')
    nnUNet_preprocessed = os.path.join(temp_dir, 'nnUNet_preprocessed')
    RESULTS_FOLDER = os.path.join(temp_dir, 'nnUNet_trained_models')

    os.environ['nnUNet_raw_data_base'] = nnUNet_raw_data_base
    os.environ['nnUNet_preprocessed'] = nnUNet_preprocessed
    os.environ['RESULTS_FOLDER'] = RESULTS_FOLDER

    os.makedirs(nnUNet_raw_data_base, exist_ok=True)
    os.makedirs(nnUNet_preprocessed, exist_ok=True)
    os.makedirs(RESULTS_FOLDER, exist_ok=True)

    return temp_dir

st.title("Étape 1: Préparation des fichiers")

uploaded_file = st.file_uploader("Téléchargez une image .nii", type="nii")
if uploaded_file is not None:
    # Enregistrer l'image téléchargée
    with tempfile.NamedTemporaryFile(delete=False, suffix='.nii.gz') as temp_file:
        temp_file.write(uploaded_file.getvalue())
        original_file = temp_file.name

    # Renommer le fichier
    new_file = '/tmp/900_0000.nii.gz'
    os.rename(original_file, new_file)
    st.write(f"Fichier renommé en : {new_file}")

    # Configurer les chemins nnU-Net
    temp_dir = set_nnunet_paths()
    st.write(f"Temporary directory for nnU-Net: {temp_dir}")

    # Dossier du modèle téléchargé
    model_folder = os.path.join(temp_dir, "nnUNet_trained_models", "seg")

    # Télécharger les fichiers du modèle
    base_model_folder_url = "https://github.com/AIxploreRCC/test_CY/raw/main/seg/"
    download_model_files(base_model_folder_url, model_folder)

    # Vérifier la présence des fichiers nécessaires
    required_files = [
        "plans.pkl",
        "postprocessing.json",
        "fold_0/model_final_checkpoint.model",
        "fold_0/debug.json",
        "fold_1/model_final_checkpoint.model",
        "fold_1/debug.json",
        "fold_2/model_final_checkpoint.model",
        "fold_2/debug.json",
        "fold_3/model_final_checkpoint.model",
        "fold_3/debug.json",
        "fold_4/model_final_checkpoint.model",
        "fold_4/debug.json"
    ]

    all_files_present = all(os.path.exists(os.path.join(model_folder, file)) for file in required_files)

    if all_files_present:
        st.write("Tous les fichiers du modèle ont été téléchargés avec succès.")
        # Sauvegarder les chemins dans les variables de session
        st.session_state['new_file'] = new_file
        st.session_state['temp_dir'] = temp_dir
        st.session_state['model_folder'] = model_folder
    else:
        st.write("Certains fichiers du modèle sont manquants. Veuillez vérifier le téléchargement des fichiers.")

    st.write("Passez à l'étape 2 pour la prédiction.")
