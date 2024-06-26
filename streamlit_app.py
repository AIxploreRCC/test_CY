import streamlit as st
import os
from nnunet.inference.predict import predict_from_folder
import shutil

# Fonction pour enregistrer le fichier téléchargé
def save_uploaded_file(uploaded_file):
    try:
        # Créer le dossier uploads s'il n'existe pas
        uploads_dir = os.path.join(os.getcwd(), "uploads")
        st.write(f"Chemin du répertoire uploads : {uploads_dir}")  # Journalisation
        if not os.path.exists(uploads_dir):
            os.makedirs(uploads_dir)
            st.write(f"Répertoire 'uploads' créé : {uploads_dir}")

        file_path = os.path.join(uploads_dir, uploaded_file.name)
        st.write(f"Chemin complet du fichier à enregistrer : {file_path}")  # Journalisation
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        return file_path
    except Exception as e:
        st.error(f"Erreur lors de l'enregistrement du fichier : {e}")
        return None

# Fonction pour exécuter la prédiction
def run_prediction(input_file_path, model_folder, output_folder):
    try:
        # Créer le dossier de sortie s'il n'existe pas
        output_dir = os.path.join(os.getcwd(), output_folder)
        st.write(f"Chemin du répertoire output : {output_dir}")  # Journalisation
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            st.write(f"Répertoire 'output' créé : {output_dir}")

        # Déplacer le fichier téléchargé vers le dossier d'entrée attendu par le modèle
        input_folder = os.path.dirname(input_file_path)
        new_file_path = os.path.join(input_folder, "300_0000.nii.gz")
        shutil.move(input_file_path, new_file_path)

        st.write(f"Chemin du modèle : {model_folder}")
        st.write(f"Chemin du dossier d'entrée : {input_folder}")
        st.write(f"Chemin du fichier d'entrée renommé : {new_file_path}")
        st.write(f"Chemin du dossier de sortie : {output_dir}")

        # Faire la prédiction
        predict_from_folder(model_folder, input_folder, output_dir, folds=[0], save_npz=False, num_threads_preprocessing=1, num_threads_nifti_save=1)
        return new_file_path
    except Exception as e:
        st.error(f"Erreur lors de l'exécution de la prédiction : {e}")
        return None

# Titre de l'application
st.title('Application de Segmentation NNUNet')

# Téléchargement du fichier
uploaded_file = st.file_uploader("Téléchargez votre fichier CT (.nii)", type=["nii"])

if uploaded_file is not None:
    # Enregistrer le fichier téléchargé
    input_file_path = save_uploaded_file(uploaded_file)
    if input_file_path:
        st.success(f"Fichier téléchargé et enregistré avec succès : {uploaded_file.name}")
        st.write(f"Chemin du fichier enregistré : {input_file_path}")

        # Dossier du modèle (local ou URL GitHub)
        model_folder = "/path/to/your/local/model/folder"  # Mettez à jour ce chemin avec le chemin de votre modèle local téléchargé
        output_folder = "output"

        # Bouton pour exécuter la prédiction
        if st.button('Exécuter la prédiction'):
            result_path = run_prediction(input_file_path, model_folder, output_folder)
            if result_path:
                st.success(f"La segmentation est terminée et les résultats sont enregistrés dans {output_folder}")

                # Afficher un lien pour télécharger le fichier de segmentation
                with open(os.path.join(output_folder, "300_0000.nii.gz"), "rb") as file:
                    st.download_button(
                        label="Télécharger la segmentation",
                        data=file,
                        file_name="segmentation_result.nii.gz",
                        mime="application/octet-stream"
                    )
