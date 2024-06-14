import os
import requests
import streamlit as st
import tempfile
from nnunet.inference.predict import predict_from_folder
import nibabel as nib
import matplotlib.pyplot as plt

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
        else:
            st.error(f"Failed to download {local_file} from {file_url}")

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

st.title("Segmentation Automatique avec nnU-Net")

uploaded_file = st.file_uploader("Téléchargez une image .nii", type="nii")
if uploaded_file is not None:
    # Enregistrer l'image téléchargée
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(uploaded_file.getvalue())
        original_file = temp_file.name

    # Renommer le fichier
    new_file = os.path.join(os.path.dirname(original_file), "900_0000.nii.gz")
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
        
        # Logs pour déboguer le contenu des dossiers
        st.write(f"Fichiers dans le dossier du modèle : {os.listdir(model_folder)}")
        st.write(f"Fichiers dans le dossier fold_0 : {os.listdir(os.path.join(model_folder, 'fold_0'))}")

        # Faire la prédiction
        input_folder = os.path.dirname(new_file)
        output_folder = os.path.join(input_folder, "output")
        os.makedirs(output_folder, exist_ok=True)

        st.write(f"Input folder: {input_folder}")
        st.write(f"Output folder: {output_folder}")
        st.write(f"Files in input folder: {os.listdir(input_folder)}")

        try:
            st.write("Starting prediction...")
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

            # Affichage du masque de segmentation
            segmentation_file = os.path.join(output_folder, "900_0000.nii.gz")
            if os.path.exists(segmentation_file):
                seg_img = nib.load(segmentation_file)
                seg_data = seg_img.get_fdata()
                slice_number = st.slider('Select Slice', 0, seg_data.shape[2] - 1, seg_data.shape[2] // 2)
                
                st.write("Segmented Image Slice:")
                plt.figure(figsize=(6, 6))
                plt.imshow(seg_data[:, :, slice_number], cmap="gray")
                plt.title(f"Segmented Image (slice {slice_number})")
                plt.axis("off")
                st.pyplot(plt)
            else:
                st.write("Le fichier de segmentation n'a pas été trouvé.")
        except Exception as e:
            st.write(f"Erreur lors de la prédiction : {str(e)}")
    else:
        st.write("Certains fichiers du modèle sont manquants. Veuillez vérifier le téléchargement des fichiers.")
