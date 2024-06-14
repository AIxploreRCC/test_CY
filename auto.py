import os
import streamlit as st
import tempfile
import SimpleITK as sitk
import nibabel as nib
import matplotlib.pyplot as plt

# Titre de l'application
st.title("Automatic Segmentation App - Part 1")

# Téléchargement du fichier CT
uploaded_ct = st.file_uploader("Upload CT Image (Part 1)", type=["nii"])

if uploaded_ct:
    with tempfile.NamedTemporaryFile(delete=False, suffix='.nii') as tmp_ct:
        tmp_ct.write(uploaded_ct.getvalue())
        tmp_ct.seek(0)
        ct_image_path = tmp_ct.name

    # Convertir .nii en .nii.gz et renommer
    try:
        patient_folder = tempfile.mkdtemp()
        renamed_file = os.path.join(patient_folder, "900_0000.nii.gz")

        # Convertir .nii à .nii.gz et renommer
        sitk_image = sitk.ReadImage(ct_image_path)
        sitk.WriteImage(sitk_image, renamed_file)

        st.success(f"File converted and renamed to {renamed_file}")

        # Charger l'image convertie pour l'affichage
        converted_image = nib.load(renamed_file)
        converted_array = converted_image.get_fdata()

        # Afficher le slider pour sélectionner les tranches
        slice_number = st.slider('Select Slice', 0, converted_array.shape[2] - 1, converted_array.shape[2] // 2)

        # Afficher l'image convertie
        plt.figure(figsize=(6, 6))
        plt.imshow(converted_array[:, :, slice_number], cmap="gray")
        plt.title(f"Converted Image (slice {slice_number})")
        plt.axis("off")
        st.pyplot(plt)

        # Sauvegarder le chemin de l'image convertie pour la partie suivante
        st.session_state.converted_image_path = renamed_file
        st.session_state.patient_folder = patient_folder

    except Exception as e:
        st.error(f"Error during file conversion: {str(e)}")




import os
import requests
import streamlit as st
import tempfile
from nnunet.inference.predict import predict_from_folder
import nibabel as nib
import matplotlib.pyplot as plt

# URL de base du dossier modèle sur GitHub
base_model_folder_url = "https://github.com/AIxploreRCC/test_CY/raw/main/seg/"

# Fonction pour télécharger tous les fichiers depuis le dossier du modèle sur GitHub
def download_model_folder(base_url, model_folder):
    os.makedirs(model_folder, exist_ok=True)
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
        st.write(f"Downloading {file_url}")
        response = requests.get(file_url)
        file_path = os.path.join(model_folder, local_file)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        if response.status_code == 200:
            with open(file_path, 'wb') as f:
                f.write(response.content)
            st.write(f"Successfully downloaded {local_file}")
        else:
            st.error(f"Failed to download {local_file} from {file_url}")

# Définir les chemins nnU-Net
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
    os.makedirs(RESULTS_FOLDER, exist.ok=True)

    return temp_dir

# Titre de l'application
st.title("Automatic Segmentation App - Part 2")

# Vérifiez si l'image convertie est disponible dans l'état de session
if 'converted_image_path' in st.session_state and 'patient_folder' in st.session_state:
    converted_image_path = st.session_state.converted_image_path
    patient_folder = st.session_state.patient_folder

    model_folder = tempfile.mkdtemp()  # Utiliser un répertoire temporaire pour le modèle
    download_model_folder(base_model_folder_url, model_folder)
    st.success(f"Model downloaded to {model_folder}")

    if st.button("Start Automatic Segmentation"):
        if not os.path.exists(os.path.join(model_folder, "plans.pkl")):
            st.error("The plans.pkl file was not found in the model folder. Please check the model download.")
        else:
            try:
                temp_dir = set_nnunet_paths()
                st.write(f"Temporary directory for nnU-Net: {temp_dir}")

                input_folder = patient_folder
                output_folder = os.path.join(patient_folder, "output")
                os.makedirs(output_folder, exist.ok=True)

                # Log the paths being used
                st.write(f"Input folder: {input_folder}")
                st.write(f"Output folder: {output_folder}")

                # Log des fichiers dans le dossier d'entrée
                st.write(f"Files in input folder: {os.listdir(input_folder)}")

                # Faire la prédiction
                predict_from_folder(model_folder, input_folder, output_folder, folds=[0], save_npz=False, num_threads_preprocessing=1, num_threads_nifti_save=1, lowres_segmentations=None, part_id=0, num_parts=1, tta=False)

                segmentation_file_path = os.path.join(output_folder, "900_0000.nii.gz")
                st.write(f"Segmentation file path: {segmentation_file_path}")

                if os.path.exists(segmentation_file_path):
                    segmented_img = nib.load(segmentation_file_path)
                    st.success("Segmentation complete and saved.")

                    # Affichage de l'image segmentée
                    segmented_array = segmented_img.get_fdata()
                    slice_number = st.slider('Select Slice', 0, segmented_array.shape[2] - 1, segmented_array.shape[2] // 2)

                    st.write("Segmented Image:")
                    plt.figure(figsize=(6, 6))
                    plt.imshow(segmented_array[:, :, slice_number], cmap="gray")
                    plt.title(f"Segmented Image (slice {slice_number})")
                    plt.axis("off")
                    st.pyplot(plt)
                else:
                    st.error("Segmentation file not found.")

            except Exception as e:
                st.error(f"Error during automatic segmentation: {str(e)}")
else:
    st.warning("Please complete Part 1 first to upload and convert the CT image.")
