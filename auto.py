import os
import requests
import streamlit as st
import tempfile
import SimpleITK as sitk
from nnunet.inference.predict import predict_from_folder

# URL du dossier modèle sur GitHub
model_folder_url = "https://github.com/AIxploreRCC/test_CY/raw/main/seg/"

# Fonction pour télécharger tous les fichiers depuis le dossier du modèle sur GitHub
def download_model_folder(url, model_folder):
    os.makedirs(model_folder, exist_ok=True)
    filenames = [
        "plans.pkl",
        "postprocessing.json",
        "fold_0/model_final_checkpoint.model",
        "fold_1/model_final_checkpoint.model",
        "fold_2/model_final_checkpoint.model",
        "fold_3/model_final_checkpoint.model",
        "fold_4/model_final_checkpoint.model",
    ]
    for filename in filenames:
        file_url = url + filename
        response = requests.get(file_url)
        file_path = os.path.join(model_folder, filename)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        if response.status_code == 200:
            with open(file_path, 'wb') as f:
                f.write(response.content)
        else:
            st.error(f"Failed to download {filename} from {file_url}")

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
    os.makedirs(RESULTS_FOLDER, exist_ok=True)

    return temp_dir

# Titre de l'application
st.title("Automatic Segmentation App")

# Téléchargement du fichier CT
uploaded_ct = st.file_uploader("Upload CT Image for Automatic Segmentation", type=["nii"])

if uploaded_ct:
    with tempfile.NamedTemporaryFile(delete=False, suffix='.nii') as tmp_ct:
        tmp_ct.write(uploaded_ct.getvalue())
        tmp_ct.seek(0)
        ct_image_path = tmp_ct.name

    model_folder = "seg"
    download_model_folder(model_folder_url, model_folder)
    st.success(f"Model downloaded to {model_folder}")

    if st.button("Start Automatic Segmentation"):
        if not os.path.exists(os.path.join(model_folder, "plans.pkl")):
            st.error("The plans.pkl file was not found in the model folder. Please check the model download.")
        else:
            try:
                temp_dir = set_nnunet_paths()
                
                patient_folder = os.path.dirname(ct_image_path)
                renamed_file = os.path.join(patient_folder, "900_0000.nii.gz")

                # Convert .nii to .nii.gz
                sitk_image = sitk.ReadImage(ct_image_path)
                sitk.WriteImage(sitk_image, renamed_file)

                input_folder = patient_folder
                output_folder = os.path.join(patient_folder, "output")
                os.makedirs(output_folder, exist_ok=True)

                # Ensure input file is named correctly for nnU-Net
                os.rename(renamed_file, os.path.join(input_folder, "900_0000.nii.gz"))

                predict_from_folder(model_folder, input_folder, output_folder, folds=[0], save_npz=False, num_threads_preprocessing=1, num_threads_nifti_save=1, lowres_segmentations=None, part_id=0, num_parts=1, tta=False)

                segmentation_file_path = os.path.join(output_folder, "900_0000.nii.gz")

                segmented_img = sitk.ReadImage(segmentation_file_path)
                st.success("Segmentation complete and saved.")

                # Affichage de l'image segmentée
                segmented_array = sitk.GetArrayFromImage(segmented_img)
                st.write("Segmented Image:")
                st.image(segmented_array[segmented_array.shape[0]//2, :, :], caption='Segmented Image (middle slice)', use_column_width=True)

            except Exception as e:
                st.error(f"Error during automatic segmentation: {str(e)}")
