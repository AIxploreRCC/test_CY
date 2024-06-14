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
