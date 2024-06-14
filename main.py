import os
import requests
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from joblib import load
from radiomics import featureextractor
from homee import homee
import SimpleITK as sitk
import tempfile
from scipy.integrate import simps
from sksurv.ensemble import RandomSurvivalForest
from sklearn.preprocessing import MinMaxScaler
import nibabel as nib
from nnunet.inference.predict import predict_from_folder

# URL des logos hébergés sur GitHub (lien brut)
logo1_url = "https://raw.githubusercontent.com/AIxploreRCC/Design/main/logo%203.png"
logo2_url = "https://raw.githubusercontent.com/AIxploreRCC/Design/main/images.png"

# URL du modèle sur GitHub
model_files = {
    "plans.pkl": "https://github.com/yourusername/yourrepository/raw/main/seg/plans.pkl",
    # Ajoutez d'autres fichiers nécessaires pour le modèle ici
}

# Charger le CSS personnalisé
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

local_css("styles2.css")

# Titre de l'application avec logos
st.markdown(f"""
    <div style="display: flex; align-items: center; justify-content: center;">
        <div style="display: flex; flex-direction: column; align-items: center; margin-right: 20px;">
            <img src="{logo1_url}" alt="Logo 1" style="width: 100px; height: 80px;">
            <img src="{logo2_url}" alt="Logo 2" style="width: 60px; height: 60px; margin-top: 10px;">
        </div>
        <h1 style="margin: 0;">RCC Clinical Radiomics Algorithm App</h1>
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

def contact():
    st.header("Contact")
    st.write("""
    For any inquiries, please contact us at: support@radiomicsapp.com
    """)

# Fonction load_model intégrée
@st.cache(allow_output_mutation=True)
def load_model():
    try:
        return load('random_survival_forest_model.joblib')
    except Exception as e:
        st.error(f"Failed to load the model: {str(e)}")
        raise

rsf_model = load_model()
scaler = load('scaler.joblib')

def setup_extractor():
    extractor = featureextractor.RadiomicsFeatureExtractor()
    extractor.settings['normalize'] = True
    extractor.settings['normalizeScale'] = 100
    extractor.settings['resampledPixelSpacing'] = [1, 1, 1]
    extractor.settings['interpolator'] = sitk.sitkBSpline
    extractor.settings['binWidth'] = 25
    extractor.enableImageTypeByName('Wavelet')
    extractor.enableImageTypeByName('LoG', customArgs={'sigma': [1.0, 2.0, 3.0, 4.0, 5.0]})
    return extractor

def display_images(ct_image, seg_image, slice_number):
    ct_array = sitk.GetArrayFromImage(ct_image)
    seg_array = sitk.GetArrayFromImage(seg_image)

    # Abdominal window setting
    window_level = 30
    window_width = 300
    min_intensity = window_level - window_width // 2
    max_intensity = window_level + window_width // 2

    # Normalize CT image for display
    ct_array = np.clip(ct_array, min_intensity, max_intensity)
    ct_array = (ct_array - min_intensity) / (max_intensity - min_intensity)

    # Resize the images
    ct_resized = ct_array[slice_number, :, :]
    seg_resized = seg_array[slice_number, :, :]

    plt.figure(figsize=(6, 6))  # Adjust the size as needed
    plt.imshow(ct_resized, cmap='gray')
    plt.imshow(seg_resized, cmap='hot', alpha=0.5)
    plt.axis('off')
    st.pyplot(plt)

def download_model(model_files, model_folder):
    os.makedirs(model_folder, exist_ok=True)
    for filename, url in model_files.items():
        response = requests.get(url)
        with open(os.path.join(model_folder, filename), 'wb') as f:
            f.write(response.content)

if choice == "Home":
    homee()
elif choice == "About":
    about()
elif choice == "Radiomics Score Generator":
    seg_choice = st.radio("Choose Segmentation Method", ("Manual Segmentation", "Automatic Segmentation"))

    if seg_choice == "Manual Segmentation":
        uploaded_ct = st.file_uploader("Upload CT Image", type=["nii", "nii.gz"])
        uploaded_seg = st.file_uploader("Upload Segmentation Mask", type=["nii", "nii.gz"])

        if uploaded_ct and uploaded_seg:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.nii.gz') as tmp_ct:
                tmp_ct.write(uploaded_ct.getvalue())
                tmp_ct.seek(0)
                ct_image = sitk.ReadImage(tmp_ct.name)

            with tempfile.NamedTemporaryFile(delete=False, suffix='.nii.gz') as tmp_seg:
                tmp_seg.write(uploaded_seg.getvalue())
                tmp_seg.seek(0)
                seg_image = sitk.ReadImage(tmp_seg.name)

            slice_number = st.slider('Select Slice', 0, ct_image.GetSize()[2] - 1, ct_image.GetSize()[2] // 2)
            display_images(ct_image, seg_image, slice_number)

            if st.button('Start Feature Extraction'):
                try:
                    extractor = setup_extractor()
                    feature_extraction_result = extractor.execute(ct_image, seg_image)
                    features_df = pd.DataFrame([feature_extraction_result])
                    
                    features_of_interest = [
                        'original_firstorder_10Percentile', 'original_firstorder_Mean', 'original_firstorder_Uniformity', 
                        'original_glcm_ClusterTendency', 'original_glcm_Idm', 'original_glcm_Imc2', 'original_glcm_JointEnergy',
                        'original_gldm_LargeDependenceEmphasis', 'original_gldm_SmallDependenceLowGrayLevelEmphasis', 
                        'original_glrlm_HighGrayLevelRunEmphasis', 'original_glrlm_LongRunEmphasis', 'original_glrlm_LongRunLowGrayLevelEmphasis', 
                        'original_glrlm_RunVariance', 'original_glrlm_ShortRunEmphasis', 'original_glrlm_ShortRunHighGrayLevelEmphasis', 
                        'original_glrlm_ShortRunLowGrayLevelEmphasis', 'original_glszm_GrayLevelVariance', 'original_glszm_HighGrayLevelZoneEmphasis', 
                        'original_glszm_LargeAreaLowGrayLevelEmphasis', 'original_glszm_SmallAreaHighGrayLevelEmphasis', 'original_glszm_ZoneVariance', 
                        'wavelet-LLH_firstorder_Entropy', 'wavelet-LLH_firstorder_InterquartileRange', 'wavelet-LLH_firstorder_Kurtosis', 
                        'wavelet-LLH_glcm_Contrast', 'wavelet-LLH_glcm_DifferenceVariance', 'wavelet-LLH_glcm_Idm', 'wavelet-LLH_glcm_Idn', 
                        'wavelet-LLH_glcm_Imc1', 'wavelet-LLH_gldm_HighGrayLevelEmphasis', 'wavelet-LLH_gldm_LargeDependenceEmphasis', 
                        'wavelet-LLH_gldm_SmallDependenceHighGrayLevelEmphasis', 'wavelet-LLH_glrlm_GrayLevelNonUniformityNormalized', 
                        'wavelet-LLH_glrlm_HighGrayLevelRunEmphasis', 'wavelet-LLH_glrlm_LongRunLowGrayLevelEmphasis', 
                        'wavelet-LLH_glrlm_RunLengthNonUniformity', 'wavelet-LLH_glrlm_RunPercentage', 'wavelet-LLH_ngtdm_Busyness', 
                        'wavelet-LHL_firstorder_RobustMeanAbsoluteDeviation', 'wavelet-LHL_glcm_ClusterTendency', 'wavelet-LHL_glcm_Correlation', 
                        'wavelet-LHL_glcm_DifferenceEntropy', 'wavelet-LHL_glcm_Idmn', 'wavelet-LHL_glcm_JointEntropy', 'wavelet-LHL_glcm_SumAverage', 
                        'wavelet-LHL_gldm_DependenceNonUniformityNormalized', 'wavelet-LHL_glrlm_LongRunEmphasis', 'wavelet-LHL_glszm_SizeZoneNonUniformityNormalized', 
                        'wavelet-LHL_ngtdm_Complexity', 'wavelet-LHH_firstorder_RootMeanSquared'
                    ]

                    selected_features_df = features_df[features_of_interest]

                    st.session_state['selected_features_df'] = selected_features_df

                    st.write("Selected Features:")
                    st.dataframe(selected_features_df)
                except Exception as e:
                    st.error(f"Error during feature extraction: {str(e)}")

        if 'selected_features_df' in st.session_state and st.button('Calculate RAD-Score for Uploaded Patient'):
            try:
                time_points = np.linspace(0, 60, 61)
                cumulative_hazards = rsf_model.predict_cumulative_hazard_function(st.session_state['selected_features_df'])
                rad_scores = np.array([simps([ch(tp) for tp in time_points], time_points) for ch in cumulative_hazards])
                normalized_rad_scores = scaler.transform(rad_scores.reshape(-1, 1)).flatten()
                st.write(f"Normalized RAD-Score for the uploaded patient: {normalized_rad_scores[0]:.5f}")
            except Exception as e:
                st.error(f"Error during RAD-Score calculation: {str(e)}")

    elif seg_choice == "Automatic Segmentation":
        uploaded_ct = st.file_uploader("Upload CT Image for Automatic Segmentation", type=["nii"])

        if uploaded_ct:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.nii') as tmp_ct:
                tmp_ct.write(uploaded_ct.getvalue())
                tmp_ct.seek(0)
                ct_image_path = tmp_ct.name

            model_folder = "seg"
            st.button("Download Model")
            download_model(model_files, model_folder)
            st.success(f"Model downloaded to {model_folder}")

            if st.button("Start Automatic Segmentation"):
                if not os.path.exists(os.path.join(model_folder, "plans.pkl")):
                    st.error("The plans.pkl file was not found in the model folder. Please check the model download.")
                else:
                    try:
                        patient_folder = os.path.dirname(ct_image_path)
                        renamed_file = os.path.join(patient_folder, "900_0000.nii.gz")

                        # Convert .nii to .nii.gz
                        sitk_image = sitk.ReadImage(ct_image_path)
                        sitk.WriteImage(sitk_image, renamed_file)

                        input_folder = patient_folder
                        output_folder = os.path.join(patient_folder, "output")
                        os.makedirs(output_folder, exist_ok=True)

                        predict_from_folder(model_folder, input_folder, output_folder, folds=[0], save_npz=False, num_threads_preprocessing=1, num_threads_nifti_save=1, lowres_segmentations=None, part_id=0, num_parts=1, tta=False)

                        segmentation_file_path = os.path.join(output_folder, "900_0000.nii.gz")

                        segmented_img = nib.load(segmentation_file_path)
                        segmented_data = segmented_img.get_fdata()

                        unique_values = np.unique(segmented_data)
                        st.write(f"Unique values in segmented image: {unique_values}")

                        tumor_value = 2

                        tumor_mask = (segmented_data == tumor_value).astype(np.uint8)
                        tumor_mask_img = nib.Nifti1Image(tumor_mask, affine=segmented_img.affine, header=segmented_img.header)
                        tumor_mask_file_path = os.path.join(patient_folder, "tumor_mask.nii.gz")

                        nib.save(tumor_mask_img, tumor_mask_file_path)

                        st.write(f"Tumor mask saved at: {tumor_mask_file_path}")

                        ct_image = sitk.ReadImage(renamed_file)
                        seg_image = sitk.ReadImage(tumor_mask_file_path)

                        slice_number = st.slider('Select Slice', 0, ct_image.GetSize()[2] - 1, ct_image.GetSize()[2] // 2)
                        display_images(ct_image, seg_image, slice_number)

                        if st.button('Start Feature Extraction'):
                            try:
                                extractor = setup_extractor()
                                feature_extraction_result = extractor.execute(ct_image, seg_image)
                                features_df = pd.DataFrame([feature_extraction_result])

                                features_of_interest = [
                                    'original_firstorder_10Percentile', 'original_firstorder_Mean', 'original_firstorder_Uniformity', 
                                    'original_glcm_ClusterTendency', 'original_glcm_Idm', 'original_glcm_Imc2', 'original_glcm_JointEnergy',
                                    'original_gldm_LargeDependenceEmphasis', 'original_gldm_SmallDependenceLowGrayLevelEmphasis', 
                                    'original_glrlm_HighGrayLevelRunEmphasis', 'original_glrlm_LongRunEmphasis', 'original_glrlm_LongRunLowGrayLevelEmphasis', 
                                    'original_glrlm_RunVariance', 'original_glrlm_ShortRunEmphasis', 'original_glrlm_ShortRunHighGrayLevelEmphasis', 
                                    'original_glrlm_ShortRunLowGrayLevelEmphasis', 'original_glszm_GrayLevelVariance', 'original_glszm_HighGrayLevelZoneEmphasis', 
                                    'original_glszm_LargeAreaLowGrayLevelEmphasis', 'original_glszm_SmallAreaHighGrayLevelEmphasis', 'original_glszm_ZoneVariance', 
                                    'wavelet-LLH_firstorder_Entropy', 'wavelet-LLH_firstorder_InterquartileRange', 'wavelet-LLH_firstorder_Kurtosis', 
                                    'wavelet-LLH_glcm_Contrast', 'wavelet-LLH_glcm_DifferenceVariance', 'wavelet-LLH_glcm_Idm', 'wavelet-LLH_glcm_Idn', 
                                    'wavelet-LLH_glcm_Imc1', 'wavelet-LLH_gldm_HighGrayLevelEmphasis', 'wavelet-LLH_gldm_LargeDependenceEmphasis', 
                                    'wavelet-LLH_gldm_SmallDependenceHighGrayLevelEmphasis', 'wavelet-LLH_glrlm_GrayLevelNonUniformityNormalized', 
                                    'wavelet-LLH_glrlm_HighGrayLevelRunEmphasis', 'wavelet-LLH_glrlm_LongRunLowGrayLevelEmphasis', 
                                    'wavelet-LLH_glrlm_RunLengthNonUniformity', 'wavelet-LLH_glrlm_RunPercentage', 'wavelet-LLH_ngtdm_Busyness', 
                                    'wavelet-LHL_firstorder_RobustMeanAbsoluteDeviation', 'wavelet-LHL_glcm_ClusterTendency', 'wavelet-LHL_glcm_Correlation', 
                                    'wavelet-LHL_glcm_DifferenceEntropy', 'wavelet-LHL_glcm_Idmn', 'wavelet-LHL_glcm_JointEntropy', 'wavelet-LHL_glcm_SumAverage', 
                                    'wavelet-LHL_gldm_DependenceNonUniformityNormalized', 'wavelet-LHL_glrlm_LongRunEmphasis', 'wavelet-LHL_glszm_SizeZoneNonUniformityNormalized', 
                                    'wavelet-LHL_ngtdm_Complexity', 'wavelet-LHH_firstorder_RootMeanSquared'
                                ]

                                selected_features_df = features_df[features_of_interest]

                                st.session_state['selected_features_df'] = selected_features_df

                                st.write("Selected Features:")
                                st.dataframe(selected_features_df)
                            except Exception as e:
                                st.error(f"Error during feature extraction: {str(e)}")

                    except Exception as e:
                        st.error(f"Error during automatic segmentation: {str(e)}")

        if 'selected_features_df' in st.session_state and st.button('Calculate RAD-Score for Processed Patients'):
            try:
                time_points = np.linspace(0, 60, 61)
                cumulative_hazards = rsf_model.predict_cumulative_hazard_function(st.session_state['selected_features_df'])
                rad_scores = np.array([simps([ch(tp) for tp in time_points], time_points) for ch in cumulative_hazards])
                normalized_rad_scores = scaler.transform(rad_scores.reshape(-1, 1)).flatten()
                st.write(f"Normalized RAD-Score for the processed patients: {normalized_rad_scores}")
            except Exception as e:
                st.error(f"Error during RAD-Score calculation: {str(e)}")

elif choice == "Contact":
    contact()
