import streamlit as st
import os
import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk
from PIL import Image
import io
from skimage import io as skio
import cv2
from region_growing import region_growing_segmentation
from utils import preprocess_image, overlay_segmentation, convert_to_grayscale
from evaluation import calculate_metrics
from preprocessor import load_medical_image

st.set_page_config(
    page_title="Brain Tumor Detection",
    page_icon="🧠",
    layout="wide",
)

def main():
    st.title("Brain Tumor Detection with Region Growing")
    st.write("""
    This application uses region growing algorithm to segment brain tumors from MRI images.
    Upload your images and adjust parameters to get the optimal segmentation results.
    """)

    # Sidebar for algorithm parameters
    st.sidebar.title("Algorithm Parameters")
    
    # Region growing parameters
    seed_point_method = st.sidebar.radio(
        "Seed Point Selection",
        ["Automatic", "Manual"],
        help="Choose how to select the starting point for region growing"
    )
    
    threshold = st.sidebar.slider(
        "Threshold", 
        min_value=0.0, 
        max_value=1.0, 
        value=0.1, 
        step=0.01,
        help="Intensity threshold for region growing"
    )
    
    connectivity = st.sidebar.radio(
        "Connectivity",
        [4, 8],
        help="4-connectivity considers only pixels sharing an edge, 8-connectivity includes diagonal pixels"
    )
    
    iterations = st.sidebar.slider(
        "Max Iterations",
        min_value=10,
        max_value=1000,
        value=100,
        step=10,
        help="Maximum number of iterations for the algorithm"
    )
    
    smoothing = st.sidebar.slider(
        "Smoothing Factor",
        min_value=0.0,
        max_value=5.0,
        value=0.5,
        step=0.1,
        help="Smoothing factor for preprocessing (Gaussian blur sigma)"
    )

    # File uploader
    uploaded_file = st.file_uploader("Upload a brain MRI image", type=["jpg", "jpeg", "png", "tif", "bmp", "dcm", "nii", "nii.gz"])

    if uploaded_file is not None:
        try:
            # Read the image based on file type
            file_extension = uploaded_file.name.split('.')[-1].lower()
            
            # Process the image
            if file_extension in ['dcm', 'nii', 'nii.gz']:
                # Medical image formats
                image, original_image = load_medical_image(uploaded_file)
            else:
                # Common image formats
                image = np.array(Image.open(uploaded_file).convert('RGB'))
                original_image = image.copy()
                
                # Convert to grayscale for processing if not already
                if len(image.shape) == 3 and image.shape[2] > 1:
                    image = convert_to_grayscale(image)
            
            # Display original image
            st.subheader("Original Image")
            fig_original, ax_original = plt.subplots(figsize=(10, 8))
            ax_original.imshow(original_image, cmap='gray' if len(original_image.shape) == 2 else None)
            ax_original.axis('off')
            st.pyplot(fig_original)
            
            # Preprocess the image
            preprocessed_image = preprocess_image(image, smoothing)
            
            # Seed point selection
            if seed_point_method == "Automatic":
                st.info("Automatic seed point selection is being used. The algorithm will try to find the most likely tumor location.")
                # For automatic, we'll use a simple heuristic to find a bright spot
                # In a real application, this would be more sophisticated
                if len(preprocessed_image.shape) == 2:
                    # For grayscale images
                    seed_point = np.unravel_index(np.argmax(preprocessed_image), preprocessed_image.shape)
                else:
                    # For colored images
                    gray = cv2.cvtColor(preprocessed_image, cv2.COLOR_RGB2GRAY)
                    seed_point = np.unravel_index(np.argmax(gray), gray.shape)
            else:
                # For manual, let the user click on the image
                st.info("Please click on the suspected tumor area in the image below to set the seed point.")
                # Create an interactive image
                fig_seed, ax_seed = plt.subplots(figsize=(10, 8))
                ax_seed.imshow(preprocessed_image, cmap='gray' if len(preprocessed_image.shape) == 2 else None)
                ax_seed.axis('off')
                st.pyplot(fig_seed)
                
                # Get coordinates from user
                seed_coords = st.text_input(
                    "Enter seed coordinates (row, column):",
                    value="100, 100",
                    help="Specify the starting point for region growing as row,column"
                )
                try:
                    seed_point = tuple(map(int, seed_coords.split(',')))
                except:
                    st.error("Invalid coordinates. Using default (100, 100).")
                    seed_point = (100, 100)

            # Run segmentation when user clicks the button
            if st.button("Run Segmentation"):
                with st.spinner("Processing..."):
                    segmented_image = region_growing_segmentation(
                        preprocessed_image, 
                        seed_point,
                        threshold,
                        connectivity,
                        iterations
                    )
                    
                    # Display segmentation result
                    st.subheader("Segmentation Result")
                    
                    # Create a figure with two subplots
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))
                    
                    # Display original image
                    ax1.imshow(original_image, cmap='gray' if len(original_image.shape) == 2 else None)
                    ax1.set_title("Original Image")
                    ax1.axis('off')
                    
                    # Display segmentation overlay
                    overlay = overlay_segmentation(original_image, segmented_image)
                    ax2.imshow(overlay)
                    ax2.set_title("Segmentation Overlay")
                    ax2.axis('off')
                    
                    st.pyplot(fig)
                    
                    # Calculate and display metrics
                    st.subheader("Evaluation Metrics")
                    st.write("""
                    For accurate evaluation metrics, a ground truth segmentation is required.
                    The following are estimated metrics based on common tumor characteristics:
                    """)
                    
                    metrics = calculate_metrics(segmented_image)
                    
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Segmented Area (pixels)", f"{metrics['area']}")
                    col2.metric("Perimeter (pixels)", f"{metrics['perimeter']:.2f}")
                    col3.metric("Circularity", f"{metrics['circularity']:.4f}")
                    
                    # Display histograms
                    st.subheader("Intensity Distribution")
                    fig_hist, (ax_hist1, ax_hist2) = plt.subplots(1, 2, figsize=(15, 5))
                    
                    # Histogram of original image
                    if len(image.shape) == 2:
                        ax_hist1.hist(image.flatten(), bins=50, color='blue', alpha=0.7)
                    else:
                        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                        ax_hist1.hist(gray.flatten(), bins=50, color='blue', alpha=0.7)
                    ax_hist1.set_title("Original Image Histogram")
                    ax_hist1.set_xlabel("Pixel Intensity")
                    ax_hist1.set_ylabel("Frequency")
                    
                    # Histogram of segmented region
                    mask = segmented_image > 0
                    if len(image.shape) == 2:
                        segmented_values = image[mask]
                    else:
                        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                        segmented_values = gray[mask]
                        
                    if len(segmented_values) > 0:
                        ax_hist2.hist(segmented_values, bins=50, color='red', alpha=0.7)
                        ax_hist2.set_title("Segmented Region Histogram")
                        ax_hist2.set_xlabel("Pixel Intensity")
                        ax_hist2.set_ylabel("Frequency")
                    else:
                        ax_hist2.set_title("No pixels in segmented region")
                    
                    st.pyplot(fig_hist)
                    
                    # Save results
                    st.subheader("Save Results")
                    
                    # Create a downloadable version of the segmentation
                    buffer = io.BytesIO()
                    plt.figure(figsize=(10, 8))
                    plt.imshow(overlay)
                    plt.axis('off')
                    plt.tight_layout(pad=0)
                    plt.savefig(buffer, format='png', bbox_inches='tight', pad_inches=0)
                    buffer.seek(0)
                    
                    st.download_button(
                        label="Download Segmentation Result",
                        data=buffer,
                        file_name="brain_tumor_segmentation.png",
                        mime="image/png"
                    )
                    
                    # More explanation about the results
                    st.subheader("Interpretation")
                    st.write("""
                    The red overlay in the segmentation result indicates the detected tumor region.
                    
                    **Analysis:**
                    - **Area**: Represents the size of the segmented region in pixels.
                    - **Perimeter**: The length of the boundary of the segmented region.
                    - **Circularity**: A measure of how close the shape is to a perfect circle (1.0). 
                      Tumors typically have irregular shapes with lower circularity values.
                    
                    The histograms show the intensity distribution of the original image and the segmented region.
                    Tumors often have different intensity patterns compared to surrounding tissue.
                    """)
                    
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")
    
    # Information about the application
    st.sidebar.header("About")
    st.sidebar.info("""
    This application uses region growing segmentation to detect brain tumors in MRI images.
    
    **Region Growing Algorithm:**
    - Starts from a seed point and grows a region by adding neighboring pixels
    - Uses intensity similarity to determine inclusion
    - Produces a binary segmentation mask
    
    Adjust the parameters on the left sidebar to improve segmentation results.
    """)

if __name__ == "__main__":
    main()
