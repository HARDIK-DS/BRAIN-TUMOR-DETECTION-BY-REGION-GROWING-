import numpy as np
import SimpleITK as sitk
import pydicom
import io
import tempfile
import os
import cv2
from PIL import Image

def load_medical_image(file_obj):
    """
    Load and preprocess medical image formats (DICOM, NIfTI).
    
    Parameters:
    -----------
    file_obj : FileUploader object
        Uploaded file object from Streamlit
    
    Returns:
    --------
    tuple
        (processed_image, original_image)
    """
    file_extension = file_obj.name.split('.')[-1].lower()
    
    # Create a temporary file to save the uploaded content
    with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{file_extension}') as tmp_file:
        tmp_file.write(file_obj.getvalue())
        tmp_file_path = tmp_file.name
    
    try:
        if file_extension == 'dcm':
            # Load DICOM file
            dicom_data = pydicom.dcmread(tmp_file_path)
            
            # Extract the pixel data
            image = dicom_data.pixel_array
            
            # Normalize the image if needed
            if image.dtype != np.uint8:
                # Scale to 0-255 range for display
                image = (image - image.min()) / (image.max() - image.min()) * 255
                image = image.astype(np.uint8)
            
            # Keep original for display
            original_image = image.copy()
            
        elif file_extension in ['nii', 'nii.gz']:
            # Load NIfTI file
            image_sitk = sitk.ReadImage(tmp_file_path)
            
            # Convert to numpy array
            image = sitk.GetArrayFromImage(image_sitk)
            
            # For 3D volumes, take a middle slice
            if len(image.shape) == 3:
                middle_slice = image.shape[0] // 2
                image = image[middle_slice, :, :]
            
            # Normalize the image
            if image.dtype != np.uint8:
                image = (image - image.min()) / (image.max() - image.min()) * 255
                image = image.astype(np.uint8)
            
            # Keep original for display
            original_image = image.copy()
        
        else:
            # For regular image formats
            image = np.array(Image.open(tmp_file_path).convert('RGB'))
            original_image = image.copy()
        
        # Clean up the temporary file
        os.unlink(tmp_file_path)
        
        return image, original_image
    
    except Exception as e:
        # Clean up the temporary file in case of error
        os.unlink(tmp_file_path)
        raise Exception(f"Error loading medical image: {str(e)}")

def normalize_image(image):
    """
    Normalize image values to 0-255 range.
    
    Parameters:
    -----------
    image : ndarray
        Input image
    
    Returns:
    --------
    ndarray
        Normalized image
    """
    if image.dtype != np.uint8:
        normalized = (image - image.min()) / (image.max() - image.min()) * 255
        return normalized.astype(np.uint8)
    return image

def enhance_contrast(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    Enhance image contrast using CLAHE.
    
    Parameters:
    -----------
    image : ndarray
        Input grayscale image
    clip_limit : float
        Threshold for contrast limiting
    tile_grid_size : tuple
        Size of grid for histogram equalization
    
    Returns:
    --------
    ndarray
        Contrast enhanced image
    """
    # Ensure grayscale
    if len(image.shape) > 2:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    
    # Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    enhanced = clahe.apply(gray.astype(np.uint8))
    
    return enhanced

def apply_windowing(image, window_center, window_width):
    """
    Apply windowing to medical images to adjust contrast.
    
    Parameters:
    -----------
    image : ndarray
        Input image
    window_center : float
        Center of the window
    window_width : float
        Width of the window
    
    Returns:
    --------
    ndarray
        Windowed image
    """
    # Compute min and max values for the window
    min_value = window_center - window_width // 2
    max_value = window_center + window_width // 2
    
    # Apply windowing
    windowed = np.clip(image, min_value, max_value)
    
    # Normalize to 0-255
    windowed = ((windowed - min_value) / (max_value - min_value)) * 255
    
    return windowed.astype(np.uint8)
