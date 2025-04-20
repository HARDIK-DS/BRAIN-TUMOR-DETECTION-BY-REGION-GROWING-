import numpy as np
import cv2
from skimage import filters, exposure, util

def preprocess_image(image, smoothing_factor=0.5):
    """
    Preprocess the image for better segmentation results.
    
    Parameters:
    -----------
    image : ndarray
        Input image
    smoothing_factor : float
        Sigma value for Gaussian blur
    
    Returns:
    --------
    ndarray
        Preprocessed image
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3 and image.shape[2] > 1:
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray_image = image.copy()
    
    # Apply Gaussian blur for noise reduction
    if smoothing_factor > 0:
        smoothed_image = cv2.GaussianBlur(gray_image, (0, 0), sigmaX=smoothing_factor)
    else:
        smoothed_image = gray_image
    
    # Enhance contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_image = clahe.apply(np.uint8(smoothed_image))
    
    return enhanced_image

def overlay_segmentation(original_image, segmentation_mask, overlay_color=(255, 0, 0), alpha=0.5):
    """
    Create an overlay of segmentation mask on the original image.
    
    Parameters:
    -----------
    original_image : ndarray
        Original input image
    segmentation_mask : ndarray
        Binary segmentation mask
    overlay_color : tuple
        RGB color for the overlay (default: red)
    alpha : float
        Transparency value (0.0 to 1.0)
    
    Returns:
    --------
    ndarray
        Image with segmentation overlay
    """
    # Ensure original image is RGB
    if len(original_image.shape) == 2:
        rgb_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2RGB)
    else:
        rgb_image = original_image.copy()
    
    # Create color overlay
    overlay = np.zeros_like(rgb_image)
    
    # Extract coordinates of the mask
    y_coords, x_coords = np.where(segmentation_mask == 1)
    
    # Apply color to the overlay at mask coordinates
    for y, x in zip(y_coords, x_coords):
        overlay[y, x] = overlay_color
    
    # Blend original image with overlay
    result = cv2.addWeighted(rgb_image, 1, overlay, alpha, 0)
    
    # Add contour for better visibility
    contours, _ = cv2.findContours(segmentation_mask.astype(np.uint8), 
                                  cv2.RETR_EXTERNAL, 
                                  cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(result, contours, -1, overlay_color, 1)
    
    return result

def convert_to_grayscale(image):
    """
    Convert an image to grayscale if it is not already.
    
    Parameters:
    -----------
    image : ndarray
        Input image
    
    Returns:
    --------
    ndarray
        Grayscale image
    """
    if len(image.shape) == 2:
        return image
    else:
        return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

def calculate_tumor_probability(image, segmentation_mask):
    """
    Calculate probability that the segmented region is a tumor.
    This is a simplified example that could be improved with ML techniques.
    
    Parameters:
    -----------
    image : ndarray
        Original image
    segmentation_mask : ndarray
        Binary segmentation mask
    
    Returns:
    --------
    float
        Probability score (0.0 to 1.0)
    """
    # This is a simplified heuristic - in a real application, 
    # this would use a trained classifier
    
    # Tumor characteristics:
    # 1. High intensity relative to surroundings
    # 2. Relatively compact shape
    # 3. Size within reasonable range for brain tumors
    
    gray_image = convert_to_grayscale(image)
    
    # Calculate mean intensity of tumor region and surrounding region
    tumor_pixels = gray_image[segmentation_mask == 1]
    non_tumor_pixels = gray_image[segmentation_mask == 0]
    
    if len(tumor_pixels) == 0:
        return 0.0
    
    tumor_mean = np.mean(tumor_pixels)
    non_tumor_mean = np.mean(non_tumor_pixels)
    
    # Contrast ratio
    contrast_ratio = tumor_mean / (non_tumor_mean + 1e-6)
    
    # Size characteristics
    tumor_size = np.sum(segmentation_mask)
    image_size = segmentation_mask.shape[0] * segmentation_mask.shape[1]
    size_ratio = tumor_size / image_size
    
    # Shape compactness (using a simple approximation)
    perimeter = calculate_perimeter(segmentation_mask)
    compactness = 4 * np.pi * tumor_size / (perimeter**2 + 1e-6)
    
    # Combine factors (simplified)
    probability = min(1.0, (
        0.4 * min(contrast_ratio / 2.0, 1.0) +
        0.3 * max(0, 1.0 - abs(size_ratio - 0.05) / 0.1) +
        0.3 * min(compactness, 1.0)
    ))
    
    return probability

def calculate_perimeter(mask):
    """
    Calculate the perimeter of a binary mask.
    
    Parameters:
    -----------
    mask : ndarray
        Binary mask
    
    Returns:
    --------
    float
        Perimeter length
    """
    contours, _ = cv2.findContours(mask.astype(np.uint8), 
                                  cv2.RETR_EXTERNAL, 
                                  cv2.CHAIN_APPROX_SIMPLE)
    
    perimeter = 0
    for contour in contours:
        perimeter += cv2.arcLength(contour, True)
    
    return perimeter
