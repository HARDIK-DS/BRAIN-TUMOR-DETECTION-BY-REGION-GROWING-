import numpy as np
from skimage import filters
import cv2
from queue import Queue

def region_growing_segmentation(image, seed_point, threshold=0.1, connectivity=8, max_iterations=100):
    """
    Perform region growing segmentation on an image.
    
    Parameters:
    -----------
    image : ndarray
        Input image, grayscale or color
    seed_point : tuple
        Starting point coordinates (row, col)
    threshold : float
        Threshold value for region growing
    connectivity : int
        4 or 8 connectivity for neighboring pixels
    max_iterations : int
        Maximum number of iterations to prevent infinite loops
    
    Returns:
    --------
    ndarray
        Binary segmentation mask (1 for segmented region, 0 for background)
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3 and image.shape[2] > 1:
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray_image = image.copy()
    
    # Create the segmentation mask
    height, width = gray_image.shape
    mask = np.zeros((height, width), dtype=np.uint8)
    
    # Check if seed point is within image bounds
    if (seed_point[0] < 0 or seed_point[0] >= height or 
        seed_point[1] < 0 or seed_point[1] >= width):
        raise ValueError(f"Seed point {seed_point} is outside image bounds ({height}x{width})")
    
    # Get the intensity value at the seed point
    seed_value = float(gray_image[seed_point])
    
    # Normalize the image for threshold comparison
    normalized_image = gray_image.astype(np.float32) / 255.0
    seed_value_normalized = seed_value / 255.0
    
    # Define neighbor offsets based on connectivity
    if connectivity == 4:
        # 4-connectivity: north, east, south, west
        neighbor_offsets = [(-1, 0), (0, 1), (1, 0), (0, -1)]
    else:
        # 8-connectivity: including diagonals
        neighbor_offsets = [(-1, 0), (-1, 1), (0, 1), (1, 1),
                            (1, 0), (1, -1), (0, -1), (-1, -1)]
    
    # Use queue for breadth-first search
    queue = Queue()
    queue.put(seed_point)
    
    # Mark the seed point as part of the region
    mask[seed_point] = 1
    
    iterations = 0
    while not queue.empty() and iterations < max_iterations:
        current_point = queue.get()
        current_row, current_col = current_point
        
        # Check neighbors
        for offset_row, offset_col in neighbor_offsets:
            neighbor_row = current_row + offset_row
            neighbor_col = current_col + offset_col
            
            # Check bounds
            if (neighbor_row < 0 or neighbor_row >= height or 
                neighbor_col < 0 or neighbor_col >= width):
                continue
            
            # Skip if already in the region
            if mask[neighbor_row, neighbor_col] == 1:
                continue
            
            # Check intensity difference
            neighbor_value = normalized_image[neighbor_row, neighbor_col]
            if abs(neighbor_value - seed_value_normalized) <= threshold:
                # Add to region
                mask[neighbor_row, neighbor_col] = 1
                queue.put((neighbor_row, neighbor_col))
        
        iterations += 1
    
    # Post-processing: remove small objects and fill holes
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    
    return mask

def region_growing_adaptive(image, seed_point, initial_threshold=0.1, max_threshold=0.3, connectivity=8, max_iterations=100):
    """
    Perform adaptive region growing segmentation on an image.
    The threshold is adaptively increased if the region is too small.
    
    Parameters:
    -----------
    image : ndarray
        Input image, grayscale or color
    seed_point : tuple
        Starting point coordinates (row, col)
    initial_threshold : float
        Initial threshold value for region growing
    max_threshold : float
        Maximum threshold value to try
    connectivity : int
        4 or 8 connectivity for neighboring pixels
    max_iterations : int
        Maximum number of iterations to prevent infinite loops
    
    Returns:
    --------
    ndarray
        Binary segmentation mask (1 for segmented region, 0 for background)
    """
    min_region_size = image.shape[0] * image.shape[1] * 0.005  # Minimum 0.5% of image
    max_region_size = image.shape[0] * image.shape[1] * 0.3  # Maximum 30% of image
    
    # Start with initial threshold
    current_threshold = initial_threshold
    mask = region_growing_segmentation(image, seed_point, current_threshold, connectivity, max_iterations)
    region_size = np.sum(mask)
    
    # If region is too small, gradually increase threshold
    while region_size < min_region_size and current_threshold < max_threshold:
        current_threshold += 0.05
        mask = region_growing_segmentation(image, seed_point, current_threshold, connectivity, max_iterations)
        region_size = np.sum(mask)
    
    # If region is too large, decrease threshold
    if region_size > max_region_size:
        current_threshold = initial_threshold
        while region_size > max_region_size and current_threshold > 0.01:
            current_threshold -= 0.01
            mask = region_growing_segmentation(image, seed_point, current_threshold, connectivity, max_iterations)
            region_size = np.sum(mask)
    
    return mask
