import numpy as np
import cv2
from skimage import measure

def calculate_metrics(segmentation_mask):
    """
    Calculate evaluation metrics for the segmentation result.
    
    Parameters:
    -----------
    segmentation_mask : ndarray
        Binary segmentation mask
    
    Returns:
    --------
    dict
        Dictionary containing calculated metrics
    """
    # Initialize metrics dictionary
    metrics = {}
    
    # Area (number of pixels)
    metrics['area'] = np.sum(segmentation_mask)
    
    # Perimeter
    contours, _ = cv2.findContours(segmentation_mask.astype(np.uint8), 
                                  cv2.RETR_EXTERNAL, 
                                  cv2.CHAIN_APPROX_SIMPLE)
    perimeter = 0
    for contour in contours:
        perimeter += cv2.arcLength(contour, True)
    metrics['perimeter'] = perimeter
    
    # Circularity: 4π(area/perimeter²)
    if perimeter > 0:
        metrics['circularity'] = 4 * np.pi * metrics['area'] / (perimeter**2)
    else:
        metrics['circularity'] = 0
    
    # Calculate region properties
    if np.sum(segmentation_mask) > 0:
        props = measure.regionprops(segmentation_mask.astype(np.uint8))
        
        if len(props) > 0:
            # Eccentricity
            metrics['eccentricity'] = props[0].eccentricity
            
            # Major and minor axis lengths
            metrics['major_axis_length'] = props[0].major_axis_length
            metrics['minor_axis_length'] = props[0].minor_axis_length
            
            # Solidity (area / convex hull area)
            metrics['solidity'] = props[0].solidity
    else:
        metrics['eccentricity'] = 0
        metrics['major_axis_length'] = 0
        metrics['minor_axis_length'] = 0
        metrics['solidity'] = 0
    
    return metrics

def calculate_dice_coefficient(predicted_mask, ground_truth_mask):
    """
    Calculate Dice similarity coefficient between predicted and ground truth masks.
    
    Parameters:
    -----------
    predicted_mask : ndarray
        Binary mask of the predicted segmentation
    ground_truth_mask : ndarray
        Binary mask of the ground truth segmentation
    
    Returns:
    --------
    float
        Dice coefficient (0.0 to 1.0)
    """
    # Ensure binary masks
    pred = predicted_mask > 0
    gt = ground_truth_mask > 0
    
    # Calculate intersection and sums
    intersection = np.logical_and(pred, gt).sum()
    sum_pred = pred.sum()
    sum_gt = gt.sum()
    
    # Calculate Dice coefficient
    if sum_pred + sum_gt > 0:
        dice = 2.0 * intersection / (sum_pred + sum_gt)
    else:
        dice = 1.0  # Both masks are empty, consider it perfect match
    
    return dice

def calculate_jaccard_index(predicted_mask, ground_truth_mask):
    """
    Calculate Jaccard index (IoU) between predicted and ground truth masks.
    
    Parameters:
    -----------
    predicted_mask : ndarray
        Binary mask of the predicted segmentation
    ground_truth_mask : ndarray
        Binary mask of the ground truth segmentation
    
    Returns:
    --------
    float
        Jaccard index (0.0 to 1.0)
    """
    # Ensure binary masks
    pred = predicted_mask > 0
    gt = ground_truth_mask > 0
    
    # Calculate intersection and union
    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    
    # Calculate Jaccard index
    if union > 0:
        jaccard = intersection / union
    else:
        jaccard = 1.0  # Both masks are empty, consider it perfect match
    
    return jaccard

def calculate_sensitivity_specificity(predicted_mask, ground_truth_mask):
    """
    Calculate sensitivity (recall) and specificity for the segmentation.
    
    Parameters:
    -----------
    predicted_mask : ndarray
        Binary mask of the predicted segmentation
    ground_truth_mask : ndarray
        Binary mask of the ground truth segmentation
    
    Returns:
    --------
    tuple
        (sensitivity, specificity)
    """
    # Ensure binary masks
    pred = predicted_mask > 0
    gt = ground_truth_mask > 0
    
    # True positive, false negative, false positive, true negative
    tp = np.logical_and(pred, gt).sum()
    fn = np.logical_and(np.logical_not(pred), gt).sum()
    fp = np.logical_and(pred, np.logical_not(gt)).sum()
    tn = np.logical_and(np.logical_not(pred), np.logical_not(gt)).sum()
    
    # Calculate sensitivity and specificity
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    return sensitivity, specificity
