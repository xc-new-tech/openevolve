"""
Image binarization algorithms for barcode image preprocessing
"""
import cv2
import numpy as np


def adaptive_threshold(image, max_value=255, adaptive_method=cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                      threshold_type=cv2.THRESH_BINARY, block_size=11, c=2):
    """
    Apply adaptive thresholding for binarization
    
    Args:
        image: Input grayscale image
        max_value: Maximum value assigned to pixel values above threshold
        adaptive_method: Adaptive method (ADAPTIVE_THRESH_MEAN_C or ADAPTIVE_THRESH_GAUSSIAN_C)
        threshold_type: Type of thresholding (THRESH_BINARY or THRESH_BINARY_INV)
        block_size: Size of the neighborhood area for threshold calculation
        c: Constant subtracted from the mean
        
    Returns:
        Binary image
    """
    return cv2.adaptiveThreshold(image, max_value, adaptive_method, threshold_type, block_size, c)


def otsu_threshold(image, threshold_type=cv2.THRESH_BINARY):
    """
    Apply Otsu's automatic threshold selection
    
    Args:
        image: Input grayscale image
        threshold_type: Type of thresholding
        
    Returns:
        Tuple of (threshold_value, binary_image)
    """
    threshold_value, binary = cv2.threshold(image, 0, 255, threshold_type + cv2.THRESH_OTSU)
    return threshold_value, binary


def global_threshold(image, threshold_value=128, max_value=255, threshold_type=cv2.THRESH_BINARY):
    """
    Apply global thresholding with fixed threshold value
    
    Args:
        image: Input grayscale image
        threshold_value: Fixed threshold value
        max_value: Maximum value assigned to pixel values above threshold
        threshold_type: Type of thresholding
        
    Returns:
        Tuple of (threshold_value, binary_image)
    """
    _, binary = cv2.threshold(image, threshold_value, max_value, threshold_type)
    return threshold_value, binary


def triangle_threshold(image):
    """
    Apply triangle threshold algorithm
    
    Args:
        image: Input grayscale image
        
    Returns:
        Tuple of (threshold_value, binary_image)
    """
    threshold_value, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE)
    return threshold_value, binary


def multi_threshold(image, thresholds=[85, 170]):
    """
    Apply multiple thresholds to create multi-level binary image
    
    Args:
        image: Input grayscale image
        thresholds: List of threshold values
        
    Returns:
        Multi-level binary image
    """
    result = np.zeros_like(image)
    
    for i, thresh in enumerate(sorted(thresholds)):
        mask = image > thresh
        result[mask] = (i + 1) * (255 // (len(thresholds) + 1))
    
    return result


def sauvola_threshold(image, window_size=15, k=0.2, r=128):
    """
    Apply Sauvola local thresholding algorithm
    
    Args:
        image: Input grayscale image
        window_size: Size of the local window
        k: Parameter to control the threshold
        r: Dynamic range of standard deviation
        
    Returns:
        Binary image
    """
    # Convert to float for calculations
    img_float = image.astype(np.float64)
    
    # Calculate local mean using convolution
    kernel = np.ones((window_size, window_size)) / (window_size * window_size)
    local_mean = cv2.filter2D(img_float, -1, kernel)
    
    # Calculate local standard deviation
    local_mean_sq = cv2.filter2D(img_float * img_float, -1, kernel)
    local_var = local_mean_sq - local_mean * local_mean
    local_std = np.sqrt(np.maximum(local_var, 0))
    
    # Calculate Sauvola threshold
    threshold = local_mean * (1 + k * (local_std / r - 1))
    
    # Apply threshold
    binary = (img_float > threshold).astype(np.uint8) * 255
    
    return binary


def niblack_threshold(image, window_size=15, k=-0.2):
    """
    Apply Niblack local thresholding algorithm
    
    Args:
        image: Input grayscale image
        window_size: Size of the local window
        k: Parameter to control the threshold
        
    Returns:
        Binary image
    """
    # Convert to float for calculations
    img_float = image.astype(np.float64)
    
    # Calculate local mean using convolution
    kernel = np.ones((window_size, window_size)) / (window_size * window_size)
    local_mean = cv2.filter2D(img_float, -1, kernel)
    
    # Calculate local standard deviation
    local_mean_sq = cv2.filter2D(img_float * img_float, -1, kernel)
    local_var = local_mean_sq - local_mean * local_mean
    local_std = np.sqrt(np.maximum(local_var, 0))
    
    # Calculate Niblack threshold
    threshold = local_mean + k * local_std
    
    # Apply threshold
    binary = (img_float > threshold).astype(np.uint8) * 255
    
    return binary


def wolf_threshold(image, window_size=15, k=0.5):
    """
    Apply Wolf-Jolion local thresholding algorithm
    
    Args:
        image: Input grayscale image
        window_size: Size of the local window
        k: Parameter to control the threshold
        
    Returns:
        Binary image
    """
    # Convert to float for calculations
    img_float = image.astype(np.float64)
    
    # Calculate local mean using convolution
    kernel = np.ones((window_size, window_size)) / (window_size * window_size)
    local_mean = cv2.filter2D(img_float, -1, kernel)
    
    # Calculate local standard deviation
    local_mean_sq = cv2.filter2D(img_float * img_float, -1, kernel)
    local_var = local_mean_sq - local_mean * local_mean
    local_std = np.sqrt(np.maximum(local_var, 0))
    
    # Calculate global mean and std
    global_mean = np.mean(img_float)
    global_std = np.std(img_float)
    
    # Calculate Wolf threshold
    threshold = local_mean - k * (local_std - global_std) * (local_mean - global_mean) / global_std
    
    # Apply threshold
    binary = (img_float > threshold).astype(np.uint8) * 255
    
    return binary


def adaptive_binarization_pipeline(image, method='auto', **kwargs):
    """
    Automatically select appropriate binarization method
    
    Args:
        image: Input grayscale image
        method: Binarization method ('auto', 'adaptive', 'otsu', 'sauvola', 'niblack')
        **kwargs: Additional parameters for specific methods
        
    Returns:
        Binary image
    """
    if method == 'auto':
        # Analyze image characteristics to select best method
        mean_intensity = np.mean(image)
        std_intensity = np.std(image)
        
        # Check for uniform lighting
        if std_intensity < 20:  # Very uniform lighting
            method = 'otsu'
        elif mean_intensity < 100 or mean_intensity > 200:  # Extreme lighting
            method = 'adaptive'
        else:  # Variable lighting
            method = 'sauvola'
    
    if method == 'adaptive':
        return adaptive_threshold(image, **kwargs)
    elif method == 'otsu':
        _, binary = otsu_threshold(image)
        return binary
    elif method == 'sauvola':
        return sauvola_threshold(image, **kwargs)
    elif method == 'niblack':
        return niblack_threshold(image, **kwargs)
    elif method == 'wolf':
        return wolf_threshold(image, **kwargs)
    else:
        raise ValueError(f"Unknown binarization method: {method}") 