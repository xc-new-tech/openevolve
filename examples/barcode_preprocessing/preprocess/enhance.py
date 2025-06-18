"""
Contrast and brightness enhancement algorithms for barcode image preprocessing
"""
import cv2
import numpy as np


def histogram_equalization(image):
    """
    Apply histogram equalization for contrast enhancement
    
    Args:
        image: Input grayscale image
        
    Returns:
        Enhanced image with equalized histogram
    """
    return cv2.equalizeHist(image)


def clahe_enhancement(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    
    Args:
        image: Input grayscale image
        clip_limit: Threshold for contrast limiting
        tile_grid_size: Size of the grid for histogram equalization
        
    Returns:
        Enhanced image
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(image)


def gamma_correction(image, gamma=1.0):
    """
    Apply gamma correction to adjust brightness
    
    Args:
        image: Input grayscale image
        gamma: Gamma value (>1 makes image darker, <1 makes it brighter)
        
    Returns:
        Gamma-corrected image
    """
    # Build a lookup table mapping pixel values [0, 255] to their adjusted gamma values
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype(np.uint8)
    
    # Apply gamma correction using the lookup table
    return cv2.LUT(image, table)


def adaptive_gamma_correction(image, target_mean=128):
    """
    Automatically adjust gamma based on image mean
    
    Args:
        image: Input grayscale image
        target_mean: Target mean intensity value
        
    Returns:
        Adaptively gamma-corrected image
    """
    current_mean = np.mean(image)
    
    if current_mean > target_mean:
        # Image is too bright, use gamma > 1 to darken
        gamma = 1.5
    elif current_mean < target_mean:
        # Image is too dark, use gamma < 1 to brighten  
        gamma = 0.7
    else:
        # Image mean is close to target, no correction needed
        gamma = 1.0
    
    return gamma_correction(image, gamma)


def linear_contrast_stretch(image, percentile_low=1, percentile_high=99):
    """
    Apply linear contrast stretching using percentiles
    
    Args:
        image: Input grayscale image
        percentile_low: Lower percentile for stretching
        percentile_high: Upper percentile for stretching
        
    Returns:
        Contrast-stretched image
    """
    # Calculate percentiles
    p_low = np.percentile(image, percentile_low)
    p_high = np.percentile(image, percentile_high)
    
    # Apply linear stretching
    stretched = np.clip((image - p_low) * 255.0 / (p_high - p_low), 0, 255)
    
    return stretched.astype(np.uint8)


def adaptive_contrast_enhancement(image, alpha=1.5, beta=0):
    """
    Apply adaptive contrast enhancement
    
    Args:
        image: Input grayscale image
        alpha: Contrast control (1.0-3.0)
        beta: Brightness control (-100 to 100)
        
    Returns:
        Enhanced image
    """
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)


def unsharp_masking(image, kernel_size=5, sigma=1.0, amount=1.0, threshold=0):
    """
    Apply unsharp masking for detail enhancement
    
    Args:
        image: Input grayscale image
        kernel_size: Size of the Gaussian kernel
        sigma: Standard deviation for Gaussian blur
        amount: Strength of the sharpening
        threshold: Minimum difference required for sharpening
        
    Returns:
        Sharpened image
    """
    # Create Gaussian blur
    blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
    
    # Calculate mask (difference between original and blurred)
    mask = image.astype(np.float32) - blurred.astype(np.float32)
    
    # Apply threshold
    if threshold > 0:
        mask = np.where(np.abs(mask) >= threshold, mask, 0)
    
    # Apply unsharp masking
    sharpened = image.astype(np.float32) + amount * mask
    
    # Clip values to valid range
    sharpened = np.clip(sharpened, 0, 255)
    
    return sharpened.astype(np.uint8)


def contrast_stretching(image, lower_percentile=2, upper_percentile=98):
    """
    Apply contrast stretching to utilize full dynamic range
    
    Args:
        image: Input grayscale image
        lower_percentile: Lower percentile for stretching
        upper_percentile: Upper percentile for stretching
        
    Returns:
        Contrast-stretched image
    """
    # Calculate percentiles
    lower = np.percentile(image, lower_percentile)
    upper = np.percentile(image, upper_percentile)
    
    # Avoid division by zero
    if upper == lower:
        return image
    
    # Apply contrast stretching
    stretched = (image.astype(np.float32) - lower) * 255.0 / (upper - lower)
    
    # Clip values to valid range
    stretched = np.clip(stretched, 0, 255)
    
    return stretched.astype(np.uint8)


def local_enhancement(image, window_size=15, k=0.5):
    """
    Apply local contrast enhancement
    
    Args:
        image: Input grayscale image
        window_size: Size of the local window
        k: Enhancement strength factor
        
    Returns:
        Locally enhanced image
    """
    # Convert to float for calculations
    img_float = image.astype(np.float32)
    
    # Calculate local mean using convolution
    kernel = np.ones((window_size, window_size)) / (window_size * window_size)
    local_mean = cv2.filter2D(img_float, -1, kernel)
    
    # Calculate local standard deviation
    local_mean_sq = cv2.filter2D(img_float * img_float, -1, kernel)
    local_var = local_mean_sq - local_mean * local_mean
    local_std = np.sqrt(np.maximum(local_var, 0))
    
    # Apply local enhancement
    enhanced = local_mean + k * (img_float - local_mean) * (1 + local_std / 255.0)
    
    # Clip values to valid range
    enhanced = np.clip(enhanced, 0, 255)
    
    return enhanced.astype(np.uint8)


def multi_scale_enhancement(image, scales=[1, 2, 4]):
    """
    Apply multi-scale enhancement using different kernel sizes
    
    Args:
        image: Input grayscale image
        scales: List of scale factors for kernel sizes
        
    Returns:
        Enhanced image
    """
    enhanced = image.astype(np.float32)
    
    for scale in scales:
        kernel_size = 2 * scale + 1
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
        
        # Calculate difference (high-frequency content)
        diff = image.astype(np.float32) - blurred.astype(np.float32)
        
        # Add weighted difference back to enhance details
        enhanced += 0.3 * diff
    
    # Clip values to valid range
    enhanced = np.clip(enhanced, 0, 255)
    
    return enhanced.astype(np.uint8)


def adaptive_enhancement_pipeline(image, method='auto', **kwargs):
    """
    Automatically select appropriate enhancement method
    
    Args:
        image: Input grayscale image
        method: Enhancement method ('auto', 'clahe', 'gamma', 'histogram', 'unsharp')
        **kwargs: Additional parameters for specific methods
        
    Returns:
        Enhanced image
    """
    if method == 'auto':
        # Analyze image characteristics to select best method
        mean_intensity = np.mean(image)
        std_intensity = np.std(image)
        
        # Check contrast
        if std_intensity < 30:  # Low contrast
            method = 'clahe'
        elif mean_intensity < 80:  # Dark image
            method = 'gamma'
        elif mean_intensity > 180:  # Bright image
            method = 'gamma'
        else:  # Normal image
            method = 'unsharp'
    
    if method == 'clahe':
        return clahe_enhancement(image, **kwargs)
    elif method == 'gamma':
        return adaptive_gamma_correction(image, **kwargs)
    elif method == 'histogram':
        return histogram_equalization(image)
    elif method == 'unsharp':
        return unsharp_masking(image, **kwargs)
    elif method == 'contrast_stretch':
        return contrast_stretching(image, **kwargs)
    elif method == 'local':
        return local_enhancement(image, **kwargs)
    else:
        raise ValueError(f"Unknown enhancement method: {method}") 