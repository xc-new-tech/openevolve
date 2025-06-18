"""
Noise reduction algorithms for barcode image preprocessing
"""
import cv2
import numpy as np


def gaussian_blur_denoise(image, kernel_size=(5, 5), sigma=0):
    """
    Apply Gaussian blur for noise reduction
    
    Args:
        image: Input grayscale image
        kernel_size: Size of the Gaussian kernel (width, height)
        sigma: Standard deviation for Gaussian kernel (0 = auto-calculate)
        
    Returns:
        Denoised image
    """
    return cv2.GaussianBlur(image, kernel_size, sigma)


def median_blur_denoise(image, kernel_size=3):
    """
    Apply median filter for salt-and-pepper noise reduction
    
    Args:
        image: Input grayscale image
        kernel_size: Size of the median filter kernel (must be odd)
        
    Returns:
        Denoised image
    """
    return cv2.medianBlur(image, kernel_size)


def bilateral_filter_denoise(image, d=9, sigma_color=75, sigma_space=75):
    """
    Apply bilateral filter for edge-preserving noise reduction
    
    Args:
        image: Input grayscale image
        d: Diameter of each pixel neighborhood
        sigma_color: Filter sigma in the color space
        sigma_space: Filter sigma in the coordinate space
        
    Returns:
        Denoised image
    """
    return cv2.bilateralFilter(image, d, sigma_color, sigma_space)


def non_local_means_denoise(image, h=10, template_window_size=7, search_window_size=21):
    """
    Apply Non-local Means denoising algorithm
    
    Args:
        image: Input grayscale image
        h: Filtering strength. Higher h value removes more noise but removes details too
        template_window_size: Size of the template patch
        search_window_size: Size of the search window
        
    Returns:
        Denoised image
    """
    return cv2.fastNlMeansDenoising(image, None, h, template_window_size, search_window_size)


def adaptive_wiener_filter(image, noise_variance=None):
    """
    Apply adaptive Wiener filter for noise reduction
    
    Args:
        image: Input grayscale image
        noise_variance: Estimated noise variance (auto-estimated if None)
        
    Returns:
        Denoised image
    """
    # Convert to float for calculations
    img_float = image.astype(np.float32)
    
    # Estimate noise variance if not provided
    if noise_variance is None:
        # Use Laplacian of Gaussian for noise estimation
        log_filtered = cv2.Laplacian(img_float, cv2.CV_32F)
        noise_variance = np.var(log_filtered) * 0.5
    
    # Calculate local statistics
    kernel = np.ones((5, 5)) / 25
    local_mean = cv2.filter2D(img_float, -1, kernel)
    local_mean_sq = cv2.filter2D(img_float ** 2, -1, kernel)
    local_variance = local_mean_sq - local_mean ** 2
    
    # Wiener filter
    wiener_filter = np.maximum(local_variance - noise_variance, 0) / np.maximum(local_variance, noise_variance)
    
    # Apply filter
    denoised = local_mean + wiener_filter * (img_float - local_mean)
    
    # Clip and convert back to uint8
    denoised = np.clip(denoised, 0, 255)
    return denoised.astype(np.uint8)


def morphological_denoise(image, kernel_size=(3, 3), operation='opening'):
    """
    Apply morphological operations for noise reduction
    
    Args:
        image: Input binary image
        kernel_size: Size of the morphological kernel
        operation: Type of operation ('opening', 'closing', 'gradient')
        
    Returns:
        Denoised image
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    
    if operation == 'opening':
        return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    elif operation == 'closing':
        return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    elif operation == 'gradient':
        return cv2.morphologyEx(image, cv2.MORPH_GRADIENT, kernel)
    else:
        raise ValueError(f"Unknown operation: {operation}")


def wavelet_denoise(image, wavelet='db4', sigma=None):
    """
    Apply wavelet denoising (requires PyWavelets)
    
    Args:
        image: Input grayscale image
        wavelet: Wavelet type
        sigma: Noise standard deviation (auto-estimated if None)
        
    Returns:
        Denoised image
    """
    try:
        import pywt
    except ImportError:
        # Fallback to Gaussian blur if PyWavelets not available
        return gaussian_blur_denoise(image, (3, 3))
    
    # Convert to float
    img_float = image.astype(np.float32) / 255.0
    
    # Estimate noise if not provided
    if sigma is None:
        sigma = np.std(img_float) * 0.1
    
    # Wavelet transform
    coeffs = pywt.wavedec2(img_float, wavelet, level=3)
    
    # Soft thresholding
    threshold = sigma * np.sqrt(2 * np.log(img_float.size))
    coeffs_thresh = list(coeffs)
    coeffs_thresh[1:] = [pywt.threshold(detail, threshold, mode='soft') for detail in coeffs[1:]]
    
    # Inverse wavelet transform
    denoised = pywt.waverec2(coeffs_thresh, wavelet)
    
    # Convert back to uint8
    denoised = np.clip(denoised * 255, 0, 255)
    return denoised.astype(np.uint8)


def anisotropic_diffusion(image, num_iter=15, delta_t=1/7, kappa=30, option=1):
    """
    Apply anisotropic diffusion for edge-preserving denoising
    
    Args:
        image: Input grayscale image
        num_iter: Number of iterations
        delta_t: Time step
        kappa: Conduction coefficient
        option: Diffusion function option (1 or 2)
        
    Returns:
        Denoised image
    """
    # Convert to float
    img = image.astype(np.float32)
    
    # Initialize
    deltaS = np.zeros_like(img)
    deltaE = np.zeros_like(img)
    deltaW = np.zeros_like(img)
    deltaN = np.zeros_like(img)
    NS = np.zeros_like(img)
    EW = np.zeros_like(img)
    
    for _ in range(num_iter):
        # Calculate gradients
        deltaS[:-1, :] = np.diff(img, axis=0)
        deltaE[:, :-1] = np.diff(img, axis=1)
        deltaW[:, 1:] = -np.diff(img, axis=1)
        deltaN[1:, :] = -np.diff(img, axis=0)
        
        # Conduction coefficients
        if option == 1:
            gS = np.exp(-(deltaS/kappa)**2)
            gE = np.exp(-(deltaE/kappa)**2)
            gW = np.exp(-(deltaW/kappa)**2)
            gN = np.exp(-(deltaN/kappa)**2)
        elif option == 2:
            gS = 1.0 / (1.0 + (deltaS/kappa)**2)
            gE = 1.0 / (1.0 + (deltaE/kappa)**2)
            gW = 1.0 / (1.0 + (deltaW/kappa)**2)
            gN = 1.0 / (1.0 + (deltaN/kappa)**2)
        
        # Update image
        NS = gN * deltaN + gS * deltaS
        EW = gE * deltaE + gW * deltaW
        img += delta_t * (NS + EW)
    
    # Clip and convert back to uint8
    img = np.clip(img, 0, 255)
    return img.astype(np.uint8)


def adaptive_denoise(image, noise_type='auto'):
    """
    Automatically select appropriate denoising method based on noise type
    
    Args:
        image: Input grayscale image
        noise_type: Type of noise ('gaussian', 'salt_pepper', 'auto')
        
    Returns:
        Denoised image
    """
    if noise_type == 'auto':
        # Simple heuristic to detect noise type
        # Calculate image variance to estimate noise level
        variance = np.var(image)
        
        # Calculate salt-and-pepper noise indicator
        hist = cv2.calcHist([image], [0], None, [256], [0, 256])
        salt_pepper_ratio = (hist[0] + hist[255]) / image.size
        
        if salt_pepper_ratio > 0.01:  # High salt-and-pepper noise
            noise_type = 'salt_pepper'
        elif variance > 1000:  # High variance suggests Gaussian noise
            noise_type = 'gaussian'
        else:  # Low noise
            noise_type = 'gaussian'
    
    if noise_type == 'gaussian':
        return gaussian_blur_denoise(image, (5, 5))
    elif noise_type == 'salt_pepper':
        return median_blur_denoise(image, 3)
    elif noise_type == 'impulse':
        return median_blur_denoise(image, 5)
    elif noise_type == 'uniform':
        return bilateral_filter_denoise(image)
    else:
        raise ValueError(f"Unknown noise type: {noise_type}")


def edge_preserving_denoise(image, method='recursive', flags=1, sigma_s=50, sigma_r=0.4):
    """
    Apply edge-preserving denoising
    
    Args:
        image: Input image
        method: Denoising method ('recursive' or 'normalized_convolution')
        flags: Edge-preserving method flag
        sigma_s: Size of the neighborhood
        sigma_r: How much an adjacent pixel influences the computation
        
    Returns:
        Denoised image
    """
    if method == 'recursive':
        flags = cv2.RECURS_FILTER
    else:
        flags = cv2.NORMCONV_FILTER
    
    return cv2.edgePreservingFilter(image, flags=flags, sigma_s=sigma_s, sigma_r=sigma_r) 