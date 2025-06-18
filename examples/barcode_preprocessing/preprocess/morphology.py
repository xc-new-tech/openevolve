"""
Morphological operations for barcode image preprocessing
"""
import cv2
import numpy as np


def morphological_opening(image, kernel_size=(3, 3), kernel_type=cv2.MORPH_RECT, iterations=1):
    """
    Apply morphological opening (erosion followed by dilation)
    
    Args:
        image: Input binary image
        kernel_size: Size of the morphological kernel
        kernel_type: Type of kernel (MORPH_RECT, MORPH_ELLIPSE, MORPH_CROSS)
        iterations: Number of iterations
        
    Returns:
        Processed image
    """
    kernel = cv2.getStructuringElement(kernel_type, kernel_size)
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel, iterations=iterations)


def morphological_closing(image, kernel_size=(3, 3), kernel_type=cv2.MORPH_RECT, iterations=1):
    """
    Apply morphological closing (dilation followed by erosion)
    
    Args:
        image: Input binary image
        kernel_size: Size of the morphological kernel
        kernel_type: Type of kernel (MORPH_RECT, MORPH_ELLIPSE, MORPH_CROSS)
        iterations: Number of iterations
        
    Returns:
        Processed image
    """
    kernel = cv2.getStructuringElement(kernel_type, kernel_size)
    return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel, iterations=iterations)


def morphological_gradient(image, kernel_size=(3, 3), kernel_type=cv2.MORPH_RECT):
    """
    Apply morphological gradient (difference between dilation and erosion)
    
    Args:
        image: Input binary image
        kernel_size: Size of the morphological kernel
        kernel_type: Type of kernel
        
    Returns:
        Gradient image
    """
    kernel = cv2.getStructuringElement(kernel_type, kernel_size)
    return cv2.morphologyEx(image, cv2.MORPH_GRADIENT, kernel)


def morphological_tophat(image, kernel_size=(9, 9), kernel_type=cv2.MORPH_RECT):
    """
    Apply morphological top hat (difference between input and opening)
    
    Args:
        image: Input grayscale image
        kernel_size: Size of the morphological kernel
        kernel_type: Type of kernel
        
    Returns:
        Top hat image
    """
    kernel = cv2.getStructuringElement(kernel_type, kernel_size)
    return cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel)


def morphological_blackhat(image, kernel_size=(9, 9), kernel_type=cv2.MORPH_RECT):
    """
    Apply morphological black hat (difference between closing and input)
    
    Args:
        image: Input grayscale image
        kernel_size: Size of the morphological kernel
        kernel_type: Type of kernel
        
    Returns:
        Black hat image
    """
    kernel = cv2.getStructuringElement(kernel_type, kernel_size)
    return cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, kernel)


def erosion(image, kernel_size=(3, 3), kernel_type=cv2.MORPH_RECT, iterations=1):
    """
    Apply morphological erosion
    
    Args:
        image: Input binary image
        kernel_size: Size of the morphological kernel
        kernel_type: Type of kernel
        iterations: Number of iterations
        
    Returns:
        Eroded image
    """
    kernel = cv2.getStructuringElement(kernel_type, kernel_size)
    return cv2.erode(image, kernel, iterations=iterations)


def dilation(image, kernel_size=(3, 3), kernel_type=cv2.MORPH_RECT, iterations=1):
    """
    Apply morphological dilation
    
    Args:
        image: Input binary image
        kernel_size: Size of the morphological kernel
        kernel_type: Type of kernel
        iterations: Number of iterations
        
    Returns:
        Dilated image
    """
    kernel = cv2.getStructuringElement(kernel_type, kernel_size)
    return cv2.dilate(image, kernel, iterations=iterations)


def remove_small_objects(image, min_size=50):
    """
    Remove small connected components from binary image
    
    Args:
        image: Input binary image
        min_size: Minimum size of objects to keep
        
    Returns:
        Cleaned image
    """
    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(image, connectivity=8)
    
    # Create output image
    output = np.zeros_like(image)
    
    # Keep only large components (skip background label 0)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_size:
            output[labels == i] = 255
    
    return output


def fill_holes(image, max_hole_size=100):
    """
    Fill small holes in binary image
    
    Args:
        image: Input binary image
        max_hole_size: Maximum size of holes to fill
        
    Returns:
        Image with filled holes
    """
    # Invert image to make holes foreground
    inverted = cv2.bitwise_not(image)
    
    # Find connected components (holes)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(inverted, connectivity=8)
    
    # Create output image
    output = image.copy()
    
    # Fill small holes (skip background label 0)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] <= max_hole_size:
            output[labels == i] = 255
    
    return output


def skeletonize(image):
    """
    Apply skeletonization to binary image
    
    Args:
        image: Input binary image
        
    Returns:
        Skeletonized image
    """
    # Create structuring element
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    
    # Initialize
    skeleton = np.zeros_like(image)
    temp = image.copy()
    
    while True:
        # Erosion
        eroded = cv2.erode(temp, kernel)
        
        # Opening
        opened = cv2.morphologyEx(eroded, cv2.MORPH_OPEN, kernel)
        
        # Subtract
        subset = cv2.subtract(eroded, opened)
        
        # Union
        skeleton = cv2.bitwise_or(skeleton, subset)
        
        # Update temp
        temp = eroded.copy()
        
        # Check if we're done
        if cv2.countNonZero(temp) == 0:
            break
    
    return skeleton


def morphological_cleanup(image, operation_sequence=None):
    """
    Apply a sequence of morphological operations for cleanup
    
    Args:
        image: Input binary image
        operation_sequence: List of operations to apply
        
    Returns:
        Cleaned image
    """
    if operation_sequence is None:
        # Default cleanup sequence for barcodes
        operation_sequence = [
            ('opening', {'kernel_size': (2, 2)}),
            ('closing', {'kernel_size': (3, 3)}),
            ('remove_small_objects', {'min_size': 30}),
            ('fill_holes', {'max_hole_size': 50})
        ]
    
    result = image.copy()
    
    for operation, params in operation_sequence:
        if operation == 'opening':
            result = morphological_opening(result, **params)
        elif operation == 'closing':
            result = morphological_closing(result, **params)
        elif operation == 'erosion':
            result = erosion(result, **params)
        elif operation == 'dilation':
            result = dilation(result, **params)
        elif operation == 'gradient':
            result = morphological_gradient(result, **params)
        elif operation == 'tophat':
            result = morphological_tophat(result, **params)
        elif operation == 'blackhat':
            result = morphological_blackhat(result, **params)
        elif operation == 'remove_small_objects':
            result = remove_small_objects(result, **params)
        elif operation == 'fill_holes':
            result = fill_holes(result, **params)
        elif operation == 'skeletonize':
            result = skeletonize(result)
        else:
            raise ValueError(f"Unknown operation: {operation}")
    
    return result


def adaptive_morphological_cleanup(image, image_type='barcode'):
    """
    Apply adaptive morphological cleanup based on image characteristics
    
    Args:
        image: Input binary image
        image_type: Type of image ('barcode', 'qr', 'text')
        
    Returns:
        Cleaned image
    """
    if image_type == 'barcode':
        # Sequence optimized for linear barcodes
        sequence = [
            ('opening', {'kernel_size': (2, 1)}),  # Remove horizontal noise
            ('closing', {'kernel_size': (1, 3)}),  # Connect vertical bars
            ('remove_small_objects', {'min_size': 20}),
            ('fill_holes', {'max_hole_size': 30})
        ]
    elif image_type == 'qr':
        # Sequence optimized for QR codes
        sequence = [
            ('opening', {'kernel_size': (2, 2)}),  # Remove small noise
            ('closing', {'kernel_size': (3, 3)}),  # Connect patterns
            ('remove_small_objects', {'min_size': 50}),
            ('fill_holes', {'max_hole_size': 100})
        ]
    elif image_type == 'text':
        # Sequence optimized for text
        sequence = [
            ('opening', {'kernel_size': (1, 1)}),  # Light cleanup
            ('closing', {'kernel_size': (2, 2)}),  # Connect characters
            ('remove_small_objects', {'min_size': 10})
        ]
    else:
        # Default sequence
        sequence = [
            ('opening', {'kernel_size': (2, 2)}),
            ('closing', {'kernel_size': (3, 3)}),
            ('remove_small_objects', {'min_size': 30})
        ]
    
    return morphological_cleanup(image, sequence) 