"""
Geometric corrections for barcode image preprocessing
"""
import cv2
import numpy as np
import math


def correct_skew(image, angle_threshold=1.0):
    """
    Detect and correct skew in barcode images
    
    Args:
        image: Input binary image
        angle_threshold: Minimum angle (degrees) to apply correction
        
    Returns:
        Skew-corrected image
    """
    # Find contours
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return image
    
    # Find the largest contour (assumed to be the barcode)
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Get the minimum area rectangle
    rect = cv2.minAreaRect(largest_contour)
    angle = rect[2]
    
    # Correct the angle (OpenCV's minAreaRect returns angles between -90 and 0)
    if angle < -45:
        angle = 90 + angle
    
    # Only correct if the angle is significant
    if abs(angle) > angle_threshold:
        # Get image center
        center = (image.shape[1] // 2, image.shape[0] // 2)
        
        # Create rotation matrix
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Apply rotation
        corrected = cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]))
        return corrected
    
    return image


def correct_skew_hough(image, angle_range=(-45, 45), angle_step=1):
    """
    Detect skew using Hough line transform
    
    Args:
        image: Input binary image
        angle_range: Range of angles to search (min, max) in degrees
        angle_step: Step size for angle search
        
    Returns:
        Skew-corrected image
    """
    # Apply Hough line transform
    lines = cv2.HoughLines(image, 1, np.pi / 180, threshold=50)
    
    if lines is None:
        return image
    
    # Collect angles
    angles = []
    for line in lines:
        rho, theta = line[0]
        angle = np.degrees(theta) - 90  # Convert to degrees relative to horizontal
        if angle_range[0] <= angle <= angle_range[1]:
            angles.append(angle)
    
    if not angles:
        return image
    
    # Calculate median angle
    median_angle = np.median(angles)
    
    # Apply correction if significant
    if abs(median_angle) > 1:
        center = (image.shape[1] // 2, image.shape[0] // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, median_angle, 1.0)
        corrected = cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]))
        return corrected
    
    return image


def correct_perspective(image, auto_detect=True, corners=None):
    """
    Correct perspective distortion in images
    
    Args:
        image: Input image
        auto_detect: Whether to automatically detect corners
        corners: Manual corner points [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
        
    Returns:
        Perspective-corrected image
    """
    height, width = image.shape[:2]
    
    if auto_detect:
        # Find contours
        contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return image
        
        # Find the largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Approximate contour to find corners
        epsilon = 0.02 * cv2.arcLength(largest_contour, True)
        approx = cv2.approxPolyDP(largest_contour, epsilon, True)
        
        if len(approx) == 4:
            corners = approx.reshape(4, 2)
        else:
            # Use bounding rectangle if we can't find 4 corners
            rect = cv2.minAreaRect(largest_contour)
            corners = cv2.boxPoints(rect).astype(np.int32)
    
    if corners is None:
        return image
    
    # Order corners: top-left, top-right, bottom-right, bottom-left
    corners = order_corners(corners)
    
    # Define destination points
    dst_corners = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]
    ], dtype=np.float32)
    
    # Calculate perspective transform matrix
    matrix = cv2.getPerspectiveTransform(corners.astype(np.float32), dst_corners)
    
    # Apply perspective correction
    corrected = cv2.warpPerspective(image, matrix, (width, height))
    
    return corrected


def order_corners(corners):
    """
    Order corners in the order: top-left, top-right, bottom-right, bottom-left
    
    Args:
        corners: Array of 4 corner points
        
    Returns:
        Ordered corners
    """
    # Calculate center point
    center = np.mean(corners, axis=0)
    
    # Calculate angles from center
    angles = []
    for corner in corners:
        angle = math.atan2(corner[1] - center[1], corner[0] - center[0])
        angles.append((angle, corner))
    
    # Sort by angle
    angles.sort(key=lambda x: x[0])
    
    # Extract corners in order (starting from top-left, going clockwise)
    ordered = np.array([angle[1] for angle in angles])
    
    # Ensure we start with top-left corner
    # Find the corner with minimum sum of coordinates (top-left)
    sums = np.sum(ordered, axis=1)
    min_idx = np.argmin(sums)
    
    # Reorder starting from top-left
    ordered = np.roll(ordered, -min_idx, axis=0)
    
    return ordered


def rotate_image(image, angle, scale=1.0):
    """
    Rotate image by specified angle
    
    Args:
        image: Input image
        angle: Rotation angle in degrees (positive = counterclockwise)
        scale: Scale factor
        
    Returns:
        Rotated image
    """
    height, width = image.shape[:2]
    center = (width // 2, height // 2)
    
    # Create rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale)
    
    # Apply rotation
    rotated = cv2.warpAffine(image, rotation_matrix, (width, height))
    
    return rotated


def resize_image(image, target_size=None, scale_factor=None, interpolation=cv2.INTER_LINEAR):
    """
    Resize image to target size or by scale factor
    
    Args:
        image: Input image
        target_size: Target size as (width, height)
        scale_factor: Scale factor (if target_size is None)
        interpolation: Interpolation method
        
    Returns:
        Resized image
    """
    if target_size is not None:
        return cv2.resize(image, target_size, interpolation=interpolation)
    elif scale_factor is not None:
        height, width = image.shape[:2]
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        return cv2.resize(image, (new_width, new_height), interpolation=interpolation)
    else:
        raise ValueError("Either target_size or scale_factor must be provided")


def crop_to_content(image, padding=10):
    """
    Crop image to content with optional padding
    
    Args:
        image: Input binary image
        padding: Padding around content in pixels
        
    Returns:
        Cropped image
    """
    # Find non-zero pixels
    coords = cv2.findNonZero(image)
    
    if coords is None:
        return image
    
    # Get bounding box
    x, y, w, h = cv2.boundingRect(coords)
    
    # Add padding
    x = max(0, x - padding)
    y = max(0, y - padding)
    w = min(image.shape[1] - x, w + 2 * padding)
    h = min(image.shape[0] - y, h + 2 * padding)
    
    # Crop image
    cropped = image[y:y+h, x:x+w]
    
    return cropped


def normalize_size(image, target_height=300, maintain_aspect_ratio=True):
    """
    Normalize image size for consistent processing
    
    Args:
        image: Input image
        target_height: Target height in pixels
        maintain_aspect_ratio: Whether to maintain aspect ratio
        
    Returns:
        Normalized image
    """
    height, width = image.shape[:2]
    
    if maintain_aspect_ratio:
        scale = target_height / height
        new_width = int(width * scale)
        new_height = target_height
    else:
        new_width = target_height  # Square image
        new_height = target_height
    
    return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)


def geometric_correction_pipeline(image, operations=None):
    """
    Apply a sequence of geometric corrections
    
    Args:
        image: Input image
        operations: List of operations to apply
        
    Returns:
        Corrected image
    """
    if operations is None:
        # Default operations for barcode preprocessing
        operations = [
            ('crop_to_content', {'padding': 20}),
            ('correct_skew', {'angle_threshold': 0.5}),
            ('normalize_size', {'target_height': 300})
        ]
    
    result = image.copy()
    
    for operation, params in operations:
        if operation == 'correct_skew':
            result = correct_skew(result, **params)
        elif operation == 'correct_skew_hough':
            result = correct_skew_hough(result, **params)
        elif operation == 'correct_perspective':
            result = correct_perspective(result, **params)
        elif operation == 'rotate':
            result = rotate_image(result, **params)
        elif operation == 'resize':
            result = resize_image(result, **params)
        elif operation == 'crop_to_content':
            result = crop_to_content(result, **params)
        elif operation == 'normalize_size':
            result = normalize_size(result, **params)
        else:
            raise ValueError(f"Unknown operation: {operation}")
    
    return result 