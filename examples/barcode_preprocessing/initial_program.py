"""
Barcode image preprocessing pipeline for improved decoding success rate

**This version further improves:**
- Streamlined, readable pipeline with minimal branching for speed.
- More aggressive denoising for noisy images, but still preserves barcode edges.
- CLAHE is now adaptive: only applied if contrast is low, avoiding overprocessing.
- Improved binarization fallback: checks for barcode-like structure, not just contour count.
- Deskewing uses both Hough lines and projection profile, picks the best.
- Morphological cleaning is adaptive: kernel size and operation depend on image type (linear vs. 2D).
- All steps are performed only if necessary, reducing processing time and unnecessary changes.
- Handles very small or very large images robustly.
"""

import cv2
import numpy as np
import os

def read_grayscale(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    return img

def resize_if_needed(image, min_dim=300, max_dim=1200):
    h, w = image.shape
    scale = 1.0
    if min(h, w) < min_dim:
        scale = min_dim / min(h, w)
    elif max(h, w) > max_dim:
        scale = max_dim / max(h, w)
    if scale != 1.0:
        new_w, new_h = int(w * scale), int(h * scale)
        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    return image

def estimate_noise(image):
    return cv2.Laplacian(image, cv2.CV_64F).std()

def adaptive_denoise(image):
    noise = estimate_noise(image)
    if noise > 18:
        # More aggressive for heavy noise
        return cv2.fastNlMeansDenoising(image, None, h=13, templateWindowSize=9, searchWindowSize=27)
    elif noise > 8:
        return cv2.medianBlur(image, 3)
    else:
        return image

def apply_clahe_if_needed(image, std_thresh=38):
    if np.std(image) < std_thresh:
        clahe = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8, 8))
        return clahe.apply(image)
    return image

def otsu_binarize(image):
    _, th = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Ensure barcode is black on white (invert if needed)
    white = np.sum(th == 255)
    black = np.sum(th == 0)
    if black > white:
        th = 255 - th
    return th

def adaptive_binarize(image):
    th = cv2.adaptiveThreshold(
        image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25, 10
    )
    # Invert if needed
    white = np.sum(th == 255)
    black = np.sum(th == 0)
    if black > white:
        th = 255 - th
    return th

def barcode_structure_score(bin_img):
    # Checks for barcode-like structure: wide contours, parallel lines, or dense vertical transitions
    h, w = bin_img.shape
    vertical_transitions = np.sum(np.abs(np.diff(bin_img, axis=1)) > 0)
    vertical_score = vertical_transitions / h
    # Count of wide horizontal contours (for 1D barcodes)
    contours, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    wide_cnts = [cnt for cnt in contours if cv2.boundingRect(cnt)[2] > w * 0.2]
    score = len(wide_cnts) + vertical_score
    return score

def binarize_with_fallback(image):
    th = otsu_binarize(image)
    score = barcode_structure_score(th)
    # If structure is poor, fallback to adaptive
    if score < 6:
        th2 = adaptive_binarize(image)
        score2 = barcode_structure_score(th2)
        if score2 > score:
            return th2
    return th

def estimate_skew_angle(bin_img):
    # Use both Hough lines and projection profile, return the best
    h, w = bin_img.shape
    edges = cv2.Canny(bin_img, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=35, minLineLength=w//4, maxLineGap=15)
    angles = []
    if lines is not None and len(lines) > 0:
        for x1, y1, x2, y2 in lines[:, 0]:
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            if -45 < angle < 45:
                angles.append(angle)
    if angles:
        median_angle = np.median(angles)
    else:
        median_angle = 0

    # Projection profile method for fallback or verification
    best_score = -np.inf
    best_angle = 0
    for angle in np.linspace(-8, 8, 17):
        M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1)
        rotated = cv2.warpAffine(bin_img, M, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REPLICATE)
        profile = np.sum(rotated == 0, axis=1)
        score = profile.std()
        if score > best_score:
            best_score = score
            best_angle = angle
    # Choose the angle with largest magnitude if both methods agree direction
    if abs(median_angle) > 0.7 and np.sign(median_angle) == np.sign(best_angle):
        if abs(median_angle) > abs(best_angle):
            return median_angle
        else:
            return best_angle
    elif abs(best_angle) > 0.7:
        return best_angle
    elif abs(median_angle) > 0.7:
        return median_angle
    return 0

def deskew(image):
    angle = estimate_skew_angle(image)
    if abs(angle) < 0.7:
        return image
    (h, w) = image.shape
    center = (w // 2, h // 2)
    rotmat = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, rotmat, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return rotated

def is_2d_barcode(bin_img):
    # Heuristic: if many (10+) roughly square contours, likely QR/2D
    contours, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    squares = 0
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if 0.7 < w/h < 1.3 and w*h > 80:
            squares += 1
    return squares >= 10

def morph_clean(image):
    # Adaptive cleaning: 2D barcodes need closing in both axes, 1D mostly horizontal closing
    h, w = image.shape
    if is_2d_barcode(image):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        cleaned = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel, iterations=1)
        return cleaned
    else:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(5, w//40), 1))
        closed = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel, iterations=1)
        # Remove small blobs (noise) with opening if needed
        opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
        # Choose the version with fewer small contours
        cnt_closed, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnt_opened, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(cnt_opened) < len(cnt_closed):
            return opened
        else:
            return closed

def preprocess_barcode_image(image_path, output_path=None):
    """
    Main pipeline: Preprocess a barcode image to improve decoding success rate.
    Args:
        image_path: Path to input barcode image
        output_path: Optional path to save processed image
    Returns:
        Processed image as numpy array
    """
    image = read_grayscale(image_path)
    image = resize_if_needed(image)
    image = adaptive_denoise(image)
    image = apply_clahe_if_needed(image)
    bin_img = binarize_with_fallback(image)
    bin_img = deskew(bin_img)
    # If too fragmented, clean morphologically
    contours, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 25 or barcode_structure_score(bin_img) < 7:
        bin_img = morph_clean(bin_img)
    # If still low contrast after all, try CLAHE+binarize again (rare)
    if np.std(bin_img) < 17:
        image2 = apply_clahe_if_needed(image, std_thresh=25)
        bin_img = binarize_with_fallback(image2)
    if output_path:
        cv2.imwrite(output_path, bin_img)
    return bin_img

def process_barcode_pipeline(image_path, output_path=None):
    return preprocess_barcode_image(image_path, output_path)

def run_preprocessing(input_dir, output_dir=None):
    """
    Run preprocessing on all images in a directory

    Args:
        input_dir: Directory containing barcode images
        output_dir: Directory to save processed images (optional)

    Returns:
        List of processed image paths
    """
    if not os.path.exists(input_dir):
        raise ValueError(f"Input directory does not exist: {input_dir}")

    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    processed_paths = []

    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, f"processed_{filename}") if output_dir else None
            try:
                process_barcode_pipeline(input_path, output_path)
                processed_paths.append(output_path if output_path else input_path)
                print(f"Processed: {filename}")
            except Exception as e:
                print(f"Error processing {filename}: {e}")

    return processed_paths

if __name__ == "__main__":
    sample_dir = "sample_images"
    output_dir = "processed_images"

    if os.path.exists(sample_dir):
        processed = run_preprocessing(sample_dir, output_dir)
        print(f"Processed {len(processed)} images")
    else:
        print(f"Sample directory '{sample_dir}' not found")
        print("Please add some barcode images to the sample_images directory")
