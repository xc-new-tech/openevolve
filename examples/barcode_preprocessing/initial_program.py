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

def resize_if_needed(image, min_dim=320, max_dim=900):
    # Slightly increase min_dim to help with tiny barcodes, lower max_dim for speed
    h, w = image.shape
    scale = 1.0
    if min(h, w) < min_dim:
        scale = min_dim / min(h, w)
        interp = cv2.INTER_CUBIC
    elif max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        interp = cv2.INTER_AREA
    else:
        interp = cv2.INTER_LINEAR
    if scale != 1.0:
        # Fix bug: cv2.resize expects (width, height)
        new_w, new_h = int(round(w * scale)), int(round(h * scale))
        return cv2.resize(image, (new_w, new_h), interpolation=interp)
    return image

def estimate_noise(image):
    return cv2.Laplacian(image, cv2.CV_64F).std()

def adaptive_denoise(image):
    noise = estimate_noise(image)
    if noise > 18:
        # More aggressive for heavy noise, but avoid over-smoothing
        denoised = cv2.fastNlMeansDenoising(image, None, h=13, templateWindowSize=7, searchWindowSize=21)
        # Sharpen after denoising to restore barcode edges
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        denoised = cv2.filter2D(denoised, -1, kernel)
        # Extra: If denoising blurs too much, blend with original for edge recovery
        denoised = cv2.addWeighted(denoised, 0.85, image, 0.15, 0)
        return denoised
    elif noise > 8:
        # Use bilateral filter for better edge preservation
        denoised = cv2.bilateralFilter(image, d=7, sigmaColor=48, sigmaSpace=48)
        # Gentle sharpening
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        denoised = cv2.filter2D(denoised, -1, kernel)
        # Extra: Blend with original for edge detail
        denoised = cv2.addWeighted(denoised, 0.9, image, 0.1, 0)
        return denoised
    else:
        return image

def apply_clahe_if_needed(image, std_thresh=38):
    if np.std(image) < std_thresh:
        clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
        return clahe.apply(image)
    return image

def otsu_binarize(image):
    # Use Gaussian blur to suppress noise before Otsu
    blur = cv2.GaussianBlur(image, (3, 3), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Ensure barcode is black on white (invert if needed)
    white = np.sum(th == 255)
    black = np.sum(th == 0)
    if black > white:
        th = 255 - th
    # Fill small holes to help with fragmented bars
    th_filled = th.copy()
    contours, _ = cv2.findContours(255-th, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 80:
            cv2.drawContours(th_filled, [cnt], 0, 0, -1)
    return th_filled

def adaptive_binarize(image):
    # Use a slightly larger block size for more robustness to uneven lighting
    th = cv2.adaptiveThreshold(
        image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 35, 11
    )
    # Invert if needed
    white = np.sum(th == 255)
    black = np.sum(th == 0)
    if black > white:
        th = 255 - th
    # Fill small holes as in Otsu
    th_filled = th.copy()
    contours, _ = cv2.findContours(255-th, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 5 < area < 80:
            cv2.drawContours(th_filled, [cnt], 0, 0, -1)
    return th_filled

def barcode_structure_score(bin_img):
    # Checks for barcode-like structure: wide contours, parallel lines, or dense vertical transitions
    h, w = bin_img.shape
    # Use only the central region to avoid noise at borders
    crop = bin_img[h//10:h-h//10, w//10:w-w//10]
    vertical_transitions = np.sum(np.abs(np.diff(crop, axis=1)) > 0)
    vertical_score = vertical_transitions / crop.shape[0]
    # Count of wide horizontal contours (for 1D barcodes)
    contours, _ = cv2.findContours(crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    wide_cnts = [cnt for cnt in contours if cv2.boundingRect(cnt)[2] > w * 0.18]
    score = len(wide_cnts) + vertical_score
    return score

def binarize_with_fallback(image):
    th = otsu_binarize(image)
    score = barcode_structure_score(th)
    # If structure is poor, fallback to adaptive
    if score < 6:
        th2 = adaptive_binarize(image)
        score2 = barcode_structure_score(th2)
        # Try a hybrid: combine Otsu and adaptive with bitwise OR if both are poor
        if score2 > score:
            return th2
        elif score2 < 4 and score < 4:
            hybrid = cv2.bitwise_or(th, th2)
            if barcode_structure_score(hybrid) > max(score, score2):
                return hybrid
        # NEW: If both are poor, try adaptiveThreshold with mean method as last resort
        th3 = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 33, 9)
        if np.sum(th3 == 0) > np.sum(th == 0) * 0.8:  # Only if it gives more black pixels (bars)
            return th3
    return th

def estimate_skew_angle(bin_img):
    # Use both Hough lines and projection profile, return the best
    h, w = bin_img.shape
    # Focus on central region for more reliable skew
    crop = bin_img[h//10:h-h//10, w//10:w-w//10]
    edges = cv2.Canny(crop, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=28, minLineLength=w//6, maxLineGap=18)
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
    rotated = cv2.warpAffine(image, rotmat, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REPLICATE)
    return rotated

def is_2d_barcode(bin_img):
    # Heuristic: if many (10+) roughly square contours, likely QR/2D
    contours, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    squares = 0
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if 0.7 < w/h < 1.3 and w*h > 60:
            squares += 1
    # If image is nearly square, and many squares, more likely 2D
    h, w = bin_img.shape
    if abs(h-w) < min(h, w)*0.2 and squares >= 7:
        return True
    return squares >= 10

def morph_clean(image):
    # Adaptive cleaning: 2D barcodes need closing in both axes, 1D mostly horizontal closing
    h, w = image.shape
    if is_2d_barcode(image):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        cleaned = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel, iterations=1)
        # Remove small noise by opening if many tiny blobs
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        small_blobs = [cnt for cnt in contours if cv2.contourArea(cnt) < 40]
        if len(small_blobs) > 8:
            cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8), iterations=1)
        # Extra: If still many small blobs, erode/dilate sequence
        if len(small_blobs) > 15:
            cleaned = cv2.erode(cleaned, np.ones((2,2), np.uint8), iterations=1)
            cleaned = cv2.dilate(cleaned, np.ones((2,2), np.uint8), iterations=2)
        return cleaned
    else:
        # Use a larger kernel for closing if image is wide, helps with fragmented bars
        kernel_width = max(7, w//30)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_width, 1))
        closed = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel, iterations=1)
        # Remove small blobs (noise) with opening if needed
        opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
        # Extra: Try both opening and closing in sequence if still many blobs
        cnt_closed, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnt_opened, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(cnt_opened) < len(cnt_closed):
            result = opened
        else:
            result = closed
        # If still >20 blobs, try a second closing with smaller kernel
        if len(cnt_opened) > 20 or len(cnt_closed) > 20:
            kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))
            result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel2, iterations=1)
        return result

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
    if len(contours) > 20 or barcode_structure_score(bin_img) < 7:
        bin_img = morph_clean(bin_img)
    # If still low contrast after all, try CLAHE+binarize again (rare)
    if np.std(bin_img) < 17:
        image2 = apply_clahe_if_needed(image, std_thresh=20)
        bin_img = binarize_with_fallback(image2)
    # Extra: If output is still too small (tiny images), upscale for decoder
    if min(bin_img.shape) < 200:
        scale = 200.0 / min(bin_img.shape)
        bin_img = cv2.resize(bin_img, (int(bin_img.shape[1]*scale), int(bin_img.shape[0]*scale)), interpolation=cv2.INTER_NEAREST)
    # Final step: ensure output is strictly 0/255
    bin_img = np.where(bin_img > 127, 255, 0).astype(np.uint8)
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


