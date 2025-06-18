#!/usr/bin/env python3
"""
Create real barcodes for testing with configurable parameters
"""

import os
import cv2
import numpy as np
from PIL import Image
import qrcode
from barcode import Code128, EAN13, Code39
from barcode.writer import ImageWriter
import random
import argparse
import json
from datetime import datetime
import sys


def create_qr_codes(count=8):
    """Create QR codes for testing"""
    qr_data_base = [
        "https://example.com",
        "Hello World",
        "Test QR Code 123",
        "OpenEvolve Project",
        "Computer Vision",
        "Barcode Processing",
        "Python OpenCV",
        "Machine Learning",
        "Data Science",
        "Artificial Intelligence",
        "Neural Networks",
        "Deep Learning",
        "Pattern Recognition",
        "Image Analysis",
        "Automation",
        "Research Lab"
    ]
    
    # Extend data if needed
    qr_data = qr_data_base * ((count // len(qr_data_base)) + 1)
    qr_data = qr_data[:count]
    
    qr_codes = []
    for i, data in enumerate(qr_data):
        qr = qrcode.QRCode(
            version=1,
            error_correction=qrcode.constants.ERROR_CORRECT_L,
            box_size=10,
            border=4,
        )
        qr.add_data(data)
        qr.make(fit=True)
        
        img = qr.make_image(fill_color="black", back_color="white")
        qr_codes.append((f"qr_{i+1:02d}_{data[:10].replace('/', '_').replace(' ', '-')}.png", img))
    
    return qr_codes


def create_code128_barcodes(count=8):
    """Create Code128 barcodes for testing"""
    barcode_data_base = [
        "1234567890",
        "HELLO-WORLD",
        "TEST-BARCODE",
        "SAMPLE-DATA",
        "EVOLUTION-",
        "OPENCV-DEMO",
        "COMPUTER-V",
        "IMAGE-PROC",
        "DATASET-01",
        "ALGORITHM-",
        "BENCHMARK-",
        "QUALITY-01",
        "METRICS-01",
        "VISION-LAB",
        "RESEARCH-",
        "AUTOMATE-1"
    ]
    
    # Extend data if needed
    barcode_data = barcode_data_base * ((count // len(barcode_data_base)) + 1)
    barcode_data = barcode_data[:count]
    
    barcodes = []
    for i, data in enumerate(barcode_data):
        try:
            code = Code128(data, writer=ImageWriter())
            # Save to a temporary buffer to get PIL Image
            from io import BytesIO
            buffer = BytesIO()
            code.write(buffer)
            buffer.seek(0)
            img = Image.open(buffer)
            barcodes.append((f"code128_{i+1:02d}_{data[:10]}.png", img))
        except Exception as e:
            print(f"Warning: Error creating barcode for {data}: {e}")
    
    return barcodes


def create_code39_barcodes(count=8):
    """Create Code39 barcodes for testing"""
    barcode_data_base = [
        "123456789",
        "HELLO-WORLD",
        "TEST-CODE39",
        "SAMPLE-39",
        "EVOLUTION",
        "OPENCV-39",
        "COMPUTER",
        "IMAGE-39",
        "DATASET-39",
        "ALGORITHM",
        "BENCHMARK",
        "QUALITY-39",
        "METRICS-39",
        "VISION-39",
        "RESEARCH",
        "AUTOMATE"
    ]
    
    # Extend data if needed
    barcode_data = barcode_data_base * ((count // len(barcode_data_base)) + 1)
    barcode_data = barcode_data[:count]
    
    barcodes = []
    for i, data in enumerate(barcode_data):
        try:
            code = Code39(data, writer=ImageWriter(), add_checksum=False)
            # Save to a temporary buffer to get PIL Image
            from io import BytesIO
            buffer = BytesIO()
            code.write(buffer)
            buffer.seek(0)
            img = Image.open(buffer)
            barcodes.append((f"code39_{i+1:02d}_{data[:10]}.png", img))
        except Exception as e:
            print(f"Warning: Error creating Code39 barcode for {data}: {e}")
    
    return barcodes



def add_degradation(image, degradation_type="noise", level=1.0, **kwargs):
    """Add various types of degradation to an image"""
    img_array = np.array(image)
    
    # Ensure image is in correct format
    if img_array.dtype == bool:
        img_array = img_array.astype(np.uint8) * 255
    elif img_array.dtype != np.uint8:
        img_array = img_array.astype(np.uint8)
    
    # Convert single channel to RGB if needed
    if len(img_array.shape) == 2:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
    elif len(img_array.shape) == 3 and img_array.shape[2] == 4:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
    
    if degradation_type == "gaussian_noise":
        noise = np.random.normal(0, 25 * level, img_array.shape).astype(np.uint8)
        degraded = cv2.add(img_array, noise)
    
    elif degradation_type == "salt_pepper_noise":
        degraded = img_array.copy()
        noise_ratio = 0.05 * level
        # Salt noise
        salt = np.random.random(img_array.shape[:2]) < noise_ratio / 2
        degraded[salt] = 255
        # Pepper noise
        pepper = np.random.random(img_array.shape[:2]) < noise_ratio / 2
        degraded[pepper] = 0
    
    elif degradation_type == "uniform_noise":
        noise = np.random.uniform(-50 * level, 50 * level, img_array.shape).astype(np.int16)
        degraded = np.clip(img_array.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    elif degradation_type == "gaussian_blur":
        kernel_size = max(3, int(3 * level))
        if kernel_size % 2 == 0:
            kernel_size += 1
        degraded = cv2.GaussianBlur(img_array, (kernel_size, kernel_size), 0)
    
    elif degradation_type == "motion_blur":
        kernel_size = max(5, int(5 * level))
        kernel = np.zeros((kernel_size, kernel_size))
        kernel[int((kernel_size-1)/2), :] = np.ones(kernel_size)
        kernel = kernel / kernel_size
        degraded = cv2.filter2D(img_array, -1, kernel)
    
    elif degradation_type == "rotation":
        h, w = img_array.shape[:2]
        center = (w // 2, h // 2)
        max_angle = kwargs.get('max_angle', 15)
        angle = random.uniform(-max_angle * level, max_angle * level)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        degraded = cv2.warpAffine(img_array, rotation_matrix, (w, h), borderValue=(255, 255, 255))
    
    elif degradation_type == "perspective":
        h, w = img_array.shape[:2]
        distortion = 0.1 * level
        src_points = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        dst_points = src_points.copy()
        for i in range(4):
            dst_points[i][0] += random.uniform(-w * distortion, w * distortion)
            dst_points[i][1] += random.uniform(-h * distortion, h * distortion)
        matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        degraded = cv2.warpPerspective(img_array, matrix, (w, h), borderValue=(255, 255, 255))
    
    elif degradation_type == "brightness":
        beta_change = kwargs.get('beta_change', -30)
        degraded = cv2.convertScaleAbs(img_array, alpha=1.0, beta=beta_change * level)
    
    elif degradation_type == "contrast":
        alpha_range = kwargs.get('alpha_range', 0.5)
        degraded = cv2.convertScaleAbs(img_array, alpha=alpha_range + (1 - alpha_range) * (1 - level), beta=0)
    
    elif degradation_type == "compression":
        # JPEG compression artifacts
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), max(10, int(100 - 80 * level))]
        _, encoded_img = cv2.imencode('.jpg', img_array, encode_param)
        degraded = cv2.imdecode(encoded_img, cv2.IMREAD_COLOR)
        if len(img_array.shape) == 2:  # Grayscale
            degraded = cv2.cvtColor(degraded, cv2.COLOR_BGR2GRAY)
    
    elif degradation_type == "distortion":
        h, w = img_array.shape[:2]
        # Barrel distortion
        k1 = 0.0005 * level
        k2 = 0.0001 * level
        
        map_x, map_y = np.meshgrid(np.arange(w), np.arange(h))
        map_x = map_x.astype(np.float32)
        map_y = map_y.astype(np.float32)
        
        # Center coordinates
        cx, cy = w / 2, h / 2
        
        # Normalize coordinates
        x_norm = (map_x - cx) / cx
        y_norm = (map_y - cy) / cy
        
        # Calculate radius
        r2 = x_norm**2 + y_norm**2
        
        # Apply distortion
        distortion_factor = 1 + k1 * r2 + k2 * r2**2
        map_x = cx + x_norm * distortion_factor * cx
        map_y = cy + y_norm * distortion_factor * cy
        
        degraded = cv2.remap(img_array, map_x, map_y, cv2.INTER_LINEAR, borderValue=(255, 255, 255))
    
    else:
        degraded = img_array
    
    return Image.fromarray(degraded)


def generate_real_barcodes(output_dir='sample_images', count=8, distortion_level=1.0, 
                          noise_types=None, include_qr=True, include_code128=True, 
                          include_code39=False, save_config=True):
    """Generate real barcodes with various degradations"""
    
    # Create timestamped subdirectory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_output_dir = os.path.join(output_dir, f"auto_{timestamp}")
    
    if os.path.exists(final_output_dir):
        # Clear existing images
        for file in os.listdir(final_output_dir):
            if file.endswith(('.png', '.jpg', '.jpeg')):
                os.remove(os.path.join(final_output_dir, file))
    else:
        os.makedirs(final_output_dir, exist_ok=True)
    
    print(f"Generating real barcodes in {final_output_dir}/...")
    
        # Generate codes
    all_codes = []
    if include_qr:
        qr_codes = create_qr_codes(count)
        all_codes.extend(qr_codes)
        print(f"Generated {len(qr_codes)} QR codes")

    if include_code128:
        code128_barcodes = create_code128_barcodes(count)
        all_codes.extend(code128_barcodes)
        print(f"Generated {len(code128_barcodes)} Code128 barcodes")
        
    if include_code39:
        code39_barcodes = create_code39_barcodes(count)
        all_codes.extend(code39_barcodes)
        print(f"Generated {len(code39_barcodes)} Code39 barcodes")
    
    # Default noise types
    if noise_types is None:
        noise_types = [
            ("clean", None, 0),
            ("gaussian_noise", "gaussian_noise", 0.8 * distortion_level),
            ("salt_pepper", "salt_pepper_noise", 0.6 * distortion_level),
            ("gaussian_blur", "gaussian_blur", 1.2 * distortion_level),
            ("motion_blur", "motion_blur", 1.0 * distortion_level),
            ("rotated", "rotation", 1.0 * distortion_level),
            ("perspective", "perspective", 0.8 * distortion_level),
            ("dark", "brightness", 1.0 * distortion_level),
            ("low_contrast", "contrast", 0.7 * distortion_level),
            ("compressed", "compression", 0.8 * distortion_level),
            ("distorted", "distortion", 0.6 * distortion_level),
        ]
    
    created_count = 0
    generation_config = {
        'timestamp': timestamp,
        'output_dir': final_output_dir,
        'count': count,
        'distortion_level': distortion_level,
        'include_qr': include_qr,
        'include_code128': include_code128,
        'include_code39': include_code39,
        'noise_types': noise_types,
        'total_base_codes': len(all_codes),
        'created_images': []
    }
    
    for filename, img in all_codes:
        base_name = os.path.splitext(filename)[0]
        
        for deg_name, deg_type, level in noise_types:
            try:
                if deg_type is None:
                    # Clean version
                    final_img = img
                else:
                    # Apply degradation
                    final_img = add_degradation(img, deg_type, level)
                
                # Save image
                output_filename = f"{deg_name}_{base_name}.png"
                output_path = os.path.join(final_output_dir, output_filename)
                final_img.save(output_path)
                
                print(f"  Created: {output_filename}")
                created_count += 1
                generation_config['created_images'].append(output_filename)
                
            except Exception as e:
                print(f"  Error creating {deg_name} version of {base_name}: {e}")
        
        # Create combined degradation
        try:
            combined = img.copy()
            combined = add_degradation(combined, "gaussian_noise", 0.5 * distortion_level)
            combined = add_degradation(combined, "gaussian_blur", 0.8 * distortion_level)
            combined = add_degradation(combined, "rotation", 0.6 * distortion_level)
            combined = add_degradation(combined, "brightness", 0.8 * distortion_level)
            
            combined_filename = f"combined_{base_name}.png"
            combined_path = os.path.join(final_output_dir, combined_filename)
            combined.save(combined_path)
            
            print(f"  Created: {combined_filename}")
            created_count += 1
            generation_config['created_images'].append(combined_filename)
            
        except Exception as e:
            print(f"  Error creating combined version of {base_name}: {e}")
    
    generation_config['total_created'] = created_count
    
    # Save configuration for reproducibility
    if save_config:
        config_path = os.path.join(final_output_dir, 'generation_config.json')
        with open(config_path, 'w') as f:
            json.dump(generation_config, f, indent=2)
        print(f"Saved generation config to: {config_path}")
    
    print(f"\nGenerated {created_count} barcode images in {final_output_dir}/")
    print("These are real barcodes that can be decoded by pyzbar.")
    
    return final_output_dir, generation_config


def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(description='Generate barcode images with configurable degradations')
    
    parser.add_argument('--count', type=int, default=8,
                       help='Number of base barcodes to generate per type (default: 8)')
    parser.add_argument('--output-dir', default='sample_images',
                       help='Output directory (default: sample_images)')
    parser.add_argument('--distortion-level', type=float, default=1.0,
                       help='Global distortion level multiplier (default: 1.0)')
    parser.add_argument('--noise-type', action='append', choices=[
        'gaussian_noise', 'salt_pepper_noise', 'uniform_noise',
        'gaussian_blur', 'motion_blur', 'rotation', 'perspective',
        'brightness', 'contrast', 'compression', 'distortion'
    ], help='Specific noise types to apply (can be used multiple times)')
    parser.add_argument('--no-qr', action='store_true',
                       help='Do not generate QR codes')
    parser.add_argument('--no-code128', action='store_true',
                       help='Do not generate Code128 barcodes')
    parser.add_argument('--code39', action='store_true',
                       help='Generate Code39 barcodes (disabled by default)')
    parser.add_argument('--no-config', action='store_true',
                       help='Do not save generation configuration')
    parser.add_argument('--quick-mode', action='store_true',
                       help='Quick mode for CI: generate minimal test data with reduced noise types')
    parser.add_argument('--types', type=str,
                       help='Comma-separated list of specific degradation types (e.g., "blurred,noisy,rotated")')
    
    args = parser.parse_args()
    
    # Prepare noise types
    noise_types = None
    if args.quick_mode:
        # Quick mode: minimal noise types for CI testing
        noise_types = [
            ("clean", None, 0),
            ("blurred", "gaussian_blur", 0.8 * args.distortion_level),
            ("noisy", "gaussian_noise", 0.6 * args.distortion_level),
            ("rotated", "rotation", 0.8 * args.distortion_level),
        ]
    elif args.types:
        # Custom types specified
        noise_types = [("clean", None, 0)]
        type_mapping = {
            'blurred': ("blurred", "gaussian_blur", 0.8),
            'noisy': ("noisy", "gaussian_noise", 0.6),
            'rotated': ("rotated", "rotation", 0.8),
            'dark': ("dark", "brightness", 1.0),
            'compressed': ("compressed", "compression", 0.8),
            'distorted': ("distorted", "distortion", 0.6),
            'motion': ("motion_blur", "motion_blur", 1.0),
            'contrast': ("low_contrast", "contrast", 0.7),
            'perspective': ("perspective", "perspective", 0.8),
            'salt_pepper': ("salt_pepper", "salt_pepper_noise", 0.6),
        }
        for typename in args.types.split(','):
            typename = typename.strip()
            if typename in type_mapping:
                name, deg_type, base_level = type_mapping[typename]
                noise_types.append((name, deg_type, base_level * args.distortion_level))
            else:
                print(f"Warning: Unknown noise type '{typename}', skipping...")
    elif args.noise_type:
        noise_types = [("clean", None, 0)]
        for nt in args.noise_type:
            noise_types.append((nt, nt, args.distortion_level))
    
    # Generate barcodes
    output_dir, config = generate_real_barcodes(
        output_dir=args.output_dir,
        count=args.count,
        distortion_level=args.distortion_level,
        noise_types=noise_types,
        include_qr=not args.no_qr,
        include_code128=not args.no_code128,
        include_code39=args.code39,
        save_config=not args.no_config
    )
    
    print(f"\nGeneration completed successfully!")
    print(f"Output directory: {output_dir}")
    print(f"Total images created: {config['total_created']}")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        # No arguments, run with defaults for backward compatibility
        generate_real_barcodes()
    else:
        main() 