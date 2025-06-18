"""
Modular barcode image preprocessing package

This package provides modular components for barcode image preprocessing:
- denoise: Noise reduction algorithms
- enhance: Contrast and brightness enhancement
- binarize: Image binarization methods  
- morphology: Morphological operations
- geometry: Geometric corrections (skew, perspective)
- pipeline: Configurable processing pipelines
"""

from .denoise import (
    gaussian_blur_denoise, median_blur_denoise, bilateral_filter_denoise,
    adaptive_wiener_filter, non_local_means_denoise
)

from .enhance import (
    histogram_equalization, clahe_enhancement, gamma_correction,
    adaptive_gamma_correction, multi_scale_enhancement, unsharp_masking
)

from .binarize import (
    adaptive_threshold, otsu_threshold, global_threshold, triangle_threshold,
    multi_threshold, sauvola_threshold, niblack_threshold, wolf_threshold,
    adaptive_binarization_pipeline
)

from .morphology import (
    morphological_opening, morphological_closing, morphological_gradient,
    morphological_tophat, morphological_blackhat, erosion, dilation,
    remove_small_objects, fill_holes, skeletonize, morphological_cleanup,
    adaptive_morphological_cleanup
)

from .geometry import (
    correct_skew, correct_skew_hough, correct_perspective, order_corners,
    rotate_image, resize_image, crop_to_content, normalize_size,
    geometric_correction_pipeline
)

from .pipeline import (
    ProcessingPipeline, create_default_pipeline, create_enhanced_pipeline,
    create_qr_pipeline, load_pipeline_from_config, benchmark_pipelines
)

__version__ = "1.0.0"
__all__ = [
    # Denoise
    'gaussian_blur_denoise', 'median_blur_denoise', 'bilateral_filter_denoise',
    'adaptive_wiener_filter', 'non_local_means_denoise',
    
    # Enhance
    'histogram_equalization', 'clahe_enhancement', 'gamma_correction',
    'adaptive_gamma_correction', 'multi_scale_enhancement', 'unsharp_masking',
    
    # Binarize
    'adaptive_threshold', 'otsu_threshold', 'global_threshold', 'triangle_threshold',
    'multi_threshold', 'sauvola_threshold', 'niblack_threshold', 'wolf_threshold',
    'adaptive_binarization_pipeline',
    
    # Morphology
    'morphological_opening', 'morphological_closing', 'morphological_gradient',
    'morphological_tophat', 'morphological_blackhat', 'erosion', 'dilation',
    'remove_small_objects', 'fill_holes', 'skeletonize', 'morphological_cleanup',
    'adaptive_morphological_cleanup',
    
    # Geometry
    'correct_skew', 'correct_skew_hough', 'correct_perspective', 'order_corners',
    'rotate_image', 'resize_image', 'crop_to_content', 'normalize_size',
    'geometric_correction_pipeline',
    
    # Pipeline
    'ProcessingPipeline', 'create_default_pipeline', 'create_enhanced_pipeline',
    'create_qr_pipeline', 'load_pipeline_from_config', 'benchmark_pipelines'
] 