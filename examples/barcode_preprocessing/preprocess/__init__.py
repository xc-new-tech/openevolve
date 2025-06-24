"""
Modular barcode image preprocessing package

This package provides modular components for barcode image preprocessing:
- denoise: Noise reduction algorithms
- enhance: Contrast and brightness enhancement
- binarize: Image binarization methods  
- morphology: Morphological operations
- geometry: Geometric corrections (skew, perspective)
- advanced_illumination: Advanced illumination correction algorithms
- pipeline: Configurable processing pipelines
- performance: Performance analysis and benchmarking tools
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

from .advanced_illumination import (
    AdvancedIllumination, process_illumination
)

from .advanced_denoise import (
    AdvancedDenoise, process_advanced_denoise
)

from .pipeline import (
    ProcessingPipeline, create_default_pipeline, create_enhanced_pipeline,
    create_qr_pipeline, load_pipeline_from_config, benchmark_pipelines
)

from .performance import (
    PerformanceMetrics, PerformanceProfiler, MemoryProfiler, BenchmarkSuite,
    profile_function, create_test_images, get_global_profiler, reset_global_profiler
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
    
    # Advanced Illumination
    'AdvancedIllumination', 'process_illumination',
    
    # Advanced Denoise
    'AdvancedDenoise', 'process_advanced_denoise',
    
    # Pipeline
    'ProcessingPipeline', 'create_default_pipeline', 'create_enhanced_pipeline',
    'create_qr_pipeline', 'load_pipeline_from_config', 'benchmark_pipelines',
    
    # Performance Analysis
    'PerformanceMetrics', 'PerformanceProfiler', 'MemoryProfiler', 'BenchmarkSuite',
    'profile_function', 'create_test_images', 'get_global_profiler', 'reset_global_profiler',
    
    # Performance Utilities
    'analyze_pipeline_performance', 'profile_preprocessing_step', 'compare_preprocessing_methods',
    'benchmark_all_modules', 'memory_usage_analysis'
] 

# Performance Analysis Utilities
import numpy as np
from typing import Dict, Any, List, Callable, Optional
import logging

logger = logging.getLogger(__name__)

def analyze_pipeline_performance(pipeline: 'ProcessingPipeline', 
                               test_images: Optional[List[np.ndarray]] = None,
                               iterations: int = 3) -> Dict[str, Any]:
    """
    Analyze the performance of a processing pipeline.
    
    Args:
        pipeline: The ProcessingPipeline to analyze
        test_images: List of test images. If None, synthetic images are created
        iterations: Number of iterations to run for each test image
        
    Returns:
        Dict containing performance analysis results
    """
    from .performance import BenchmarkSuite, create_test_images
    
    # Create test images if not provided
    if test_images is None:
        test_images = create_test_images(
            sizes=[(480, 640), (720, 1280), (1080, 1920)],
            noise_levels=[0.0, 0.1, 0.2]
        )
    
    # Create benchmark suite
    benchmark = BenchmarkSuite(test_images)
    
    # Define the pipeline function for benchmarking
    def pipeline_func(image):
        return pipeline.process(image)
    
    # Run benchmark
    results = benchmark.run_benchmark({'pipeline': pipeline_func}, iterations)
    
    # Add pipeline-specific information
    results['pipeline_info'] = {
        'name': getattr(pipeline, 'name', 'Unknown'),
        'steps': [step.__class__.__name__ for step in pipeline.steps] if hasattr(pipeline, 'steps') else [],
        'test_image_count': len(test_images),
        'iterations': iterations
    }
    
    return results

def profile_preprocessing_step(func: Callable, 
                             image: np.ndarray, 
                             step_name: Optional[str] = None) -> Dict[str, Any]:
    """
    Profile a single preprocessing step.
    
    Args:
        func: The preprocessing function to profile
        image: Input image
        step_name: Name of the preprocessing step
        
    Returns:
        Dict containing profiling results
    """
    from .performance import PerformanceProfiler
    
    profiler = PerformanceProfiler()
    name = step_name or func.__name__
    
    with profiler.profile(name, image.shape[:2]):
        result = func(image.copy())
    
    metrics = profiler.get_summary()
    
    return {
        'step_name': name,
        'input_shape': image.shape,
        'output_shape': result.shape if hasattr(result, 'shape') else None,
        'metrics': metrics.get(name, {}),
        'success': True
    }

def compare_preprocessing_methods(methods: Dict[str, Callable],
                                test_images: Optional[List[np.ndarray]] = None,
                                criteria: List[str] = ['speed', 'memory', 'reliability']) -> Dict[str, Any]:
    """
    Compare multiple preprocessing methods on the same dataset.
    
    Args:
        methods: Dictionary mapping method names to functions
        test_images: List of test images. If None, synthetic images are created
        criteria: List of comparison criteria ('speed', 'memory', 'reliability')
        
    Returns:
        Dict containing comparison results and rankings
    """
    from .performance import BenchmarkSuite, create_test_images
    
    # Create test images if not provided
    if test_images is None:
        test_images = create_test_images(
            sizes=[(480, 640), (720, 1280)],
            noise_levels=[0.0, 0.1]
        )
    
    # Create benchmark suite
    benchmark = BenchmarkSuite(test_images)
    
    # Run comparison
    results = benchmark.compare_algorithms(methods)
    
    # Filter rankings by requested criteria
    filtered_rankings = {
        criterion: results['rankings'][criterion] 
        for criterion in criteria 
        if criterion in results['rankings']
    }
    
    return {
        'comparison_results': results['detailed_results'],
        'rankings': filtered_rankings,
        'summary_statistics': results['summary'],
        'test_configuration': {
            'num_test_images': len(test_images),
            'image_sizes': [img.shape for img in test_images],
            'methods_compared': list(methods.keys())
        }
    }

def benchmark_all_modules(test_images: Optional[List[np.ndarray]] = None) -> Dict[str, Any]:
    """
    Benchmark all available preprocessing modules.
    
    Args:
        test_images: List of test images. If None, synthetic images are created
        
    Returns:
        Dict containing benchmark results for all modules
    """
    from .performance import create_test_images
    
    # Create test images if not provided
    if test_images is None:
        test_images = create_test_images(
            sizes=[(480, 640)],
            noise_levels=[0.1]
        )
    
    # Define methods to benchmark from each module
    methods = {
        # Denoise methods
        'gaussian_blur': gaussian_blur_denoise,
        'median_blur': median_blur_denoise,
        'bilateral_filter': bilateral_filter_denoise,
        
        # Enhancement methods
        'histogram_eq': histogram_equalization,
        'clahe': clahe_enhancement,
        'gamma_correction': lambda img: gamma_correction(img, gamma=1.2),
        
        # Binarization methods
        'adaptive_threshold': adaptive_threshold,
        'otsu_threshold': otsu_threshold,
        
        # Morphology methods
        'opening': morphological_opening,
        'closing': morphological_closing,
        
        # Geometry methods
        'correct_skew': correct_skew,
        'resize': lambda img: resize_image(img, (640, 480))
    }
    
    return compare_preprocessing_methods(methods, test_images)

def memory_usage_analysis(pipeline: 'ProcessingPipeline',
                        image_sizes: List[tuple] = [(480, 640), (720, 1280), (1080, 1920)]) -> Dict[str, Any]:
    """
    Analyze memory usage patterns for different image sizes.
    
    Args:
        pipeline: The ProcessingPipeline to analyze
        image_sizes: List of (height, width) tuples to test
        
    Returns:
        Dict containing memory usage analysis
    """
    from .performance import MemoryProfiler, create_test_images
    
    memory_profiler = MemoryProfiler()
    results = {}
    
    memory_profiler.take_snapshot("baseline")
    
    for i, size in enumerate(image_sizes):
        # Create test image of specific size
        test_image = create_test_images([size], [0.0])[0]
        
        memory_profiler.take_snapshot(f"before_size_{size}")
        
        # Process image
        try:
            result = pipeline.process(test_image)
            memory_profiler.take_snapshot(f"after_size_{size}")
            
            # Calculate memory difference
            memory_diff = memory_profiler.get_memory_diff(-2, -1)
            
            results[f"size_{size}"] = {
                'input_size_mb': test_image.nbytes / 1024 / 1024,
                'output_size_mb': result.nbytes / 1024 / 1024 if hasattr(result, 'nbytes') else 0,
                'memory_usage_mb': memory_diff.get('memory_diff_mb', 0),
                'success': True
            }
            
        except Exception as e:
            results[f"size_{size}"] = {
                'error': str(e),
                'success': False
            }
    
    return {
        'memory_analysis': results,
        'total_snapshots': len(memory_profiler.snapshots),
        'baseline_memory': memory_profiler.baseline_memory
    } 