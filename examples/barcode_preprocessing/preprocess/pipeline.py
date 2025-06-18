"""
Configurable preprocessing pipelines for barcode image processing
"""
import cv2
import numpy as np
import json
import os
from datetime import datetime
import logging

from .denoise import *
from .enhance import *
from .binarize import *
from .morphology import *
from .geometry import *


class ProcessingPipeline:
    """
    Configurable image processing pipeline
    """
    
    def __init__(self, pipeline_config=None, name="default"):
        """
        Initialize processing pipeline
        
        Args:
            pipeline_config: List of processing steps or path to config file
            name: Pipeline name for logging
        """
        self.name = name
        self.steps = []
        self.metadata = {
            'created': datetime.now().isoformat(),
            'name': name,
            'version': '1.0.0'
        }
        
        if pipeline_config is not None:
            self.load_config(pipeline_config)
        
        self.logger = logging.getLogger(f"pipeline.{name}")
    
    def load_config(self, config):
        """
        Load pipeline configuration
        
        Args:
            config: List of steps or path to JSON config file
        """
        if isinstance(config, str):
            # Load from file
            with open(config, 'r') as f:
                config_data = json.load(f)
            self.steps = config_data.get('steps', [])
            self.metadata.update(config_data.get('metadata', {}))
        elif isinstance(config, list):
            # Direct list of steps
            self.steps = config
        else:
            raise ValueError("Config must be a list of steps or path to JSON file")
    
    def save_config(self, filepath):
        """
        Save pipeline configuration to file
        
        Args:
            filepath: Path to save configuration
        """
        config_data = {
            'metadata': self.metadata,
            'steps': self.steps
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(config_data, f, indent=2)
    
    def add_step(self, module, function, params=None, condition=None):
        """
        Add a processing step to the pipeline
        
        Args:
            module: Module name (e.g., 'denoise', 'enhance')
            function: Function name within the module
            params: Parameters for the function
            condition: Optional condition for applying the step
        """
        step = {
            'module': module,
            'function': function,
            'params': params or {},
            'condition': condition
        }
        self.steps.append(step)
    
    def process_image(self, image, debug=False):
        """
        Process image through the pipeline
        
        Args:
            image: Input image (BGR or grayscale)
            debug: Whether to return intermediate results
            
        Returns:
            Processed image (and intermediate results if debug=True)
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            current = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            current = image.copy()
        
        intermediate_results = [('input', current.copy())] if debug else []
        
        for i, step in enumerate(self.steps):
            try:
                # Check condition if specified
                if step.get('condition'):
                    if not self._evaluate_condition(current, step['condition']):
                        self.logger.debug(f"Skipping step {i}: condition not met")
                        continue
                
                # Get function from module
                func = self._get_function(step['module'], step['function'])
                
                # Apply function
                params = step.get('params', {})
                current = func(current, **params)
                
                if debug:
                    step_name = f"{step['module']}.{step['function']}"
                    intermediate_results.append((step_name, current.copy()))
                
                self.logger.debug(f"Applied step {i}: {step['module']}.{step['function']}")
                
            except Exception as e:
                self.logger.error(f"Error in step {i}: {e}")
                if debug:
                    intermediate_results.append((f"error_{i}", current.copy()))
        
        if debug:
            return current, intermediate_results
        return current
    
    def _get_function(self, module, function_name):
        """
        Get function from module
        
        Args:
            module: Module name
            function_name: Function name
            
        Returns:
            Function object
        """
        module_map = {
            'denoise': globals(),
            'enhance': globals(),
            'binarize': globals(),
            'morphology': globals(),
            'geometry': globals()
        }
        
        if module not in module_map:
            raise ValueError(f"Unknown module: {module}")
        
        module_funcs = module_map[module]
        if function_name not in module_funcs:
            raise ValueError(f"Unknown function: {function_name} in module {module}")
        
        return module_funcs[function_name]
    
    def _evaluate_condition(self, image, condition):
        """
        Evaluate condition for applying a step
        
        Args:
            image: Current image
            condition: Condition dictionary
            
        Returns:
            Boolean result
        """
        condition_type = condition.get('type')
        
        if condition_type == 'mean_intensity':
            mean_val = np.mean(image)
            operator = condition.get('operator', '>')
            threshold = condition.get('value', 128)
            
            if operator == '>':
                return mean_val > threshold
            elif operator == '<':
                return mean_val < threshold
            elif operator == '==':
                return abs(mean_val - threshold) < 5
        
        elif condition_type == 'std_intensity':
            std_val = np.std(image)
            operator = condition.get('operator', '>')
            threshold = condition.get('value', 30)
            
            if operator == '>':
                return std_val > threshold
            elif operator == '<':
                return std_val < threshold
        
        return True  # Default: always apply
    
    def optimize_parameters(self, images, ground_truth=None, metric='auto'):
        """
        Optimize pipeline parameters using sample images
        
        Args:
            images: List of sample images
            ground_truth: List of expected results (optional)
            metric: Optimization metric
            
        Returns:
            Optimized pipeline configuration
        """
        # This is a placeholder for parameter optimization
        # In a real implementation, this would use techniques like:
        # - Grid search
        # - Genetic algorithms
        # - Bayesian optimization
        
        self.logger.info("Parameter optimization not yet implemented")
        return self.steps


def create_default_pipeline():
    """
    Create a default preprocessing pipeline for barcodes
    
    Returns:
        ProcessingPipeline instance
    """
    pipeline = ProcessingPipeline(name="default_barcode")
    
    # Step 1: Denoise
    pipeline.add_step('denoise', 'median_blur_denoise', {'kernel_size': 3})
    
    # Step 2: Enhance contrast (conditional on low contrast)
    pipeline.add_step('enhance', 'clahe_enhancement', 
                     {'clip_limit': 2.0, 'tile_grid_size': (8, 8)},
                     condition={'type': 'std_intensity', 'operator': '<', 'value': 40})
    
    # Step 3: Binarize
    pipeline.add_step('binarize', 'adaptive_binarization_pipeline', {'method': 'auto'})
    
    # Step 4: Geometric correction
    pipeline.add_step('geometry', 'correct_skew', {'angle_threshold': 1.0})
    
    # Step 5: Morphological cleanup
    pipeline.add_step('morphology', 'adaptive_morphological_cleanup', {'image_type': 'barcode'})
    
    return pipeline


def create_enhanced_pipeline():
    """
    Create an enhanced preprocessing pipeline with more steps
    
    Returns:
        ProcessingPipeline instance
    """
    pipeline = ProcessingPipeline(name="enhanced_barcode")
    
    # Step 1: Geometric normalization
    pipeline.add_step('geometry', 'crop_to_content', {'padding': 20})
    
    # Step 2: Advanced denoising
    pipeline.add_step('denoise', 'bilateral_filter_denoise', 
                     {'d': 9, 'sigma_color': 75, 'sigma_space': 75})
    
    # Step 3: Multi-scale enhancement
    pipeline.add_step('enhance', 'multi_scale_enhancement', {'scales': [1, 2, 4]})
    
    # Step 4: Advanced binarization
    pipeline.add_step('binarize', 'sauvola_threshold', {'window_size': 15, 'k': 0.2})
    
    # Step 5: Skew correction with Hough transform
    pipeline.add_step('geometry', 'correct_skew_hough', {'angle_range': (-45, 45)})
    
    # Step 6: Advanced morphological operations
    pipeline.add_step('morphology', 'morphological_cleanup', {
        'operation_sequence': [
            ('opening', {'kernel_size': (2, 1)}),
            ('closing', {'kernel_size': (1, 3)}),
            ('remove_small_objects', {'min_size': 25}),
            ('fill_holes', {'max_hole_size': 40})
        ]
    })
    
    # Step 7: Final size normalization
    pipeline.add_step('geometry', 'normalize_size', {'target_height': 300})
    
    return pipeline


def create_qr_pipeline():
    """
    Create a preprocessing pipeline optimized for QR codes
    
    Returns:
        ProcessingPipeline instance
    """
    pipeline = ProcessingPipeline(name="qr_code")
    
    # Step 1: Denoise
    pipeline.add_step('denoise', 'gaussian_blur_denoise', {'kernel_size': (3, 3)})
    
    # Step 2: Enhance for dark images
    pipeline.add_step('enhance', 'gamma_correction', {'gamma': 0.8},
                     condition={'type': 'mean_intensity', 'operator': '<', 'value': 100})
    
    # Step 3: Adaptive binarization
    pipeline.add_step('binarize', 'adaptive_threshold', 
                     {'block_size': 15, 'c': 5})
    
    # Step 4: Perspective correction
    pipeline.add_step('geometry', 'correct_perspective', {'auto_detect': True})
    
    # Step 5: QR-specific morphology
    pipeline.add_step('morphology', 'adaptive_morphological_cleanup', {'image_type': 'qr'})
    
    return pipeline


def load_pipeline_from_config(config_path):
    """
    Load pipeline from configuration file
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        ProcessingPipeline instance
    """
    pipeline = ProcessingPipeline()
    pipeline.load_config(config_path)
    return pipeline


def benchmark_pipelines(pipelines, test_images, output_dir=None):
    """
    Benchmark multiple pipelines on test images
    
    Args:
        pipelines: List of ProcessingPipeline instances
        test_images: List of test image paths
        output_dir: Directory to save results (optional)
        
    Returns:
        Benchmark results
    """
    results = {}
    
    for pipeline in pipelines:
        pipeline_results = {
            'name': pipeline.name,
            'processing_times': [],
            'success_count': 0,
            'error_count': 0
        }
        
        for img_path in test_images:
            try:
                # Load image
                image = cv2.imread(img_path)
                if image is None:
                    continue
                
                # Time processing
                start_time = datetime.now()
                processed = pipeline.process_image(image)
                end_time = datetime.now()
                
                processing_time = (end_time - start_time).total_seconds()
                pipeline_results['processing_times'].append(processing_time)
                pipeline_results['success_count'] += 1
                
                # Save result if output directory specified
                if output_dir:
                    output_path = os.path.join(output_dir, pipeline.name, os.path.basename(img_path))
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)
                    cv2.imwrite(output_path, processed)
                
            except Exception as e:
                pipeline_results['error_count'] += 1
                logging.error(f"Error processing {img_path} with {pipeline.name}: {e}")
        
        # Calculate statistics
        if pipeline_results['processing_times']:
            pipeline_results['avg_time'] = np.mean(pipeline_results['processing_times'])
            pipeline_results['std_time'] = np.std(pipeline_results['processing_times'])
        
        results[pipeline.name] = pipeline_results
    
    return results 