"""
Performance analysis and monitoring utilities for barcode preprocessing.
"""

import time
import psutil
import tracemalloc
import functools
import logging
from typing import Dict, Any, List, Callable, Optional, Tuple
from dataclasses import dataclass
from contextlib import contextmanager
import numpy as np
import cv2

logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
    """Performance metrics container."""
    execution_time: float
    memory_usage: float  # MB
    cpu_usage: float  # percentage
    image_size: Tuple[int, int]
    operation_name: str
    timestamp: float

class PerformanceProfiler:
    """Advanced performance profiler for image processing operations."""
    
    def __init__(self, enable_memory_tracking: bool = True):
        self.enable_memory_tracking = enable_memory_tracking
        self.metrics_history: List[PerformanceMetrics] = []
        self.process = psutil.Process()
        
        if enable_memory_tracking:
            tracemalloc.start()
    
    @contextmanager
    def profile(self, operation_name: str, image_shape: Optional[Tuple[int, int]] = None):
        """Context manager for profiling operations."""
        # Start measurements
        start_time = time.perf_counter()
        start_cpu = self.process.cpu_percent()
        
        if self.enable_memory_tracking:
            tracemalloc.clear_traces()
            start_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        
        try:
            yield
        finally:
            # End measurements
            end_time = time.perf_counter()
            execution_time = end_time - start_time
            
            # CPU usage (average during operation)
            cpu_usage = self.process.cpu_percent()
            
            # Memory usage
            if self.enable_memory_tracking:
                current, peak = tracemalloc.get_traced_memory()
                memory_usage = current / 1024 / 1024  # MB
            else:
                memory_usage = self.process.memory_info().rss / 1024 / 1024
            
            # Store metrics
            metrics = PerformanceMetrics(
                execution_time=execution_time,
                memory_usage=memory_usage,
                cpu_usage=cpu_usage,
                image_size=image_shape if image_shape else (0, 0),
                operation_name=operation_name,
                timestamp=time.time()
            )
            
            self.metrics_history.append(metrics)
            
            logger.debug(f"Performance: {operation_name} - "
                        f"Time: {execution_time:.3f}s, "
                        f"Memory: {memory_usage:.1f}MB, "
                        f"CPU: {cpu_usage:.1f}%")
    
    def get_summary(self) -> Dict[str, Any]:
        """Get performance summary statistics."""
        if not self.metrics_history:
            return {}
        
        # Group by operation
        operations = {}
        for metric in self.metrics_history:
            op_name = metric.operation_name
            if op_name not in operations:
                operations[op_name] = []
            operations[op_name].append(metric)
        
        # Calculate statistics
        summary = {}
        for op_name, metrics in operations.items():
            times = [m.execution_time for m in metrics]
            memories = [m.memory_usage for m in metrics]
            cpus = [m.cpu_usage for m in metrics]
            
            summary[op_name] = {
                'count': len(metrics),
                'execution_time': {
                    'mean': np.mean(times),
                    'std': np.std(times),
                    'min': np.min(times),
                    'max': np.max(times)
                },
                'memory_usage': {
                    'mean': np.mean(memories),
                    'std': np.std(memories),
                    'min': np.min(memories),
                    'max': np.max(memories)
                },
                'cpu_usage': {
                    'mean': np.mean(cpus),
                    'std': np.std(cpus),
                    'min': np.min(cpus),
                    'max': np.max(cpus)
                }
            }
        
        return summary
    
    def clear_history(self):
        """Clear performance metrics history."""
        self.metrics_history.clear()
    
    def export_metrics(self, filename: str):
        """Export metrics to CSV file."""
        import csv
        
        with open(filename, 'w', newline='') as csvfile:
            fieldnames = ['timestamp', 'operation_name', 'execution_time', 
                         'memory_usage', 'cpu_usage', 'image_width', 'image_height']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for metric in self.metrics_history:
                writer.writerow({
                    'timestamp': metric.timestamp,
                    'operation_name': metric.operation_name,
                    'execution_time': metric.execution_time,
                    'memory_usage': metric.memory_usage,
                    'cpu_usage': metric.cpu_usage,
                    'image_width': metric.image_size[0],
                    'image_height': metric.image_size[1]
                })

def profile_function(operation_name: Optional[str] = None, 
                    profiler: Optional[PerformanceProfiler] = None):
    """Decorator for profiling individual functions."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Use global profiler if none provided
            if profiler is None:
                global_profiler = getattr(wrapper, '_profiler', None)
                if global_profiler is None:
                    global_profiler = PerformanceProfiler()
                    wrapper._profiler = global_profiler
                current_profiler = global_profiler
            else:
                current_profiler = profiler
            
            # Determine operation name
            op_name = operation_name or func.__name__
            
            # Try to get image shape from arguments
            image_shape = None
            for arg in args:
                if isinstance(arg, np.ndarray) and len(arg.shape) >= 2:
                    image_shape = arg.shape[:2]
                    break
            
            with current_profiler.profile(op_name, image_shape):
                return func(*args, **kwargs)
        
        return wrapper
    return decorator

class MemoryProfiler:
    """Specialized memory profiler for tracking memory usage patterns."""
    
    def __init__(self):
        self.snapshots = []
        self.baseline_memory = None
        
    def take_snapshot(self, label: str = ""):
        """Take a memory snapshot."""
        if not tracemalloc.is_tracing():
            tracemalloc.start()
        
        snapshot = tracemalloc.take_snapshot()
        current_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        self.snapshots.append({
            'label': label,
            'snapshot': snapshot,
            'memory_mb': current_memory,
            'timestamp': time.time()
        })
        
        if self.baseline_memory is None:
            self.baseline_memory = current_memory
    
    def get_memory_diff(self, snapshot1_idx: int = 0, snapshot2_idx: int = -1) -> Dict[str, Any]:
        """Get memory difference between two snapshots."""
        if len(self.snapshots) < 2:
            return {}
        
        snap1 = self.snapshots[snapshot1_idx]
        snap2 = self.snapshots[snapshot2_idx]
        
        # Compare snapshots
        top_stats = snap2['snapshot'].compare_to(snap1['snapshot'], 'lineno')
        
        # Get top memory consumers
        top_10 = top_stats[:10]
        
        return {
            'memory_diff_mb': snap2['memory_mb'] - snap1['memory_mb'],
            'label1': snap1['label'],
            'label2': snap2['label'],
            'top_consumers': [
                {
                    'filename': stat.traceback.format()[0],
                    'size_diff_mb': stat.size_diff / 1024 / 1024,
                    'count_diff': stat.count_diff
                }
                for stat in top_10
            ]
        }
    
    def clear_snapshots(self):
        """Clear all snapshots."""
        self.snapshots.clear()
        self.baseline_memory = None

class BenchmarkSuite:
    """Benchmark suite for comparing different preprocessing approaches."""
    
    def __init__(self, test_images: List[np.ndarray]):
        self.test_images = test_images
        self.results = {}
        self.profiler = PerformanceProfiler()
    
    def run_benchmark(self, function_map: Dict[str, Callable], 
                     iterations: int = 5) -> Dict[str, Any]:
        """Run benchmark on multiple functions."""
        results = {}
        
        for func_name, func in function_map.items():
            func_results = []
            
            for iteration in range(iterations):
                iteration_results = []
                
                for img_idx, image in enumerate(self.test_images):
                    with self.profiler.profile(f"{func_name}_img_{img_idx}"):
                        try:
                            result = func(image.copy())
                            iteration_results.append({
                                'success': True,
                                'error': None
                            })
                        except Exception as e:
                            iteration_results.append({
                                'success': False,
                                'error': str(e)
                            })
                
                func_results.append(iteration_results)
            
            # Calculate success rate
            total_runs = len(self.test_images) * iterations
            successful_runs = sum(
                1 for iter_results in func_results 
                for result in iter_results 
                if result['success']
            )
            
            results[func_name] = {
                'success_rate': successful_runs / total_runs,
                'total_runs': total_runs,
                'successful_runs': successful_runs,
                'performance_metrics': self.profiler.get_summary().get(f"{func_name}_img_0", {})
            }
        
        return results
    
    def compare_algorithms(self, algorithms: Dict[str, Callable]) -> Dict[str, Any]:
        """Compare different algorithms on the same dataset."""
        benchmark_results = self.run_benchmark(algorithms)
        
        # Rank algorithms by different criteria
        rankings = {}
        
        # Rank by execution time
        time_ranking = sorted(
            benchmark_results.items(),
            key=lambda x: x[1]['performance_metrics'].get('execution_time', {}).get('mean', float('inf'))
        )
        rankings['speed'] = [name for name, _ in time_ranking]
        
        # Rank by memory usage
        memory_ranking = sorted(
            benchmark_results.items(),
            key=lambda x: x[1]['performance_metrics'].get('memory_usage', {}).get('mean', float('inf'))
        )
        rankings['memory'] = [name for name, _ in memory_ranking]
        
        # Rank by success rate
        success_ranking = sorted(
            benchmark_results.items(),
            key=lambda x: x[1]['success_rate'],
            reverse=True
        )
        rankings['reliability'] = [name for name, _ in success_ranking]
        
        return {
            'detailed_results': benchmark_results,
            'rankings': rankings,
            'summary': self.profiler.get_summary()
        }

def create_test_images(sizes: List[Tuple[int, int]], 
                      noise_levels: List[float] = [0.0, 0.1, 0.2]) -> List[np.ndarray]:
    """Create synthetic test images for benchmarking."""
    test_images = []
    
    for size in sizes:
        for noise_level in noise_levels:
            # Create a simple barcode-like pattern
            image = np.zeros(size, dtype=np.uint8)
            
            # Add vertical bars
            bar_width = size[1] // 20
            for i in range(0, size[1], bar_width * 2):
                image[:, i:i+bar_width] = 255
            
            # Add noise if specified
            if noise_level > 0:
                noise = np.random.normal(0, noise_level * 255, size)
                image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
            
            test_images.append(image)
    
    return test_images

# Global profiler instance
global_profiler = PerformanceProfiler()

def get_global_profiler() -> PerformanceProfiler:
    """Get the global profiler instance."""
    return global_profiler

def reset_global_profiler():
    """Reset the global profiler."""
    global global_profiler
    global_profiler = PerformanceProfiler() 