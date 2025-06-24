"""
管道性能优化模块
优化算法执行顺序、实现并行处理、内存优化和缓存机制
"""

import numpy as np
import cv2
import time
import threading
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from typing import Tuple, Dict, List, Any, Optional, Union, Callable
import logging
import hashlib
import os
import pickle
from functools import wraps, lru_cache
import psutil

logger = logging.getLogger(__name__)


class MemoryManager:
    """内存管理器"""
    
    def __init__(self, max_memory_mb: int = 1024):
        self.max_memory_mb = max_memory_mb
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.current_usage = 0
        self.temp_arrays = []
        
    def get_memory_usage(self) -> Dict[str, float]:
        """获取当前内存使用情况"""
        process = psutil.Process()
        memory_info = process.memory_info()
        
        return {
            'rss_mb': memory_info.rss / 1024 / 1024,  # 物理内存
            'vms_mb': memory_info.vms / 1024 / 1024,  # 虚拟内存
            'percent': process.memory_percent(),       # 使用百分比
            'available_mb': psutil.virtual_memory().available / 1024 / 1024
        }
    
    def allocate_optimized_array(self, shape: Tuple, dtype=np.uint8) -> np.ndarray:
        """优化的数组分配"""
        # 计算所需内存
        size_bytes = np.prod(shape) * np.dtype(dtype).itemsize
        
        # 检查内存限制
        memory_usage = self.get_memory_usage()
        if memory_usage['available_mb'] * 1024 * 1024 < size_bytes * 2:
            # 内存不足，使用内存映射
            return self._create_memory_mapped_array(shape, dtype)
        
        # 正常分配
        array = np.empty(shape, dtype=dtype)
        self.temp_arrays.append(array)
        return array
    
    def _create_memory_mapped_array(self, shape: Tuple, dtype=np.uint8) -> np.ndarray:
        """创建内存映射数组"""
        import tempfile
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        array = np.memmap(temp_file.name, dtype=dtype, mode='w+', shape=shape)
        return array
    
    def cleanup(self):
        """清理临时内存"""
        self.temp_arrays.clear()
        
    def optimize_image_copy(self, image: np.ndarray) -> np.ndarray:
        """优化的图像复制"""
        # 如果图像连续，直接view
        if image.flags['C_CONTIGUOUS']:
            return image.view()
        
        # 否则创建优化的副本
        optimized = self.allocate_optimized_array(image.shape, image.dtype)
        optimized[:] = image
        return optimized


class CacheManager:
    """缓存管理器"""
    
    def __init__(self, cache_dir: str = ".cache", max_size_mb: int = 500):
        self.cache_dir = cache_dir
        self.max_size_mb = max_size_mb
        self.max_size_bytes = max_size_mb * 1024 * 1024
        
        # 创建缓存目录
        os.makedirs(cache_dir, exist_ok=True)
        
        # 内存缓存（小结果）
        self.memory_cache = {}
        self.memory_cache_size = 0
        self.max_memory_cache_mb = 100
    
    def _generate_cache_key(self, image: np.ndarray, algorithm: str, params: Dict) -> str:
        """生成缓存键"""
        # 基于图像内容和参数生成哈希
        image_hash = hashlib.md5(image.tobytes()).hexdigest()[:16]
        params_str = str(sorted(params.items())) if params else ""
        params_hash = hashlib.md5(params_str.encode()).hexdigest()[:8]
        
        return f"{algorithm}_{image_hash}_{params_hash}"
    
    def get_from_cache(self, image: np.ndarray, algorithm: str, params: Dict = None) -> Optional[np.ndarray]:
        """从缓存获取结果"""
        if params is None:
            params = {}
            
        cache_key = self._generate_cache_key(image, algorithm, params)
        
        # 先检查内存缓存
        if cache_key in self.memory_cache:
            logger.debug(f"Cache hit (memory): {cache_key}")
            return self.memory_cache[cache_key].copy()
        
        # 检查磁盘缓存
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.npz")
        if os.path.exists(cache_file):
            try:
                cached_data = np.load(cache_file)
                result = cached_data['result']
                logger.debug(f"Cache hit (disk): {cache_key}")
                
                # 如果结果较小，加入内存缓存
                result_size = result.nbytes
                if result_size < 10 * 1024 * 1024:  # < 10MB
                    self._add_to_memory_cache(cache_key, result.copy())
                
                return result.copy()
            except Exception as e:
                logger.warning(f"Cache read error: {e}")
                # 删除损坏的缓存文件
                os.remove(cache_file)
        
        return None
    
    def store_in_cache(self, image: np.ndarray, algorithm: str, result: np.ndarray, params: Dict = None):
        """存储结果到缓存"""
        if params is None:
            params = {}
            
        cache_key = self._generate_cache_key(image, algorithm, params)
        result_size = result.nbytes
        
        # 内存缓存（小结果）
        if result_size < 5 * 1024 * 1024:  # < 5MB
            self._add_to_memory_cache(cache_key, result.copy())
        
        # 磁盘缓存
        try:
            cache_file = os.path.join(self.cache_dir, f"{cache_key}.npz")
            np.savez_compressed(cache_file, result=result, params=params)
            logger.debug(f"Cached result: {cache_key} ({result_size / 1024:.1f} KB)")
        except Exception as e:
            logger.warning(f"Cache write error: {e}")
    
    def _add_to_memory_cache(self, key: str, result: np.ndarray):
        """添加到内存缓存"""
        result_size = result.nbytes
        
        # 检查内存缓存大小限制
        while (self.memory_cache_size + result_size) > (self.max_memory_cache_mb * 1024 * 1024):
            if not self.memory_cache:
                break
            # 移除最老的条目（简单FIFO）
            oldest_key = next(iter(self.memory_cache))
            old_size = self.memory_cache[oldest_key].nbytes
            del self.memory_cache[oldest_key]
            self.memory_cache_size -= old_size
        
        self.memory_cache[key] = result
        self.memory_cache_size += result_size
    
    def clear_cache(self):
        """清理缓存"""
        # 清理内存缓存
        self.memory_cache.clear()
        self.memory_cache_size = 0
        
        # 清理磁盘缓存
        for filename in os.listdir(self.cache_dir):
            if filename.endswith('.npz'):
                os.remove(os.path.join(self.cache_dir, filename))


class ParallelProcessor:
    """并行处理器"""
    
    def __init__(self, max_workers: Optional[int] = None):
        self.max_workers = max_workers or min(4, mp.cpu_count())
        self.use_threading = True  # 默认使用线程（IO密集型）
        
    def process_batch_parallel(self, images: List[np.ndarray], 
                             process_func: Callable, 
                             use_processes: bool = False) -> List[np.ndarray]:
        """并行处理图像批次"""
        if len(images) <= 1:
            return [process_func(img) for img in images]
        
        executor_class = ProcessPoolExecutor if use_processes else ThreadPoolExecutor
        
        with executor_class(max_workers=self.max_workers) as executor:
            results = list(executor.map(process_func, images))
        
        return results
    
    def process_pipeline_stages_parallel(self, image: np.ndarray, 
                                       stage_functions: List[Callable]) -> np.ndarray:
        """并行处理管道中的独立阶段"""
        if len(stage_functions) <= 1:
            current = image.copy()
            for func in stage_functions:
                current = func(current)
            return current
        
        # 识别可并行化的阶段
        parallelizable_stages = self._identify_parallelizable_stages(stage_functions)
        
        # 执行并行阶段
        return self._execute_parallel_stages(image, stage_functions, parallelizable_stages)
    
    def _identify_parallelizable_stages(self, stage_functions: List[Callable]) -> List[List[int]]:
        """识别可以并行执行的阶段组"""
        # 简单策略：假设相邻的某些类型算法可以并行
        parallelizable_groups = []
        current_group = [0]
        
        for i in range(1, len(stage_functions)):
            # 这里可以根据算法类型来判断是否可以并行
            # 当前简化为顺序执行
            parallelizable_groups.append([i-1])
            current_group = [i]
        
        if current_group:
            parallelizable_groups.append(current_group)
        
        return parallelizable_groups
    
    def _execute_parallel_stages(self, image: np.ndarray, 
                               stage_functions: List[Callable],
                               parallelizable_stages: List[List[int]]) -> np.ndarray:
        """执行并行阶段"""
        current = image.copy()
        
        for stage_group in parallelizable_stages:
            if len(stage_group) == 1:
                # 单一阶段，直接执行
                current = stage_functions[stage_group[0]](current)
            else:
                # 多个阶段，并行执行
                with ThreadPoolExecutor(max_workers=len(stage_group)) as executor:
                    futures = []
                    for stage_idx in stage_group:
                        future = executor.submit(stage_functions[stage_idx], current.copy())
                        futures.append(future)
                    
                    # 等待第一个完成的结果（简化策略）
                    current = futures[0].result()
        
        return current


class PerformanceProfiler:
    """性能分析器"""
    
    def __init__(self):
        self.timings = {}
        self.memory_usage = {}
        
    def start_timing(self, stage_name: str):
        """开始计时"""
        if stage_name not in self.timings:
            self.timings[stage_name] = []
        
        setattr(self, f"_start_time_{stage_name}", time.perf_counter())
    
    def end_timing(self, stage_name: str):
        """结束计时"""
        start_time = getattr(self, f"_start_time_{stage_name}", None)
        if start_time is not None:
            elapsed = time.perf_counter() - start_time
            self.timings[stage_name].append(elapsed)
            
            # 记录内存使用
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            if stage_name not in self.memory_usage:
                self.memory_usage[stage_name] = []
            self.memory_usage[stage_name].append(memory_mb)
            
            delattr(self, f"_start_time_{stage_name}")
    
    def get_performance_report(self) -> Dict[str, Any]:
        """获取性能报告"""
        report = {}
        
        for stage, times in self.timings.items():
            if times:
                report[stage] = {
                    'avg_time_ms': np.mean(times) * 1000,
                    'min_time_ms': np.min(times) * 1000,
                    'max_time_ms': np.max(times) * 1000,
                    'std_time_ms': np.std(times) * 1000,
                    'total_time_ms': np.sum(times) * 1000,
                    'count': len(times)
                }
                
                if stage in self.memory_usage:
                    memories = self.memory_usage[stage]
                    report[stage].update({
                        'avg_memory_mb': np.mean(memories),
                        'max_memory_mb': np.max(memories),
                        'min_memory_mb': np.min(memories)
                    })
        
        return report
    
    def print_performance_summary(self):
        """打印性能摘要"""
        report = self.get_performance_report()
        
        print("\n=== Performance Summary ===")
        for stage, metrics in report.items():
            print(f"\n{stage}:")
            print(f"  Average time: {metrics['avg_time_ms']:.2f}ms")
            print(f"  Memory usage: {metrics.get('avg_memory_mb', 0):.1f}MB")
            print(f"  Count: {metrics['count']}")


def cached_algorithm(cache_manager: CacheManager, algorithm_name: str):
    """缓存装饰器"""
    def decorator(func):
        @wraps(func)
        def wrapper(image: np.ndarray, **kwargs):
            # 尝试从缓存获取
            cached_result = cache_manager.get_from_cache(image, algorithm_name, kwargs)
            if cached_result is not None:
                return cached_result
            
            # 执行算法
            result = func(image, **kwargs)
            
            # 存储到缓存
            cache_manager.store_in_cache(image, algorithm_name, result, kwargs)
            
            return result
        return wrapper
    return decorator


def timed_algorithm(profiler: PerformanceProfiler, stage_name: str):
    """计时装饰器"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            profiler.start_timing(stage_name)
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                profiler.end_timing(stage_name)
        return wrapper
    return decorator


class OptimizedPipelineManager:
    """优化的管道管理器"""
    
    def __init__(self, enable_cache: bool = True, enable_profiling: bool = True,
                 max_workers: Optional[int] = None):
        self.cache_manager = CacheManager() if enable_cache else None
        self.profiler = PerformanceProfiler() if enable_profiling else None
        self.parallel_processor = ParallelProcessor(max_workers)
        self.memory_manager = MemoryManager()
        
        # 算法性能历史
        self.algorithm_performance = {}
        
        # GPU 支持检测
        self.gpu_available = self._check_gpu_support()
        
    def _check_gpu_support(self) -> bool:
        """检测GPU支持"""
        try:
            import cupy as cp
            # 测试GPU可用性
            cp.cuda.Device(0).compute_capability
            return True
        except:
            return False
    
    def optimize_algorithm_order(self, algorithms: List[str], 
                               image_characteristics: Dict) -> List[str]:
        """根据图像特征和历史性能优化算法顺序"""
        if not self.algorithm_performance:
            return algorithms  # 首次运行，使用默认顺序
        
        # 根据历史性能数据排序
        algorithm_scores = {}
        for alg in algorithms:
            if alg in self.algorithm_performance:
                perf_data = self.algorithm_performance[alg]
                # 计算评分（时间越短分数越高）
                avg_time = np.mean(perf_data.get('times', [1.0]))
                success_rate = np.mean(perf_data.get('successes', [1.0]))
                score = success_rate / max(avg_time, 0.001)  # 避免除零
                algorithm_scores[alg] = score
            else:
                algorithm_scores[alg] = 0.5  # 默认分数
        
        # 按分数排序（分数高的优先）
        sorted_algorithms = sorted(algorithms, key=lambda x: algorithm_scores[x], reverse=True)
        
        logger.info(f"Optimized algorithm order: {sorted_algorithms}")
        return sorted_algorithms
    
    def process_with_optimization(self, image: np.ndarray, 
                                algorithms: Dict[str, Callable],
                                image_characteristics: Dict = None) -> np.ndarray:
        """使用优化策略处理图像"""
        if image_characteristics is None:
            image_characteristics = self._analyze_image_characteristics(image)
        
        # 优化算法顺序
        algorithm_names = list(algorithms.keys())
        optimized_order = self.optimize_algorithm_order(algorithm_names, image_characteristics)
        
        current_image = image.copy()
        
        for alg_name in optimized_order:
            if alg_name not in algorithms:
                continue
                
            alg_func = algorithms[alg_name]
            
            # 应用缓存和计时装饰器
            if self.cache_manager:
                alg_func = cached_algorithm(self.cache_manager, alg_name)(alg_func)
            
            if self.profiler:
                alg_func = timed_algorithm(self.profiler, alg_name)(alg_func)
            
            # 执行算法
            start_time = time.perf_counter()
            try:
                current_image = alg_func(current_image)
                success = True
            except Exception as e:
                logger.error(f"Algorithm {alg_name} failed: {e}")
                success = False
                continue
            
            end_time = time.perf_counter()
            
            # 记录性能数据
            self._record_algorithm_performance(alg_name, end_time - start_time, success)
        
        return current_image
    
    def batch_process_optimized(self, images: List[np.ndarray],
                              algorithms: Dict[str, Callable]) -> List[np.ndarray]:
        """批量优化处理"""
        
        def process_single(image):
            return self.process_with_optimization(image, algorithms)
        
        return self.parallel_processor.process_batch_parallel(images, process_single)
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """获取性能摘要"""
        summary = {}
        
        if self.profiler:
            summary['profiler'] = self.profiler.get_performance_report()
        
        if self.algorithm_performance:
            summary['algorithm_performance'] = {}
            for alg, data in self.algorithm_performance.items():
                if data['times']:
                    summary['algorithm_performance'][alg] = {
                        'avg_time_ms': np.mean(data['times']) * 1000,
                        'success_rate': np.mean(data['successes']),
                        'total_runs': len(data['times'])
                    }
        
        if self.cache_manager:
            summary['cache'] = {
                'memory_cache_size': len(self.cache_manager.memory_cache),
                'memory_usage_mb': self.cache_manager.memory_cache_size / 1024 / 1024
            }
        
        return summary
    
    def cleanup(self):
        """清理资源"""
        if self.cache_manager:
            self.cache_manager.clear_cache()
        
        if self.memory_manager:
            self.memory_manager.cleanup()
    
    def _analyze_image_characteristics(self, image: np.ndarray) -> Dict[str, Any]:
        """分析图像特征"""
        return {
            'size': image.shape,
            'area': image.shape[0] * image.shape[1],
            'dtype': str(image.dtype),
            'channels': len(image.shape),
            'mean_intensity': np.mean(image),
            'std_intensity': np.std(image)
        }
    
    def _record_algorithm_performance(self, algorithm: str, execution_time: float, success: bool):
        """记录算法性能"""
        if algorithm not in self.algorithm_performance:
            self.algorithm_performance[algorithm] = {
                'times': [],
                'successes': []
            }
        
        self.algorithm_performance[algorithm]['times'].append(execution_time)
        self.algorithm_performance[algorithm]['successes'].append(1.0 if success else 0.0)


# GPU加速支持类
class GPUAccelerator:
    """GPU加速器"""
    
    def __init__(self):
        self.gpu_available = False
        self.cupy_available = False
        
        try:
            import cupy as cp
            self.cp = cp
            self.cupy_available = True
            # 测试GPU设备
            with cp.cuda.Device(0):
                test_array = cp.zeros((10, 10))
            self.gpu_available = True
            logger.info("GPU acceleration available")
        except Exception as e:
            logger.info(f"GPU acceleration not available: {e}")
    
    def to_gpu(self, image: np.ndarray) -> 'cp.ndarray':
        """将图像转移到GPU"""
        if not self.gpu_available:
            return image
        
        return self.cp.asarray(image)
    
    def to_cpu(self, gpu_image: 'cp.ndarray') -> np.ndarray:
        """将图像从GPU转移到CPU"""
        if not self.gpu_available:
            return gpu_image
        
        return self.cp.asnumpy(gpu_image)
    
    def gpu_gaussian_blur(self, image: np.ndarray, kernel_size: int = 5, sigma: float = 1.0) -> np.ndarray:
        """GPU加速的高斯模糊"""
        if not self.gpu_available:
            return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
        
        gpu_image = self.to_gpu(image)
        
        # 使用CuPy的过滤器
        from cupyx.scipy import ndimage
        result = ndimage.gaussian_filter(gpu_image, sigma)
        
        return self.to_cpu(result)
    
    def gpu_median_filter(self, image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
        """GPU加速的中值滤波"""
        if not self.gpu_available:
            return cv2.medianBlur(image, kernel_size)
        
        gpu_image = self.to_gpu(image)
        
        # 使用CuPy的中值滤波
        from cupyx.scipy import ndimage
        result = ndimage.median_filter(gpu_image, size=kernel_size)
        
        return self.to_cpu(result)
    
    def gpu_histogram_equalization(self, image: np.ndarray) -> np.ndarray:
        """GPU加速的直方图均衡化"""
        if not self.gpu_available:
            return cv2.equalizeHist(image)
        
        gpu_image = self.to_gpu(image)
        
        # 简化的直方图均衡化
        hist, bins = self.cp.histogram(gpu_image, bins=256, range=(0, 256))
        cdf = self.cp.cumsum(hist)
        cdf_normalized = cdf * 255 / cdf[-1]
        
        # 映射像素值
        result = self.cp.interp(gpu_image.flatten(), bins[:-1], cdf_normalized)
        result = result.reshape(gpu_image.shape).astype(self.cp.uint8)
        
        return self.to_cpu(result)


class SmartAlgorithmSelector:
    """智能算法选择器"""
    
    def __init__(self):
        self.algorithm_database = {
            'high_noise': ['bilateral_filter', 'non_local_means', 'adaptive_wiener'],
            'low_contrast': ['clahe_enhancement', 'histogram_equalization', 'gamma_correction'],
            'geometric_distortion': ['correct_skew', 'correct_perspective'],
            'small_image': ['super_resolution', 'resize_image'],
            'large_image': ['gaussian_blur', 'morphological_opening'],
            'barcode_specific': ['adaptive_threshold', 'morphological_cleanup']
        }
        
        self.performance_weights = {
            'speed': 0.4,
            'quality': 0.4,
            'memory': 0.2
        }
    
    def analyze_image_problems(self, image: np.ndarray) -> List[str]:
        """分析图像问题"""
        problems = []
        
        # 噪声检测
        if self._detect_noise(image):
            problems.append('high_noise')
        
        # 对比度检测
        if self._detect_low_contrast(image):
            problems.append('low_contrast')
        
        # 尺寸检测
        height, width = image.shape[:2]
        if height * width < 100000:  # < 100k pixels
            problems.append('small_image')
        elif height * width > 2000000:  # > 2M pixels
            problems.append('large_image')
        
        # 几何畸变检测（简化）
        if self._detect_skew(image):
            problems.append('geometric_distortion')
        
        # 条码特定问题
        problems.append('barcode_specific')
        
        return problems
    
    def recommend_algorithms(self, image: np.ndarray, 
                           performance_priority: str = 'balanced') -> List[str]:
        """推荐算法"""
        problems = self.analyze_image_problems(image)
        
        recommended = []
        for problem in problems:
            if problem in self.algorithm_database:
                recommended.extend(self.algorithm_database[problem])
        
        # 去重并排序
        unique_algorithms = list(dict.fromkeys(recommended))
        
        # 根据性能优先级排序
        if performance_priority == 'speed':
            return self._sort_by_speed(unique_algorithms)
        elif performance_priority == 'quality':
            return self._sort_by_quality(unique_algorithms)
        else:  # balanced
            return unique_algorithms
    
    def _detect_noise(self, image: np.ndarray) -> bool:
        """检测噪声"""
        # 使用拉普拉斯算子检测噪声
        laplacian_var = cv2.Laplacian(image, cv2.CV_64F).var()
        return laplacian_var < 100  # 阈值可调整
    
    def _detect_low_contrast(self, image: np.ndarray) -> bool:
        """检测低对比度"""
        return image.std() < 50  # 标准差低表示对比度低
    
    def _detect_skew(self, image: np.ndarray) -> bool:
        """检测倾斜（简化版）"""
        edges = cv2.Canny(image, 50, 150)
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
        
        if lines is not None:
            angles = []
            for line in lines[:10]:  # 只检查前10条线
                rho, theta = line[0]
                angle = theta * 180 / np.pi
                angles.append(angle)
            
            if angles:
                angle_std = np.std(angles)
                return angle_std > 5  # 角度标准差大表示可能有倾斜
        
        return False
    
    def _sort_by_speed(self, algorithms: List[str]) -> List[str]:
        """按速度排序算法"""
        # 简化的速度排序（实际应基于性能测试）
        speed_order = {
            'gaussian_blur': 1,
            'median_blur': 2,
            'bilateral_filter': 3,
            'histogram_equalization': 4,
            'adaptive_threshold': 5,
            'morphological_opening': 6,
            'clahe_enhancement': 7,
            'correct_skew': 8,
            'non_local_means': 9,
            'super_resolution': 10
        }
        
        return sorted(algorithms, key=lambda x: speed_order.get(x, 99))
    
    def _sort_by_quality(self, algorithms: List[str]) -> List[str]:
        """按质量排序算法"""
        # 简化的质量排序
        quality_order = {
            'non_local_means': 1,
            'bilateral_filter': 2,
            'clahe_enhancement': 3,
            'super_resolution': 4,
            'adaptive_threshold': 5,
            'correct_perspective': 6,
            'morphological_cleanup': 7,
            'gaussian_blur': 8,
            'median_blur': 9,
            'histogram_equalization': 10
        }
        
        return sorted(algorithms, key=lambda x: quality_order.get(x, 99))


# 示例使用
if __name__ == "__main__":
    # 创建测试数据
    def mock_denoise(image):
        return cv2.GaussianBlur(image, (5, 5), 0)
    
    def mock_enhance(image):
        return cv2.equalizeHist(image)
    
    # 测试优化管道
    test_image = np.random.randint(0, 255, (480, 640), dtype=np.uint8)
    
    optimizer = OptimizedPipelineManager()
    algorithms = {
        'denoise': mock_denoise,
        'enhance': mock_enhance
    }
    
    result = optimizer.process_with_optimization(test_image, algorithms)
    print("Optimization test completed")
    print("Performance summary:", optimizer.get_performance_summary())
    
    # 测试GPU加速（如果可用）
    gpu_accelerator = GPUAccelerator()
    if gpu_accelerator.gpu_available:
        gpu_result = gpu_accelerator.gpu_gaussian_blur(test_image)
        print("GPU acceleration test completed")
    
    # 测试智能算法选择
    selector = SmartAlgorithmSelector()
    problems = selector.analyze_image_problems(test_image)
    recommendations = selector.recommend_algorithms(test_image, 'speed')
    print(f"Detected problems: {problems}")
    print(f"Recommended algorithms: {recommendations}")
    
    optimizer.cleanup() 