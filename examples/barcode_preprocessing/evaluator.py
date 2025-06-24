"""
增强的条形码预处理评估器

Features:
- 并行条形码解码以提高性能
- 可配置详细程度的详细日志记录
- 失败图像分析和保存
- 性能分析和指标
- 批量处理优化
- 内存使用监控
- CI友好的执行模式
- 统一的预处理算法评估
- 基准测试集成
- 算法性能分析
- 详细的评估报告生成
"""

import importlib.util
import os
import sys
import time
import traceback
import tempfile
import subprocess
import pickle
import logging
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from datetime import datetime
import json
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, asdict
import shutil
import glob

import cv2
import numpy as np
from pyzbar import pyzbar
from PIL import Image
try:
    import psutil
    HAS_OPTIONAL_DEPS = True
except ImportError:
    HAS_OPTIONAL_DEPS = False

# 尝试导入可选依赖
try:
    import matplotlib.pyplot as plt
    import psutil
    HAS_OPTIONAL_DEPS = True
except ImportError:
    HAS_OPTIONAL_DEPS = False

# Configure logging
class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors for different log levels"""
    
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
    }
    RESET = '\033[0m'
    
    def format(self, record):
        color = self.COLORS.get(record.levelname, self.RESET)
        record.levelname = f"{color}{record.levelname}{self.RESET}"
        return super().format(record)

def setup_logging(verbose: bool = False, log_file: Optional[str] = None):
    """Setup logging configuration"""
    level = logging.DEBUG if verbose else logging.INFO
    
    # Create logger
    logger = logging.getLogger('barcode_evaluator')
    logger.setLevel(level)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Console handler with colors
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_formatter = ColoredFormatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(funcName)s - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger

logger = setup_logging()

class TimeoutError(Exception):
    pass

@dataclass
class TestResult:
    """测试结果数据类"""
    algorithm_name: str
    total_images: int
    successful_decodes: int
    success_rate: float
    processing_time: float
    throughput: float
    memory_usage: float = 0.0
    cpu_usage: float = 0.0
    decoded_barcodes_count: int = 0
    failed_images: List[str] = None
    type_statistics: Dict[str, Any] = None
    quality_metrics: Dict[str, float] = None
    timestamp: str = None
    
    def __post_init__(self):
        if self.failed_images is None:
            self.failed_images = []
        if self.type_statistics is None:
            self.type_statistics = {}
        if self.quality_metrics is None:
            self.quality_metrics = {}
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()

@dataclass 
class ComparisonReport:
    """对比报告数据类"""
    baseline_result: TestResult
    processed_result: TestResult
    improvement_metrics: Dict[str, float]
    quality_improvement: Dict[str, float]
    performance_impact: Dict[str, float]
    overall_score: float
    recommendation: str

class BarcodeDecodeResult:
    """Container for barcode decode results with additional metadata"""
    
    def __init__(self, image_path: str, success: bool, 
                 decoded_data: List[Dict] = None, 
                 error: str = None,
                 processing_time: float = 0.0):
        self.image_path = image_path
        self.filename = os.path.basename(image_path)
        self.success = success
        self.decoded_data = decoded_data or []
        self.error = error
        self.processing_time = processing_time
        self.timestamp = datetime.now()

def decode_barcode_single(image_path: str) -> BarcodeDecodeResult:
    """
    Decode a single barcode image with detailed result tracking
    
    Args:
        image_path: Path to barcode image
        
    Returns:
        BarcodeDecodeResult with decode information
    """
    start_time = time.time()
    
    try:
        logger.debug(f"Decoding {os.path.basename(image_path)}")
        
        # Try reading with OpenCV first
        image = cv2.imread(image_path)
        if image is not None:
            # Convert BGR to RGB for pyzbar
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            # Fallback to PIL
            image_pil = Image.open(image_path)
            image_rgb = np.array(image_pil)
            if len(image_rgb.shape) == 3 and image_rgb.shape[2] == 4:
                # Convert RGBA to RGB
                image_rgb = image_rgb[:, :, :3]
        
        # Decode barcodes
        barcodes = pyzbar.decode(image_rgb)
        
        # Extract data from barcodes
        decoded_data = []
        for barcode in barcodes:
            try:
                barcode_data = barcode.data.decode('utf-8')
                barcode_type = barcode.type
                decoded_data.append({
                    'data': barcode_data,
                    'type': barcode_type,
                    'rect': barcode.rect,
                    'polygon': barcode.polygon
                })
            except UnicodeDecodeError:
                # Handle non-UTF8 data
                barcode_data = str(barcode.data)
                decoded_data.append({
                    'data': barcode_data,
                    'type': barcode.type,
                    'rect': barcode.rect,
                    'polygon': barcode.polygon,
                    'encoding_error': True
                })
        
        processing_time = time.time() - start_time
        
        if decoded_data:
            logger.debug(f"✓ {os.path.basename(image_path)}: {len(decoded_data)} barcode(s) in {processing_time:.3f}s")
            return BarcodeDecodeResult(image_path, True, decoded_data, processing_time=processing_time)
        else:
            logger.debug(f"✗ {os.path.basename(image_path)}: No barcodes found in {processing_time:.3f}s")
            return BarcodeDecodeResult(image_path, False, processing_time=processing_time)
    
    except Exception as e:
        processing_time = time.time() - start_time
        error_msg = f"Error decoding {os.path.basename(image_path)}: {str(e)}"
        logger.warning(error_msg)
        return BarcodeDecodeResult(image_path, False, error=error_msg, processing_time=processing_time)

def decode_barcode_parallel(image_paths: List[str], max_workers: int = None) -> List[BarcodeDecodeResult]:
    """
    Decode multiple barcodes in parallel using process pool
    
    Args:
        image_paths: List of image file paths
        max_workers: Maximum number of worker processes (default: CPU count)
        
    Returns:
        List of BarcodeDecodeResult objects
    """
    if not image_paths:
        return []
    
    if max_workers is None:
        max_workers = min(len(image_paths), os.cpu_count() or 1)
    
    logger.info(f"Processing {len(image_paths)} images with {max_workers} workers")
    
    results = []
    start_time = time.time()
    
    # Use ThreadPoolExecutor for I/O bound operations like image reading
    # ProcessPoolExecutor can cause issues with OpenCV/PIL in some environments
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all jobs
        future_to_path = {executor.submit(decode_barcode_single, path): path 
                         for path in image_paths}
        
        # Collect results as they complete
        for future in as_completed(future_to_path):
            try:
                result = future.result(timeout=10)  # 10 second timeout per image
                results.append(result)
            except Exception as e:
                path = future_to_path[future]
                logger.error(f"Failed to process {os.path.basename(path)}: {e}")
                results.append(BarcodeDecodeResult(path, False, error=str(e)))
    
    total_time = time.time() - start_time
    successful = sum(1 for r in results if r.success)
    
    logger.info(f"Parallel processing completed: {successful}/{len(results)} successful in {total_time:.2f}s")
    logger.info(f"Average time per image: {total_time/len(results):.3f}s")
    
    return results

def save_failed_images(failed_results: List[BarcodeDecodeResult], 
                      original_dir: str, 
                      output_dir: str = "failed_images") -> str:
    """
    Save failed images and analysis to a directory
    
    Args:
        failed_results: List of failed BarcodeDecodeResult objects
        original_dir: Original directory containing the images
        output_dir: Directory to save failed images and analysis
        
    Returns:
        Path to the created failed images directory
    """
    if not failed_results:
        logger.info("No failed images to save")
        return ""
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    failed_dir = f"{output_dir}_{timestamp}"
    os.makedirs(failed_dir, exist_ok=True)
    
    logger.info(f"Saving {len(failed_results)} failed images to {failed_dir}")
    
    # Copy failed images
    import shutil
    failed_analysis = []
    
    for result in failed_results:
        try:
            # Copy the image file
            dest_path = os.path.join(failed_dir, result.filename)
            shutil.copy2(result.image_path, dest_path)
            
            # Analyze the image
            analysis = analyze_failed_image(result.image_path)
            failed_analysis.append({
                'filename': result.filename,
                'error': result.error,
                'processing_time': result.processing_time,
                'analysis': analysis
            })
            
        except Exception as e:
            logger.warning(f"Failed to save {result.filename}: {e}")
    
    # Save analysis report
    analysis_path = os.path.join(failed_dir, "failure_analysis.json")
    with open(analysis_path, 'w') as f:
        json.dump({
            'timestamp': timestamp,
            'total_failed': len(failed_results),
            'failed_images': failed_analysis,
            'summary': generate_failure_summary(failed_analysis)
        }, f, indent=2)
    
    logger.info(f"Failed image analysis saved to {analysis_path}")
    return failed_dir

def analyze_failed_image(image_path: str) -> Dict:
    """
    Analyze a failed image to determine potential causes of failure
    
    Args:
        image_path: Path to the failed image
        
    Returns:
        Dictionary with analysis results
    """
    try:
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            return {'error': 'Could not read image'}
        
        # Basic image properties
        height, width = image.shape[:2]
        channels = image.shape[2] if len(image.shape) > 2 else 1
        
        # Convert to grayscale for analysis
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Image quality metrics
        mean_brightness = np.mean(gray)
        std_brightness = np.std(gray)
        
        # Edge detection to check for blur
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Contrast analysis
        contrast = std_brightness / mean_brightness if mean_brightness > 0 else 0
        
        # Noise estimation (high frequency content)
        kernel = np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]])
        noise_level = np.mean(np.abs(cv2.filter2D(gray, cv2.CV_64F, kernel)))
        
        analysis = {
            'dimensions': {'width': int(width), 'height': int(height), 'channels': int(channels)},
            'brightness': {'mean': float(mean_brightness), 'std': float(std_brightness)},
            'sharpness': float(laplacian_var),
            'contrast': float(contrast),
            'noise_level': float(noise_level),
            'issues': []
        }
        
        # Identify potential issues
        if mean_brightness < 50:
            analysis['issues'].append('very_dark')
        elif mean_brightness > 200:
            analysis['issues'].append('very_bright')
        
        if laplacian_var < 100:
            analysis['issues'].append('blurry')
        
        if contrast < 0.2:
            analysis['issues'].append('low_contrast')
        
        if noise_level > 50:
            analysis['issues'].append('noisy')
        
        if width < 100 or height < 50:
            analysis['issues'].append('too_small')
        
        return analysis
        
    except Exception as e:
        return {'error': f'Analysis failed: {str(e)}'}

def generate_failure_summary(failed_analysis: List[Dict]) -> Dict:
    """Generate summary statistics of failure analysis"""
    if not failed_analysis:
        return {}
    
    # Count issues
    issue_counts = {}
    total_images = len(failed_analysis)
    
    for analysis in failed_analysis:
        if 'analysis' in analysis and 'issues' in analysis['analysis']:
            for issue in analysis['analysis']['issues']:
                issue_counts[issue] = issue_counts.get(issue, 0) + 1
    
    # Calculate percentages
    issue_percentages = {issue: (count / total_images) * 100 
                        for issue, count in issue_counts.items()}
    
    return {
        'total_failed_images': total_images,
        'common_issues': issue_counts,
        'issue_percentages': issue_percentages,
        'most_common_issue': max(issue_counts.items(), key=lambda x: x[1])[0] if issue_counts else None
    }

def test_barcode_set(image_dir: str, 
                   parallel: bool = True,
                   max_workers: int = None,
                   save_failures: bool = False,
                   verbose: bool = False,
                   algorithm_name: str = "unknown") -> TestResult:
    """
    条形码解码测试
    
    Args:
        image_dir: 包含条形码图像的目录
        parallel: 是否使用并行处理
        max_workers: 并行处理的最大工作线程数
        save_failures: 是否保存失败的图像进行分析
        verbose: 是否显示详细进度
        algorithm_name: 算法名称
        
    Returns:
        TestResult对象，包含全面的指标
    """
    if not os.path.exists(image_dir):
        logger.warning(f"Image directory does not exist: {image_dir}")
        return TestResult(
            algorithm_name=algorithm_name,
            total_images=0,
            successful_decodes=0,
            success_rate=0.0,
            processing_time=0.0,
            throughput=0.0
        )
    
    # Find all image files
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif'}
    image_paths = []
    
    for filename in os.listdir(image_dir):
        if any(filename.lower().endswith(ext) for ext in image_extensions):
            image_paths.append(os.path.join(image_dir, filename))
    
    if not image_paths:
        logger.warning(f"No image files found in {image_dir}")
        return TestResult(
            algorithm_name=algorithm_name,
            total_images=0,
            successful_decodes=0,
            success_rate=0.0,
            processing_time=0.0,
            throughput=0.0
        )
    
    logger.info(f"Found {len(image_paths)} images in {image_dir}")
    start_time = time.time()
    
    # Process images
    if parallel and len(image_paths) > 1:
        results = decode_barcode_parallel(image_paths, max_workers)
    else:
        results = [decode_barcode_single(path) for path in image_paths]
    
    processing_time = time.time() - start_time
    
    # Analyze results
    successful_results = [r for r in results if r.success]
    failed_results = [r for r in results if not r.success]
    
    all_decoded_barcodes = []
    for result in successful_results:
        all_decoded_barcodes.extend(result.decoded_data)
    
    # Calculate metrics
    total_images = len(results)
    successful_decodes = len(successful_results)
    success_rate = successful_decodes / total_images if total_images > 0 else 0.0
    throughput = total_images / processing_time if processing_time > 0 else 0.0
    
    # Statistics by barcode type
    type_stats = {}
    for result in results:
        # Determine expected barcode type from filename
        filename = result.filename.lower()
        if 'qr' in filename:
            expected_type = 'QRCODE'
        elif 'code128' in filename:
            expected_type = 'CODE128'
        elif 'code39' in filename:
            expected_type = 'CODE39'
        elif 'datamatrix' in filename:
            expected_type = 'DATAMATRIX'
        else:
            expected_type = 'UNKNOWN'
        
        if expected_type not in type_stats:
            type_stats[expected_type] = {'total': 0, 'successful': 0, 'failed': 0}
        
        type_stats[expected_type]['total'] += 1
        
        if result.success:
            type_stats[expected_type]['successful'] += 1
            
            # Check if detected type matches expected type
            detected_types = set()
            for decoded in result.decoded_data:
                detected_types.add(str(decoded.get('type', 'UNKNOWN')))
            
            # Add detected type information
            if 'detected_types' not in type_stats[expected_type]:
                type_stats[expected_type]['detected_types'] = {}
            
            for detected_type in detected_types:
                if detected_type not in type_stats[expected_type]['detected_types']:
                    type_stats[expected_type]['detected_types'][detected_type] = 0
                type_stats[expected_type]['detected_types'][detected_type] += 1
        else:
            type_stats[expected_type]['failed'] += 1
    
    # Calculate success rates by type
    for barcode_type in type_stats:
        stats = type_stats[barcode_type]
        stats['success_rate'] = stats['successful'] / stats['total'] if stats['total'] > 0 else 0.0
    
    # Log summary
    logger.info(f"Processing complete: {successful_decodes}/{total_images} successful ({success_rate:.1%})")
    logger.info(f"Total time: {processing_time:.2f}s, Throughput: {throughput:.1f} images/sec")
    
    # Log type-specific statistics
    if type_stats:
        logger.info("Success rates by barcode type:")
        for barcode_type, stats in type_stats.items():
            logger.info(f"  {barcode_type}: {stats['successful']}/{stats['total']} ({stats['success_rate']:.1%})")
            if verbose and 'detected_types' in stats:
                logger.debug(f"    Detected types: {stats['detected_types']}")
    
    if failed_results and verbose:
        logger.info(f"Failed images: {[r.filename for r in failed_results[:10]]}")
        if len(failed_results) > 10:
            logger.info(f"... and {len(failed_results) - 10} more")
    
    # Save failed images if requested
    failed_dir = ""
    if save_failures and failed_results:
        failed_dir = save_failed_images(failed_results, image_dir)
    
    # 计算质量指标
    quality_metrics = calculate_quality_metrics(results, type_stats)
    
    # 内存和CPU使用（如果可用）
    memory_usage = 0.0
    cpu_usage = 0.0
    if HAS_OPTIONAL_DEPS:
        try:
            memory_usage = psutil.virtual_memory().percent
            cpu_usage = psutil.cpu_percent()
        except:
            pass
    
    return TestResult(
        algorithm_name=algorithm_name,
        total_images=total_images,
        successful_decodes=successful_decodes,
        success_rate=success_rate,
        processing_time=processing_time,
        throughput=throughput,
        memory_usage=memory_usage,
        cpu_usage=cpu_usage,
        decoded_barcodes_count=len(all_decoded_barcodes),
        failed_images=[r.filename for r in failed_results],
        type_statistics=type_stats,
        quality_metrics=quality_metrics
    )

def calculate_quality_metrics(results: List[BarcodeDecodeResult], 
                            type_stats: Dict[str, Any]) -> Dict[str, float]:
    """计算图像质量指标"""
    if not results:
        return {}
    
    # 计算平均处理时间
    avg_processing_time = np.mean([r.processing_time for r in results])
    
    # 计算成功率方差（稳定性指标）
    success_rates_by_type = []
    for type_name, stats in type_stats.items():
        if stats['total'] > 0:
            success_rates_by_type.append(stats['success_rate'])
    
    stability_score = 1.0 - np.var(success_rates_by_type) if success_rates_by_type else 0.0
    
    # 处理时间一致性
    processing_times = [r.processing_time for r in results]
    time_consistency = 1.0 / (1.0 + np.std(processing_times)) if processing_times else 0.0
    
    return {
        'average_processing_time': avg_processing_time,
        'stability_score': max(0.0, stability_score),
        'time_consistency': max(0.0, time_consistency),
        'total_barcode_types': len(type_stats),
        'successful_types': sum(1 for stats in type_stats.values() if stats['successful'] > 0)
    }

def generate_comparison_report(baseline: TestResult, 
                             processed: TestResult) -> ComparisonReport:
    """生成对比报告"""
    
    # 计算改进指标
    improvement_metrics = {
        'success_rate_improvement': processed.success_rate - baseline.success_rate,
        'throughput_improvement': processed.throughput - baseline.throughput,
        'processing_time_improvement': baseline.processing_time - processed.processing_time,
        'memory_efficiency': baseline.memory_usage - processed.memory_usage if baseline.memory_usage > 0 else 0,
    }
    
    # 质量改进
    quality_improvement = {}
    if baseline.quality_metrics and processed.quality_metrics:
        for key in baseline.quality_metrics:
            if key in processed.quality_metrics:
                quality_improvement[key] = processed.quality_metrics[key] - baseline.quality_metrics[key]
    
    # 性能影响
    performance_impact = {
        'speed_factor': processed.throughput / baseline.throughput if baseline.throughput > 0 else 1.0,
        'time_reduction_percent': (improvement_metrics['processing_time_improvement'] / baseline.processing_time * 100) 
                                if baseline.processing_time > 0 else 0.0,
        'success_rate_delta': improvement_metrics['success_rate_improvement'] * 100
    }
    
    # 综合评分 (0-100)
    success_score = min(50, improvement_metrics['success_rate_improvement'] * 200)  # 最高50分
    speed_score = min(30, max(0, performance_impact['time_reduction_percent']))     # 最高30分  
    efficiency_score = min(20, improvement_metrics['memory_efficiency'])           # 最高20分
    
    overall_score = max(0, success_score + speed_score + efficiency_score)
    
    # 生成建议
    if overall_score >= 70:
        recommendation = "强烈推荐：预处理算法显著改善了解码性能"
    elif overall_score >= 40:
        recommendation = "推荐：预处理算法有明显改进，值得采用"
    elif overall_score >= 10:
        recommendation = "谨慎考虑：预处理算法有轻微改进，需权衡成本效益"
    else:
        recommendation = "不推荐：预处理算法未显示明显改进或有负面影响"
    
    return ComparisonReport(
        baseline_result=baseline,
        processed_result=processed,
        improvement_metrics=improvement_metrics,
        quality_improvement=quality_improvement,
        performance_impact=performance_impact,
        overall_score=overall_score,
        recommendation=recommendation
    )

def run_with_timeout(program_path: str, 
                    sample_images_dir: str, 
                    timeout_seconds: int = 30) -> str:
    """
    Run the preprocessing program with timeout
    
    Args:
        program_path: Path to the preprocessing program
        sample_images_dir: Directory with sample barcode images
        timeout_seconds: Maximum execution time
        
    Returns:
        Path to directory with processed images
    """
    # Create temporary directory for processed images
    temp_dir = tempfile.mkdtemp(prefix="barcode_processed_")
    
    logger.info(f"Running preprocessing with {timeout_seconds}s timeout")
    logger.debug(f"Input: {sample_images_dir}")
    logger.debug(f"Output: {temp_dir}")
    
    # Create script to run preprocessing
    with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as temp_file:
        script = f"""
import sys
import os
import traceback
import pickle

# Add the directory to sys.path
sys.path.insert(0, os.path.dirname('{program_path}'))

try:
    # Import the program
    spec = __import__('importlib.util').util.spec_from_file_location("program", '{program_path}')
    program = __import__('importlib.util').util.module_from_spec(spec)
    spec.loader.exec_module(program)
    
    # Run preprocessing
    print("Running preprocessing...")
    processed_paths = program.run_preprocessing('{sample_images_dir}', '{temp_dir}')
    print(f"Preprocessing completed: {{len(processed_paths)}} images processed")
    
    # Save results
    results = {{
        'processed_dir': '{temp_dir}',
        'processed_count': len(processed_paths)
    }}
    
    with open('{temp_file.name}.results', 'wb') as f:
        pickle.dump(results, f)
        
except Exception as e:
    print(f"Error in preprocessing: {{str(e)}}")
    traceback.print_exc()
    with open('{temp_file.name}.results', 'wb') as f:
        pickle.dump({{'error': str(e)}}, f)
"""
        temp_file.write(script.encode())
        temp_file_path = temp_file.name
    
    results_path = f"{temp_file_path}.results"
    
    try:
        # Run the script with timeout
        process = subprocess.Popen(
            [sys.executable, temp_file_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        try:
            stdout, stderr = process.communicate(timeout=timeout_seconds)
            exit_code = process.returncode
            
            if stdout:
                logger.debug(f"Preprocessing stdout: {stdout.decode()}")
            if stderr:
                logger.warning(f"Preprocessing stderr: {stderr.decode()}")
            
            if exit_code != 0:
                raise RuntimeError(f"Preprocessing failed with exit code {exit_code}")
            
            # Load results
            if os.path.exists(results_path):
                with open(results_path, 'rb') as f:
                    results = pickle.load(f)
                
                if 'error' in results:
                    raise RuntimeError(f"Preprocessing error: {results['error']}")
                
                logger.info(f"Preprocessing successful: {results['processed_count']} images processed")
                return results['processed_dir']
            else:
                raise RuntimeError("Results file not found")
        
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
            raise TimeoutError(f"Preprocessing timed out after {timeout_seconds} seconds")
    
    finally:
        # Clean up temporary files
        if os.path.exists(temp_file_path):
            os.unlink(temp_file_path)
        if os.path.exists(results_path):
            os.unlink(results_path)

def evaluate(program_path: str,
            parallel: bool = True,
            max_workers: int = None,
            save_failures: bool = False,
            verbose: bool = False,
            timeout_seconds: int = 30) -> Dict:
    """
    统一的条码预处理算法评估函数
    
    Args:
        program_path: Path to the preprocessing program
        parallel: Whether to use parallel processing
        max_workers: Maximum number of workers for parallel processing
        save_failures: Whether to save failed images for analysis
        verbose: Whether to show detailed progress
        timeout_seconds: Timeout for preprocessing execution
        
    Returns:
        Dictionary of comprehensive evaluation metrics
    """
    try:
        start_time = time.time()
        
        # Get the directory containing the program
        program_dir = os.path.dirname(program_path)
        if not program_dir:
            program_dir = os.path.dirname(os.path.abspath(program_path))
        sample_images_dir = os.path.join(program_dir, 'sample_images')
        
        # If running from project root, look for the correct path
        if not os.path.exists(sample_images_dir):
            # Try relative to current working directory
            sample_images_dir = 'sample_images'
            if not os.path.exists(sample_images_dir):
                # Try in the same directory as this evaluator
                evaluator_dir = os.path.dirname(os.path.abspath(__file__))
                sample_images_dir = os.path.join(evaluator_dir, 'sample_images')
        
        # Check if sample images exist
        if not os.path.exists(sample_images_dir):
            logger.error(f"Sample images directory not found: {sample_images_dir}")
            return {
                'success_rate': 0.0,
                'total_images': 0,
                'successful_decodes': 0,
                'error': 'No sample images found',
                'execution_time': 0.0
            }
        
        # Count images in directory
        image_files = [f for f in os.listdir(sample_images_dir) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
        if not image_files:
            logger.error(f"No image files found in {sample_images_dir}")
            return {
                'success_rate': 0.0,
                'total_images': 0,
                'successful_decodes': 0,
                'error': 'No image files found',
                'execution_time': 0.0
            }
        
        logger.info(f"Found {len(image_files)} sample images")
        
        # Test original images first (baseline)
        logger.info("Testing baseline (原始图像)...")
        baseline_results = test_barcode_set(
            sample_images_dir, 
            parallel=parallel,
            max_workers=max_workers,
            save_failures=save_failures,
            verbose=verbose,
            algorithm_name="Baseline"
        )
        
        # Run preprocessing with timeout
        logger.info("Running preprocessing algorithm...")
        preprocessing_start = time.time()
        processed_dir = run_with_timeout(program_path, sample_images_dir, timeout_seconds)
        preprocessing_time = time.time() - preprocessing_start
        
        # Test processed images
        logger.info("Testing preprocessed images...")
        processed_results = test_barcode_set(
            processed_dir,
            parallel=parallel,
            max_workers=max_workers,
            save_failures=save_failures,
            verbose=verbose,
            algorithm_name="Preprocessed"
        )
        
        end_time = time.time()
        total_execution_time = end_time - start_time
        
        # Calculate improvement metrics
        baseline_rate = baseline_results.success_rate
        processed_rate = processed_results.success_rate
        improvement = processed_rate - baseline_rate
        
        # Performance metrics
        baseline_throughput = baseline_results.throughput
        processed_throughput = processed_results.throughput
        
        # Clean up processed directory
        try:
            import shutil
            shutil.rmtree(processed_dir)
            logger.debug(f"Cleaned up temporary directory: {processed_dir}")
        except Exception as e:
            logger.warning(f"Failed to clean up {processed_dir}: {e}")
        
        # Generate comparison report
        comparison_report = generate_comparison_report(baseline_results, processed_results)
        
        # Calculate final score
        if processed_results.total_images == 0:
            score = 0.0
        else:
            # 主要基于成功率改进，加权处理时间和吞吐量
            success_weight = 0.7
            throughput_weight = 0.2
            time_weight = 0.1
            
            # 成功率得分 (0-100)
            success_score = processed_rate * 100
            
            # 改进得分 (考虑相对基线的提升)
            improvement_factor = improvement / baseline_rate if baseline_rate > 0 else 0
            improvement_score = min(20, improvement_factor * 100)  # 最多额外加20分
            
            # 吞吐量得分 (标准化到0-100)
            throughput_score = min(100, (processed_throughput / 100) * 100)  # 假设100图像/秒为满分
            
            # 综合得分
            score = (success_score * success_weight + 
                    throughput_score * throughput_weight + 
                    improvement_score * time_weight)
        
        # Comprehensive results
        results = {
            'score': max(0.0, score),
            'success_rate': processed_rate,
            'baseline_success_rate': baseline_rate,
            'improvement': improvement,
            'improvement_percentage': (improvement / baseline_rate * 100) if baseline_rate > 0 else 0,
            'total_images': processed_results.total_images,
            'successful_decodes': processed_results.successful_decodes,
            'baseline_successful_decodes': baseline_results.successful_decodes,
            'execution_time': total_execution_time,
            'preprocessing_time': preprocessing_time,
            'evaluation_time': total_execution_time - preprocessing_time,
            'decoded_barcodes_count': processed_results.decoded_barcodes_count,
            'baseline_throughput': baseline_throughput,
            'processed_throughput': processed_throughput,
            'throughput_improvement': processed_throughput - baseline_throughput,
            'failed_images': processed_results.failed_images,
            'baseline_failed_count': len(baseline_results.failed_images),
            'processed_failed_count': len(processed_results.failed_images),
            'average_processing_time': processed_results.quality_metrics.get('average_processing_time', 0),
            'parallel_processing': parallel,
            'max_workers': max_workers or 1,
            'comparison_report': comparison_report
        }
        
        # Log comprehensive summary
        logger.info("\n" + "="*60)
        logger.info("算法评估结果")
        logger.info("="*60)
        logger.info(f"综合得分: {results['score']:.2f}")
        logger.info(f"成功率: {results['success_rate']:.2%} (基线: {results['baseline_success_rate']:.2%})")
        logger.info(f"改进幅度: {results['improvement']:.2%} ({results['improvement_percentage']:+.1f}%)")
        logger.info(f"成功解码: {results['successful_decodes']}/{results['total_images']} 张")
        logger.info(f"执行时间: {results['execution_time']:.2f}s (预处理: {results['preprocessing_time']:.2f}s)")
        logger.info(f"处理速度: {results['processed_throughput']:.1f} 图像/秒")
        logger.info(f"推荐: {comparison_report.recommendation}")
        logger.info("="*60)
        
        return results
    
    except Exception as e:
        logger.error(f"Evaluation error: {e}")
        traceback.print_exc()
        return {
            'score': 0.0,
            'success_rate': 0.0,
            'error': str(e),
            'execution_time': 0.0
        }

def main():
    """Command line interface for the evaluator"""
    parser = argparse.ArgumentParser(description='Barcode Preprocessing Evaluator')
    parser.add_argument('program_path', help='Path to the preprocessing program')
    parser.add_argument('--parallel', action='store_true', default=True,
                       help='Use parallel processing (default: True)')
    parser.add_argument('--no-parallel', dest='parallel', action='store_false',
                       help='Disable parallel processing')
    parser.add_argument('--max-workers', type=int, default=None,
                       help='Maximum number of worker processes')
    parser.add_argument('--save-failures', action='store_true', default=False,
                       help='Save failed images for analysis')
    parser.add_argument('--no-save-failures', dest='save_failures', action='store_false',
                       help='Disable saving failed images (CI-friendly)')
    parser.add_argument('--verbose', '-v', action='store_true', default=False,
                       help='Enable verbose logging')
    parser.add_argument('--timeout', type=int, default=30,
                       help='Timeout for preprocessing execution (seconds)')
    parser.add_argument('--log-file', type=str, default=None,
                       help='Save logs to file')

    args = parser.parse_args()
    
    # Setup logging
    global logger
    logger = setup_logging(args.verbose, args.log_file)
    
    logger.info(f"Starting evaluation of {args.program_path}")
    
    results = evaluate(
        args.program_path,
        parallel=args.parallel,
        max_workers=args.max_workers,
        save_failures=args.save_failures,
        verbose=args.verbose,
        timeout_seconds=args.timeout
    )
    
    # Output results in a format that can be parsed by the evolution system
    print(f"\nFinal Score: {results.get('score', 0):.2f}")
    
    if 'error' in results:
        sys.exit(1)

if __name__ == "__main__":
    main() 