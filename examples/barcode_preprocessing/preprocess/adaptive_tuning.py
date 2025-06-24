"""
自适应参数调优系统
基于图像质量动态调整预处理参数和算法策略
"""

import numpy as np
import cv2
from typing import Tuple, Dict, List, Any, Optional, Union
import logging
from scipy import stats
from sklearn.metrics import mean_squared_error
import math

logger = logging.getLogger(__name__)


class ImageQualityAssessment:
    """图像质量评估器"""
    
    @staticmethod
    def calculate_gradient_metrics(image: np.ndarray) -> Dict[str, float]:
        """计算梯度相关指标"""
        # Sobel梯度
        grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        
        # 梯度幅值
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        return {
            'gradient_mean': float(np.mean(gradient_magnitude)),
            'gradient_std': float(np.std(gradient_magnitude)),
            'gradient_max': float(np.max(gradient_magnitude)),
            'edge_density': float(np.mean(gradient_magnitude > np.mean(gradient_magnitude))),
            'sharpness': float(np.var(cv2.Laplacian(image, cv2.CV_64F)))
        }
    
    @staticmethod
    def calculate_texture_metrics(image: np.ndarray) -> Dict[str, float]:
        """计算纹理相关指标"""
        # 灰度共生矩阵简化计算
        hist = cv2.calcHist([image], [0], None, [256], [0, 256])
        hist_normalized = hist / (image.size + 1e-7)
        
        # 熵
        entropy = -np.sum(hist_normalized * np.log2(hist_normalized + 1e-7))
        
        # 对比度
        i, j = np.meshgrid(range(256), range(256))
        contrast = np.sum(hist_normalized * (i - j)**2)
        
        # 齐次性（均匀性）
        homogeneity = np.sum(hist_normalized / (1 + (i - j)**2))
        
        return {
            'entropy': float(entropy),
            'contrast': float(contrast),
            'homogeneity': float(homogeneity),
            'texture_complexity': float(np.std(hist_normalized))
        }
    
    @staticmethod
    def calculate_noise_metrics(image: np.ndarray) -> Dict[str, float]:
        """计算噪声相关指标"""
        # 拉普拉斯方差（噪声指标）
        laplacian_var = cv2.Laplacian(image, cv2.CV_64F).var()
        
        # 高频噪声分析
        f_transform = np.fft.fft2(image)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = np.log(np.abs(f_shift) + 1)
        
        # 高频能量比例
        h, w = image.shape
        center_y, center_x = h // 2, w // 2
        radius = min(h, w) // 4
        y, x = np.ogrid[:h, :w]
        mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2
        
        low_freq_energy = np.sum(magnitude_spectrum[mask])
        total_energy = np.sum(magnitude_spectrum)
        high_freq_ratio = 1 - (low_freq_energy / (total_energy + 1e-7))
        
        # SNR估算
        signal_power = np.mean(image**2)
        noise_power = laplacian_var / 6  # 拉普拉斯噪声方差估计
        snr = 10 * np.log10(signal_power / (noise_power + 1e-7))
        
        return {
            'laplacian_variance': float(laplacian_var),
            'high_freq_ratio': float(high_freq_ratio),
            'estimated_snr': float(snr),
            'noise_level': float(min(100, max(0, (laplacian_var - 100) / 10)))
        }
    
    @staticmethod
    def calculate_lighting_metrics(image: np.ndarray) -> Dict[str, float]:
        """计算光照相关指标"""
        # 亮度统计
        mean_brightness = float(np.mean(image))
        brightness_std = float(np.std(image))
        
        # 动态范围
        dynamic_range = float(np.max(image) - np.min(image))
        
        # 光照均匀性（通过分块分析）
        h, w = image.shape
        block_size = min(h, w) // 8
        if block_size > 0:
            block_means = []
            for i in range(0, h - block_size, block_size):
                for j in range(0, w - block_size, block_size):
                    block = image[i:i+block_size, j:j+block_size]
                    block_means.append(np.mean(block))
            
            illumination_uniformity = 1.0 / (1.0 + np.std(block_means))
        else:
            illumination_uniformity = 1.0
        
        # 对比度
        contrast_ratio = brightness_std / (mean_brightness + 1e-7)
        
        return {
            'mean_brightness': mean_brightness,
            'brightness_std': brightness_std,
            'dynamic_range': dynamic_range,
            'illumination_uniformity': float(illumination_uniformity),
            'contrast_ratio': float(contrast_ratio),
            'is_low_contrast': contrast_ratio < 0.3,
            'is_overexposed': mean_brightness > 200,
            'is_underexposed': mean_brightness < 50
        }
    
    @classmethod
    def comprehensive_assessment(cls, image: np.ndarray) -> Dict[str, Any]:
        """综合图像质量评估"""
        # 确保是灰度图像
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 计算各类指标
        gradient_metrics = cls.calculate_gradient_metrics(image)
        texture_metrics = cls.calculate_texture_metrics(image)
        noise_metrics = cls.calculate_noise_metrics(image)
        lighting_metrics = cls.calculate_lighting_metrics(image)
        
        # 综合评分
        quality_score = cls.calculate_overall_quality(
            gradient_metrics, texture_metrics, noise_metrics, lighting_metrics
        )
        
        # 合并所有指标
        assessment = {
            **gradient_metrics,
            **texture_metrics,
            **noise_metrics,
            **lighting_metrics,
            'overall_quality': quality_score,
            'image_size': image.shape,
            'assessment_summary': cls.generate_summary(
                gradient_metrics, texture_metrics, noise_metrics, lighting_metrics, quality_score
            )
        }
        
        return assessment
    
    @staticmethod
    def calculate_overall_quality(gradient_metrics: Dict, texture_metrics: Dict, 
                                 noise_metrics: Dict, lighting_metrics: Dict) -> float:
        """计算整体质量评分 (0-100)"""
        # 权重设计
        weights = {
            'sharpness': 0.25,
            'contrast': 0.20,
            'noise': 0.25,
            'lighting': 0.20,
            'texture': 0.10
        }
        
        # 标准化各指标到0-100
        sharpness_score = min(100, gradient_metrics['sharpness'] / 1000 * 100)
        contrast_score = min(100, lighting_metrics['contrast_ratio'] * 200)
        noise_score = max(0, 100 - noise_metrics['noise_level'])
        lighting_score = lighting_metrics['illumination_uniformity'] * 100
        texture_score = min(100, texture_metrics['entropy'] / 8 * 100)
        
        overall_score = (
            weights['sharpness'] * sharpness_score +
            weights['contrast'] * contrast_score +
            weights['noise'] * noise_score +
            weights['lighting'] * lighting_score +
            weights['texture'] * texture_score
        )
        
        return float(overall_score)
    
    @staticmethod
    def generate_summary(gradient_metrics: Dict, texture_metrics: Dict,
                        noise_metrics: Dict, lighting_metrics: Dict, quality_score: float) -> Dict[str, str]:
        """生成评估摘要"""
        summary = {}
        
        # 质量等级
        if quality_score >= 80:
            summary['quality_level'] = 'excellent'
        elif quality_score >= 60:
            summary['quality_level'] = 'good'
        elif quality_score >= 40:
            summary['quality_level'] = 'fair'
        else:
            summary['quality_level'] = 'poor'
        
        # 主要问题识别
        issues = []
        if noise_metrics['noise_level'] > 30:
            issues.append('high_noise')
        if lighting_metrics['is_low_contrast']:
            issues.append('low_contrast')
        if lighting_metrics['illumination_uniformity'] < 0.7:
            issues.append('uneven_lighting')
        if gradient_metrics['sharpness'] < 100:
            issues.append('blur')
        if lighting_metrics['is_overexposed'] or lighting_metrics['is_underexposed']:
            issues.append('exposure_problems')
        
        summary['identified_issues'] = issues
        summary['primary_issue'] = issues[0] if issues else 'none'
        
        return summary


class AdaptiveParameterTuner:
    """自适应参数调优器"""
    
    def __init__(self):
        self.parameter_templates = self._initialize_parameter_templates()
        self.algorithm_combinations = self._initialize_algorithm_combinations()
    
    def _initialize_parameter_templates(self) -> Dict[str, Dict]:
        """初始化参数模板"""
        return {
            # 光照校正参数
            'illumination': {
                'excellent': {'method': 'gamma', 'gamma': 1.0},
                'good': {'method': 'clahe', 'clip_limit': 2.0},
                'fair': {'method': 'msr', 'scales': [15, 80, 250]},
                'poor': {'method': 'ssr', 'scale': 125}
            },
            
            # 去噪参数
            'denoising': {
                'excellent': {'method': 'none'},
                'good': {'method': 'bilateral', 'd': 5, 'sigmaColor': 50, 'sigmaSpace': 50},
                'fair': {'method': 'nlm', 'h': 10, 'templateWindowSize': 7, 'searchWindowSize': 21},
                'poor': {'method': 'deep', 'model': 'auto'}
            },
            
            # 边缘保护参数
            'edge_preserving': {
                'excellent': {'method': 'none'},
                'good': {'method': 'guided', 'radius': 4, 'epsilon': 0.01},
                'fair': {'method': 'adaptive', 'flags': cv2.RECURS_FILTER, 'sigma_s': 50, 'sigma_r': 0.4},
                'poor': {'method': 'anisotropic', 'iterations': 10, 'k': 20, 'gamma': 0.1}
            },
            
            # 二值化参数
            'binarization': {
                'excellent': {'method': 'otsu'},
                'good': {'method': 'adaptive_gaussian', 'blockSize': 11, 'C': 2},
                'fair': {'method': 'adaptive_mean', 'blockSize': 15, 'C': 8},
                'poor': {'method': 'adaptive_gaussian', 'blockSize': 25, 'C': 15}
            },
            
            # 形态学操作参数
            'morphology': {
                'excellent': {'operations': []},
                'good': {'operations': [('close', (3, 3))]},
                'fair': {'operations': [('open', (2, 2)), ('close', (3, 3))]},
                'poor': {'operations': [('open', (3, 3)), ('close', (5, 5)), ('erode', (1, 1))]}
            }
        }
    
    def _initialize_algorithm_combinations(self) -> Dict[str, List]:
        """初始化算法组合策略"""
        return {
            'high_quality': [
                'super_resolution', 'geometry_correction', 'illumination', 'binarization'
            ],
            'medium_quality': [
                'super_resolution', 'geometry_correction', 'illumination', 
                'edge_preserving', 'binarization', 'morphology'
            ],
            'low_quality': [
                'super_resolution', 'geometry_correction', 'denoising', 
                'illumination', 'edge_preserving', 'binarization', 'morphology'
            ],
            'very_poor': [
                'super_resolution', 'denoising', 'geometry_correction', 
                'illumination', 'edge_preserving', 'binarization', 'morphology'
            ]
        }
    
    def select_optimal_parameters(self, assessment: Dict[str, Any]) -> Dict[str, Dict]:
        """根据图像评估选择最优参数"""
        quality_level = assessment['assessment_summary']['quality_level']
        primary_issue = assessment['assessment_summary']['primary_issue']
        
        # 基础参数选择
        optimal_params = {}
        for param_type, templates in self.parameter_templates.items():
            optimal_params[param_type] = templates.get(quality_level, templates['fair']).copy()
        
        # 根据主要问题调整参数
        optimal_params = self._adjust_for_specific_issues(optimal_params, assessment)
        
        return optimal_params
    
    def _adjust_for_specific_issues(self, params: Dict, assessment: Dict) -> Dict:
        """根据具体问题调整参数"""
        issues = assessment['assessment_summary']['identified_issues']
        
        # 针对高噪声调整
        if 'high_noise' in issues:
            if assessment['noise_level'] > 50:
                params['denoising'] = {'method': 'deep', 'model': 'dncnn'}
            elif assessment['noise_level'] > 30:
                params['denoising'] = {'method': 'nlm', 'h': 15, 'templateWindowSize': 7, 'searchWindowSize': 21}
        
        # 针对低对比度调整
        if 'low_contrast' in issues:
            params['illumination'] = {'method': 'clahe', 'clip_limit': 4.0, 'tile_grid_size': (8, 8)}
            params['binarization'] = {'method': 'adaptive_gaussian', 'blockSize': 21, 'C': 10}
        
        # 针对光照不均调整
        if 'uneven_lighting' in issues:
            if assessment['illumination_uniformity'] < 0.5:
                params['illumination'] = {'method': 'msr', 'scales': [15, 80, 250]}
            params['binarization'] = {'method': 'adaptive_mean', 'blockSize': 19, 'C': 8}
        
        # 针对模糊调整
        if 'blur' in issues:
            params['edge_preserving'] = {'method': 'guided', 'radius': 8, 'epsilon': 0.001}
            params['morphology']['operations'].append(('sharpen', None))
        
        return params
    
    def select_algorithm_sequence(self, assessment: Dict[str, Any]) -> List[str]:
        """选择算法执行顺序"""
        quality_score = assessment['overall_quality']
        
        if quality_score >= 70:
            return self.algorithm_combinations['high_quality']
        elif quality_score >= 50:
            return self.algorithm_combinations['medium_quality']
        elif quality_score >= 30:
            return self.algorithm_combinations['low_quality']
        else:
            return self.algorithm_combinations['very_poor']
    
    def calculate_dynamic_thresholds(self, image: np.ndarray, assessment: Dict) -> Dict[str, float]:
        """计算动态阈值"""
        # Otsu阈值作为基准
        otsu_threshold, _ = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 根据图像特征调整
        adjustments = {
            'base_threshold': float(otsu_threshold),
            'adaptive_threshold_c': self._calculate_adaptive_c(assessment),
            'adaptive_block_size': self._calculate_block_size(image.shape, assessment),
            'morphology_kernel_size': self._calculate_kernel_size(assessment)
        }
        
        return adjustments
    
    def _calculate_adaptive_c(self, assessment: Dict) -> float:
        """计算自适应阈值的C参数"""
        base_c = 5.0
        
        # 根据噪声级别调整
        if assessment['noise_level'] > 30:
            base_c += assessment['noise_level'] / 10
        
        # 根据对比度调整
        if assessment['is_low_contrast']:
            base_c += 5.0
        
        # 根据光照均匀性调整
        if assessment['illumination_uniformity'] < 0.7:
            base_c += (0.7 - assessment['illumination_uniformity']) * 20
        
        return min(25.0, max(2.0, base_c))
    
    def _calculate_block_size(self, image_shape: Tuple, assessment: Dict) -> int:
        """计算自适应阈值的块大小"""
        min_dim = min(image_shape)
        base_size = max(11, min_dim // 20)
        
        # 确保是奇数
        if base_size % 2 == 0:
            base_size += 1
        
        # 根据纹理复杂度调整
        if assessment.get('texture_complexity', 0) > 5:
            base_size = max(base_size - 4, 11)
        
        # 根据噪声调整
        if assessment['noise_level'] > 30:
            base_size = min(base_size + 6, 31)
        
        return base_size
    
    def _calculate_kernel_size(self, assessment: Dict) -> Tuple[int, int]:
        """计算形态学操作的核大小"""
        base_size = 3
        
        # 根据噪声级别调整
        if assessment['noise_level'] > 40:
            base_size = 5
        elif assessment['noise_level'] < 20:
            base_size = 2
        
        return (base_size, base_size)


def process_adaptive_tuning(image: np.ndarray, target_quality: str = 'auto') -> Dict[str, Any]:
    """自适应参数调优统一接口"""
    # 图像质量评估
    assessor = ImageQualityAssessment()
    assessment = assessor.comprehensive_assessment(image)
    
    # 参数调优
    tuner = AdaptiveParameterTuner()
    optimal_params = tuner.select_optimal_parameters(assessment)
    algorithm_sequence = tuner.select_algorithm_sequence(assessment)
    dynamic_thresholds = tuner.calculate_dynamic_thresholds(image, assessment)
    
    return {
        'assessment': assessment,
        'optimal_parameters': optimal_params,
        'algorithm_sequence': algorithm_sequence,
        'dynamic_thresholds': dynamic_thresholds,
        'tuning_summary': {
            'quality_score': assessment['overall_quality'],
            'quality_level': assessment['assessment_summary']['quality_level'],
            'primary_optimization': assessment['assessment_summary']['primary_issue'],
            'recommended_pipeline': ' -> '.join(algorithm_sequence)
        }
    }


if __name__ == "__main__":
    # 测试自适应参数调优系统
    print("=== 自适应参数调优系统测试 ===")
    
    # 创建测试图像
    test_images = {
        'clean': np.random.randint(100, 200, (200, 200), dtype=np.uint8),
        'noisy': np.random.randint(0, 255, (200, 200), dtype=np.uint8),
        'low_contrast': np.random.randint(80, 120, (200, 200), dtype=np.uint8)
    }
    
    for image_type, test_image in test_images.items():
        print(f"\n--- 测试 {image_type} 图像 ---")
        
        # 进行自适应调优
        tuning_result = process_adaptive_tuning(test_image)
        
        print(f"质量评分: {tuning_result['tuning_summary']['quality_score']:.1f}")
        print(f"质量等级: {tuning_result['tuning_summary']['quality_level']}")
        print(f"主要问题: {tuning_result['tuning_summary']['primary_optimization']}")
        print(f"推荐管道: {tuning_result['tuning_summary']['recommended_pipeline']}")
        
        # 显示关键参数
        print("关键参数:")
        for param_type, params in tuning_result['optimal_parameters'].items():
            if param_type in ['denoising', 'binarization']:
                print(f"  {param_type}: {params}")
    
    print("\n�� 自适应参数调优系统测试成功！") 