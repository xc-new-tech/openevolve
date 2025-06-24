#!/usr/bin/env python3
"""
重构后的图像增强算法模块
基于BaseProcessor的统一接口实现
"""

import numpy as np
import cv2
from typing import Tuple, Optional, List, Dict, Any
from base import BaseProcessor, ProcessorConfig

class UnsharpMaskingProcessor(BaseProcessor):
    """反锐化掩模处理器"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        default_config = {
            'sigma': 1.0,
            'strength': 1.5,
            'threshold': 0
        }
        if config:
            default_config.update(config)
        super().__init__(default_config)
    
    def _process(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """执行反锐化掩模处理"""
        sigma = kwargs.get('sigma', self.config.get('sigma'))
        strength = kwargs.get('strength', self.config.get('strength'))
        threshold = kwargs.get('threshold', self.config.get('threshold'))
        
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
            
        # 高斯模糊
        blurred = cv2.GaussianBlur(gray, (0, 0), sigma)
        
        # 计算锐化掩模
        mask = cv2.subtract(gray, blurred)
        
        # 应用阈值
        if threshold > 0:
            mask = np.where(np.abs(mask) >= threshold, mask, 0)
        
        # 应用锐化
        sharpened = cv2.addWeighted(gray, 1.0, mask, strength, 0)
        
        # 截断到有效范围
        result = np.clip(sharpened, 0, 255).astype(np.uint8)
        
        return result

class LaplacianSharpeningProcessor(BaseProcessor):
    """拉普拉斯锐化处理器"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        default_config = {
            'strength': 1.0,
            'kernel_type': 'standard'
        }
        if config:
            default_config.update(config)
        super().__init__(default_config)
    
    def _process(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """执行拉普拉斯锐化处理"""
        strength = kwargs.get('strength', self.config.get('strength'))
        kernel_type = kwargs.get('kernel_type', self.config.get('kernel_type'))
        
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
            
        # 选择拉普拉斯核
        kernels = {
            'standard': np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=np.float32),
            'enhanced': np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=np.float32),
            'diagonal': np.array([[-1, -2, -1], [-2, 12, -2], [-1, -2, -1]], dtype=np.float32)
        }
        
        kernel = kernels.get(kernel_type, kernels['standard'])
        if kernel_type not in kernels:
            self.logger.warning(f"未知的核类型: {kernel_type}, 使用标准核")
        
        # 应用拉普拉斯滤波
        laplacian = cv2.filter2D(gray, cv2.CV_32F, kernel)
        
        # 锐化图像
        sharpened = gray.astype(np.float32) - strength * laplacian
        
        # 截断到有效范围
        result = np.clip(sharpened, 0, 255).astype(np.uint8)
        
        return result

class CLAHEProcessor(BaseProcessor):
    """自适应直方图均衡化处理器"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        default_config = {
            'clip_limit': 2.0,
            'tile_grid_size': (8, 8)
        }
        if config:
            default_config.update(config)
        super().__init__(default_config)
    
    def _process(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """执行CLAHE处理"""
        clip_limit = kwargs.get('clip_limit', self.config.get('clip_limit'))
        tile_grid_size = kwargs.get('tile_grid_size', self.config.get('tile_grid_size'))
        
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
            
        # 创建CLAHE对象
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        
        # 应用CLAHE
        result = clahe.apply(gray)
        
        return result

class ContrastStretchingProcessor(BaseProcessor):
    """对比度拉伸处理器"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        default_config = {
            'percentile_low': 2,
            'percentile_high': 98
        }
        if config:
            default_config.update(config)
        super().__init__(default_config)
    
    def _process(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """执行对比度拉伸处理"""
        percentile_low = kwargs.get('percentile_low', self.config.get('percentile_low'))
        percentile_high = kwargs.get('percentile_high', self.config.get('percentile_high'))
        
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
            
        # 计算百分位数
        p_low = np.percentile(gray, percentile_low)
        p_high = np.percentile(gray, percentile_high)
        
        # 线性拉伸
        if p_high > p_low:
            stretched = 255 * (gray - p_low) / (p_high - p_low)
        else:
            stretched = gray.astype(np.float32)
            
        # 截断到有效范围
        result = np.clip(stretched, 0, 255).astype(np.uint8)
        
        return result

class GammaCorrectionProcessor(BaseProcessor):
    """伽马校正处理器"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        default_config = {
            'gamma': 1.0,
            'adaptive': True
        }
        if config:
            default_config.update(config)
        super().__init__(default_config)
    
    def _process(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """执行伽马校正处理"""
        gamma = kwargs.get('gamma', self.config.get('gamma'))
        adaptive = kwargs.get('adaptive', self.config.get('adaptive'))
        
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
            
        if adaptive:
            # 基于图像均值自适应选择伽马值
            mean_val = np.mean(gray) / 255.0
            
            if mean_val < 0.3:
                gamma = 1.5  # 图像较暗，提高伽马值
            elif mean_val > 0.7:
                gamma = 0.7  # 图像较亮，降低伽马值
            else:
                gamma = 1.0  # 中等亮度，不调整
                
            self.logger.debug(f"自适应伽马值: {gamma:.2f} (均值: {mean_val:.3f})")
        
        # 构建查找表
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype(np.uint8)
        
        # 应用伽马校正
        result = cv2.LUT(gray, table)
        
        return result

class HighFrequencyEmphasisProcessor(BaseProcessor):
    """高频强调处理器"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        default_config = {
            'cutoff': 0.1,
            'amplification': 2.0
        }
        if config:
            default_config.update(config)
        super().__init__(default_config)
    
    def _process(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """执行高频强调处理"""
        cutoff = kwargs.get('cutoff', self.config.get('cutoff'))
        amplification = kwargs.get('amplification', self.config.get('amplification'))
        
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
            
        # 转换为浮点数
        f_image = gray.astype(np.float32) / 255.0
        
        # FFT变换
        f_transform = np.fft.fft2(f_image)
        f_shift = np.fft.fftshift(f_transform)
        
        # 创建高通滤波器
        rows, cols = f_image.shape
        crow, ccol = rows // 2, cols // 2
        
        # 创建高频强调滤波器
        y, x = np.ogrid[:rows, :cols]
        mask = np.sqrt((x - ccol)**2 + (y - crow)**2)
        mask = mask / np.max(mask)
        
        # 高频强调函数: H(u,v) = 1 + amplification * (1 - exp(-mask^2 / cutoff^2))
        hfe_filter = 1 + amplification * (1 - np.exp(-(mask**2) / (cutoff**2)))
        
        # 应用滤波器
        f_shift_filtered = f_shift * hfe_filter
        
        # 逆变换
        f_ishift = np.fft.ifftshift(f_shift_filtered)
        img_back = np.fft.ifft2(f_ishift)
        img_back = np.real(img_back)
        
        # 归一化到0-255
        result = np.clip(img_back * 255, 0, 255).astype(np.uint8)
        
        return result

class BarcodeSpecificEnhancementProcessor(BaseProcessor):
    """条形码专用增强处理器"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        default_config = {
            'bar_direction': 'vertical'
        }
        if config:
            default_config.update(config)
        super().__init__(default_config)
    
    def _process(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """执行条形码专用增强处理"""
        bar_direction = kwargs.get('bar_direction', self.config.get('bar_direction'))
        
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
            
        # 方向相关的形态学操作
        if bar_direction == 'vertical':
            # 垂直条形码：水平方向的开闭运算
            kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1))
            kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
        else:
            # 水平条形码：垂直方向的开闭运算
            kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))
            kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 9))
        
        # 开运算去除噪声
        opened = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel_v)
        
        # 闭运算连接断裂的条纹
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel_h)
        
        # 增强对比度
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(closed)
        
        return enhanced

class MultiScaleEnhancementProcessor(BaseProcessor):
    """多尺度增强处理器"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        default_config = {
            'scales': [0.5, 1.0, 1.5],
            'weights': None
        }
        if config:
            default_config.update(config)
        super().__init__(default_config)
    
    def _process(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """执行多尺度增强处理"""
        scales = kwargs.get('scales', self.config.get('scales'))
        weights = kwargs.get('weights', self.config.get('weights'))
        
        if weights is None:
            weights = [1.0 / len(scales)] * len(scales)
        
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
            
        original_size = gray.shape[::-1]  # (width, height)
        enhanced_images = []
        
        for scale in scales:
            # 缩放图像
            if scale != 1.0:
                new_size = (int(original_size[0] * scale), int(original_size[1] * scale))
                scaled = cv2.resize(gray, new_size, interpolation=cv2.INTER_CUBIC)
            else:
                scaled = gray.copy()
            
            # 增强处理
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(scaled)
            
            # 锐化
            kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]], dtype=np.float32)
            sharpened = cv2.filter2D(enhanced, -1, kernel)
            
            # 缩放回原始尺寸
            if scale != 1.0:
                resized = cv2.resize(sharpened, original_size, interpolation=cv2.INTER_CUBIC)
            else:
                resized = sharpened
                
            enhanced_images.append(resized.astype(np.float32))
        
        # 加权融合
        result = np.zeros_like(enhanced_images[0])
        for img, weight in zip(enhanced_images, weights):
            result += weight * img
        
        # 归一化
        result = np.clip(result, 0, 255).astype(np.uint8)
        
        return result

# 工厂函数
def create_enhancement_processor(method: str, config: Optional[Dict[str, Any]] = None) -> BaseProcessor:
    """
    创建增强处理器的工厂函数
    
    Args:
        method: 处理方法名称
        config: 配置参数
        
    Returns:
        处理器实例
    """
    processors = {
        'unsharp': UnsharpMaskingProcessor,
        'laplacian': LaplacianSharpeningProcessor,
        'clahe': CLAHEProcessor,
        'contrast_stretch': ContrastStretchingProcessor,
        'gamma': GammaCorrectionProcessor,
        'high_freq': HighFrequencyEmphasisProcessor,
        'barcode_specific': BarcodeSpecificEnhancementProcessor,
        'multi_scale': MultiScaleEnhancementProcessor
    }
    
    if method not in processors:
        raise ValueError(f"未知的增强方法: {method}. 可用方法: {list(processors.keys())}")
    
    return processors[method](config)

# 兼容性函数 - 保持与原始接口的兼容
def process_enhancement(image: np.ndarray, method: str = 'unsharp', **kwargs) -> np.ndarray:
    """
    处理图像增强的主函数（兼容性接口）
    
    Args:
        image: 输入图像
        method: 增强方法
        **kwargs: 其他参数
        
    Returns:
        增强后的图像
    """
    processor = create_enhancement_processor(method, kwargs)
    result = processor.process(image)
    
    if result.success:
        return result.image
    else:
        raise RuntimeError(f"图像增强失败: {result.message}") 