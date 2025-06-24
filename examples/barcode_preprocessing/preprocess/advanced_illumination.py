"""
高级光照校正算法模块
实现多种光照校正技术以改善条形码图像在复杂光照环境下的识别率
"""

import numpy as np
import cv2
from typing import Union, Tuple, List, Optional
import logging

logger = logging.getLogger(__name__)


class AdvancedIllumination:
    """高级光照校正算法集合"""
    
    @staticmethod
    def multi_scale_retinex(image: np.ndarray, 
                           scales: List[float] = [15, 80, 250],
                           alpha: float = 125,
                           beta: float = 46) -> np.ndarray:
        """
        Multi-Scale Retinex (MSR) 多尺度视网膜增强算法
        
        Args:
            image: 输入图像 (H, W) 或 (H, W, C)
            scales: 高斯核尺度列表
            alpha: 增益参数
            beta: 偏移参数
            
        Returns:
            处理后的图像
        """
        if len(image.shape) == 3:
            # 转换为灰度图
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
            
        # 转换为浮点数并添加小常数避免log(0)
        img_float = gray.astype(np.float64) + 1.0
        img_log = np.log(img_float)
        
        # 多尺度处理
        msr_result = np.zeros_like(img_log)
        for scale in scales:
            # 创建高斯核
            kernel_size = int(6 * scale + 1)
            if kernel_size % 2 == 0:
                kernel_size += 1
                
            # 高斯模糊
            blurred = cv2.GaussianBlur(img_float, (kernel_size, kernel_size), scale)
            blurred_log = np.log(blurred + 1e-10)
            
            # 单尺度retinex
            ssr = img_log - blurred_log
            msr_result += ssr
            
        # 平均
        msr_result /= len(scales)
        
        # 缩放和偏移
        msr_result = alpha * msr_result + beta
        
        # 截断到有效范围
        msr_result = np.clip(msr_result, 0, 255)
        
        return msr_result.astype(np.uint8)
    
    @staticmethod
    def single_scale_retinex(image: np.ndarray, 
                            scale: float = 80) -> np.ndarray:
        """
        Single Scale Retinex (SSR) 单尺度视网膜增强
        
        Args:
            image: 输入图像
            scale: 高斯核尺度
            
        Returns:
            处理后的图像
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
            
        # 转换为浮点数
        img_float = gray.astype(np.float64) + 1.0
        
        # 高斯模糊
        kernel_size = int(6 * scale + 1)
        if kernel_size % 2 == 0:
            kernel_size += 1
            
        blurred = cv2.GaussianBlur(img_float, (kernel_size, kernel_size), scale)
        
        # SSR计算
        ssr = np.log(img_float + 1e-10) - np.log(blurred + 1e-10)
        
        # 归一化到[0, 255]
        ssr_norm = cv2.normalize(ssr, None, 0, 255, cv2.NORM_MINMAX)
        
        return ssr_norm.astype(np.uint8)
    
    @staticmethod
    def adaptive_gamma_correction(image: np.ndarray, 
                                 auto_gamma: bool = True,
                                 gamma: float = 1.0) -> np.ndarray:
        """
        自适应伽马校正
        
        Args:
            image: 输入图像
            auto_gamma: 是否自动计算gamma值
            gamma: 手动指定的gamma值（当auto_gamma=False时使用）
            
        Returns:
            伽马校正后的图像
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
            
        if auto_gamma:
            # 基于图像统计特性自动计算gamma
            mean_intensity = np.mean(gray) / 255.0
            
            # 根据平均亮度调整gamma
            if mean_intensity < 0.3:
                # 暗图像，使用gamma < 1增强亮度
                gamma = 0.5 + 0.3 * (mean_intensity / 0.3)
            elif mean_intensity > 0.7:
                # 亮图像，使用gamma > 1降低亮度
                gamma = 1.0 + 0.5 * ((mean_intensity - 0.7) / 0.3)
            else:
                # 中等亮度，轻微调整
                gamma = 0.8 + 0.4 * ((mean_intensity - 0.3) / 0.4)
                
        # 构建查找表
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 
                         for i in np.arange(0, 256)]).astype(np.uint8)
        
        # 应用伽马校正
        corrected = cv2.LUT(gray, table)
        
        return corrected
    
    @staticmethod
    def clahe_enhanced(image: np.ndarray,
                      clip_limit: float = 2.0,
                      tile_grid_size: Tuple[int, int] = (8, 8),
                      adaptive_clip: bool = True) -> np.ndarray:
        """
        增强版CLAHE（对比度限制自适应直方图均衡化）
        
        Args:
            image: 输入图像
            clip_limit: 对比度限制阈值
            tile_grid_size: 网格大小
            adaptive_clip: 是否自适应调整clip_limit
            
        Returns:
            CLAHE处理后的图像
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
            
        if adaptive_clip:
            # 根据图像特征自适应调整clip_limit
            contrast = np.std(gray)
            brightness = np.mean(gray)
            
            # 低对比度图像需要更高的clip_limit
            if contrast < 30:
                clip_limit = 3.0 + (30 - contrast) / 10
            elif contrast > 80:
                clip_limit = 1.5 + (contrast - 80) / 50
            else:
                clip_limit = 2.0
                
            # 根据亮度微调
            if brightness < 60:
                clip_limit *= 1.2
            elif brightness > 180:
                clip_limit *= 0.8
                
        # 创建CLAHE对象
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        
        # 应用CLAHE
        enhanced = clahe.apply(gray)
        
        return enhanced
    
    @staticmethod
    def retinex_msrcr(image: np.ndarray,
                     scales: List[float] = [15, 80, 250],
                     alpha: float = 125,
                     beta: float = 46,
                     color_restoration_factor: float = 1.2) -> np.ndarray:
        """
        Multi-Scale Retinex with Color Restoration (MSRCR)
        带颜色恢复的多尺度视网膜算法
        
        Args:
            image: 输入图像
            scales: 尺度列表
            alpha: 增益参数
            beta: 偏移参数
            color_restoration_factor: 颜色恢复因子
            
        Returns:
            处理后的图像
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
            
        # 基本MSR处理
        msr = AdvancedIllumination.multi_scale_retinex(gray, scales, alpha, beta)
        
        # 颜色恢复处理（针对灰度图的增强）
        if len(image.shape) == 3:
            # 计算颜色恢复权重
            intensity_sum = np.sum(image.astype(np.float64), axis=2)
            intensity_sum[intensity_sum == 0] = 1  # 避免除零
            
            # 应用MSR到每个通道
            result = np.zeros_like(image)
            for c in range(3):
                channel_weight = (image[:, :, c].astype(np.float64) * color_restoration_factor) / intensity_sum
                result[:, :, c] = np.clip(msr * channel_weight, 0, 255)
                
            return result.astype(np.uint8)
        else:
            return msr
    
    @staticmethod
    def histogram_specification(image: np.ndarray,
                               target_hist: Optional[np.ndarray] = None) -> np.ndarray:
        """
        直方图规定化/匹配
        将图像直方图匹配到目标分布
        
        Args:
            image: 输入图像
            target_hist: 目标直方图（如果为None，使用理想均匀分布）
            
        Returns:
            直方图匹配后的图像
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
            
        # 计算原始直方图和CDF
        hist, bins = np.histogram(gray.flatten(), 256, [0, 256])
        cdf = hist.cumsum()
        cdf_normalized = cdf * (255 / cdf[-1])
        
        if target_hist is None:
            # 使用理想的均匀分布作为目标
            target_cdf = np.linspace(0, 255, 256)
        else:
            # 计算目标直方图的CDF
            target_cdf = target_hist.cumsum()
            target_cdf = target_cdf * (255 / target_cdf[-1])
        
        # 构建映射表
        mapping = np.interp(cdf_normalized, target_cdf, np.arange(256))
        
        # 应用映射
        result = np.interp(gray.flatten(), np.arange(256), mapping)
        result = result.reshape(gray.shape).astype(np.uint8)
        
        return result
    
    @staticmethod
    def local_contrast_enhancement(image: np.ndarray,
                                  kernel_size: int = 9,
                                  strength: float = 1.5) -> np.ndarray:
        """
        局部对比度增强
        
        Args:
            image: 输入图像
            kernel_size: 局部窗口大小
            strength: 增强强度
            
        Returns:
            对比度增强后的图像
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
            
        # 计算局部均值
        mean_local = cv2.blur(gray.astype(np.float32), (kernel_size, kernel_size))
        
        # 计算局部标准差
        mean_local_sq = cv2.blur((gray.astype(np.float32))**2, (kernel_size, kernel_size))
        std_local = np.sqrt(np.maximum(mean_local_sq - mean_local**2, 0))
        
        # 避免除零
        std_local[std_local < 1] = 1
        
        # 局部对比度增强
        enhanced = mean_local + strength * (gray.astype(np.float32) - mean_local) * (50 / std_local)
        
        # 截断到有效范围
        enhanced = np.clip(enhanced, 0, 255)
        
        return enhanced.astype(np.uint8)


def process_illumination(image: np.ndarray, 
                        method: str = 'msr',
                        **kwargs) -> np.ndarray:
    """
    光照校正主函数
    
    Args:
        image: 输入图像
        method: 处理方法 ('msr', 'ssr', 'gamma', 'clahe', 'msrcr', 'hist_spec', 'local_contrast')
        **kwargs: 方法特定参数
        
    Returns:
        处理后的图像
    """
    try:
        if method == 'msr':
            return AdvancedIllumination.multi_scale_retinex(image, **kwargs)
        elif method == 'ssr':
            return AdvancedIllumination.single_scale_retinex(image, **kwargs)
        elif method == 'gamma':
            return AdvancedIllumination.adaptive_gamma_correction(image, **kwargs)
        elif method == 'clahe':
            return AdvancedIllumination.clahe_enhanced(image, **kwargs)
        elif method == 'msrcr':
            return AdvancedIllumination.retinex_msrcr(image, **kwargs)
        elif method == 'hist_spec':
            return AdvancedIllumination.histogram_specification(image, **kwargs)
        elif method == 'local_contrast':
            return AdvancedIllumination.local_contrast_enhancement(image, **kwargs)
        else:
            logger.warning(f"未知的光照校正方法: {method}")
            return image
            
    except Exception as e:
        logger.error(f"光照校正处理失败: {e}")
        return image


def test_illumination_algorithms():
    """测试所有光照校正算法"""
    import os
    from pathlib import Path
    
    # 创建测试图像（模拟不均匀光照）
    test_img = np.zeros((200, 300), dtype=np.uint8)
    
    # 添加渐变光照
    for i in range(200):
        for j in range(300):
            # 创建从左上到右下的光照渐变
            intensity = int(50 + 150 * (i + j) / (200 + 300))
            test_img[i, j] = min(255, intensity)
    
    # 添加条形码样式的模式
    for i in range(80, 120):
        for j in range(50, 250, 10):
            if (j // 10) % 2 == 0:
                test_img[i:i+5, j:j+5] = 20  # 暗条
            else:
                test_img[i:i+5, j:j+5] = 200  # 亮条
    
    methods = ['msr', 'ssr', 'gamma', 'clahe', 'msrcr', 'local_contrast']
    
    print("测试所有光照校正算法...")
    for method in methods:
        try:
            result = process_illumination(test_img, method)
            print(f"✓ {method}: 处理成功，输出shape: {result.shape}")
        except Exception as e:
            print(f"✗ {method}: 处理失败 - {e}")
    
    print("光照校正算法测试完成！")


if __name__ == "__main__":
    test_illumination_algorithms() 