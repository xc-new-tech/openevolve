"""
图像增强算法模块
专门针对条形码识别的图像增强技术
"""

import numpy as np
import cv2
from typing import Tuple, Optional, List
import logging

logger = logging.getLogger(__name__)


class ImageEnhancement:
    """图像增强算法集合"""
    
    @staticmethod
    def unsharp_masking(image: np.ndarray,
                       sigma: float = 1.0,
                       strength: float = 1.5,
                       threshold: float = 0) -> np.ndarray:
        """
        反锐化掩模 (Unsharp Masking)
        经典的图像锐化技术
        
        Args:
            image: 输入图像
            sigma: 高斯模糊的标准差
            strength: 锐化强度
            threshold: 阈值，只对超过此值的像素进行锐化
            
        Returns:
            锐化后的图像
        """
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
    
    @staticmethod
    def laplacian_sharpening(image: np.ndarray,
                           strength: float = 1.0,
                           kernel_type: str = 'standard') -> np.ndarray:
        """
        拉普拉斯锐化
        基于拉普拉斯算子的边缘增强
        
        Args:
            image: 输入图像
            strength: 锐化强度
            kernel_type: 核类型 ('standard', 'enhanced', 'diagonal')
            
        Returns:
            锐化后的图像
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
            
        # 选择拉普拉斯核
        if kernel_type == 'standard':
            kernel = np.array([[0, -1, 0],
                             [-1, 4, -1],
                             [0, -1, 0]], dtype=np.float32)
        elif kernel_type == 'enhanced':
            kernel = np.array([[-1, -1, -1],
                             [-1, 8, -1],
                             [-1, -1, -1]], dtype=np.float32)
        elif kernel_type == 'diagonal':
            kernel = np.array([[-1, -2, -1],
                             [-2, 12, -2],
                             [-1, -2, -1]], dtype=np.float32)
        else:
            logger.warning(f"未知的核类型: {kernel_type}")
            kernel = np.array([[0, -1, 0],
                             [-1, 4, -1],
                             [0, -1, 0]], dtype=np.float32)
        
        # 应用拉普拉斯滤波
        laplacian = cv2.filter2D(gray, cv2.CV_32F, kernel)
        
        # 锐化图像
        sharpened = gray.astype(np.float32) - strength * laplacian
        
        # 截断到有效范围
        result = np.clip(sharpened, 0, 255).astype(np.uint8)
        
        return result
    
    @staticmethod
    def adaptive_histogram_equalization(image: np.ndarray,
                                      clip_limit: float = 2.0,
                                      tile_grid_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
        """
        自适应直方图均衡化 (CLAHE)
        改进版本，针对条形码优化
        
        Args:
            image: 输入图像
            clip_limit: 对比度限制阈值
            tile_grid_size: 瓦片网格大小
            
        Returns:
            均衡化后的图像
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
            
        # 创建CLAHE对象
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        
        # 应用CLAHE
        result = clahe.apply(gray)
        
        return result
    
    @staticmethod
    def contrast_stretching(image: np.ndarray,
                          percentile_low: float = 2,
                          percentile_high: float = 98) -> np.ndarray:
        """
        对比度拉伸
        基于百分位数的线性拉伸
        
        Args:
            image: 输入图像
            percentile_low: 低百分位数
            percentile_high: 高百分位数
            
        Returns:
            拉伸后的图像
        """
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
    
    @staticmethod
    def gamma_correction(image: np.ndarray,
                        gamma: float = 1.0,
                        adaptive: bool = True) -> np.ndarray:
        """
        伽马校正
        支持自适应伽马值选择
        
        Args:
            image: 输入图像
            gamma: 伽马值（当adaptive=False时使用）
            adaptive: 是否自适应选择伽马值
            
        Returns:
            校正后的图像
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
            
        if adaptive:
            # 基于图像均值自适应选择伽马值
            mean_val = np.mean(gray) / 255.0
            
            if mean_val < 0.3:
                # 图像较暗，提高伽马值
                gamma = 0.5
            elif mean_val > 0.7:
                # 图像较亮，降低伽马值
                gamma = 1.5
            else:
                # 中等亮度，使用默认值
                gamma = 1.0
        
        # 构建查找表
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 
                         for i in np.arange(0, 256)]).astype(np.uint8)
        
        # 应用伽马校正
        result = cv2.LUT(gray, table)
        
        return result
    
    @staticmethod
    def local_adaptive_threshold_enhance(image: np.ndarray,
                                       window_size: int = 15,
                                       c: float = 10) -> np.ndarray:
        """
        局部自适应阈值增强
        基于局部统计的对比度增强
        
        Args:
            image: 输入图像
            window_size: 局部窗口大小
            c: 常数项
            
        Returns:
            增强后的图像
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
            
        # 计算局部均值
        kernel = np.ones((window_size, window_size), np.float32) / (window_size * window_size)
        local_mean = cv2.filter2D(gray.astype(np.float32), -1, kernel)
        
        # 计算局部标准差
        local_sqr_mean = cv2.filter2D((gray.astype(np.float32))**2, -1, kernel)
        local_var = local_sqr_mean - local_mean**2
        local_std = np.sqrt(np.maximum(local_var, 0))
        
        # 自适应增强
        enhanced = gray.astype(np.float32)
        mask = gray.astype(np.float32) > (local_mean + c)
        
        # 对高于局部均值的像素进行增强
        enhanced = np.where(mask, 
                          np.minimum(255, gray + local_std * 0.5),
                          np.maximum(0, gray - local_std * 0.3))
        
        result = np.clip(enhanced, 0, 255).astype(np.uint8)
        
        return result
    
    @staticmethod
    def high_frequency_emphasis(image: np.ndarray,
                              cutoff: float = 0.1,
                              amplification: float = 2.0) -> np.ndarray:
        """
        高频强调滤波
        在频域增强高频成分
        
        Args:
            image: 输入图像
            cutoff: 截止频率（归一化）
            amplification: 高频放大倍数
            
        Returns:
            增强后的图像
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
            
        # 转换为浮点数
        img_float = gray.astype(np.float64)
        
        # 傅里叶变换
        f_transform = np.fft.fft2(img_float)
        f_shift = np.fft.fftshift(f_transform)
        
        # 创建高频强调滤波器
        rows, cols = gray.shape
        crow, ccol = rows // 2, cols // 2
        
        # 创建网格
        u = np.arange(rows).reshape(-1, 1) - crow
        v = np.arange(cols) - ccol
        D = np.sqrt(u**2 + v**2)
        
        # 归一化距离
        D_norm = D / np.max(D)
        
        # 高频强调滤波器 H(u,v) = 1 + amplification * (1 - exp(-D^2/cutoff^2))
        H = 1 + amplification * (1 - np.exp(-(D_norm**2) / (cutoff**2)))
        
        # 应用滤波器
        filtered = f_shift * H
        
        # 逆傅里叶变换
        f_ishift = np.fft.ifftshift(filtered)
        result = np.fft.ifft2(f_ishift)
        result = np.abs(result)
        
        # 截断到有效范围
        result = np.clip(result, 0, 255).astype(np.uint8)
        
        return result
    
    @staticmethod
    def barcode_specific_enhance(image: np.ndarray,
                               bar_direction: str = 'vertical') -> np.ndarray:
        """
        条形码专用增强
        针对条形码特征的定向增强
        
        Args:
            image: 输入图像
            bar_direction: 条形码方向 ('vertical', 'horizontal', 'auto')
            
        Returns:
            增强后的图像
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
            
        if bar_direction == 'auto':
            # 自动检测条形码方向
            # 计算水平和垂直方向的梯度
            grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            
            # 计算梯度强度
            grad_x_mean = np.mean(np.abs(grad_x))
            grad_y_mean = np.mean(np.abs(grad_y))
            
            if grad_x_mean > grad_y_mean:
                bar_direction = 'vertical'
            else:
                bar_direction = 'horizontal'
        
        # 创建定向增强核
        if bar_direction == 'vertical':
            # 垂直条形码：增强水平边缘
            kernel = np.array([[-1, -1, -1],
                             [ 2,  2,  2],
                             [-1, -1, -1]], dtype=np.float32)
        else:
            # 水平条形码：增强垂直边缘
            kernel = np.array([[-1,  2, -1],
                             [-1,  2, -1],
                             [-1,  2, -1]], dtype=np.float32)
        
        # 应用定向滤波
        enhanced = cv2.filter2D(gray, cv2.CV_32F, kernel)
        
        # 与原图像组合
        result = cv2.addWeighted(gray, 0.8, enhanced.astype(np.uint8), 0.2, 0)
        
        return result
    
    @staticmethod
    def multi_scale_enhancement(image: np.ndarray,
                              scales: List[float] = [0.5, 1.0, 1.5],
                              weights: Optional[List[float]] = None) -> np.ndarray:
        """
        多尺度增强
        结合不同尺度的图像特征
        
        Args:
            image: 输入图像
            scales: 尺度列表
            weights: 权重列表（如果为None，使用均等权重）
            
        Returns:
            增强后的图像
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
            
        if weights is None:
            weights = [1.0 / len(scales)] * len(scales)
        
        # 确保权重归一化
        weights = np.array(weights)
        weights = weights / np.sum(weights)
        
        h, w = gray.shape
        enhanced = np.zeros((h, w), dtype=np.float32)
        
        for scale, weight in zip(scales, weights):
            if scale == 1.0:
                # 原始尺度
                scaled_img = gray.astype(np.float32)
            else:
                # 缩放图像
                new_h, new_w = int(h * scale), int(w * scale)
                scaled = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
                # 缩放回原始尺寸
                scaled_img = cv2.resize(scaled, (w, h), interpolation=cv2.INTER_CUBIC).astype(np.float32)
            
            # 累加权重
            enhanced += weight * scaled_img
        
        # 截断到有效范围
        result = np.clip(enhanced, 0, 255).astype(np.uint8)
        
        return result


def process_enhancement(image: np.ndarray,
                       method: str = 'unsharp',
                       **kwargs) -> np.ndarray:
    """
    图像增强主函数
    
    Args:
        image: 输入图像
        method: 增强方法 ('unsharp', 'laplacian', 'clahe', 'contrast_stretch',
                'gamma', 'local_adaptive', 'high_freq', 'barcode_specific', 'multi_scale')
        **kwargs: 方法特定参数
        
    Returns:
        增强后的图像
    """
    try:
        if method == 'unsharp':
            return ImageEnhancement.unsharp_masking(image, **kwargs)
        elif method == 'laplacian':
            return ImageEnhancement.laplacian_sharpening(image, **kwargs)
        elif method == 'clahe':
            return ImageEnhancement.adaptive_histogram_equalization(image, **kwargs)
        elif method == 'contrast_stretch':
            return ImageEnhancement.contrast_stretching(image, **kwargs)
        elif method == 'gamma':
            return ImageEnhancement.gamma_correction(image, **kwargs)
        elif method == 'local_adaptive':
            return ImageEnhancement.local_adaptive_threshold_enhance(image, **kwargs)
        elif method == 'high_freq':
            return ImageEnhancement.high_frequency_emphasis(image, **kwargs)
        elif method == 'barcode_specific':
            return ImageEnhancement.barcode_specific_enhance(image, **kwargs)
        elif method == 'multi_scale':
            return ImageEnhancement.multi_scale_enhancement(image, **kwargs)
        else:
            logger.warning(f"未知的增强方法: {method}")
            return image
            
    except Exception as e:
        logger.error(f"图像增强处理失败: {e}")
        return image


def test_enhancement_algorithms():
    """测试所有图像增强算法"""
    # 创建测试图像：低对比度的条形码
    test_img = np.full((200, 300), 128, dtype=np.uint8)  # 灰色背景
    
    # 添加低对比度条形码图案
    for i in range(80, 120):
        for j in range(50, 250, 10):
            if (j // 10) % 2 == 0:
                test_img[i:i+5, j:j+5] = 100  # 稍暗的条
            else:
                test_img[i:i+5, j:j+5] = 156  # 稍亮的条
    
    # 添加轻微模糊
    blurred_img = cv2.GaussianBlur(test_img, (3, 3), 1.0)
    
    methods = [
        'unsharp', 'laplacian', 'clahe', 'contrast_stretch', 
        'gamma', 'local_adaptive', 'high_freq', 'barcode_specific', 'multi_scale'
    ]
    
    print("测试所有图像增强算法...")
    print(f"原始图像: shape={test_img.shape}")
    print(f"测试图像: shape={blurred_img.shape}, 对比度={np.std(blurred_img):.2f}")
    
    for method in methods:
        try:
            result = process_enhancement(blurred_img, method)
            
            # 计算对比度改善
            orig_contrast = np.std(blurred_img)
            enhanced_contrast = np.std(result)
            contrast_improvement = enhanced_contrast / orig_contrast
            
            # 计算锐度改善（基于拉普拉斯方差）
            orig_sharpness = cv2.Laplacian(blurred_img, cv2.CV_64F).var()
            enhanced_sharpness = cv2.Laplacian(result, cv2.CV_64F).var()
            if orig_sharpness > 0:
                sharpness_improvement = enhanced_sharpness / orig_sharpness
            else:
                sharpness_improvement = 1.0
            
            print(f"✓ {method}: 处理成功，对比度提升: {contrast_improvement:.2f}x, 锐度提升: {sharpness_improvement:.2f}x")
            
        except Exception as e:
            print(f"✗ {method}: 处理失败 - {e}")
    
    print("图像增强算法测试完成！")


if __name__ == "__main__":
    test_enhancement_algorithms() 