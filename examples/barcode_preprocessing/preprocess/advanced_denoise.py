"""
高级去噪算法模块
扩展现有去噪功能，添加更强大的去噪技术
"""

import numpy as np
import cv2
from scipy import ndimage, signal
from typing import Tuple, Optional, Union
import logging

logger = logging.getLogger(__name__)


class AdvancedDenoise:
    """高级去噪算法集合"""
    
    @staticmethod
    def non_local_means_denoise(image: np.ndarray,
                               h: float = 10,
                               template_window_size: int = 7,
                               search_window_size: int = 21) -> np.ndarray:
        """
        非局部均值去噪 (Non-local Means)
        利用图像自相似性进行强力去噪
        
        Args:
            image: 输入图像
            h: 滤波器强度。h越大，去噪越强，但也会移除图像细节
            template_window_size: 模板窗口大小，应该是奇数
            search_window_size: 搜索窗口大小，应该是奇数
            
        Returns:
            去噪后的图像
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
            
        # OpenCV的非局部均值去噪
        denoised = cv2.fastNlMeansDenoising(
            gray, 
            None, 
            h=h,
            templateWindowSize=template_window_size,
            searchWindowSize=search_window_size
        )
        
        return denoised
    
    @staticmethod
    def bilateral_filter_enhanced(image: np.ndarray,
                                 d: int = 9,
                                 sigma_color: float = 75,
                                 sigma_space: float = 75,
                                 adaptive: bool = True) -> np.ndarray:
        """
        增强版双边滤波
        改进的边缘保护滤波
        
        Args:
            image: 输入图像
            d: 像素邻域直径
            sigma_color: 颜色空间的标准方差
            sigma_space: 坐标空间的标准方差
            adaptive: 是否自适应调整参数
            
        Returns:
            滤波后的图像
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
            
        if adaptive:
            # 根据图像噪声水平自适应调整参数
            noise_level = np.std(gray)
            
            if noise_level < 10:
                # 低噪声图像，减少滤波强度
                sigma_color = 50
                sigma_space = 50
                d = 5
            elif noise_level > 30:
                # 高噪声图像，增加滤波强度
                sigma_color = 100
                sigma_space = 100
                d = 13
            else:
                # 中等噪声，使用默认参数
                pass
                
        # 应用双边滤波
        filtered = cv2.bilateralFilter(gray, d, sigma_color, sigma_space)
        
        return filtered
    
    @staticmethod
    def wiener_filter(image: np.ndarray,
                     noise_variance: Optional[float] = None,
                     psf: Optional[np.ndarray] = None) -> np.ndarray:
        """
        维纳滤波器 - 频域噪声抑制
        
        Args:
            image: 输入图像
            noise_variance: 噪声方差（如果为None，会自动估计）
            psf: 点扩散函数（如果为None，假设无模糊）
            
        Returns:
            滤波后的图像
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
            
        # 转换为浮点数
        img_float = gray.astype(np.float64)
        
        # 估计噪声方差
        if noise_variance is None:
            # 使用Laplacian算子估计噪声
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            noise_variance = np.var(laplacian) * 0.5
            
        # 如果没有PSF，创建delta函数（假设无模糊）
        if psf is None:
            psf = np.zeros_like(gray)
            psf[gray.shape[0]//2, gray.shape[1]//2] = 1
            
        # 傅里叶变换
        img_fft = np.fft.fft2(img_float)
        psf_fft = np.fft.fft2(psf, s=img_float.shape)
        
        # 维纳滤波
        psf_conj = np.conj(psf_fft)
        psf_abs_sq = np.abs(psf_fft) ** 2
        
        # 防止除零
        denominator = psf_abs_sq + noise_variance
        wiener_filter = psf_conj / np.maximum(denominator, 1e-10)
        
        # 应用滤波器
        result_fft = img_fft * wiener_filter
        result = np.fft.ifft2(result_fft).real
        
        # 截断到有效范围
        result = np.clip(result, 0, 255)
        
        return result.astype(np.uint8)
    
    @staticmethod
    def morphological_denoise(image: np.ndarray,
                             operation: str = 'opening',
                             kernel_size: Tuple[int, int] = (3, 3),
                             iterations: int = 1) -> np.ndarray:
        """
        形态学去噪
        针对条形码特征的结构化去噪
        
        Args:
            image: 输入图像
            operation: 形态学操作 ('opening', 'closing', 'gradient', 'tophat', 'blackhat')
            kernel_size: 结构元素大小
            iterations: 迭代次数
            
        Returns:
            处理后的图像
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
            
        # 创建结构元素
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
        
        if operation == 'opening':
            result = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel, iterations=iterations)
        elif operation == 'closing':
            result = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel, iterations=iterations)
        elif operation == 'gradient':
            result = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel, iterations=iterations)
        elif operation == 'tophat':
            result = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel, iterations=iterations)
        elif operation == 'blackhat':
            result = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel, iterations=iterations)
        else:
            logger.warning(f"未知的形态学操作: {operation}")
            result = gray
            
        return result
    
    @staticmethod
    def morphological_combo_denoise(image: np.ndarray,
                                   sequence: str = 'open-close') -> np.ndarray:
        """
        形态学组合去噪
        针对条形码特征的多步骤结构化去噪
        
        Args:
            image: 输入图像
            sequence: 操作序列 ('open-close', 'close-open', 'gradient-close', 'tophat-closing')
            
        Returns:
            处理后的图像
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
            
        # 针对条形码的结构元素：水平和垂直
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))  # 水平条纹
        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))  # 垂直条纹
        rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))  # 矩形
        
        if sequence == 'open-close':
            # 开运算去除小噪声，闭运算填充空洞
            result = cv2.morphologyEx(gray, cv2.MORPH_OPEN, rect_kernel)
            result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, rect_kernel)
            
        elif sequence == 'close-open':
            # 闭运算填充空洞，开运算去除噪声
            result = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, rect_kernel)
            result = cv2.morphologyEx(result, cv2.MORPH_OPEN, rect_kernel)
            
        elif sequence == 'gradient-close':
            # 梯度运算突出边缘，闭运算连接断裂
            result = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, rect_kernel)
            result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, h_kernel)
            
        elif sequence == 'tophat-closing':
            # 顶帽运算提取亮细节，闭运算连接
            result = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, rect_kernel)
            result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, h_kernel)
            # 与原图像组合
            result = cv2.add(gray, result)
            
        else:
            logger.warning(f"未知的形态学序列: {sequence}")
            result = gray
            
        return result
    
    @staticmethod
    def median_filter_enhanced(image: np.ndarray,
                              kernel_size: int = 5,
                              adaptive: bool = True) -> np.ndarray:
        """
        增强版中值滤波
        
        Args:
            image: 输入图像
            kernel_size: 滤波核大小
            adaptive: 是否使用自适应核大小
            
        Returns:
            滤波后的图像
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
            
        if adaptive:
            # 根据图像特征自适应调整核大小
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges) / (edges.shape[0] * edges.shape[1])
            
            if edge_density < 0.1:
                # 边缘较少，可以使用较大的核
                kernel_size = 7
            elif edge_density > 0.3:
                # 边缘较多，使用较小的核保护边缘
                kernel_size = 3
            else:
                kernel_size = 5
                
        # 应用中值滤波
        filtered = cv2.medianBlur(gray, kernel_size)
        
        return filtered
    
    @staticmethod
    def gaussian_bilateral_combo(image: np.ndarray,
                                sigma_spatial: float = 1.5,
                                sigma_color: float = 50) -> np.ndarray:
        """
        高斯-双边滤波组合
        先用高斯去除高频噪声，再用双边滤波保护边缘
        
        Args:
            image: 输入图像
            sigma_spatial: 空间标准差
            sigma_color: 颜色标准差
            
        Returns:
            滤波后的图像
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
            
        # 第一步：轻微的高斯滤波去除高频噪声
        gaussian_filtered = cv2.GaussianBlur(gray, (3, 3), sigma_spatial)
        
        # 第二步：双边滤波保护边缘
        result = cv2.bilateralFilter(gaussian_filtered, 7, sigma_color, sigma_color)
        
        return result


def process_advanced_denoise(image: np.ndarray,
                           method: str = 'nlm',
                           **kwargs) -> np.ndarray:
    """
    高级去噪主函数
    
    Args:
        image: 输入图像
        method: 去噪方法 ('nlm', 'bilateral_enhanced', 'wiener', 'morphological', 
                'morph_combo', 'median_enhanced', 'gaussian_bilateral')
        **kwargs: 方法特定参数
        
    Returns:
        去噪后的图像
    """
    try:
        if method == 'nlm':
            return AdvancedDenoise.non_local_means_denoise(image, **kwargs)
        elif method == 'bilateral_enhanced':
            return AdvancedDenoise.bilateral_filter_enhanced(image, **kwargs)
        elif method == 'wiener':
            return AdvancedDenoise.wiener_filter(image, **kwargs)
        elif method == 'morphological':
            return AdvancedDenoise.morphological_denoise(image, **kwargs)
        elif method == 'morph_combo':
            return AdvancedDenoise.morphological_combo_denoise(image, **kwargs)
        elif method == 'median_enhanced':
            return AdvancedDenoise.median_filter_enhanced(image, **kwargs)
        elif method == 'gaussian_bilateral':
            return AdvancedDenoise.gaussian_bilateral_combo(image, **kwargs)
        else:
            logger.warning(f"未知的去噪方法: {method}")
            return image
            
    except Exception as e:
        logger.error(f"高级去噪处理失败: {e}")
        return image


def test_advanced_denoise_algorithms():
    """测试所有高级去噪算法"""
    # 创建带噪声的测试图像
    test_img = np.zeros((200, 300), dtype=np.uint8)
    
    # 添加条形码图案
    for i in range(80, 120):
        for j in range(50, 250, 10):
            if (j // 10) % 2 == 0:
                test_img[i:i+5, j:j+5] = 20   # 暗条
            else:
                test_img[i:i+5, j:j+5] = 200  # 亮条
    
    # 添加噪声
    noise = np.random.normal(0, 25, test_img.shape)
    noisy_img = np.clip(test_img.astype(np.float64) + noise, 0, 255).astype(np.uint8)
    
    methods = [
        'nlm', 'bilateral_enhanced', 'wiener', 'morphological', 
        'morph_combo', 'median_enhanced', 'gaussian_bilateral'
    ]
    
    print("测试所有高级去噪算法...")
    print(f"原始图像: shape={test_img.shape}")
    print(f"噪声图像: shape={noisy_img.shape}, 噪声水平={np.std(noisy_img - test_img):.2f}")
    
    for method in methods:
        try:
            result = process_advanced_denoise(noisy_img, method)
            # 计算PSNR
            mse = np.mean((test_img.astype(np.float64) - result.astype(np.float64)) ** 2)
            if mse > 0:
                psnr = 20 * np.log10(255.0 / np.sqrt(mse))
            else:
                psnr = float('inf')
            print(f"✓ {method}: 处理成功，输出shape: {result.shape}, PSNR: {psnr:.2f}dB")
        except Exception as e:
            print(f"✗ {method}: 处理失败 - {e}")
    
    print("高级去噪算法测试完成！")


if __name__ == "__main__":
    test_advanced_denoise_algorithms() 