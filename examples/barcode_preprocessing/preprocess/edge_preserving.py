"""
边缘保护滤波算法模块
在去噪的同时保持边缘信息的滤波技术
"""

import numpy as np
import cv2
from typing import Tuple, Optional, Union
import logging

logger = logging.getLogger(__name__)


class EdgePreservingFilters:
    """边缘保护滤波算法集合"""
    
    @staticmethod
    def guided_filter(image: np.ndarray, guide: Optional[np.ndarray] = None, 
                      radius: int = 8, epsilon: float = 0.01) -> np.ndarray:
        """
        导向滤波 (Guided Filter)
        基于引导图像的边缘保护平滑滤波
        
        Args:
            image: 输入图像
            guide: 引导图像，如果为None则使用输入图像
            radius: 滤波半径
            epsilon: 正则化参数
            
        Returns:
            滤波后的图像
        """
        if guide is None:
            guide = image.copy()
            
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if len(guide.shape) == 3:
            guide = cv2.cvtColor(guide, cv2.COLOR_BGR2GRAY)
            
        # 转换为浮点数
        I = image.astype(np.float64) / 255.0
        p = I.copy()
        
        # 计算均值
        mean_I = cv2.boxFilter(I, cv2.CV_64F, (radius, radius))
        mean_p = cv2.boxFilter(p, cv2.CV_64F, (radius, radius))
        
        # 计算协方差
        mean_Ip = cv2.boxFilter(I * p, cv2.CV_64F, (radius, radius))
        cov_Ip = mean_Ip - mean_I * mean_p
        
        # 计算方差
        mean_II = cv2.boxFilter(I * I, cv2.CV_64F, (radius, radius))
        var_I = mean_II - mean_I * mean_I
        
        # 计算线性参数
        a = cov_Ip / (var_I + epsilon)
        b = mean_p - a * mean_I
        
        # 平滑参数
        mean_a = cv2.boxFilter(a, cv2.CV_64F, (radius, radius))
        mean_b = cv2.boxFilter(b, cv2.CV_64F, (radius, radius))
        
        # 计算输出
        result = mean_a * I + mean_b
        
        # 转换回uint8
        result = np.clip(result * 255, 0, 255).astype(np.uint8)
        
        return result
    
    @staticmethod
    def edge_preserving_smooth(image: np.ndarray, d: int = 5, 
                              sigma_color: float = 80.0, 
                              sigma_space: float = 80.0) -> np.ndarray:
        """
        边缘保护平滑滤波
        使用OpenCV的边缘保护滤波器
        
        Args:
            image: 输入图像
            d: 像素邻域直径
            sigma_color: 颜色空间的sigma值
            sigma_space: 坐标空间的sigma值
            
        Returns:
            滤波后的图像
        """
        if len(image.shape) == 3:
            # 彩色图像，使用edgePreservingFilter
            result = cv2.edgePreservingFilter(image, flags=cv2.RECURS_FILTER, 
                                            sigma_s=sigma_space, sigma_r=sigma_color)
        else:
            # 灰度图像，使用双边滤波
            result = cv2.bilateralFilter(image, d, sigma_color, sigma_space)
            
        return result
    
    @staticmethod
    def anisotropic_diffusion(image: np.ndarray, iterations: int = 15, 
                             kappa: float = 50, gamma: float = 0.1,
                             option: int = 1) -> np.ndarray:
        """
        各向异性扩散滤波 (Perona-Malik Diffusion)
        
        Args:
            image: 输入图像
            iterations: 迭代次数
            kappa: 扩散常数
            gamma: 时间步长
            option: 扩散函数选项 (1 or 2)
            
        Returns:
            滤波后的图像
        """
        if len(image.shape) == 3:
            img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            img = image.copy()
            
        # 转换为浮点数
        img = img.astype(np.float64)
        
        # 定义核
        dx = np.array([[-1, 1], [-1, 1]], dtype=np.float64)
        dy = np.array([[-1, -1], [1, 1]], dtype=np.float64)
        
        for i in range(iterations):
            # 计算梯度
            deltaE = cv2.filter2D(img, -1, dx)
            deltaS = cv2.filter2D(img, -1, dy)
            deltaN = cv2.filter2D(img, -1, -dy)
            deltaW = cv2.filter2D(img, -1, -dx)
            
            # 计算扩散系数
            if option == 1:
                # 选项1: c(x,y,t) = exp(-(||grad I||/K)^2)
                cE = np.exp(-(deltaE/kappa)**2)
                cS = np.exp(-(deltaS/kappa)**2)
                cN = np.exp(-(deltaN/kappa)**2)
                cW = np.exp(-(deltaW/kappa)**2)
            elif option == 2:
                # 选项2: c(x,y,t) = 1/(1+(||grad I||/K)^2)
                cE = 1.0 / (1.0 + (deltaE/kappa)**2)
                cS = 1.0 / (1.0 + (deltaS/kappa)**2)
                cN = 1.0 / (1.0 + (deltaN/kappa)**2)
                cW = 1.0 / (1.0 + (deltaW/kappa)**2)
            else:
                raise ValueError("Option must be 1 or 2")
            
            # 更新图像
            img = img + gamma * (cN*deltaN + cS*deltaS + cE*deltaE + cW*deltaW)
            
        # 转换回uint8
        result = np.clip(img, 0, 255).astype(np.uint8)
        
        return result
    
    @staticmethod
    def adaptive_edge_preserving(image: np.ndarray, noise_level: float = 15.0) -> np.ndarray:
        """
        自适应边缘保护滤波
        根据图像特征自动选择最适合的滤波方法
        
        Args:
            image: 输入图像
            noise_level: 噪声等级
            
        Returns:
            滤波后的图像
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
            
        # 分析图像特征
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (gray.shape[0] * gray.shape[1])
        
        # 计算图像标准差作为纹理复杂度指标
        texture_complexity = np.std(gray)
        
        if edge_density > 0.15 and texture_complexity > 40:
            # 高边缘密度和高纹理复杂度：使用导向滤波
            return EdgePreservingFilters.guided_filter(image, radius=6, epsilon=0.02)
        elif noise_level > 20:
            # 高噪声：使用各向异性扩散
            return EdgePreservingFilters.anisotropic_diffusion(image, iterations=10, kappa=30)
        else:
            # 一般情况：使用边缘保护平滑
            return EdgePreservingFilters.edge_preserving_smooth(image, d=7, 
                                                              sigma_color=60, sigma_space=60)


def process_edge_preserving(image: np.ndarray, method: str = 'adaptive', **kwargs) -> np.ndarray:
    """
    统一的边缘保护滤波接口
    
    Args:
        image: 输入图像
        method: 滤波方法 ('guided', 'smooth', 'diffusion', 'adaptive')
        **kwargs: 方法特定参数
        
    Returns:
        滤波后的图像
    """
    try:
        if method == 'guided':
            return EdgePreservingFilters.guided_filter(
                image, 
                radius=kwargs.get('radius', 8),
                epsilon=kwargs.get('epsilon', 0.01)
            )
        elif method == 'smooth':
            return EdgePreservingFilters.edge_preserving_smooth(
                image,
                d=kwargs.get('d', 5),
                sigma_color=kwargs.get('sigma_color', 80.0),
                sigma_space=kwargs.get('sigma_space', 80.0)
            )
        elif method == 'diffusion':
            return EdgePreservingFilters.anisotropic_diffusion(
                image,
                iterations=kwargs.get('iterations', 15),
                kappa=kwargs.get('kappa', 50),
                gamma=kwargs.get('gamma', 0.1),
                option=kwargs.get('option', 1)
            )
        elif method == 'adaptive':
            return EdgePreservingFilters.adaptive_edge_preserving(
                image,
                noise_level=kwargs.get('noise_level', 15.0)
            )
        else:
            raise ValueError(f"Unknown method: {method}")
            
    except Exception as e:
        logger.error(f"Error in edge preserving filter {method}: {e}")
        return image


def main():
    """测试边缘保护滤波算法"""
    print("测试边缘保护滤波算法...")
    
    # 创建测试图像
    test_image = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
    
    # 测试各种算法
    methods = ['guided', 'smooth', 'diffusion', 'adaptive']
    
    for method in methods:
        try:
            result = process_edge_preserving(test_image, method=method)
            print(f"{method}: {result.shape} - OK")
        except Exception as e:
            print(f"{method}: Error - {e}")
    
    # 测试自适应算法
    result = EdgePreservingFilters.adaptive_edge_preserving(test_image)
    print(f"自适应边缘保护: {result.shape} - OK")
    
    print("所有边缘保护滤波算法测试完成！")


if __name__ == "__main__":
    main() 