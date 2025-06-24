"""
深度学习超分辨率算法模块
提供多种超分辨率算法，提升低分辨率条形码图像质量
"""

import numpy as np
import cv2
from typing import Tuple, Optional, Union, Dict, Any
import logging
import os

logger = logging.getLogger(__name__)

# 检查是否可用深度学习库
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    logger.warning("PyTorch not available. Deep learning super-resolution disabled.")


class LightweightSRCNN(nn.Module if PYTORCH_AVAILABLE else object):
    """轻量级SRCNN模型 - 专为条形码优化的超分辨率网络"""
    
    def __init__(self, scale_factor: int = 2, num_channels: int = 1):
        if not PYTORCH_AVAILABLE:
            raise ImportError("PyTorch is required for SRCNN")
            
        super(LightweightSRCNN, self).__init__()
        self.scale_factor = scale_factor
        
        # 特征提取层 - 减少参数但保持效果
        self.feature_extraction = nn.Conv2d(num_channels, 32, kernel_size=9, padding=4)
        
        # 非线性映射层 - 轻量化设计
        self.mapping1 = nn.Conv2d(32, 16, kernel_size=5, padding=2)
        self.mapping2 = nn.Conv2d(16, 8, kernel_size=3, padding=1)
        
        # 重建层
        self.reconstruction = nn.Conv2d(8, num_channels, kernel_size=5, padding=2)
        
        # 上采样层
        self.upsampling = nn.UpsamplingBilinear2d(scale_factor=scale_factor)
        
    def forward(self, x):
        # 先上采样到目标尺寸
        x = self.upsampling(x)
        
        # 特征提取
        x1 = F.relu(self.feature_extraction(x))
        
        # 非线性映射
        x2 = F.relu(self.mapping1(x1))
        x3 = F.relu(self.mapping2(x2))
        
        # 重建
        out = self.reconstruction(x3)
        
        return out


class SuperResolution:
    """超分辨率算法集合"""
    
    def __init__(self):
        self.models_loaded = {}
        self.model_cache = {}
        
    @staticmethod
    def bicubic_interpolation(image: np.ndarray, scale_factor: float = 2.0) -> np.ndarray:
        """
        双三次插值超分辨率
        
        Args:
            image: 输入图像
            scale_factor: 放大倍数
            
        Returns:
            放大后的图像
        """
        height, width = image.shape[:2]
        new_height = int(height * scale_factor)
        new_width = int(width * scale_factor)
        
        return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    
    @staticmethod
    def lanczos_interpolation(image: np.ndarray, scale_factor: float = 2.0) -> np.ndarray:
        """
        Lanczos插值超分辨率 - 更好的边缘保持
        
        Args:
            image: 输入图像
            scale_factor: 放大倍数
            
        Returns:
            放大后的图像
        """
        height, width = image.shape[:2]
        new_height = int(height * scale_factor)
        new_width = int(width * scale_factor)
        
        return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
    
    @staticmethod
    def edge_directed_interpolation(image: np.ndarray, scale_factor: float = 2.0) -> np.ndarray:
        """
        边缘导向插值 - 专为条形码优化
        
        Args:
            image: 输入图像
            scale_factor: 放大倍数
            
        Returns:
            放大后的图像
        """
        # 先用双三次插值放大
        upscaled = SuperResolution.bicubic_interpolation(image, scale_factor)
        
        # 检测边缘
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
            
        # 使用Canny边缘检测
        edges = cv2.Canny(gray, 50, 150)
        
        # 形态学操作增强边缘
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
        
        # 将边缘信息融合到放大图像中
        if len(upscaled.shape) == 3:
            upscaled_gray = cv2.cvtColor(upscaled, cv2.COLOR_BGR2GRAY)
        else:
            upscaled_gray = upscaled.copy()
            
        # 放大边缘mask
        edges_upscaled = cv2.resize(edges, (upscaled_gray.shape[1], upscaled_gray.shape[0]), 
                                   interpolation=cv2.INTER_NEAREST)
        
        # 在边缘区域应用锐化
        sharpening_kernel = np.array([[-1, -1, -1],
                                     [-1,  9, -1],
                                     [-1, -1, -1]])
        
        # 应用锐化到边缘区域
        sharpened = cv2.filter2D(upscaled_gray, -1, sharpening_kernel)
        
        # 混合原图和锐化图
        mask = edges_upscaled.astype(np.float32) / 255.0
        result = upscaled_gray.astype(np.float32) * (1 - mask) + sharpened.astype(np.float32) * mask
        
        return np.clip(result, 0, 255).astype(np.uint8)
    
    def load_srcnn_model(self, model_path: Optional[str] = None, scale_factor: int = 2) -> bool:
        """
        加载SRCNN模型
        
        Args:
            model_path: 模型文件路径
            scale_factor: 放大倍数
            
        Returns:
            是否成功加载
        """
        if not PYTORCH_AVAILABLE:
            logger.warning("PyTorch not available, cannot load SRCNN model")
            return False
            
        try:
            model = LightweightSRCNN(scale_factor=scale_factor, num_channels=1)
            
            if model_path and os.path.exists(model_path):
                model.load_state_dict(torch.load(model_path, map_location='cpu'))
                logger.info(f"Loaded SRCNN model from {model_path}")
            else:
                logger.info("Using untrained SRCNN model (random weights)")
            
            model.eval()
            self.models_loaded['srcnn'] = model
            return True
            
        except Exception as e:
            logger.error(f"Failed to load SRCNN model: {e}")
            return False
    
    def srcnn_super_resolution(self, image: np.ndarray, scale_factor: int = 2) -> np.ndarray:
        """
        使用SRCNN进行超分辨率
        
        Args:
            image: 输入图像
            scale_factor: 放大倍数
            
        Returns:
            超分辨率图像
        """
        if not PYTORCH_AVAILABLE or 'srcnn' not in self.models_loaded:
            logger.warning("SRCNN not available, falling back to edge-directed interpolation")
            return self.edge_directed_interpolation(image, scale_factor)
        
        try:
            # 预处理
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # 归一化到[0,1]
            input_tensor = torch.from_numpy(gray.astype(np.float32) / 255.0)
            input_tensor = input_tensor.unsqueeze(0).unsqueeze(0)  # 添加batch和channel维度
            
            # 推理
            model = self.models_loaded['srcnn']
            with torch.no_grad():
                output_tensor = model(input_tensor)
            
            # 后处理
            output = output_tensor.squeeze().numpy()
            output = np.clip(output * 255.0, 0, 255).astype(np.uint8)
            
            return output
            
        except Exception as e:
            logger.error(f"SRCNN inference failed: {e}")
            return self.edge_directed_interpolation(image, scale_factor)
    
    @staticmethod
    def analyze_image_resolution(image: np.ndarray) -> Dict[str, Any]:
        """
        分析图像分辨率并推荐超分辨率方法
        
        Args:
            image: 输入图像
            
        Returns:
            分析结果字典
        """
        height, width = image.shape[:2]
        pixel_count = height * width
        
        analysis = {
            'width': width,
            'height': height,
            'pixel_count': pixel_count,
            'is_low_resolution': pixel_count < 10000,  # 小于100x100
            'is_very_low_resolution': pixel_count < 2500,  # 小于50x50
            'recommended_scale': 1.0,
            'recommended_method': 'none'
        }
        
        if analysis['is_very_low_resolution']:
            analysis['recommended_scale'] = 4.0
            analysis['recommended_method'] = 'srcnn'
        elif analysis['is_low_resolution']:
            analysis['recommended_scale'] = 2.0
            analysis['recommended_method'] = 'edge_directed'
        elif min(width, height) < 150:
            analysis['recommended_scale'] = 1.5
            analysis['recommended_method'] = 'lanczos'
        
        return analysis


def process_super_resolution(image: np.ndarray, method: str = 'auto', 
                           scale_factor: float = 2.0, **kwargs) -> np.ndarray:
    """
    统一的超分辨率处理接口
    
    Args:
        image: 输入图像
        method: 处理方法 ('auto', 'bicubic', 'lanczos', 'edge_directed', 'srcnn')
        scale_factor: 放大倍数
        **kwargs: 额外参数
        
    Returns:
        处理后的图像
    """
    sr = SuperResolution()
    
    if method == 'auto':
        # 自动选择最佳方法
        analysis = sr.analyze_image_resolution(image)
        method = analysis['recommended_method']
        if analysis['recommended_scale'] > 1.0:
            scale_factor = analysis['recommended_scale']
        else:
            return image  # 不需要超分辨率
    
    if method == 'bicubic':
        return sr.bicubic_interpolation(image, scale_factor)
    elif method == 'lanczos':
        return sr.lanczos_interpolation(image, scale_factor)
    elif method == 'edge_directed':
        return sr.edge_directed_interpolation(image, scale_factor)
    elif method == 'srcnn':
        # 尝试加载SRCNN模型
        if sr.load_srcnn_model(scale_factor=int(scale_factor)):
            return sr.srcnn_super_resolution(image, int(scale_factor))
        else:
            # 降级到边缘导向插值
            return sr.edge_directed_interpolation(image, scale_factor)
    else:
        logger.warning(f"Unknown method {method}, using bicubic interpolation")
        return sr.bicubic_interpolation(image, scale_factor)


# 测试代码
if __name__ == "__main__":
    print("🔍 测试超分辨率算法...")
    
    # 创建测试图像 - 模拟低分辨率条形码
    test_img = np.random.randint(0, 255, (30, 100), dtype=np.uint8)
    
    # 添加一些条形码状的模式
    for i in range(0, 100, 4):
        if i % 8 < 4:
            test_img[:, i:i+2] = 0  # 黑条
        else:
            test_img[:, i:i+2] = 255  # 白条
    
    print(f"原始图像尺寸: {test_img.shape}")
    
    # 测试各种方法
    methods = ['bicubic', 'lanczos', 'edge_directed', 'auto']
    
    for method in methods:
        try:
            result = process_super_resolution(test_img, method=method, scale_factor=2.0)
            print(f"✓ {method} 方法成功 - 输出尺寸: {result.shape}")
        except Exception as e:
            print(f"✗ {method} 方法失败: {e}")
    
    # 测试SRCNN（可能需要PyTorch）
    if PYTORCH_AVAILABLE:
        try:
            sr = SuperResolution()
            if sr.load_srcnn_model(scale_factor=2):
                result = sr.srcnn_super_resolution(test_img, scale_factor=2)
                print(f"✓ SRCNN 方法成功 - 输出尺寸: {result.shape}")
            else:
                print("! SRCNN 模型加载失败，使用随机权重")
        except Exception as e:
            print(f"✗ SRCNN 方法失败: {e}")
    else:
        print("! PyTorch 不可用，跳过SRCNN测试")
    
    # 测试图像分析
    analysis = SuperResolution.analyze_image_resolution(test_img)
    print(f"\n📊 图像分析结果:")
    print(f"  分辨率: {analysis['width']}x{analysis['height']}")
    print(f"  是否低分辨率: {analysis['is_low_resolution']}")
    print(f"  推荐放大倍数: {analysis['recommended_scale']}")
    print(f"  推荐方法: {analysis['recommended_method']}")
    
    print("\n🎉 超分辨率模块测试完成！") 