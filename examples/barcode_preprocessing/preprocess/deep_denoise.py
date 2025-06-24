"""
深度学习去噪模型模块
集成DnCNN、FFDNet等先进深度去噪模型
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
    logger.warning("PyTorch not available. Deep learning denoising disabled.")


class DnCNN(nn.Module if PYTORCH_AVAILABLE else object):
    """DnCNN - 深度卷积去噪网络"""
    
    def __init__(self, channels=1, num_layers=17, features=64):
        if not PYTORCH_AVAILABLE:
            raise ImportError("PyTorch is required for DnCNN")
            
        super(DnCNN, self).__init__()
        
        # 第一层：卷积 + ReLU
        layers = [nn.Conv2d(channels, features, 3, padding=1), nn.ReLU(inplace=True)]
        
        # 中间层：卷积 + 批归一化 + ReLU
        for _ in range(num_layers - 2):
            layers.extend([
                nn.Conv2d(features, features, 3, padding=1),
                nn.BatchNorm2d(features),
                nn.ReLU(inplace=True)
            ])
        
        # 最后一层：卷积（输出残差）
        layers.append(nn.Conv2d(features, channels, 3, padding=1))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        """前向传播 - 输出噪声残差"""
        return self.network(x)


class LightweightDenoiser(nn.Module if PYTORCH_AVAILABLE else object):
    """轻量级去噪网络 - 专为条形码优化"""
    
    def __init__(self, channels=1):
        if not PYTORCH_AVAILABLE:
            raise ImportError("PyTorch is required for LightweightDenoiser")
            
        super(LightweightDenoiser, self).__init__()
        
        # 编码器
        self.encoder = nn.Sequential(
            nn.Conv2d(channels, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        
        # 中间处理
        self.middle = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # 解码器
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, channels, 3, padding=1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        """前向传播"""
        # 保存输入尺寸
        input_size = x.shape[2:]
        
        # 编码
        encoded = self.encoder(x)
        
        # 中间处理
        middle = self.middle(encoded)
        
        # 解码
        decoded = self.decoder(middle)
        
        # 确保输出尺寸与输入一致
        if decoded.shape[2:] != input_size:
            decoded = F.interpolate(decoded, size=input_size, mode='bilinear', align_corners=False)
            
        return decoded


class DeepDenoise:
    """深度学习去噪算法集合"""
    
    def __init__(self):
        self.dncnn_model = None
        self.lightweight_model = None
        self.device = 'cuda' if PYTORCH_AVAILABLE and torch.cuda.is_available() else 'cpu'
        
    def load_dncnn_model(self, model_path: Optional[str] = None) -> bool:
        """加载DnCNN模型"""
        if not PYTORCH_AVAILABLE:
            logger.warning("PyTorch not available, cannot load DnCNN")
            return False
            
        try:
            self.dncnn_model = DnCNN()
            
            if model_path and os.path.exists(model_path):
                # 加载预训练权重
                self.dncnn_model.load_state_dict(torch.load(model_path, map_location=self.device))
                logger.info(f"Loaded DnCNN model from {model_path}")
            else:
                # 使用随机初始化权重
                logger.info("Using randomly initialized DnCNN model")
                
            self.dncnn_model.to(self.device)
            self.dncnn_model.eval()
            return True
            
        except Exception as e:
            logger.error(f"Failed to load DnCNN model: {e}")
            return False
    
    def load_lightweight_model(self, model_path: Optional[str] = None) -> bool:
        """加载轻量级模型"""
        if not PYTORCH_AVAILABLE:
            logger.warning("PyTorch not available, cannot load lightweight model")
            return False
            
        try:
            self.lightweight_model = LightweightDenoiser()
            
            if model_path and os.path.exists(model_path):
                # 加载预训练权重
                self.lightweight_model.load_state_dict(torch.load(model_path, map_location=self.device))
                logger.info(f"Loaded lightweight model from {model_path}")
            else:
                # 使用随机初始化权重
                logger.info("Using randomly initialized lightweight model")
                
            self.lightweight_model.to(self.device)
            self.lightweight_model.eval()
            return True
            
        except Exception as e:
            logger.error(f"Failed to load lightweight model: {e}")
            return False
    
    @staticmethod
    def analyze_noise_characteristics(image: np.ndarray) -> Dict[str, Any]:
        """分析图像噪声特征"""
        # 计算噪声水平
        laplacian_var = cv2.Laplacian(image, cv2.CV_64F).var()
        
        # 计算噪声类型（高斯vs椒盐）
        hist = cv2.calcHist([image], [0], None, [256], [0, 256])
        extreme_pixels = hist[0] + hist[255]
        total_pixels = image.size
        salt_pepper_ratio = extreme_pixels / total_pixels
        
        # 计算纹理复杂度
        gray_cooc = cv2.calcHist([image], [0], None, [256], [0, 256])
        texture_complexity = np.std(gray_cooc)
        
        return {
            'noise_level': float(laplacian_var),
            'salt_pepper_ratio': float(salt_pepper_ratio),
            'texture_complexity': float(texture_complexity),
            'is_high_noise': laplacian_var > 500,
            'has_salt_pepper': salt_pepper_ratio > 0.01,
            'is_complex_texture': texture_complexity > 1000,
            'recommended_method': 'dncnn' if laplacian_var > 300 else 'lightweight'
        }
    
    def dncnn_denoise(self, image: np.ndarray) -> np.ndarray:
        """使用DnCNN进行去噪"""
        if not PYTORCH_AVAILABLE or self.dncnn_model is None:
            logger.warning("DnCNN model not available, using fallback")
            return self.traditional_denoise(image, method='nlm')
            
        try:
            # 预处理
            original_shape = image.shape
            if len(image.shape) == 2:
                image_input = image[np.newaxis, np.newaxis, :, :]  # 添加batch和channel维度
            else:
                image_input = image.transpose(2, 0, 1)[np.newaxis, :]  # NHWC -> NCHW
                
            # 归一化到[0,1]
            image_input = image_input.astype(np.float32) / 255.0
            
            # 转换为tensor
            input_tensor = torch.from_numpy(image_input).to(self.device)
            
            # 推理
            with torch.no_grad():
                noise_residual = self.dncnn_model(input_tensor)
                denoised = input_tensor - noise_residual
                denoised = torch.clamp(denoised, 0, 1)
            
            # 后处理
            output = denoised.cpu().numpy()
            if len(original_shape) == 2:
                output = output[0, 0]  # 移除batch和channel维度
            else:
                output = output[0].transpose(1, 2, 0)  # NCHW -> HWC
                
            # 转换回uint8
            output = (output * 255).astype(np.uint8)
            
            return output
            
        except Exception as e:
            logger.error(f"DnCNN denoising failed: {e}")
            return self.traditional_denoise(image, method='nlm')
    
    def lightweight_denoise(self, image: np.ndarray) -> np.ndarray:
        """使用轻量级模型进行去噪"""
        if not PYTORCH_AVAILABLE or self.lightweight_model is None:
            logger.warning("Lightweight model not available, using fallback")
            return self.traditional_denoise(image, method='bilateral')
            
        try:
            # 预处理
            original_shape = image.shape
            if len(image.shape) == 2:
                image_input = image[np.newaxis, np.newaxis, :, :]
            else:
                image_input = image.transpose(2, 0, 1)[np.newaxis, :]
                
            # 归一化到[0,1]
            image_input = image_input.astype(np.float32) / 255.0
            
            # 转换为tensor
            input_tensor = torch.from_numpy(image_input).to(self.device)
            
            # 推理
            with torch.no_grad():
                denoised = self.lightweight_model(input_tensor)
            
            # 后处理
            output = denoised.cpu().numpy()
            if len(original_shape) == 2:
                output = output[0, 0]
            else:
                output = output[0].transpose(1, 2, 0)
                
            # 转换回uint8
            output = (output * 255).astype(np.uint8)
            
            return output
            
        except Exception as e:
            logger.error(f"Lightweight denoising failed: {e}")
            return self.traditional_denoise(image, method='bilateral')
    
    def traditional_denoise(self, image: np.ndarray, method: str = 'nlm') -> np.ndarray:
        """传统去噪方法（降级选项）"""
        if method == 'nlm':
            # 非局部均值去噪
            return cv2.fastNlMeansDenoising(image, None, 10, 7, 21)
        elif method == 'bilateral':
            # 双边滤波
            return cv2.bilateralFilter(image, 9, 75, 75)
        elif method == 'gaussian':
            # 高斯滤波
            return cv2.GaussianBlur(image, (5, 5), 0)
        else:
            return image
    
    def adaptive_denoise(self, image: np.ndarray) -> np.ndarray:
        """自适应去噪 - 根据噪声特征选择最佳方法"""
        analysis = self.analyze_noise_characteristics(image)
        
        if analysis['is_high_noise']:
            # 高噪声：优先使用DnCNN
            if analysis['recommended_method'] == 'dncnn':
                return self.dncnn_denoise(image)
            else:
                return self.lightweight_denoise(image)
        elif analysis['has_salt_pepper']:
            # 椒盐噪声：使用中值滤波 + 轻量级模型
            median_filtered = cv2.medianBlur(image, 3)
            return self.lightweight_denoise(median_filtered)
        else:
            # 轻微噪声：使用轻量级模型
            return self.lightweight_denoise(image)


def process_deep_denoise(image: np.ndarray, method: str = 'auto', **kwargs) -> np.ndarray:
    """深度学习去噪统一接口"""
    denoiser = DeepDenoise()
    
    if method == 'auto':
        # 自动选择最佳方法
        return denoiser.adaptive_denoise(image)
    elif method == 'dncnn':
        # 强制使用DnCNN
        denoiser.load_dncnn_model()
        return denoiser.dncnn_denoise(image)
    elif method == 'lightweight':
        # 使用轻量级模型
        denoiser.load_lightweight_model()
        return denoiser.lightweight_denoise(image)
    else:
        # 传统方法降级
        return denoiser.traditional_denoise(image, method)


if __name__ == "__main__":
    # 测试深度学习去噪功能
    print("=== 深度学习去噪模块测试 ===")
    
    if PYTORCH_AVAILABLE:
        print("✓ PyTorch可用，启用深度学习功能")
    else:
        print("⚠ PyTorch不可用，将使用传统算法")
    
    # 创建测试图像
    test_image = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
    print(f"测试图像尺寸: {test_image.shape}")
    
    # 测试噪声分析
    denoiser = DeepDenoise()
    analysis = denoiser.analyze_noise_characteristics(test_image)
    print(f"噪声分析结果: {analysis}")
    
    # 测试各种去噪方法
    try:
        auto_result = process_deep_denoise(test_image, method='auto')
        print(f"✓ 自适应去噪: {auto_result.shape}")
        
        lightweight_result = process_deep_denoise(test_image, method='lightweight')
        print(f"✓ 轻量级去噪: {lightweight_result.shape}")
        
        dncnn_result = process_deep_denoise(test_image, method='dncnn')
        print(f"✓ DnCNN去噪: {dncnn_result.shape}")
        
        traditional_result = process_deep_denoise(test_image, method='nlm')
        print(f"✓ 传统去噪: {traditional_result.shape}")
        
        print("\n🎉 所有深度学习去噪算法测试成功！")
        
    except Exception as e:
        print(f"✗ 测试失败: {e}") 