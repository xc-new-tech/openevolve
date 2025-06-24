#!/usr/bin/env python3
"""
统一的图像处理器基类

提供所有算法的标准接口和公共功能
"""

import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union, Tuple
import numpy as np
import cv2
from datetime import datetime

# 导入统一的异常和日志模块
from exceptions import (
    ProcessorError, ValidationError, ConfigurationError, 
    ProcessingError, ResourceError, EXCEPTION_MAPPING
)
from logging_config import ProcessorLogger, PerformanceMonitor

class ProcessorConfig:
    """处理器配置基类"""
    
    def __init__(self, **kwargs):
        """初始化配置"""
        self.config = kwargs
        
    def get(self, key: str, default: Any = None) -> Any:
        """获取配置值"""
        return self.config.get(key, default)
        
    def set(self, key: str, value: Any) -> None:
        """设置配置值"""
        self.config[key] = value
        
    def update(self, config: Dict[str, Any]) -> None:
        """更新配置"""
        self.config.update(config)
        
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return self.config.copy()

class ProcessingResult:
    """处理结果封装"""
    
    def __init__(self, 
                 image: np.ndarray,
                 success: bool = True,
                 message: str = "",
                 processing_time: float = 0.0,
                 metadata: Optional[Dict[str, Any]] = None):
        self.image = image
        self.success = success
        self.message = message
        self.processing_time = processing_time
        self.metadata = metadata or {}
        
    def __bool__(self) -> bool:
        return self.success

class BaseProcessor(ABC):
    """
    图像处理器基类
    
    所有图像处理算法都应该继承这个基类，实现统一的接口
    """
    
    def __init__(self, config: Optional[Union[Dict[str, Any], ProcessorConfig]] = None):
        """
        初始化处理器
        
        Args:
            config: 处理器配置
        """
        if isinstance(config, dict):
            self.config = ProcessorConfig(**config)
        elif isinstance(config, ProcessorConfig):
            self.config = config
        else:
            self.config = ProcessorConfig()
            
        self.logger = logging.getLogger(self.__class__.__name__)
        self._setup_logging()
        
    def _setup_logging(self):
        """设置日志"""
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
    
    @abstractmethod
    def _process(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """
        实际的图像处理逻辑（子类必须实现）
        
        Args:
            image: 输入图像
            **kwargs: 其他参数
            
        Returns:
            处理后的图像
        """
        pass
    
    def process(self, image: np.ndarray, **kwargs) -> ProcessingResult:
        """
        处理图像的主入口
        
        Args:
            image: 输入图像
            **kwargs: 其他参数
            
        Returns:
            处理结果
        """
        start_time = time.time()
        
        try:
            # 输入验证
            if not self._validate_input(image):
                return ProcessingResult(
                    image=image,
                    success=False,
                    message="输入图像验证失败",
                    processing_time=time.time() - start_time
                )
            
            self.logger.debug(f"开始处理图像，尺寸: {image.shape}")
            
            # 实际处理
            processed_image = self._process(image, **kwargs)
            
            # 输出验证
            if not self._validate_output(processed_image):
                return ProcessingResult(
                    image=image,
                    success=False,
                    message="输出图像验证失败",
                    processing_time=time.time() - start_time
                )
            
            processing_time = time.time() - start_time
            self.logger.debug(f"处理完成，耗时: {processing_time:.3f}秒")
            
            return ProcessingResult(
                image=processed_image,
                success=True,
                message="处理成功",
                processing_time=processing_time,
                metadata=self._get_metadata()
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"处理失败: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            
            return ProcessingResult(
                image=image,
                success=False,
                message=error_msg,
                processing_time=processing_time
            )
    
    def _validate_input(self, image: np.ndarray) -> bool:
        """
        验证输入图像
        
        Args:
            image: 输入图像
            
        Returns:
            是否有效
        """
        if image is None:
            self.logger.error("输入图像为空")
            return False
            
        if not isinstance(image, np.ndarray):
            self.logger.error(f"输入必须是numpy数组，当前类型: {type(image)}")
            return False
            
        if image.size == 0:
            self.logger.error("输入图像为空数组")
            return False
            
        if len(image.shape) not in [2, 3]:
            self.logger.error(f"不支持的图像维度: {len(image.shape)}")
            return False
            
        return True
    
    def _validate_output(self, image: np.ndarray) -> bool:
        """
        验证输出图像
        
        Args:
            image: 输出图像
            
        Returns:
            是否有效
        """
        return self._validate_input(image)
    
    def _get_metadata(self) -> Dict[str, Any]:
        """
        获取处理元数据
        
        Returns:
            元数据字典
        """
        return {
            "processor": self.__class__.__name__,
            "config": self.config.to_dict()
        }
    
    def get_config(self) -> ProcessorConfig:
        """获取配置"""
        return self.config
    
    def set_config(self, config: Union[Dict[str, Any], ProcessorConfig]) -> None:
        """设置配置"""
        if isinstance(config, dict):
            self.config = ProcessorConfig(**config)
        elif isinstance(config, ProcessorConfig):
            self.config = config
    
    def update_config(self, **kwargs) -> None:
        """更新配置"""
        self.config.update(kwargs)
    
    @property
    def name(self) -> str:
        """处理器名称"""
        return self.__class__.__name__
    
    def __str__(self) -> str:
        return f"{self.name}(config={self.config.to_dict()})"
    
    def __repr__(self) -> str:
        return self.__str__()

class Pipeline(BaseProcessor):
    """
    处理管道
    
    将多个处理器串联起来
    """
    
    def __init__(self, processors: list, config: Optional[Dict[str, Any]] = None):
        """
        初始化管道
        
        Args:
            processors: 处理器列表
            config: 管道配置
        """
        super().__init__(config)
        self.processors = processors
        
    def _process(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """
        执行管道处理
        
        Args:
            image: 输入图像
            **kwargs: 其他参数
            
        Returns:
            处理后的图像
        """
        current_image = image
        
        for i, processor in enumerate(self.processors):
            self.logger.debug(f"执行处理器 {i+1}/{len(self.processors)}: {processor.name}")
            
            result = processor.process(current_image, **kwargs)
            
            if not result.success:
                raise ProcessorError(f"处理器 {processor.name} 失败: {result.message}")
                
            current_image = result.image
            
        return current_image
    
    def add_processor(self, processor: BaseProcessor) -> None:
        """添加处理器"""
        self.processors.append(processor)
        
    def remove_processor(self, index: int) -> None:
        """移除处理器"""
        if 0 <= index < len(self.processors):
            self.processors.pop(index)
            
    def get_processors(self) -> list:
        """获取所有处理器"""
        return self.processors.copy() 