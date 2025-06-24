#!/usr/bin/env python3
"""
统一的日志配置模块

提供所有处理器的标准化日志配置
"""

import logging
import sys
import os
from typing import Optional, Dict, Any
from datetime import datetime
import json

class ProcessorLogger:
    """处理器专用日志器"""
    
    _loggers: Dict[str, logging.Logger] = {}
    _initialized = False
    
    @classmethod
    def setup_logging(cls, 
                     log_level: str = 'INFO',
                     log_file: Optional[str] = None,
                     enable_console: bool = True,
                     log_format: Optional[str] = None):
        """
        设置全局日志配置
        
        Args:
            log_level: 日志级别
            log_file: 日志文件路径
            enable_console: 是否启用控制台输出
            log_format: 自定义日志格式
        """
        if cls._initialized:
            return
            
        # 设置根日志器
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, log_level.upper()))
        
        # 清除现有处理器
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        
        # 默认格式
        if log_format is None:
            log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        
        formatter = logging.Formatter(log_format)
        
        # 控制台处理器
        if enable_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            root_logger.addHandler(console_handler)
        
        # 文件处理器
        if log_file:
            # 确保日志目录存在
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)
        
        cls._initialized = True
    
    @classmethod
    def get_logger(cls, name: str) -> logging.Logger:
        """
        获取处理器专用日志器
        
        Args:
            name: 日志器名称（通常是处理器类名）
            
        Returns:
            日志器实例
        """
        if not cls._initialized:
            cls.setup_logging()
            
        if name not in cls._loggers:
            logger = logging.getLogger(f"processor.{name}")
            cls._loggers[name] = logger
            
        return cls._loggers[name]
    
    @classmethod
    def log_processing_start(cls, logger: logging.Logger, 
                           processor_name: str, 
                           image_shape: tuple,
                           config: Dict[str, Any]):
        """记录处理开始"""
        logger.info(f"开始处理 - 处理器: {processor_name}, 图像尺寸: {image_shape}")
        logger.debug(f"配置参数: {json.dumps(config, indent=2, ensure_ascii=False)}")
    
    @classmethod
    def log_processing_end(cls, logger: logging.Logger,
                          processor_name: str,
                          success: bool,
                          processing_time: float,
                          message: str = ""):
        """记录处理结束"""
        if success:
            logger.info(f"处理完成 - 处理器: {processor_name}, 耗时: {processing_time:.3f}秒")
        else:
            logger.error(f"处理失败 - 处理器: {processor_name}, 耗时: {processing_time:.3f}秒, 错误: {message}")
    
    @classmethod
    def log_error(cls, logger: logging.Logger,
                  processor_name: str,
                  error: Exception,
                  context: Dict[str, Any] = None):
        """记录错误详情"""
        error_info = {
            "processor": processor_name,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "timestamp": datetime.now().isoformat()
        }
        
        if context:
            error_info["context"] = context
            
        logger.error(f"详细错误信息: {json.dumps(error_info, indent=2, ensure_ascii=False)}")
    
    @classmethod
    def log_performance_metrics(cls, logger: logging.Logger,
                              processor_name: str,
                              metrics: Dict[str, Any]):
        """记录性能指标"""
        logger.info(f"性能指标 - 处理器: {processor_name}")
        for key, value in metrics.items():
            logger.info(f"  {key}: {value}")

class PerformanceMonitor:
    """性能监控器"""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.start_time = None
        self.metrics = {}
    
    def start(self):
        """开始监控"""
        self.start_time = datetime.now()
        
    def record_metric(self, name: str, value: Any):
        """记录指标"""
        self.metrics[name] = value
        
    def end_and_log(self, processor_name: str):
        """结束监控并记录"""
        if self.start_time:
            end_time = datetime.now()
            processing_time = (end_time - self.start_time).total_seconds()
            self.metrics['processing_time_seconds'] = processing_time
            
            ProcessorLogger.log_performance_metrics(
                self.logger, processor_name, self.metrics
            )

# 全局日志配置函数
def configure_global_logging(config: Dict[str, Any]):
    """
    配置全局日志设置
    
    Args:
        config: 日志配置字典
    """
    ProcessorLogger.setup_logging(
        log_level=config.get('log_level', 'INFO'),
        log_file=config.get('log_file'),
        enable_console=config.get('enable_console', True),
        log_format=config.get('log_format')
    )

# 默认配置
DEFAULT_LOG_CONFIG = {
    'log_level': 'INFO',
    'enable_console': True,
    'log_format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
} 