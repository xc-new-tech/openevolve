#!/usr/bin/env python3
"""
统一的异常处理模块

定义所有处理器可能抛出的异常类型
"""

class ProcessorError(Exception):
    """处理器异常基类"""
    
    def __init__(self, message: str, processor_name: str = None, cause: Exception = None):
        super().__init__(message)
        self.processor_name = processor_name
        self.cause = cause
        
    def __str__(self):
        if self.processor_name:
            return f"[{self.processor_name}] {super().__str__()}"
        return super().__str__()

class ValidationError(ProcessorError):
    """输入验证错误"""
    pass

class ConfigurationError(ProcessorError):
    """配置错误"""
    pass

class ProcessingError(ProcessorError):
    """处理过程中的错误"""
    pass

class ResourceError(ProcessorError):
    """资源相关错误（内存、GPU等）"""
    pass

class DependencyError(ProcessorError):
    """依赖模块错误"""
    pass

class TimeoutError(ProcessorError):
    """处理超时错误"""
    pass

# 异常映射 - 将常见异常映射到我们的异常类型
EXCEPTION_MAPPING = {
    ValueError: ValidationError,
    TypeError: ValidationError,
    KeyError: ConfigurationError,
    ImportError: DependencyError,
    MemoryError: ResourceError,
    OSError: ResourceError,
    RuntimeError: ProcessingError,
} 