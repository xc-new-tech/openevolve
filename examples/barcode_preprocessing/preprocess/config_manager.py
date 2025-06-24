#!/usr/bin/env python3
"""
增强的配置管理系统

支持分层配置、模板和验证机制
"""

import yaml
import json
import os
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field
from pathlib import Path
import copy
import logging

logger = logging.getLogger(__name__)

@dataclass
class ConfigValidationRule:
    """配置验证规则"""
    field_name: str
    data_type: type
    required: bool = True
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    allowed_values: Optional[List[Any]] = None
    default_value: Optional[Any] = None
    description: str = ""

class ConfigTemplate:
    """配置模板类"""
    
    def __init__(self, name: str, description: str, config: Dict[str, Any]):
        self.name = name
        self.description = description
        self.config = config
        self.validation_rules: List[ConfigValidationRule] = []
    
    def add_validation_rule(self, rule: ConfigValidationRule):
        """添加验证规则"""
        self.validation_rules.append(rule)
    
    def validate(self, config: Dict[str, Any]) -> Dict[str, List[str]]:
        """验证配置"""
        errors = {}
        
        for rule in self.validation_rules:
            field_errors = []
            value = config.get(rule.field_name)
            
            # 检查必填字段
            if rule.required and value is None:
                field_errors.append(f"字段 {rule.field_name} 是必填的")
                continue
            
            # 如果字段不存在且不是必填，跳过后续检查
            if value is None:
                continue
            
            # 检查数据类型
            if not isinstance(value, rule.data_type):
                field_errors.append(f"字段 {rule.field_name} 类型应为 {rule.data_type.__name__}")
                continue
            
            # 检查数值范围
            if rule.min_value is not None and isinstance(value, (int, float)):
                if value < rule.min_value:
                    field_errors.append(f"字段 {rule.field_name} 值不能小于 {rule.min_value}")
            
            if rule.max_value is not None and isinstance(value, (int, float)):
                if value > rule.max_value:
                    field_errors.append(f"字段 {rule.field_name} 值不能大于 {rule.max_value}")
            
            # 检查允许值
            if rule.allowed_values is not None:
                if value not in rule.allowed_values:
                    field_errors.append(f"字段 {rule.field_name} 值必须是 {rule.allowed_values} 之一")
            
            if field_errors:
                errors[rule.field_name] = field_errors
        
        return errors

class HierarchicalConfig:
    """分层配置类"""
    
    def __init__(self, name: str = ""):
        self.name = name
        self.global_config = {}
        self.algorithm_configs = {}
        self.instance_configs = {}
        self.templates = {}
        
    def set_global_config(self, config: Dict[str, Any]):
        """设置全局配置"""
        self.global_config = copy.deepcopy(config)
        
    def set_algorithm_config(self, algorithm_name: str, config: Dict[str, Any]):
        """设置算法特定配置"""
        self.algorithm_configs[algorithm_name] = copy.deepcopy(config)
        
    def set_instance_config(self, instance_id: str, config: Dict[str, Any]):
        """设置实例配置"""
        self.instance_configs[instance_id] = copy.deepcopy(config)
        
    def get_merged_config(self, algorithm_name: str = None, instance_id: str = None) -> Dict[str, Any]:
        """获取合并后的配置"""
        # 从全局配置开始
        merged = copy.deepcopy(self.global_config)
        
        # 合并算法配置
        if algorithm_name and algorithm_name in self.algorithm_configs:
            merged = self._deep_merge(merged, self.algorithm_configs[algorithm_name])
        
        # 合并实例配置
        if instance_id and instance_id in self.instance_configs:
            merged = self._deep_merge(merged, self.instance_configs[instance_id])
            
        return merged
    
    def _deep_merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """深度合并两个字典"""
        result = copy.deepcopy(base)
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = copy.deepcopy(value)
                
        return result
    
    def add_template(self, template: ConfigTemplate):
        """添加配置模板"""
        self.templates[template.name] = template
        
    def apply_template(self, template_name: str, level: str = "global", 
                      algorithm_name: str = None, instance_id: str = None):
        """应用配置模板"""
        if template_name not in self.templates:
            raise ValueError(f"模板 {template_name} 不存在")
            
        template = self.templates[template_name]
        
        if level == "global":
            self.global_config = self._deep_merge(self.global_config, template.config)
        elif level == "algorithm" and algorithm_name:
            if algorithm_name not in self.algorithm_configs:
                self.algorithm_configs[algorithm_name] = {}
            self.algorithm_configs[algorithm_name] = self._deep_merge(
                self.algorithm_configs[algorithm_name], template.config
            )
        elif level == "instance" and instance_id:
            if instance_id not in self.instance_configs:
                self.instance_configs[instance_id] = {}
            self.instance_configs[instance_id] = self._deep_merge(
                self.instance_configs[instance_id], template.config
            )

class ConfigManager:
    """配置管理器"""
    
    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)
        
        self.hierarchical_config = HierarchicalConfig()
        self.templates = {}
        self._load_default_templates()
        
    def _load_default_templates(self):
        """加载默认配置模板"""
        
        # 条形码识别优化模板
        barcode_template = ConfigTemplate(
            name="barcode_optimization",
            description="条形码识别优化配置",
            config={
                "enhancement": {
                    "enable": True,
                    "methods": ["clahe", "unsharp"],
                    "clahe": {"clip_limit": 2.0, "tile_grid_size": [8, 8]},
                    "unsharp": {"sigma": 1.0, "strength": 1.5}
                },
                "denoise": {
                    "enable": True,
                    "method": "bilateral",
                    "bilateral": {"d": 9, "sigma_color": 75, "sigma_space": 75}
                },
                "geometry": {
                    "enable": True,
                    "auto_rotate": True,
                    "perspective_correction": True
                }
            }
        )
        
        # 添加验证规则
        barcode_template.add_validation_rule(
            ConfigValidationRule("enhancement.clahe.clip_limit", float, min_value=0.1, max_value=10.0)
        )
        barcode_template.add_validation_rule(
            ConfigValidationRule("denoise.bilateral.d", int, min_value=3, max_value=15)
        )
        
        self.add_template(barcode_template)
        
        # 高质量处理模板
        high_quality_template = ConfigTemplate(
            name="high_quality",
            description="高质量图像处理配置",
            config={
                "enhancement": {
                    "enable": True,
                    "methods": ["multi_scale", "high_freq"],
                    "multi_scale": {"scales": [0.5, 1.0, 1.5, 2.0]},
                    "high_freq": {"cutoff": 0.05, "amplification": 1.5}
                },
                "super_resolution": {
                    "enable": True,
                    "model": "Real-ESRGAN",
                    "scale_factor": 2
                },
                "deep_denoise": {
                    "enable": True,
                    "model": "DnCNN",
                    "noise_level": "medium"
                }
            }
        )
        
        self.add_template(high_quality_template)
        
        # 快速处理模板
        fast_processing_template = ConfigTemplate(
            name="fast_processing",
            description="快速处理配置",
            config={
                "enhancement": {
                    "enable": True,
                    "methods": ["gamma"],
                    "gamma": {"adaptive": True}
                },
                "denoise": {
                    "enable": True,
                    "method": "gaussian",
                    "gaussian": {"kernel_size": 3, "sigma": 0.5}
                },
                "parallel_processing": {
                    "enable": True,
                    "num_workers": 4
                }
            }
        )
        
        self.add_template(fast_processing_template)
    
    def add_template(self, template: ConfigTemplate):
        """添加配置模板"""
        self.templates[template.name] = template
        self.hierarchical_config.add_template(template)
        
    def get_template(self, name: str) -> Optional[ConfigTemplate]:
        """获取配置模板"""
        return self.templates.get(name)
        
    def list_templates(self) -> List[str]:
        """列出所有模板"""
        return list(self.templates.keys())
    
    def load_config_file(self, file_path: str) -> Dict[str, Any]:
        """加载配置文件"""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {file_path}")
        
        if file_path.suffix.lower() == '.yaml' or file_path.suffix.lower() == '.yml':
            with open(file_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        elif file_path.suffix.lower() == '.json':
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            raise ValueError(f"不支持的配置文件格式: {file_path.suffix}")
    
    def save_config_file(self, config: Dict[str, Any], file_path: str, format: str = 'yaml'):
        """保存配置文件"""
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format.lower() == 'yaml':
            with open(file_path, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        elif format.lower() == 'json':
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
        else:
            raise ValueError(f"不支持的保存格式: {format}")
    
    def validate_config(self, config: Dict[str, Any], template_name: str = None) -> Dict[str, List[str]]:
        """验证配置"""
        if template_name and template_name in self.templates:
            return self.templates[template_name].validate(config)
        
        # 基本验证
        errors = {}
        
        # 检查必要的顶级字段
        required_fields = ["enhancement", "denoise", "geometry"]
        for field in required_fields:
            if field not in config:
                errors[field] = [f"缺少必要的配置节: {field}"]
        
        return errors
    
    def get_config_for_processor(self, processor_name: str, instance_id: str = None) -> Dict[str, Any]:
        """获取处理器的配置"""
        return self.hierarchical_config.get_merged_config(
            algorithm_name=processor_name,
            instance_id=instance_id
        )
    
    def set_global_config(self, config: Dict[str, Any]):
        """设置全局配置"""
        self.hierarchical_config.set_global_config(config)
        
    def set_processor_config(self, processor_name: str, config: Dict[str, Any]):
        """设置处理器配置"""
        self.hierarchical_config.set_algorithm_config(processor_name, config)
        
    def apply_template_to_global(self, template_name: str):
        """将模板应用到全局配置"""
        self.hierarchical_config.apply_template(template_name, level="global")
        
    def apply_template_to_processor(self, template_name: str, processor_name: str):
        """将模板应用到特定处理器"""
        self.hierarchical_config.apply_template(
            template_name, level="algorithm", algorithm_name=processor_name
        )
    
    def export_config_documentation(self, output_file: str):
        """导出配置文档"""
        doc_content = {
            "配置文档": {
                "说明": "这是自动生成的配置文档",
                "模板列表": {}
            }
        }
        
        for name, template in self.templates.items():
            template_doc = {
                "描述": template.description,
                "配置": template.config,
                "验证规则": []
            }
            
            for rule in template.validation_rules:
                rule_doc = {
                    "字段": rule.field_name,
                    "类型": rule.data_type.__name__,
                    "必填": rule.required,
                    "描述": rule.description
                }
                
                if rule.min_value is not None:
                    rule_doc["最小值"] = rule.min_value
                if rule.max_value is not None:
                    rule_doc["最大值"] = rule.max_value
                if rule.allowed_values is not None:
                    rule_doc["允许值"] = rule.allowed_values
                if rule.default_value is not None:
                    rule_doc["默认值"] = rule.default_value
                    
                template_doc["验证规则"].append(rule_doc)
            
            doc_content["配置文档"]["模板列表"][name] = template_doc
        
        self.save_config_file(doc_content, output_file, format='yaml')

# 全局配置管理器实例
config_manager = ConfigManager()

def get_config_manager() -> ConfigManager:
    """获取全局配置管理器"""
    return config_manager 