#!/usr/bin/env python3
"""
OpenEvolve API 配置测试脚本
用于验证不同LLM提供商的API密钥配置是否正确
"""

import asyncio
import os
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 加载.env文件
try:
    from dotenv import load_dotenv
    env_file = project_root / ".env"
    if env_file.exists():
        load_dotenv(env_file)
        print(f"✅ 已加载环境变量文件: {env_file}")
    else:
        print(f"⚠️  未找到.env文件: {env_file}")
except ImportError:
    print("⚠️  未安装python-dotenv，尝试安装: pip install python-dotenv")
except Exception as e:
    print(f"⚠️  加载.env文件时出错: {e}")

from openevolve.config import load_config, LLMModelConfig
from openevolve.llm.openai import OpenAILLM


def check_env_vars():
    """检查环境变量中的API密钥"""
    print("=== 检查环境变量 ===")
    
    api_keys = {
        "OpenAI": "OPENAI_API_KEY",
        "Anthropic": "ANTHROPIC_API_KEY", 
        "Google": "GOOGLE_API_KEY",
        "Mistral": "MISTRAL_API_KEY",
        "OpenRouter": "OPENROUTER_API_KEY",
        "xAI": "XAI_API_KEY",
        "Azure OpenAI": "AZURE_OPENAI_API_KEY",
        "Perplexity": "PERPLEXITY_API_KEY",
    }
    
    found_keys = []
    for provider, env_var in api_keys.items():
        value = os.getenv(env_var)
        if value:
            # 只显示密钥的前几位和后几位
            masked = f"{value[:8]}...{value[-4:]}" if len(value) > 12 else "***"
            print(f"✅ {provider}: {masked}")
            found_keys.append(provider)
        else:
            print(f"❌ {provider}: 未设置")
    
    if not found_keys:
        print("\n⚠️  未找到任何API密钥！请设置至少一个提供商的API密钥。")
        return False
    
    print(f"\n✅ 找到 {len(found_keys)} 个API密钥")
    return True


async def test_api_connection(provider, model_config):
    """测试API连接"""
    try:
        print(f"\n--- 测试 {provider} ---")
        llm = OpenAILLM(model_config)
        
        # 发送简单测试请求
        response = await llm.generate("说'你好'", max_tokens=10)
        print(f"✅ {provider} 测试成功")
        print(f"   模型: {model_config.name}")
        print(f"   响应: {response.strip()}")
        return True
        
    except Exception as e:
        print(f"❌ {provider} 测试失败: {str(e)}")
        return False


async def test_config_file(config_path):
    """测试配置文件中的设置"""
    print(f"\n=== 测试配置文件: {config_path} ===")
    
    try:
        config = load_config(config_path)
        print(f"✅ 配置文件加载成功")
        print(f"   语言: {config.language}")
        print(f"   模型数量: {len(config.llm.models)}")
        
        # 测试每个模型
        success_count = 0
        for i, model in enumerate(config.llm.models):
            if model.name:
                provider = f"模型{i+1} ({model.name})"
                success = await test_api_connection(provider, model)
                if success:
                    success_count += 1
        
        print(f"\n📊 测试结果: {success_count}/{len(config.llm.models)} 个模型可用")
        return success_count > 0
        
    except Exception as e:
        print(f"❌ 配置文件测试失败: {str(e)}")
        return False


def create_test_models():
    """创建测试模型配置"""
    test_models = []
    
    # OpenAI
    if os.getenv("OPENAI_API_KEY"):
        test_models.append(LLMModelConfig(
            name="gpt-3.5-turbo",
            api_base="https://api.openai.com/v1",
            api_key=os.getenv("OPENAI_API_KEY"),
            temperature=0.7,
            max_tokens=50,
            timeout=30,
            retries=2,
            retry_delay=2
        ))
    
    # Google Gemini
    if os.getenv("GOOGLE_API_KEY"):
        test_models.append(LLMModelConfig(
            name="gemini-2.0-flash-lite", 
            api_base="https://generativelanguage.googleapis.com/v1beta/openai/",
            api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=0.7,
            max_tokens=50,
            timeout=30,
            retries=2,
            retry_delay=2
        ))
    
    # Anthropic
    if os.getenv("ANTHROPIC_API_KEY"):
        test_models.append(LLMModelConfig(
            name="claude-3-haiku-20240307",
            api_base="https://api.anthropic.com", 
            api_key=os.getenv("ANTHROPIC_API_KEY"),
            temperature=0.7,
            max_tokens=50,
            timeout=30,
            retries=2,
            retry_delay=2
        ))
    
    # OpenRouter
    if os.getenv("OPENROUTER_API_KEY"):
        test_models.append(LLMModelConfig(
            name="anthropic/claude-3-haiku",
            api_base="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY"),
            temperature=0.7,
            max_tokens=50,
            timeout=30,
            retries=2,
            retry_delay=2
        ))
    
    return test_models


async def main():
    """主测试函数"""
    print("🧪 OpenEvolve API 配置测试工具")
    print("=" * 50)
    
    # 检查环境变量
    if not check_env_vars():
        return
    
    # 测试环境变量中的API密钥
    print("\n=== 测试环境变量中的API配置 ===")
    test_models = create_test_models()
    
    if not test_models:
        print("❌ 没有找到可测试的API密钥")
        return
    
    success_count = 0
    for model in test_models:
        provider = model.name.split("-")[0].upper()
        success = await test_api_connection(provider, model)
        if success:
            success_count += 1
    
    print(f"\n📊 环境变量测试结果: {success_count}/{len(test_models)} 个API可用")
    
    # 测试配置文件（如果存在）
    config_files = [
        "config.yaml",
        "configs/default_config.yaml",
        "examples/circle_packing/config_phase_1.yaml"
    ]
    
    for config_file in config_files:
        config_path = project_root / config_file
        if config_path.exists():
            await test_config_file(config_path)
            break
    
    print("\n✅ 测试完成!")
    print("\n💡 提示:")
    print("- 如果测试失败，请检查API密钥是否正确")
    print("- 确保网络连接正常")
    print("- 检查API配额是否充足")
    print("- 参考 docs/api_configuration.md 获取详细配置说明")


if __name__ == "__main__":
    asyncio.run(main()) 