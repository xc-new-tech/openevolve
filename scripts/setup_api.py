#!/usr/bin/env python3
"""
OpenEvolve API 快速设置脚本
交互式配置API密钥和基本设置
"""

import os
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent

def main():
    print("🚀 OpenEvolve API 快速设置")
    print("=" * 50)
    
    # 检查 .env 文件是否存在
    env_file = project_root / ".env"
    if env_file.exists():
        print(f"发现现有的 .env 文件: {env_file}")
        overwrite = input("是否覆盖现有配置? (y/N): ").lower().strip()
        if overwrite != 'y':
            print("保持现有配置不变。")
            return
    
    print("\n请选择您想要配置的LLM提供商（可以配置多个）:")
    print("1. OpenAI (GPT-4o, GPT-4) - 最强性能")
    print("2. Google Gemini - 免费配额高，性价比好")  
    print("3. Anthropic Claude - 代码理解能力强")
    print("4. 其他提供商")
    print("5. 我已经有 API 密钥文件了")
    
    choice = input("\n请输入选择 (1-5): ").strip()
    
    env_content = []
    env_content.append("# OpenEvolve API 配置")
    env_content.append("# 由设置脚本自动生成")
    env_content.append("")
    
    if choice == "1":
        setup_openai(env_content)
    elif choice == "2":
        setup_google(env_content)
    elif choice == "3":
        setup_anthropic(env_content)
    elif choice == "4":
        setup_other(env_content)
    elif choice == "5":
        print("\n请手动编辑 .env 文件或参考 docs/env_example.txt")
        return
    else:
        print("无效选择，退出。")
        return
    
    # 写入 .env 文件
    with open(env_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(env_content))
    
    print(f"\n✅ 配置已保存到: {env_file}")
    
    # 提示下一步
    print("\n🎯 下一步:")
    print("1. 测试配置: python scripts/test_api_config.py")
    print("2. 运行示例: python openevolve-run.py --config examples/symbolic_regression/config.yaml")
    print("3. 查看文档: docs/api_configuration.md")


def setup_openai(env_content):
    print("\n🔑 OpenAI 配置")
    print("获取API密钥: https://platform.openai.com/api-keys")
    
    api_key = input("请输入您的 OpenAI API 密钥 (sk-...): ").strip()
    if api_key and api_key.startswith('sk-'):
        env_content.append("# OpenAI Configuration")
        env_content.append(f"OPENAI_API_KEY={api_key}")
        env_content.append("")
        print("✅ OpenAI 配置完成")
    else:
        print("❌ 无效的 OpenAI API 密钥格式")


def setup_google(env_content):
    print("\n🔑 Google Gemini 配置")
    print("获取API密钥: https://makersuite.google.com/app/apikey")
    print("💡 提示: Google 提供较高的免费配额")
    
    api_key = input("请输入您的 Google API 密钥: ").strip()
    if api_key:
        env_content.append("# Google Gemini Configuration")
        env_content.append(f"GOOGLE_API_KEY={api_key}")
        env_content.append("")
        print("✅ Google Gemini 配置完成")
    else:
        print("❌ 请输入有效的 Google API 密钥")


def setup_anthropic(env_content):
    print("\n🔑 Anthropic Claude 配置")
    print("获取API密钥: https://console.anthropic.com/")
    
    api_key = input("请输入您的 Anthropic API 密钥: ").strip()
    if api_key:
        env_content.append("# Anthropic Claude Configuration")
        env_content.append(f"ANTHROPIC_API_KEY={api_key}")
        env_content.append("")
        print("✅ Anthropic Claude 配置完成")
    else:
        print("❌ 请输入有效的 Anthropic API 密钥")


def setup_other(env_content):
    print("\n🔑 其他提供商配置")
    
    providers = {
        "1": ("Mistral AI", "MISTRAL_API_KEY", "https://console.mistral.ai/"),
        "2": ("OpenRouter", "OPENROUTER_API_KEY", "https://openrouter.ai/keys"),
        "3": ("xAI (Grok)", "XAI_API_KEY", "https://console.x.ai/"),
        "4": ("Perplexity", "PERPLEXITY_API_KEY", "https://www.perplexity.ai/settings/api"),
    }
    
    print("选择提供商:")
    for key, (name, _, url) in providers.items():
        print(f"{key}. {name}")
    
    choice = input("请选择 (1-4): ").strip()
    
    if choice in providers:
        name, env_var, url = providers[choice]
        print(f"\n配置 {name}")
        print(f"获取API密钥: {url}")
        
        api_key = input(f"请输入您的 {name} API 密钥: ").strip()
        if api_key:
            env_content.append(f"# {name} Configuration")
            env_content.append(f"{env_var}={api_key}")
            env_content.append("")
            print(f"✅ {name} 配置完成")
        else:
            print(f"❌ 请输入有效的 {name} API 密钥")
    else:
        print("无效选择")


if __name__ == "__main__":
    main() 