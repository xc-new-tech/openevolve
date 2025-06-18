# OpenEvolve 大语言模型 API 配置指南

OpenEvolve 支持多种大语言模型提供商。本指南将详细说明如何配置API密钥和相关设置。

## 支持的LLM提供商

- **OpenAI** (GPT-3.5, GPT-4, GPT-4o)
- **Anthropic** (Claude 3.5 Sonnet, Claude 3 Haiku)
- **Google** (Gemini Pro, Gemini Flash)
- **Mistral AI**
- **OpenRouter** (访问多种开源模型)
- **xAI** (Grok)
- **Azure OpenAI** (企业版)
- **Ollama** (本地部署)
- **Perplexity** (用于研究功能)

## 配置方法

### 方法1: 环境变量配置（推荐）

在项目根目录创建 `.env` 文件：

```bash
# 复制示例文件
cp .env.example .env
```

然后编辑 `.env` 文件，添加您需要的API密钥：

```bash
# OpenAI (GPT-4o, GPT-4, GPT-3.5)
OPENAI_API_KEY=sk-your_openai_api_key_here

# Anthropic (Claude)
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# Google (Gemini)
GOOGLE_API_KEY=your_google_api_key_here

# Mistral AI
MISTRAL_API_KEY=your_mistral_api_key_here

# OpenRouter
OPENROUTER_API_KEY=your_openrouter_api_key_here

# xAI (Grok)
XAI_API_KEY=your_xai_api_key_here

# Azure OpenAI
AZURE_OPENAI_API_KEY=your_azure_openai_key_here
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/

# Ollama (本地部署)
OLLAMA_BASE_URL=http://localhost:11434/api
# OLLAMA_API_KEY=optional_if_auth_enabled

# Perplexity (研究功能)
PERPLEXITY_API_KEY=your_perplexity_api_key_here
```

### 方法2: 配置文件直接设置

在您的配置文件 (如 `config.yaml`) 中直接设置API密钥：

```yaml
llm:
  api_key: "your_api_key_here"
  api_base: "https://api.openai.com/v1"
  models:
    - name: "gpt-4o"
      weight: 0.8
    - name: "gpt-3.5-turbo"
      weight: 0.2
```

**注意**: 方法2不推荐用于生产环境，因为API密钥会明文存储在配置文件中。

## 不同提供商的配置示例

### OpenAI 配置

```yaml
llm:
  api_base: "https://api.openai.com/v1"
  api_key: null  # 使用环境变量 OPENAI_API_KEY
  models:
    - name: "gpt-4o"
      weight: 0.7
    - name: "gpt-4o-mini"
      weight: 0.3
  temperature: 0.7
  max_tokens: 4096
```

### Anthropic (Claude) 配置

```yaml
llm:
  api_base: "https://api.anthropic.com"
  api_key: null  # 使用环境变量 ANTHROPIC_API_KEY
  models:
    - name: "claude-3-5-sonnet-20241022"
      weight: 0.8
    - name: "claude-3-haiku-20240307"
      weight: 0.2
```

### Google (Gemini) 配置

```yaml
llm:
  api_base: "https://generativelanguage.googleapis.com/v1beta/openai/"
  api_key: null  # 使用环境变量 GOOGLE_API_KEY
  models:
    - name: "gemini-2.0-flash-lite"
      weight: 0.8
    - name: "gemini-2.0-flash"
      weight: 0.2
```

### OpenRouter 配置

```yaml
llm:
  api_base: "https://openrouter.ai/api/v1"
  api_key: null  # 使用环境变量 OPENROUTER_API_KEY
  models:
    - name: "anthropic/claude-3.5-sonnet"
      weight: 0.6
    - name: "meta-llama/llama-3.1-70b-instruct"
      weight: 0.4
```

### Ollama (本地部署) 配置

```yaml
llm:
  api_base: "http://localhost:11434/v1"
  api_key: null  # Ollama通常不需要API密钥
  models:
    - name: "llama3.1:70b"
      weight: 0.7
    - name: "codellama:34b"
      weight: 0.3
```

## 获取API密钥

### OpenAI
1. 访问 [OpenAI Platform](https://platform.openai.com/api-keys)
2. 登录并点击 "Create new secret key"
3. 复制生成的密钥 (格式: sk-...)

### Anthropic
1. 访问 [Anthropic Console](https://console.anthropic.com/)
2. 在 API Keys 部分创建新密钥
3. 复制生成的密钥

### Google (Gemini)
1. 访问 [Google AI Studio](https://makersuite.google.com/app/apikey)
2. 创建新的API密钥
3. 复制生成的密钥

### Mistral
1. 访问 [Mistral Console](https://console.mistral.ai/)
2. 在 API Keys 部分创建新密钥
3. 复制生成的密钥

### OpenRouter
1. 访问 [OpenRouter](https://openrouter.ai/keys)
2. 创建账户并生成API密钥
3. 复制生成的密钥

### xAI
1. 访问 [xAI Console](https://console.x.ai/)
2. 创建账户并生成API密钥
3. 复制生成的密钥

### Perplexity
1. 访问 [Perplexity API](https://www.perplexity.ai/settings/api)
2. 创建账户并生成API密钥
3. 复制生成的密钥

## 多模型配置

OpenEvolve 支持使用多个模型的集成，您可以设置权重来控制每个模型的使用频率：

```yaml
llm:
  models:
    - name: "gpt-4o"
      weight: 0.5
      api_base: "https://api.openai.com/v1"
      api_key: null  # 使用 OPENAI_API_KEY
    - name: "claude-3-5-sonnet-20241022"
      weight: 0.3
      api_base: "https://api.anthropic.com"
      api_key: null  # 使用 ANTHROPIC_API_KEY
    - name: "gemini-2.0-flash"
      weight: 0.2
      api_base: "https://generativelanguage.googleapis.com/v1beta/openai/"
      api_key: null  # 使用 GOOGLE_API_KEY
  
  # 评估模型可以与进化模型不同
  evaluator_models:
    - name: "gpt-4o"
      weight: 1.0
```

## 安全注意事项

1. **永远不要**将API密钥提交到版本控制系统
2. 将 `.env` 文件添加到 `.gitignore` 中
3. 使用环境变量而不是硬编码API密钥
4. 定期轮换API密钥
5. 监控API使用情况和费用

## 测试配置

运行以下命令测试您的API配置：

```bash
# 测试基本配置
python -c "
from openevolve.config import load_config
from openevolve.llm.openai import OpenAILLM
import asyncio

config = load_config('your_config.yaml')
llm = OpenAILLM(config.llm.models[0])
result = asyncio.run(llm.generate('Hello, world!'))
print(f'测试成功: {result}')
"
```

## 故障排除

### 常见错误

1. **"Invalid API key"**: 检查API密钥格式和有效性
2. **"Rate limit exceeded"**: 降低请求频率或升级API计划
3. **"Model not found"**: 确认模型名称正确
4. **连接超时**: 检查网络连接和API端点

### 调试技巧

1. 启用调试日志：
```yaml
log_level: "DEBUG"
```

2. 检查环境变量：
```bash
echo $OPENAI_API_KEY
```

3. 验证网络连接：
```bash
curl -H "Authorization: Bearer $OPENAI_API_KEY" https://api.openai.com/v1/models
```

## 性能优化

1. **模型选择**: 根据任务复杂度选择合适的模型
2. **批处理**: 使用并行评估减少延迟
3. **缓存**: 启用结果缓存避免重复调用
4. **超时设置**: 合理设置超时时间

```yaml
llm:
  timeout: 60      # API调用超时时间
  retries: 3       # 重试次数
  retry_delay: 5   # 重试间隔
  
evaluator:
  parallel_evaluations: 4  # 并行评估数量
``` 