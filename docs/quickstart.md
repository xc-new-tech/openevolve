# OpenEvolve 快速开始

这个指南将帮助您在 10 分钟内开始使用 OpenEvolve。

## 1. 安装 (2 分钟)

```bash
# 克隆仓库
git clone https://github.com/codelion/openevolve.git
cd openevolve

# 安装依赖
pip install -e .
```

## 2. 设置 API 密钥 (1 分钟)

创建 `.env` 文件：

```bash
# 复制示例文件
cp docs/env_example.txt .env

# 编辑 .env 文件，添加您的 API 密钥
# 至少需要设置一个 LLM 提供商的密钥
```

最少配置示例：
```bash
# 使用 OpenAI
OPENAI_API_KEY=sk-your_api_key_here

# 或使用 Google Gemini（免费配额更高）
GOOGLE_API_KEY=your_google_api_key_here
```

## 3. 运行第一个示例 (3 分钟)

```bash
# 运行函数优化示例
python openevolve-run.py \
  examples/function_minimization/initial_program.py \
  examples/function_minimization/evaluator.py \
  --config examples/function_minimization/config.yaml \
  --iterations 50
```

## 4. 查看结果 (2 分钟)

进化完成后：

```bash
# 查看最佳程序
cat examples/function_minimization/openevolve_output/best_program.py

# 查看进化历史
ls examples/function_minimization/openevolve_output/checkpoints/
```

## 5. 可视化进化过程 (2 分钟)

```bash
# 安装可视化依赖
pip install -r scripts/requirements.txt

# 启动可视化服务器
python scripts/visualizer.py

# 在浏览器中打开 http://localhost:8000
```

## 下一步

1. **阅读完整指南**: [用户指南](user_guide.md)
2. **尝试其他示例**: 查看 `examples/` 目录
3. **创建自己的项目**: 参考教程创建自定义进化任务

## 快速问题解决

### 常见错误

**错误: "No API key found"**
```bash
# 确保 .env 文件中有正确的 API 密钥
cat .env | grep API_KEY
```

**错误: "Module not found"**
```bash
# 确保正确安装了包
pip install -e .
```

**错误: "Permission denied"**
```bash
# 确保有执行权限
chmod +x openevolve-run.py
```

### 性能优化

如果运行缓慢，尝试：
- 减少迭代次数：`--iterations 20`
- 使用更快的模型（如 `gpt-3.5-turbo` 或 `gemini-flash`）
- 减少种群大小：在配置文件中设置 `population_size: 20`

## 获取帮助

- [用户指南](user_guide.md) - 详细文档
- [API 配置](api_configuration.md) - API 密钥设置
- [GitHub Issues](https://github.com/codelion/openevolve/issues) - 报告问题

---

*现在您已经可以开始使用 OpenEvolve 了！* 