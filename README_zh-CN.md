# OpenEvolve - 进化编程框架

<div align="center">
  <img src="openevolve-logo.png" alt="OpenEvolve Logo" width="300">
</div>

**OpenEvolve** 是一个开源的进化编程框架，结合大语言模型 (LLM) 的强大能力来自动化改进代码和程序。

## 🚀 项目特色

- **多语言支持**：支持 Python、C、C++ 等编程语言
- **LLM 驱动**：利用 GPT-4、Claude 等大语言模型进行智能代码生成和优化
- **并行评估**：支持多种群、岛屿模型的并行进化
- **灵活配置**：YAML 配置驱动，支持各种问题域定制
- **安全执行**：内置安全约束和沙箱机制
- **丰富示例**：包含符号回归、函数优化、在线判题等多种应用场景

## 📋 系统要求

- Python 3.9+
- C/C++ 编译器（gcc、g++、clang 或 clang++）- 用于 C/C++ 支持
- 支持的 LLM API 密钥（OpenAI、Anthropic、Google 等）

## 🛠 安装

### 基础安装

```bash
# 克隆仓库
git clone https://github.com/your-org/openevolve.git
cd openevolve

# 安装 Python 依赖
pip install -e .
```

### API 密钥配置

OpenEvolve 需要大语言模型 API 来驱动代码进化。支持多种提供商：

1. **创建环境变量文件**：
```bash
# 将示例内容复制到 .env 文件
cp docs/env_example.txt .env
```

2. **添加 API 密钥**（选择您使用的提供商）：
```bash
# OpenAI (推荐)
OPENAI_API_KEY=sk-your_openai_api_key_here

# 或者 Google Gemini (免费配额更高)
GOOGLE_API_KEY=your_google_api_key_here

# 或者 Anthropic Claude
ANTHROPIC_API_KEY=your_anthropic_api_key_here
```

3. **快速设置**（推荐新用户）：
```bash
# 交互式设置向导
python scripts/setup_api.py
```

4. **测试配置**：
```bash
python scripts/test_api_config.py
```

> 📝 **获取 API 密钥**：
> - [OpenAI](https://platform.openai.com/api-keys)
> - [Google AI Studio](https://makersuite.google.com/app/apikey)  
> - [Anthropic](https://console.anthropic.com/)
> 
> 详细配置说明请参考 [docs/api_configuration.md](docs/api_configuration.md)

### C/C++ 支持安装

#### Ubuntu/Debian
```bash
sudo apt-get update
sudo apt-get install -y build-essential gcc g++ clang clang++ libc6-dev make
```

#### macOS
```bash
# 安装 Xcode Command Line Tools
xcode-select --install

# 或使用 Homebrew
brew install gcc llvm
```

#### Windows
```bash
# 使用 Windows Subsystem for Linux (WSL)
# 或安装 Visual Studio Build Tools
```

## 🏃‍♂️ 快速开始

### Python 示例

```bash
# 运行 Python 符号回归示例
python openevolve-run.py --config examples/symbolic_regression/config.yaml \
                         --initial_program examples/symbolic_regression/initial_program.py \
                         --evaluation examples/symbolic_regression/eval.py
```

### C++ 示例

```bash
# 运行 C++ 符号回归示例
python openevolve-run.py --config examples/symbolic_regression_c/config.yaml \
                         --initial_program examples/symbolic_regression_c/initial_program.cpp \
                         --evaluation examples/symbolic_regression_c/evaluator.py
```

## 🏗 架构概述

<div align="center">
  <img src="openevolve-architecture.png" alt="OpenEvolve Architecture" width="600">
</div>

### 核心组件

| 组件 | 功能 |
|------|------|
| **Controller** | 主控制器，协调整个进化过程 |
| **LLM Interface** | 与大语言模型交互的统一接口 |
| **Evaluator** | 程序评估器，支持多种语言和评估策略 |
| **Database** | 程序版本管理和进化历史存储 |
| **Prompt Manager** | 智能提示词生成和模板管理 |

### 支持的编程语言

- **Python** ✅ 完全支持
- **C** ✅ 完全支持（v2.0+）
- **C++** ✅ 完全支持（v2.0+）
- **JavaScript** 🚧 计划中
- **Rust** 🚧 计划中

## 📁 项目结构

```
openevolve/
├── openevolve/           # 核心框架代码
│   ├── llm/             # LLM 接口和适配器
│   ├── prompt/          # 提示词模板和采样器
│   ├── utils/           # 工具函数
│   ├── controller.py    # 主控制器
│   ├── evaluator.py     # 评估器（支持多语言）
│   ├── config.py        # 配置管理
│   └── database.py      # 数据存储
├── examples/            # 示例和用例
│   ├── symbolic_regression/     # Python 符号回归
│   ├── symbolic_regression_c/   # C++ 符号回归
│   ├── function_minimization/   # 函数优化
│   └── online_judge_programming/ # 在线判题
├── configs/             # 配置文件模板
├── scripts/             # 可视化和工具脚本
└── tests/              # 单元测试
```

## 🎯 应用场景

### 1. 符号回归
自动发现数据背后的数学公式：

```cpp
// 目标：找到逼近 y = x² + x 的函数
double evaluate_function(double x) {
    return x * x + x;  // 进化后的结果
}
```

### 2. 算法优化
改进现有算法的性能和准确性：

```python
def optimized_algorithm(data):
    # LLM 自动生成的优化版本
    pass
```

### 3. 代码重构
自动重构代码以提高可读性和性能：

```c
// 重构前
int old_function(int a, int b) {
    // 复杂的实现
}

// 重构后
int new_function(int a, int b) {
    // 简化且高效的实现
}
```

## ⚙️ 配置指南

### 语言特定配置

#### C/C++ 配置
```yaml
# config.yaml
language: cpp  # 选项: c, cpp, python
compiler: auto  # 选项: auto, gcc, g++, clang, clang++
compile_flags: ["-O2", "-Wall", "-Wextra", "-std=c++17"]
compile_timeout: 10.0
run_timeout: 30.0
```

#### LLM 配置
```yaml
llm:
  api_base: "https://api.openai.com/v1"
  temperature: 0.7
  models:
    - name: "gpt-4"
      weight: 1.0
```

### 安全约束

OpenEvolve 内置多层安全机制：

- **代码沙箱**：限制文件 I/O 和系统调用
- **执行超时**：防止无限循环
- **内存限制**：控制资源使用
- **编译安全**：安全的编译标志

## 📊 可视化界面

OpenEvolve 提供了直观的 Web 界面来监控进化过程：

<div align="center">
  <img src="openevolve-visualizer.png" alt="OpenEvolve Visualizer" width="500">
</div>

启动可视化界面：
```bash
python scripts/visualizer.py
```

访问 `http://localhost:8080` 查看：
- 实时进化图表
- 程序性能统计
- 代码版本历史
- 评估指标趋势

## 🤝 贡献指南

我们欢迎各种形式的贡献！

### 开发环境设置

```bash
# Fork 并克隆仓库
git clone https://github.com/your-username/openevolve.git
cd openevolve

# 创建开发环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或 venv\Scripts\activate  # Windows

# 安装开发依赖
pip install -e .[dev]

# 运行测试
python -m pytest tests/
```

### 贡献类型

- 🐛 **Bug 修复**：修复现有问题
- ✨ **新功能**：添加新的语言支持或功能
- 📚 **文档**：改进文档和示例
- 🧪 **测试**：增加测试覆盖率
- 🎨 **代码质量**：代码重构和优化

### 提交 Pull Request

1. 创建功能分支：`git checkout -b feature/amazing-feature`
2. 提交更改：`git commit -m 'Add amazing feature'`
3. 推送分支：`git push origin feature/amazing-feature`
4. 创建 Pull Request

## 📝 许可证

本项目基于 [MIT License](LICENSE) 开源。

## 🙏 致谢

感谢以下项目和贡献者：

- OpenAI GPT 系列模型
- Anthropic Claude 模型
- 所有开源贡献者

## 📞 联系方式

- **GitHub Issues**: [报告问题](https://github.com/your-org/openevolve/issues)
- **讨论**: [GitHub Discussions](https://github.com/your-org/openevolve/discussions)
- **邮件**: openevolve@example.com

---

<div align="center">
  <p>⭐ 如果这个项目对您有帮助，请给我们一个星标！</p>
</div> 