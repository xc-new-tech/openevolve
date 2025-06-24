# OpenEvolve 用户指南

这是一份全面的 OpenEvolve 用户指南，从基础使用到高级功能，帮助您充分利用这个进化编程框架。

## 目录

1. [快速开始](#快速开始)
2. [基础概念](#基础概念)
3. [详细教程](#详细教程)
4. [配置指南](#配置指南)
5. [高级功能](#高级功能)
6. [最佳实践](#最佳实践)
7. [常见问题](#常见问题)
8. [示例项目](#示例项目)

## 快速开始

### 安装

选择以下任一方式安装 OpenEvolve：

**方式一：从源码安装（推荐）**
```bash
git clone https://github.com/codelion/openevolve.git
cd openevolve
pip install -e .
```

**方式二：使用 Docker**
```bash
docker build -t openevolve .
```

### 第一个例子

让我们从一个简单的函数优化例子开始：

```python
from openevolve import OpenEvolve

# 初始化系统
evolve = OpenEvolve(
    initial_program_path="examples/function_minimization/initial_program.py",
    evaluation_file="examples/function_minimization/evaluator.py",
    config_path="examples/function_minimization/config.yaml"
)

# 运行进化
best_program = await evolve.run(iterations=100)
print(f"最佳程序指标: {best_program.metrics}")
```

**命令行运行：**
```bash
python openevolve-run.py \
  examples/function_minimization/initial_program.py \
  examples/function_minimization/evaluator.py \
  --config examples/function_minimization/config.yaml \
  --iterations 100
```

## 基础概念

### 进化编程原理

OpenEvolve 基于以下核心概念：

1. **种群 (Population)**: 一组候选程序
2. **适应度 (Fitness)**: 程序性能的评估指标
3. **变异 (Mutation)**: 通过 LLM 对程序进行修改
4. **选择 (Selection)**: 根据适应度选择优秀程序
5. **进化 (Evolution)**: 重复变异和选择的过程

### 系统架构

```
初始程序 → 提示采样器 → LLM集成 → 候选程序 → 评估器 → 程序数据库
    ↑                                                              ↓
    ←←←←←←←←←← 反馈循环 ←←←←←←←←←←←←←←←←←←←←←←←
```

#### 组件说明

- **提示采样器**: 创建包含历史程序和评分的上下文提示
- **LLM集成**: 使用多个语言模型生成代码修改
- **评估器池**: 测试生成的程序并分配评分
- **程序数据库**: 存储程序及其评估指标，指导未来进化

## 详细教程

### 教程 1: 创建您的第一个进化任务

#### 步骤 1: 准备初始程序

创建一个简单的初始程序 `initial_program.py`：

```python
def solve():
    """找到使函数 f(x) = x^2 + 2x + 1 最小的 x 值"""
    x = 0.0  # 初始猜测
    return x

if __name__ == "__main__":
    result = solve()
    print(f"x = {result}")
```

#### 步骤 2: 创建评估器

创建评估器 `evaluator.py`：

```python
import importlib.util
import sys
from openevolve.evaluation_result import EvaluationResult

def evaluate(program_path: str) -> EvaluationResult:
    """评估程序的性能"""
    try:
        # 动态导入程序
        spec = importlib.util.spec_from_file_location("program", program_path)
        program = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(program)
        
        # 运行程序
        x = program.solve()
        
        # 计算目标函数值 f(x) = x^2 + 2x + 1
        f_value = x**2 + 2*x + 1
        
        # 适应度是负的函数值（最小化问题）
        fitness = -f_value
        
        return EvaluationResult(
            metrics={
                "fitness": fitness,
                "f_value": f_value,
                "x": x
            }
        )
        
    except Exception as e:
        return EvaluationResult(
            metrics={"fitness": -float('inf'), "error": 1.0},
            artifacts={"error": str(e)}
        )
```

#### 步骤 3: 配置文件

创建 `config.yaml`：

```yaml
# 进化参数
max_iterations: 100
population_size: 50

# LLM 配置
llm:
  models:
    - name: "gpt-4o-mini"
      weight: 1.0
  temperature: 0.7
  max_tokens: 2048

# 数据库设置
database:
  num_islands: 3
  
# 输出设置
output:
  save_best_program: true
  checkpoint_interval: 10
```

#### 步骤 4: 运行进化

```bash
python openevolve-run.py initial_program.py evaluator.py --config config.yaml --iterations 100
```

### 教程 2: 多目标优化

有时您需要同时优化多个目标。以下是一个例子：

```python
# evaluator.py
def evaluate(program_path: str) -> EvaluationResult:
    # ... 程序执行代码 ...
    
    return EvaluationResult(
        metrics={
            "accuracy": accuracy_score,     # 准确性
            "speed": -execution_time,       # 速度（负值，因为要最小化时间）
            "memory": -memory_usage,        # 内存使用（负值）
            "complexity": -code_complexity  # 代码复杂度（负值）
        }
    )
```

配置多目标优化：

```yaml
database:
  map_elites:
    enabled: true
    dimensions:
      - feature: "accuracy"
        bins: 10
      - feature: "speed"
        bins: 10
```

### 教程 3: 使用人工制品 (Artifacts) 系统

人工制品系统允许评估器向 LLM 提供额外的反馈信息：

```python
def evaluate(program_path: str) -> EvaluationResult:
    try:
        # 尝试编译/运行程序
        result = run_program(program_path)
        
        return EvaluationResult(
            metrics={"score": result.score},
            artifacts={
                "compilation_log": result.compilation_output,
                "execution_time": result.runtime,
                "profiling_data": result.profile,
                "test_results": result.test_output
            }
        )
    except SyntaxError as e:
        return EvaluationResult(
            metrics={"score": 0.0, "compile_error": 1.0},
            artifacts={
                "stderr": str(e),
                "suggestion": "修复语法错误：检查括号和缩进"
            }
        )
```

## 配置指南

### 基础配置

#### LLM 配置

```yaml
llm:
  # API 设置
  api_base: "https://api.openai.com/v1"
  api_key: null  # 使用环境变量
  
  # 模型配置
  models:
    - name: "gpt-4o"
      weight: 0.7
    - name: "gpt-4o-mini"
      weight: 0.3
  
  # 生成参数
  temperature: 0.7
  max_tokens: 4096
  top_p: 1.0
```

#### 数据库配置

```yaml
database:
  # 种群大小
  population_size: 100
  
  # 岛屿模型
  num_islands: 5
  migration_interval: 10
  migration_rate: 0.1
  
  # MAP-Elites
  map_elites:
    enabled: true
    dimensions:
      - feature: "accuracy"
        bins: 10
        range: [0.0, 1.0]
```

#### 进化参数

```yaml
evolution:
  # 选择策略
  selection_strategy: "tournament"
  tournament_size: 3
  
  # 变异概率
  mutation_rate: 0.8
  
  # 精英保留
  elitism_rate: 0.1
```

### 高级配置

#### 分布式评估

```yaml
evaluation:
  # 并行评估数量
  max_concurrent_evaluations: 10
  
  # 超时设置
  evaluation_timeout: 300  # 秒
  
  # 分布式设置
  distributed:
    enabled: true
    worker_nodes:
      - "worker1.example.com:8080"
      - "worker2.example.com:8080"
```

#### 提示工程

```yaml
prompts:
  # 系统提示
  system_prompt: |
    你是一个专家程序员，专门优化代码性能。
    
  # 变异提示模板
  mutation_templates:
    - "改进这个程序的性能: {code}"
    - "修复这个程序中的bug: {code}"
    - "重构这个程序使其更简洁: {code}"
  
  # 包含历史信息
  include_history: true
  max_history_items: 5
```

## 高级功能

### 1. 检查点和恢复

OpenEvolve 自动保存检查点，允许您中断和恢复进化过程：

```bash
# 运行进化并保存检查点
python openevolve-run.py program.py evaluator.py --iterations 1000

# 从检查点恢复
python openevolve-run.py program.py evaluator.py \
  --checkpoint output/checkpoints/checkpoint_100 \
  --iterations 200
```

检查点包含：
- 所有进化的程序和指标
- 进化状态和参数
- 最佳程序副本

### 2. 进化可视化

使用内置的可视化工具监控进化过程：

```bash
# 安装可视化依赖
pip install -r scripts/requirements.txt

# 启动可视化服务器
python scripts/visualizer.py

# 或指定特定检查点
python scripts/visualizer.py --path output/checkpoints/checkpoint_100/
```

可视化功能：
- 进化树网络图
- 性能指标趋势
- 程序代码对比
- MAP-Elites 热图

### 3. 自定义进化算子

您可以定义自定义的进化算子：

```python
from openevolve.evolution.operators import MutationOperator

class CustomMutationOperator(MutationOperator):
    def mutate(self, program: str, context: dict) -> str:
        # 实现您的自定义变异逻辑
        modified_program = your_mutation_logic(program, context)
        return modified_program

# 在配置中使用
evolution:
  mutation_operators:
    - type: "custom"
      class: "path.to.CustomMutationOperator"
      weight: 0.5
```

### 4. 动态配置调整

在进化过程中动态调整参数：

```python
# callback.py
def evolution_callback(generation: int, best_fitness: float, config: dict):
    """进化回调函数"""
    if generation > 100 and best_fitness > 0.9:
        # 降低变异率
        config["evolution"]["mutation_rate"] *= 0.9
    
    if generation % 50 == 0:
        # 增加种群多样性
        config["database"]["migration_rate"] *= 1.1
```

## 最佳实践

### 1. 程序设计

**好的初始程序：**
```python
def solve_problem():
    """清晰的问题描述"""
    # 简单但合理的初始实现
    result = simple_approach()
    return result
```

**避免：**
- 过于复杂的初始程序
- 缺少文档字符串
- 硬编码的常量

### 2. 评估器设计

**鲁棒的评估器：**
```python
def evaluate(program_path: str) -> EvaluationResult:
    try:
        # 加载和运行程序
        result = safe_execute(program_path)
        
        # 多重检查
        if not validate_output(result):
            return failure_result("输出验证失败")
        
        # 计算多个指标
        metrics = {
            "primary_score": calculate_score(result),
            "efficiency": measure_efficiency(result),
            "correctness": verify_correctness(result)
        }
        
        return EvaluationResult(metrics=metrics)
        
    except Exception as e:
        # 详细的错误报告
        return EvaluationResult(
            metrics={"score": 0.0},
            artifacts={"error": str(e), "traceback": traceback.format_exc()}
        )
```

### 3. 配置优化

**分阶段进化：**
```yaml
# 阶段1：探索
phase1:
  iterations: 200
  temperature: 1.0
  mutation_rate: 0.9
  population_size: 50

# 阶段2：优化
phase2:
  iterations: 300
  temperature: 0.5
  mutation_rate: 0.5
  population_size: 100
```

### 4. 性能优化

**并行化设置：**
```yaml
evaluation:
  max_concurrent_evaluations: 8  # 根据 CPU 核心数调整
  batch_size: 4
  
llm:
  request_parallel: true
  max_parallel_requests: 4
```

### 5. 调试技巧

**启用详细日志：**
```yaml
logging:
  level: "DEBUG"
  file: "openevolve.log"
  console: true
```

**保存中间结果：**
```yaml
output:
  save_all_programs: true
  save_artifacts: true
  detailed_metrics: true
```

## 常见问题

### Q: 进化过程卡住，适应度不提升怎么办？

**A:** 尝试以下方法：
1. 增加温度参数：`temperature: 1.0`
2. 增加种群多样性：`num_islands: 8`
3. 调整提示模板，提供更多上下文
4. 检查评估器是否过于严格

### Q: 内存使用过高怎么办？

**A:**
1. 减少种群大小：`population_size: 50`
2. 限制历史记录：`max_history_items: 3`
3. 启用定期清理：`cleanup_interval: 20`

### Q: LLM 生成的代码质量差怎么办？

**A:**
1. 改进提示模板，提供更好的示例
2. 使用更强的模型：`gpt-4o`
3. 增加上下文信息，包含评估反馈
4. 调整温度参数平衡创新性和稳定性

### Q: 如何处理编译错误？

**A:** 使用人工制品系统：
```python
return EvaluationResult(
    metrics={"score": 0.0, "compile_error": 1.0},
    artifacts={
        "stderr": compilation_error,
        "suggestion": "修复语法错误"
    }
)
```

### Q: 多目标优化如何平衡不同目标？

**A:** 使用 MAP-Elites：
```yaml
database:
  map_elites:
    enabled: true
    dimensions:
      - feature: "accuracy"
        weight: 0.6
      - feature: "speed"
        weight: 0.4
```

## 示例项目

### 1. 函数优化
- **路径**: `examples/function_minimization/`
- **描述**: 找到使数学函数最小的参数值
- **学习要点**: 基础进化算法使用

### 2. 符号回归
- **路径**: `examples/symbolic_regression/`
- **描述**: 从数据点发现数学公式
- **学习要点**: 程序合成，表达式进化

### 3. 圆形装箱
- **路径**: `examples/circle_packing/`
- **描述**: 优化圆形在容器中的排列
- **学习要点**: 几何优化，可视化

### 4. 条形码预处理
- **路径**: `examples/barcode_preprocessing/`
- **描述**: 优化图像预处理算法
- **学习要点**: 图像处理，性能优化

### 5. 在线编程竞赛
- **路径**: `examples/online_judge_programming/`
- **描述**: 自动生成竞赛题目解决方案
- **学习要点**: 算法设计，复杂约束

### 6. 语言模型评估
- **路径**: `examples/lm_eval/`
- **描述**: 优化语言模型的推理过程
- **学习要点**: AI模型优化，高级提示工程

## 下一步

1. **阅读 API 文档**: 了解详细的 API 参考
2. **查看高级示例**: 研究复杂的使用案例
3. **参与社区**: 在 GitHub 上提问和分享经验
4. **贡献代码**: 帮助改进 OpenEvolve

## 相关链接

- [API 配置指南](api_configuration.md)
- [开发者指南](../CONTRIBUTING.md)
- [GitHub 仓库](https://github.com/codelion/openevolve)
- [论文原文](https://arxiv.org/abs/xxx) <!-- 更新实际链接 -->

---

*本指南会持续更新。如果您发现任何问题或有改进建议，请在 GitHub 上提出 issue。* 