# Symbolic Regression C++ Example

这个示例演示了如何使用 OpenEvolve 来进化 C++ 代码，以逼近数学函数。

## 目标

本示例的目标是进化一个 C++ 函数来逼近目标函数：`y = x² + x`

## 文件说明

- `config.yaml` - OpenEvolve 配置文件，设置了 C++ 编译参数
- `initial_program.cpp` - 初始的 C++ 程序（简单的线性函数 f(x) = x）
- `evaluator.py` - 自定义评估器，编译并运行 C++ 程序测试准确性
- `README.md` - 本说明文档

## 运行示例

1. 确保已安装 C++ 编译器（gcc, g++, clang, 或 clang++）
2. 设置环境变量中的 OpenAI API Key（或其他 LLM 提供商的密钥）
3. 运行示例：

```bash
cd examples/symbolic_regression_c
python ../../openevolve-run.py --config config.yaml --initial_program initial_program.cpp --evaluation evaluator.py
```

## 工作原理

1. **初始程序**：从简单的线性函数 `f(x) = x` 开始
2. **编译执行**：每个候选程序被编译成可执行文件并运行
3. **评估**：
   - 程序输出被解析，提取函数值
   - 与目标函数 `y = x² + x` 比较计算准确性
   - 同时考虑代码长度和复杂度
4. **进化**：LLM 基于评估结果生成改进的代码变体

## 评估指标

- `accuracy`: 与目标函数的匹配程度（0-1，越高越好）
- `error`: 误差（0-1，越低越好）
- `code_length`: 代码长度（标准化到 0-1）
- `complexity`: 代码复杂度（基于控制结构和数学运算）
- `runtime_success`: 程序是否成功运行

## 安全约束

本示例的 C++ 代码受到以下安全约束：
- 禁止文件 I/O 操作
- 禁止系统调用
- 禁止网络访问
- 限制运行时间（5秒超时）
- 内存使用限制

## 扩展建议

1. **更复杂的目标函数**：尝试更复杂的数学函数，如三角函数、指数函数等
2. **多变量函数**：扩展到多变量函数逼近
3. **性能优化**：在准确性基础上增加执行速度评估
4. **代码质量**：增加代码风格和可读性评估

## 注意事项

- 确保系统安装了 C++ 编译器
- 某些复杂的数学函数可能需要链接数学库（-lm）
- 编译错误会导致评估失败，返回低分
- 建议在开始时使用较低的 `max_iterations` 进行测试 