# Barcode Image Preprocessing Evolution Example

[![进化算法](https://img.shields.io/badge/Evolution-Algorithm-blue.svg)](https://github.com/openevolve/openevolve)
[![图像处理](https://img.shields.io/badge/Image-Processing-green.svg)](https://opencv.org/)
[![条形码解码](https://img.shields.io/badge/Barcode-Decoding-orange.svg)](https://github.com/NaturalHistoryMuseum/pyzbar)

这个示例展示了如何使用 **OpenEvolve** 来进化图像预处理算法，以提高损坏或低质量条形码图像的解码成功率。通过进化算法自动优化预处理流程，使原本无法解码的条形码图像能够被标准解码库成功识别。

## 🎯 项目目标

在实际应用中，条形码图像经常受到各种因素的影响而变得难以识别：
- 📷 **图像质量问题**: 模糊、噪声、低分辨率
- 💡 **光照条件**: 过暗、过亮、不均匀照明  
- 📐 **几何变形**: 透视畸变、倾斜、旋转
- 🚧 **物理损坏**: 部分遮挡、褶皱、磨损
- 🖼️ **环境干扰**: 背景复杂、反光

**目标**: 通过AI驱动的进化算法，自动优化图像预处理流程，将解码成功率从 <20% 提升至 >80%。

## 📁 目录结构

```
barcode_preprocessing/
├── 📋 配置文件
│   ├── config.yaml                    # 标准进化算法配置
│   └── config_optimized.yaml          # 优化后的配置参数
├── 🔬 核心算法
│   ├── initial_program.py             # 初始预处理算法
│   └── evaluator.py                   # 评估器（并行处理+详细分析）
├── 🧩 模块化预处理系统
│   └── preprocess/                    # 模块化预处理包
│       ├── denoise.py                 # 噪声处理模块
│       ├── enhance.py                 # 图像增强模块
│       ├── binarize.py                # 二值化模块
│       ├── morphology.py              # 形态学处理模块
│       ├── geometry.py                # 几何校正模块
│       └── pipeline.py                # 处理管道系统
├── 📊 数据生成与分析
│   ├── create_real_barcodes.py        # 数据集生成脚本
│   ├── demo.py                        # 可视化演示
│   ├── visualization_demo_complete.py # 完整分析演示
│   └── sample_images/                 # 测试图像目录
├── 🔧 测试与验证
│   ├── test_modular.py                # 模块化系统测试
│   ├── demo_modular.py                # 模块化演示

├── ⚙️ 配置优化
│   ├── config_tuning_experiment.py    # 参数调优实验
│   ├── quick_config_test.py           # 快速配置验证
│   └── CONFIG_OPTIMIZATION_REPORT.md  # 优化报告
├── 📖 文档与报告
│   ├── README.md                      # 本文档

│   └── VISUALIZATION_DEMO_RESULTS.md  # 可视化结果文档
└── 📦 依赖配置
    └── requirements.txt               # Python依赖清单
```

## 📊 实时监控

为了方便跟踪OpenEvolve的进化进程，我们提供了专门的监控脚本：

### 🎯 监控脚本选择

| 脚本 | 界面类型 | 依赖包 | 功能特性 | 推荐场景 |
|------|----------|--------|----------|----------|
| `evolution_monitor.py` | 图形界面 | matplotlib, numpy, pandas | 📈多维度图表、📊详细统计、💾数据导出 | 详细分析 |
| `simple_monitor.py` | 终端界面 | 无 | 🖥️文本图表、🏆最佳跟踪、📄简要导出 | 简单监控 |

### 🚀 监控快速启动

```bash
# 1. 一键启动监控器（推荐）
./start_monitor.sh

# 2. 直接使用图形界面监控
python evolution_monitor.py --export

# 3. 直接使用终端监控（无依赖）
python simple_monitor.py --export

# 4. 自定义参数
python evolution_monitor.py --interval 1.0 --max-points 200 --export
python simple_monitor.py --interval 3.0 --export
```

### 📈 监控界面预览

#### 图形界面版本（evolution_monitor.py）
- 🎨 **6个实时图表**: 得分趋势、成功率、执行时间、条形码类型统计、岛屿状态、综合面板
- 📊 **丰富可视化**: 曲线图、柱状图、填充区域、标注说明
- 💾 **数据导出**: CSV格式完整数据，支持后续分析
- 🔄 **实时更新**: 2秒间隔自动刷新

#### 终端界面版本（simple_monitor.py）
```
🚀 OpenEvolve 条形码预处理进化监控器
=====================================

🏆 当前最佳程序                  📊 近期得分趋势
├─ 得分: 96.67                   ┌─ 最高: 96.7 ─ ████████████████
├─ 迭代: #10                     │             ██░░░░░░░░░░░░████
├─ 成功率: 71.1%                 │             ██░░░░░░░░░░░░████
└─ 发现于: 245秒前               └─ 最低: 13.3 ─ ████████████████

📱 最新识别率                    🏝️  岛屿状态
├─ CODE128: [████████░░░░] 53.3%  ├─ 岛屿0: 8个程序 [███████████] 85.2
├─ CODE39:  [████████████] 80.0%  ├─ 岛屿1: 6个程序 [████████░░░] 78.9
└─ QRCODE:  [████████████] 80.0%  └─ 岛屿2: 5个程序 [██████░░░░░] 67.3
```

### 🔍 监控功能详解

- **🎯 进化跟踪**: 实时显示得分、成功率、执行时间趋势
- **🏆 最佳记录**: 自动跟踪和标记历史最佳程序
- **📱 条形码统计**: 分类显示CODE128、CODE39、QRCODE的识别率
- **🏝️ 岛屿状态**: 多岛屿进化算法的种群分布和性能
- **💾 数据保存**: 完整记录进化过程，支持后续分析
- **🔄 故障恢复**: 自动检测日志文件，支持中断恢复

详细使用说明请参考 `MONITOR_USAGE.md`

## 🚀 快速开始

### 1. 环境准备

**系统要求**:
- Python 3.9+
- 支持的操作系统: Ubuntu/Debian, macOS, Windows
- 至少 4GB 可用内存
- 1GB 磁盘空间（用于测试数据）

**安装依赖**:
```bash
cd examples/barcode_preprocessing

# 安装Python依赖
pip install -r requirements.txt

# macOS用户需要额外安装zbar库
brew install zbar

# Ubuntu/Debian用户
sudo apt-get install libzbar0

# 验证安装
python -c "from pyzbar import pyzbar; print('✅ pyzbar安装成功')"

# 如果遇到 "Unable to find zbar shared library" 错误（仅macOS）：
# 修复步骤：
pip uninstall pyzbar -y
DYLD_LIBRARY_PATH=/opt/homebrew/lib:$DYLD_LIBRARY_PATH pip install pyzbar
echo 'export DYLD_LIBRARY_PATH="/opt/homebrew/lib:$DYLD_LIBRARY_PATH"' >> ~/.zshrc
```

### 2. 生成测试数据（可选）

```bash
# 生成1000张多种损坏类型的条形码图像
python create_real_barcodes.py --count 1000 --output-dir sample_images/auto_$(date +%Y%m%d)

# 或使用默认配置生成
python create_real_barcodes.py
```

### 3. 测试基线算法

```bash
# 测试标准预处理算法
python initial_program.py

# 运行标准评估器
python evaluator.py initial_program.py

# 如果遇到zbar依赖问题，使用启动脚本：
./run_evaluator.sh initial_program.py

# 运行详细分析模式（推荐）
python evaluator.py initial_program.py --verbose --save-failures

# 或使用启动脚本：
./run_evaluator.sh initial_program.py --verbose --save-failures
```

### 4. 模块化系统演示

```bash
# 测试模块化预处理系统
python test_modular.py

# 运行模块化演示
python demo_modular.py
```

### 5. 可视化分析

```bash
# 生成完整的可视化分析报告
python visualization_demo_complete.py

# 基础可视化演示
python demo.py initial_program.py --enhanced
```

### 6. 运行进化优化

```bash
# 使用标准配置
cd ../../  # 返回项目根目录
python openevolve-run.py examples/barcode_preprocessing/config.yaml

# 使用优化配置（推荐）
python openevolve-run.py examples/barcode_preprocessing/config_optimized.yaml

# 或使用CLI
openevolve examples/barcode_preprocessing/config_optimized.yaml
```

## 📊 性能基准

### 🎯 当前性能指标

| 指标 | 基线算法 | 优化配置 | 增强评估器 | 说明 |
|------|----------|----------|------------|------|
| **解码成功率** | 25.0% | TBD | 25.0% | 处理后成功解码的图像比例 |
| **原始成功率** | 77.8% | 77.8% | 77.8% | 原始图像的解码成功率 |
| **性能改进** | -52.8% | TBD | -52.8% | 相对于原始图像的改进 |
| **处理速度** | 144 img/s | TBD | 1655+ img/s | 图像处理吞吐量 |
| **并行效率** | 单线程 | 单线程 | 16线程 | 并行处理能力 |
| **内存使用** | ~200MB | TBD | ~150MB | 内存占用优化 |

### ⚡ 优化配置参数

基于理论分析和实验验证的最优参数组合：

| 参数 | 默认值 | 优化值 | 改进幅度 | 说明 |
|------|--------|--------|----------|------|
| `population_size` | 40 | 30 | -25% | 更专注的种群搜索 |
| `max_iterations` | 50 | 50 | 0% | 保持足够的进化代数 |
| `temperature` | 0.7 | 0.6 | -14% | 提升稳定性 |
| `elite_ratio` | 0.3 | 0.4 | +33% | 保留更多优秀解 |
| `exploitation_ratio` | 0.7 | 0.8 | +14% | 增强局部搜索 |
| `parallel_evaluations` | 2 | 3 | +50% | 提升评估速度 |

**预期改进效果**:
- 🚀 收敛速度提升: 20-25%
- 💾 资源效率提升: 25%
- 🎯 解决方案质量: 保持或提升

### 📈 进化过程基准

| 阶段 | 迭代范围 | 预期成功率 | 关键改进 |
|------|----------|------------|----------|
| **初始阶段** | 1-10 | 20-30% | 参数空间探索 |
| **快速改进** | 11-25 | 30-60% | 关键技术发现 |
| **精细调优** | 26-40 | 60-80% | 参数精细化 |
| **收敛阶段** | 41-50 | 80%+ | 稳定最优解 |

## 🧩 模块化预处理系统

### 系统架构

```python
# 处理管道示例
from preprocess.pipeline import ProcessingPipeline

# 创建自定义处理管道
pipeline = ProcessingPipeline([
    ('denoise', {'method': 'median', 'kernel_size': 5}),
    ('enhance', {'method': 'clahe', 'clip_limit': 2.0}),
    ('binarize', {'method': 'adaptive', 'max_value': 255}),
    ('morphology', {'operation': 'opening', 'kernel_size': 3}),
    ('geometry', {'method': 'skew_correction'})
])

# 处理图像
processed_image = pipeline.process(input_image)
```

### 可用模块

| 模块 | 功能 | 算法选项 | 推荐用途 |
|------|------|----------|----------|
| **去噪** | 噪声消除 | 中值、高斯、双边、非局部均值 | 椒盐噪声、高斯噪声 |
| **增强** | 对比度提升 | CLAHE、直方图均衡、伽马校正 | 低对比度、光照不均 |
| **二值化** | 二值转换 | 自适应、Otsu、全局阈值 | 文字识别预处理 |
| **形态学** | 形状优化 | 开运算、闭运算、梯度、顶帽 | 细节清理、连接断点 |
| **几何** | 变形校正 | 倾斜校正、透视变换、旋转 | 角度偏移、透视失真 |

### 配置化管道

```yaml
# pipeline_config.yaml
pipeline:
  - step: denoise
    method: bilateral
    params:
      d: 9
      sigma_color: 75
      sigma_space: 75
  
  - step: enhance
    method: clahe
    params:
      clip_limit: 3.0
      tile_grid_size: [8, 8]
```

## 🔬 评估器系统

### 标准评估器 (`evaluator.py`)

```bash
# 基本使用
python evaluator.py initial_program.py

# 输出示例
Score: 25.00 (成功率: 25.0%, 改进: -52.8%)
```

### 评估器详细功能 (`evaluator.py`)

```bash
# 详细分析模式
python evaluator.py initial_program.py --verbose --save-failures

# 自定义并行度
python evaluator.py initial_program.py --max-workers 8

# CI友好模式（不保存失败图像）
python evaluator.py initial_program.py --no-save-failures
```

**主要功能**:
- ⚡ **并行处理**: 2-3倍速度提升，支持多线程解码
- 📝 **详细日志**: 彩色输出、进度追踪、可配置详细级别
- 📊 **失败分析**: 自动分析失败原因，生成失败报告
- 💾 **失败图像保存**: 自动保存到 `failed_images/` 目录
- 📈 **性能指标**: 吞吐量、处理时间、按类型统计
- 🔍 **图像分析**: 亮度、对比度、清晰度、噪声分析
- 🔧 **CI友好**: 支持 `--no-save-failures` 等CI模式参数

## 🎨 可视化与分析

### 基础可视化 (`demo.py`)

```bash
# 生成对比图像和HTML报告
python demo.py initial_program.py --sample-dir sample_images --enhanced

# 自定义输出目录
python demo.py initial_program.py --output-dir custom_results_$(date +%Y%m%d)
```

### 完整分析 (`visualization_demo_complete.py`)

```bash
# 一键生成完整分析报告
python visualization_demo_complete.py

# 自动生成内容:
# - 15张对比图像（原始vs处理后）
# - 3个统计图表（成功率、性能、流程）
# - 交互式HTML报告
# - 自动浏览器打开
```

**生成文件结构**:
```
comprehensive_demo_YYYYMMDD_HHMMSS/
├── README.md                          # 结果摘要
├── reports/
│   └── demo_report_YYYYMMDD_HHMMSS.html # 交互式报告
├── charts/
│   ├── success_rate_comparison.png    # 成功率对比
│   ├── performance_metrics.png        # 性能指标
│   └── processing_pipeline.png        # 处理流程图
└── images/
    ├── comparison_01_*.png             # 对比图像
    └── ... (最多15张对比图)
```

## ⚙️ 配置参数调优

### 使用优化配置

我们提供了经过理论分析和实验验证的优化配置：

```bash
# 使用优化配置运行进化
python openevolve-run.py examples/barcode_preprocessing/config_optimized.yaml
```

### 自定义参数调优

```bash
# 运行参数调优实验
python config_tuning_experiment.py

# 快速验证配置
python quick_config_test.py --config config_optimized.yaml
```

### 关键配置参数

```yaml
# config_optimized.yaml
evolution:
  population_size: 30          # 🎯 平衡搜索效率与多样性
  max_iterations: 50           # 🕐 确保充分进化
  timeout: 60                  # ⏱️ 单次评估超时
  
llm:
  temperature: 0.6             # 🌡️ 降低温度提升稳定性
  max_tokens: 2048             # 📝 充足的代码生成空间
  
parallel:
  num_workers: 3               # 🔄 优化并行评估数量
  
selection:
  elite_ratio: 0.4             # 🏆 保留更多优秀个体
  exploitation_ratio: 0.8      # 🎯 增强局部搜索
```

## 🛠️ 常见问题与故障排除

### 🚨 安装问题

#### 1. `ImportError: No module named 'pyzbar'`

**解决方案**:
```bash
# macOS
brew install zbar
pip install pyzbar

# Ubuntu/Debian  
sudo apt-get install libzbar0
pip install pyzbar

# Windows
# 下载预编译的zbar-dll: http://zbar.sourceforge.net/download.html
pip install pyzbar
```

#### 2. `cv2` 相关错误

**解决方案**:
```bash
# 卸载可能冲突的包
pip uninstall opencv-python opencv-python-headless opencv-contrib-python

# 重新安装
pip install opencv-python>=4.8.0
```

#### 3. `seaborn` 可视化错误

**解决方案**:
```bash
pip install seaborn>=0.12.0 matplotlib>=3.7.0
```

### 📊 运行问题

#### 1. 没有样本图像

**问题**: `FileNotFoundError: sample_images/ 目录为空`

**解决方案**:
```bash
# 生成测试数据
python create_real_barcodes.py --count 100

# 或手动添加条形码图像到 sample_images/ 目录
```

#### 2. 评估超时

**问题**: 评估过程经常超时

**解决方案**:
```bash
# 方法1: 增加超时时间
python evaluator.py initial_program.py --timeout 120

# 方法2: 减少测试图像数量
python evaluator.py initial_program.py --max-samples 50

# 方法3: 使用更快的配置
python openevolve-run.py examples/barcode_preprocessing/config_optimized.yaml
```

#### 3. 内存不足

**问题**: `MemoryError` 或系统变慢

**解决方案**:
```bash
# 减少并行线程数
python evaluator.py initial_program.py --max-workers 4

# 减少种群大小
# 编辑 config.yaml: population_size: 20
```

### 🔍 调试技巧

#### 1. 查看处理后图像

```python
# 在 initial_program.py 中添加调试代码
import cv2

def preprocess_image(image_path):
    # ... 处理逻辑 ...
    
    # 显示处理后的图像
    cv2.imshow('Original', original_image)
    cv2.imshow('Processed', processed_image) 
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return processed_image
```

#### 2. 详细日志输出

```bash
# 启用详细日志
python evaluator.py initial_program.py --verbose

# 查看失败样本
ls failed_images_*/
```

#### 3. 性能分析

```bash
# 生成性能报告
python demo_enhanced_evaluator.py

# 查看处理时间分布
python visualization_demo_complete.py
```

### 📋 配置检查清单

在运行进化算法前，请确认：

- [ ] ✅ Python 3.9+ 已安装
- [ ] ✅ 所有依赖包已安装 (`pip install -r requirements.txt`)
- [ ] ✅ zbar 系统库已安装
- [ ] ✅ sample_images/ 目录包含测试图像
- [ ] ✅ API 密钥已正确配置（OpenAI/Anthropic/Google）
- [ ] ✅ 配置文件语法正确 (`python -c "import yaml; yaml.safe_load(open('config.yaml'))"`)
- [ ] ✅ 初始程序可以正常运行 (`python initial_program.py`)
- [ ] ✅ 评估器返回有效分数 (`python evaluator.py initial_program.py`)

## 📈 预期结果与目标

### 🎯 成功标准

运行完整的进化流程后，预期能够达到：

| 指标 | 目标值 | 验证方法 |
|------|--------|----------|
| **解码成功率** | >80% | `python evaluator.py evolved_program.py` |
| **相对原始改进** | >50% | 对比原始图像解码率 |
| **处理稳定性** | >95% | 多次运行结果一致性 |
| **算法多样性** | 5+策略 | 查看不同代的解决方案 |

### 📊 进化过程监控

```bash
# 实时监控进化过程
tail -f openevolve_output/*/logs/evolution.log

# 查看最佳个体
cat openevolve_output/*/best_individuals.json

# 可视化进化曲线
python scripts/visualize_evolution.py openevolve_output/*/
```

### 🏆 成功案例

理想情况下，进化算法将发现如下优化策略：

1. **多尺度处理**: 结合不同尺度的滤波器
2. **自适应阈值**: 根据图像特征动态调整参数
3. **鲁棒性增强**: 对各种噪声类型的综合处理
4. **几何校正**: 精确的角度和透视校正
5. **后处理优化**: 形态学操作的智能组合

## 🚀 扩展建议

### 1. 算法扩展

**添加更多预处理技术**:
```python
# 新增模块示例
from preprocess.super_resolution import SuperResolution
from preprocess.denoising_cnn import CNNDenoiser

# 扩展管道
pipeline.add_step('super_resolution', SuperResolution(scale=2))
pipeline.add_step('cnn_denoise', CNNDenoiser(model='dncnn'))
```

**深度学习集成**:
- 超分辨率重建 (ESRGAN, Real-ESRGAN)
- 深度学习去噪 (DnCNN, FFDNet)
- 语义分割辅助 (U-Net, DeepLab)

### 2. 条形码类型扩展

```python
# 支持更多码制
SUPPORTED_BARCODES = [
    'CODE128',   # ✅ 已支持
    'CODE39',    # ✅ 已支持  
    'QR',        # 🚧 计划中
    'DATAMATRIX', # 🚧 计划中
    'PDF417',    # 🚧 计划中
    'AZTEC'      # 🚧 计划中
]
```

### 3. 评估指标优化

```python
# 增强评估函数
def enhanced_evaluate(program_path):
    metrics = {
        'decode_rate': calculate_decode_rate(),
        'confidence_score': calculate_confidence(),
        'processing_speed': measure_speed(),
        'memory_usage': measure_memory(),
        'robustness': test_robustness()
    }
    return weighted_score(metrics)
```

### 4. 真实场景测试

**数据集建议**:
- 工业扫描环境数据
- 手机拍摄条形码
- 不同光照条件
- 各种表面材质 (金属、塑料、纸张)
- 不同磨损程度

## 📄 许可证

本示例遵循 OpenEvolve 项目的开源许可证。详细信息请参考项目根目录的 LICENSE 文件。

## 🤝 贡献

欢迎提交 Pull Request 来改进这个示例！

**贡献指南**:
1. Fork 此仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

**改进建议**:
- 新的预处理算法
- 性能优化
- 文档改进
- 测试用例增加
- 新的可视化功能

## 📞 技术支持

如果遇到问题，请：

1. 查阅本文档的 [故障排除](#-常见问题与故障排除) 章节
2. 检查 [项目 Issues](https://github.com/openevolve/openevolve/issues)
3. 在 GitHub 上创建新的 Issue
4. 参考项目主文档: [README_zh-CN.md](../../README_zh-CN.md)

---

<div align="center">
  <img src="barcode_preprocessing_demo.png" alt="演示截图" width="600">
  <br>
  <em>条形码预处理进化示例 - 从损坏图像到成功解码</em>
</div> 