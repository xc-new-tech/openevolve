# 条形码图像预处理进化示例

[![进化算法](https://img.shields.io/badge/Evolution-Algorithm-blue.svg)](https://github.com/openevolve/openevolve)
[![图像处理](https://img.shields.io/badge/Image-Processing-green.svg)](https://opencv.org/)
[![条形码解码](https://img.shields.io/badge/Barcode-Decoding-orange.svg)](https://github.com/NaturalHistoryMuseum/pyzbar)

这个示例展示了如何使用 **OpenEvolve** 来进化图像预处理算法，以提高损坏或低质量条形码图像的解码成功率。通过AI驱动的进化算法自动优化预处理流程，使原本无法解码的条形码图像能够被标准解码库成功识别。

## 🎯 项目目标

在实际应用中，条形码图像经常受到各种因素的影响而变得难以识别：
- 📷 **图像质量问题**: 模糊、噪声、低分辨率
- 💡 **光照条件**: 过暗、过亮、不均匀照明  
- 📐 **几何变形**: 透视畸变、倾斜、旋转
- 🚧 **物理损坏**: 部分遮挡、褶皱、磨损
- 🖼️ **环境干扰**: 背景复杂、反光

**目标**: 通过AI驱动的进化算法，自动优化图像预处理流程，将解码成功率从 <20% 提升至 >80%。

## 📁 项目结构

```
barcode_preprocessing/
├── 📋 配置文件
│   └── config.yaml                    # OpenEvolve进化配置
├── 🔬 核心算法
│   ├── initial_program.py             # 初始预处理算法
│   └── evaluator.py                   # 高性能评估器（OpenEvolve兼容）
├── 🧩 模块化预处理系统
│   └── preprocess/                    # 模块化预处理包
│       ├── __init__.py                # 包初始化
│       ├── denoise.py                 # 噪声处理模块
│       ├── enhance.py                 # 图像增强模块
│       ├── binarize.py                # 二值化模块
│       ├── morphology.py              # 形态学处理模块
│       ├── geometry.py                # 几何校正模块
│       └── pipeline.py                # 处理管道系统
├── 📊 数据生成与演示
│   ├── create_real_barcodes.py        # 数据集生成脚本（支持Code39）
│   ├── demo.py                        # 可视化演示脚本
│   └── sample_images/                 # 测试图像目录（45张优化数据）
├── 📖 文档与状态
│   ├── README.md                      # 英文文档
│   ├── README_zh-CN.md                # 本文档（中文）
│   └── CLEAN_STATUS.md                # 项目清理状态说明
├── 🔄 进化结果（自动生成）
│   ├── openevolve_output/             # OpenEvolve进化输出
│   ├── openevolve_output_v2/          # 后续进化输出
│   ├── evolution_results/             # 进化结果目录
│   └── improved_processed/            # 处理结果示例
└── 📦 依赖配置
    └── requirements.txt               # Python依赖清单
```

## 🚀 快速开始

### 1. 环境准备

**系统要求**:
- Python 3.9 或更高版本
- 支持的操作系统: Ubuntu/Debian, macOS, Windows
- 至少 4GB 可用内存
- 1GB 磁盘空间（用于测试数据和结果）

**安装依赖**:
```bash
cd examples/barcode_preprocessing

# 安装Python依赖包
pip install -r requirements.txt

# macOS用户需要额外安装zbar库
brew install zbar

# Ubuntu/Debian用户需要安装系统依赖
sudo apt-get install libzbar0

# Windows用户请参考故障排除章节

# 验证安装是否成功
python -c "from pyzbar import pyzbar; print('✅ pyzbar安装成功')"
```

**解决zbar依赖问题（macOS）**:
如果遇到 `ImportError: Unable to find zbar shared library` 错误：

```bash
# 修复步骤
pip uninstall pyzbar -y
DYLD_LIBRARY_PATH=/opt/homebrew/lib:$DYLD_LIBRARY_PATH pip install pyzbar
echo 'export DYLD_LIBRARY_PATH="/opt/homebrew/lib:$DYLD_LIBRARY_PATH"' >> ~/.zshrc

# 重新加载shell配置
source ~/.zshrc
```

### 2. 生成测试数据（可选）

```bash
# 快速生成测试数据（推荐）
python create_real_barcodes.py --count 50 --types clean,blurred,noisy --quick-mode

# 生成包含Code39条形码的数据
python create_real_barcodes.py --count 30 --code39 --types clean,blurred

# 生成完整数据集
python create_real_barcodes.py --count 1000

# 查看生成的图像
ls sample_images/
```

### 3. 测试基线算法

```bash
# 使用标准evaluator评估基线算法
python evaluator.py initial_program.py --verbose --max-workers 4

# 快速测试（减少并行度和保存失败图像）
python evaluator.py initial_program.py --max-workers 2 --no-save-failures

# 基本测试
python evaluator.py initial_program.py
```

### 4. 生成测试数据和演示

```bash
# 生成对比演示
python demo.py initial_program.py --max-samples 10

# 查看处理效果
ls processed_images_*/
```

### 5. 运行OpenEvolve进化

```bash
# 返回项目根目录
cd ../../

# 使用标准OpenEvolve命令运行进化算法
python openevolve-run.py initial_program.py evaluator.py --config examples/barcode_preprocessing/config.yaml

# 或使用单独的配置文件
cd examples/barcode_preprocessing
python ../../openevolve-run.py initial_program.py evaluator.py --config config.yaml

# 监控进化过程
tail -f openevolve_output/*/logs/evolution.log
```

## 📊 性能基准测试

### 🎯 当前性能指标

基于45张优化测试图像的性能评估：

| 性能指标 | 基线算法 | 改进算法 | 说明 |
|----------|----------|----------|------|
| **解码成功率** | 13.3% (6/45) | 97.8% (44/45) | 处理后成功解码的图像百分比 |
| **原始成功率** | 60.0% (27/45) | 60.0% (27/45) | 原始图像直接解码的成功率 |
| **性能改进** | -77.8% | +63.0% | 相对于原始图像的改进幅度 |
| **处理速度** | 966 图/秒 | 1200+ 图/秒 | 图像处理吞吐量 |
| **并行效率** | 16线程 | 16线程 | 并行处理能力 |
| **算法评分** | 23分 | 137分 | 综合评估评分 |

### 📈 按条形码类型的性能

| 条形码类型 | 基线成功率 | 改进成功率 | 提升幅度 |
|------------|------------|------------|----------|
| **CODE128** | 40.0% | 93.3% | +53.3% |
| **CODE39** | 0.0% | 100.0% | +100.0% |
| **QRCODE** | 0.0% | 100.0% | +100.0% |

### 🔧 核心技术改进

改进算法采用的关键技术：

1. **自适应码型检测**: 根据图像特征自动识别QR码vs线性码
2. **分策略预处理**: 
   - QR码: 温和预处理，使用Otsu阈值化
   - 线性码: 强化预处理，优化自适应阈值化参数
3. **多级备用策略**: 自适应→最小预处理→灰度原图
4. **实时效果验证**: 每个步骤都验证解码成功率

## 🧩 模块化预处理系统

### 系统架构设计

```python
# 处理管道使用示例
from preprocess.pipeline import ProcessingPipeline

# 创建自定义处理管道
pipeline = ProcessingPipeline([
    ('denoise', {'method': 'median', 'kernel_size': 5}),
    ('enhance', {'method': 'clahe', 'clip_limit': 2.0}),
    ('binarize', {'method': 'adaptive', 'max_value': 255}),
    ('morphology', {'operation': 'opening', 'kernel_size': 3}),
    ('geometry', {'method': 'skew_correction'})
])

# 处理单张图像
processed_image = pipeline.process(input_image)

# 批量处理
results = pipeline.process_batch(image_list)
```

### 可用预处理模块

| 模块名称 | 主要功能 | 可选算法 | 最佳应用场景 |
|----------|----------|----------|--------------|
| **去噪模块** | 消除图像噪声 | 中值滤波、高斯滤波、双边滤波、非局部均值 | 椒盐噪声、高斯噪声处理 |
| **增强模块** | 提升图像对比度 | CLAHE、直方图均衡化、伽马校正、对比度拉伸 | 低对比度、光照不均处理 |
| **二值化模块** | 图像二值转换 | 自适应阈值、Otsu算法、全局阈值 | 条形码识别预处理 |
| **形态学模块** | 形状结构优化 | 开运算、闭运算、梯度、顶帽变换 | 细节清理、断点连接 |
| **几何模块** | 几何变形校正 | 倾斜校正、透视变换、旋转校正 | 角度偏移、透视失真校正 |

## 🔬 评估器系统

### 统一评估器 (`evaluator.py`)

高性能评估器，提供全面的功能，兼容OpenEvolve标准调用方式：

```bash
# 基本使用方法
python evaluator.py initial_program.py

# 详细分析模式（推荐）
python evaluator.py initial_program.py --verbose --save-failures

# 自定义并行线程数
python evaluator.py initial_program.py --max-workers 8

# CI友好模式（不保存失败图像）
python evaluator.py initial_program.py --no-save-failures

# 超时设置
python evaluator.py initial_program.py --timeout 60
```

**主要功能特性**:
- ⚡ **并行处理**: 多线程并行，2-3倍速度提升
- 📝 **详细日志**: 彩色控制台输出，实时进度追踪
- 📊 **失败分析**: 自动分析失败原因并分类
- 💾 **失败图像保存**: 自动保存到时间戳目录
- 📈 **性能指标**: 详细的吞吐量、处理时间、按类型统计
- 🔍 **图像质量分析**: 亮度、对比度、清晰度、噪声自动分析
- 🔧 **OpenEvolve兼容**: 完全兼容标准进化算法调用
- ⏱️ **超时控制**: 防止单个评估占用过多时间

## 🎨 可视化与分析工具

### 基础可视化工具 (`demo.py`)

生成对比图像和基本分析：

```bash
# 生成对比图像和HTML报告
./run_evaluator.sh demo.py initial_program.py --max-samples 10

# 对比多个算法
python demo.py initial_program.py --output-dir baseline_results
python demo.py improved_program.py --output-dir improved_results
```

### 调试工具 (`simple_debug.py`)

可视化预处理的每个步骤：

```bash
# 文本模式调试（避免matplotlib问题）
python simple_debug.py

# 输出每个预处理步骤的效果分析
# 包括：原始→去噪→增强→二值化→形态学处理
```

## 🛠️ 常见问题与解决方案

### 🚨 依赖问题

#### 1. zbar库依赖问题

**问题症状**: `ImportError: Unable to find zbar shared library`

**自动解决方案**:
```bash
# 使用提供的启动脚本（推荐）
./run_evaluator.sh initial_program.py

# 手动修复（macOS）
pip uninstall pyzbar -y
DYLD_LIBRARY_PATH=/opt/homebrew/lib:$DYLD_LIBRARY_PATH pip install pyzbar
echo 'export DYLD_LIBRARY_PATH="/opt/homebrew/lib:$DYLD_LIBRARY_PATH"' >> ~/.zshrc
```

**Ubuntu/Debian解决方案**:
```bash
sudo apt-get update
sudo apt-get install libzbar0 libzbar-dev
pip install pyzbar
```

#### 2. OpenCV相关错误

```bash
# 完全清理后重新安装
pip uninstall opencv-python opencv-python-headless opencv-contrib-python -y
pip install opencv-python>=4.8.0
```

### 📊 运行时问题

#### 1. 测试数据缺失

```bash
# 快速生成测试数据
python create_real_barcodes.py --count 50 --quick-mode

# 生成多种条形码类型
python create_real_barcodes.py --count 30 --code39 --types clean,blurred,noisy
```

#### 2. 性能问题

```bash
# 减少并行线程数
./run_evaluator.sh initial_program.py --max-workers 2

# 限制测试图像数量
./run_evaluator.sh initial_program.py --quick-test

# 禁用失败图像保存
./run_evaluator.sh initial_program.py --no-save-failures
```

### 🔍 调试技巧

#### 1. 预处理步骤可视化

```bash
# 使用调试工具查看每个步骤的效果
python simple_debug.py

# 输出示例：
# ✅ 原始图像: [图像信息]
# ✅ 去噪后: [处理效果]
# ✅ 增强后: [对比度改善]
# ❌ 二值化后: [可能的问题]
```

#### 2. 性能分析

```bash
# 详细性能报告
./run_evaluator.sh initial_program.py --verbose

# 失败分析
ls failed_images_*/
cat failed_images_*/analysis_report.json
```

## 📈 预期结果与成功指标

### 🎯 进化目标

OpenEvolve进化算法的预期改进目标：

| 评估指标 | 当前基线 | 目标数值 | 验证方法 |
|----------|----------|----------|----------|
| **解码成功率** | 13.3% | >80% | `./run_evaluator.sh evolved_program.py` |
| **相对原始改进** | -77.8% | >50% | 对比原始图像解码率 |
| **处理稳定性** | - | >95% | 多次运行结果一致性检查 |
| **算法评分** | 23分 | >100分 | 综合评估评分 |

### 📊 进化过程监控

```bash
# 实时监控进化过程
tail -f openevolve_output/*/logs/evolution.log

# 查看当前最佳个体
cat openevolve_output/*/best_individuals.json | python -m json.tool

# 检查中间结果
ls -la openevolve_output/*/generation_*/
```

### 🏆 典型成功案例参考

手动优化版本(`improved_program.py`)展示了可能的进化方向：

**核心改进策略**:
```python
# 1. 自适应码型检测
def detect_barcode_type(image):
    # 基于图像特征判断QR码vs线性码
    
# 2. 分策略预处理
def preprocess_qr_code(image):
    # QR码专用的温和预处理
    
def preprocess_linear_code(image):
    # 线性码专用的强化预处理
    
# 3. 多级备用策略
def multi_level_processing(image):
    # 自适应→最小预处理→原图
```

## 🚀 扩展开发建议

### 1. 支持更多条形码类型

```python
# 当前支持状态
SUPPORTED_FORMATS = {
    'CODE128': '✅ 完全支持',
    'CODE39': '✅ 完全支持（v1.1新增）',  
    'EAN13': '✅ 基础支持',
    'QRCODE': '✅ 完全支持',
    'DATAMATRIX': '🚧 开发中（占位符已添加）',
    'PDF417': '🚧 计划中',
    'AZTEC': '🚧 计划中'
}
```

### 2. 深度学习集成

```python
# 新增深度学习模块示例
from preprocess.super_resolution import ESRGAN
from preprocess.denoising_cnn import DnCNN

# 扩展处理管道
pipeline.add_step('super_resolution', ESRGAN(scale_factor=2))
pipeline.add_step('dl_denoise', DnCNN(model_path='models/dncnn.pth'))
```

### 3. 实时处理优化

```python
# GPU加速处理
import cupy as cp  # GPU NumPy

def gpu_accelerated_preprocessing(image):
    gpu_image = cp.asarray(image)
    # GPU加速的图像处理
    return cp.asnumpy(processed_image)
```

## 📄 项目状态与维护

### 🧹 最近更新 (2024-12)

1. **项目清理**: 删除了500MB+的临时文件和实验代码
2. **评估器整合**: 统一为单一高功能评估器
3. **依赖修复**: 添加了zbar依赖问题的自动化解决方案
4. **码型扩展**: 新增Code39支持，DataMatrix占位符
5. **文档同步**: 中英文档完全同步更新

### 📊 当前项目指标

- **代码质量**: 简洁高效，核心功能完整
- **测试覆盖**: 45张优化测试图像，涵盖主要场景
- **文档完整度**: 100%（中英文同步）
- **依赖稳定性**: 所有已知问题已修复
- **进化就绪**: 完全兼容OpenEvolve进化算法

### 🤝 参与贡献

我们欢迎社区贡献！参与方式：

1. **Fork 仓库**: 克隆项目到您的账户
2. **创建分支**: `git checkout -b feature/YourFeatureName`
3. **开发功能**: 实现您的改进
4. **测试验证**: 使用 `./run_evaluator.sh` 确保功能正常
5. **提交代码**: 创建Pull Request

**优先贡献领域**:
- 🔧 **新预处理算法**: 深度学习方法、传统算法优化
- 📊 **性能优化**: GPU加速、内存优化
- 🎯 **新条形码类型**: DataMatrix、PDF417、Aztec支持
- 🌍 **多平台支持**: Windows、ARM架构优化

## 🎯 OpenEvolve标准用法

### 标准启动命令

```bash
# 进入项目目录
cd examples/barcode_preprocessing

# 使用标准OpenEvolve命令启动进化
python ../../openevolve-run.py initial_program.py evaluator.py --config config.yaml
```

### 与其他OpenEvolve示例保持一致

本示例现已完全遵循OpenEvolve项目的标准模式：
- ✅ `initial_program.py` - 初始算法实现
- ✅ `evaluator.py` - 性能评估脚本（统一整合版）
- ✅ `config.yaml` - 进化算法配置（标准格式）
- ✅ 标准命令格式：`python openevolve-run.py initial_program.py evaluator.py --config config.yaml`
- ✅ 简化项目结构，专注核心功能

**重要说明**: 本项目已删除所有非标准脚本（如`run_evaluator.sh`、`openevolve_evaluator.py`等），确保与其他OpenEvolve示例的一致性。

---

<div align="center">
  <br>
  <em>🎯 条形码预处理进化示例 - 从13.3%到97.8%成功率的AI驱动优化</em>
  <br><br>
  
  [![GitHub stars](https://img.shields.io/github/stars/openevolve/openevolve?style=social)](https://github.com/openevolve/openevolve)
  [![GitHub forks](https://img.shields.io/github/forks/openevolve/openevolve?style=social)](https://github.com/openevolve/openevolve)
  [![GitHub issues](https://img.shields.io/github/issues/openevolve/openevolve)](https://github.com/openevolve/openevolve/issues)
  
  <br>
  Made with ❤️ by the OpenEvolve Community
</div> 