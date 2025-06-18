# OpenEvolve 进化监控脚本使用说明

本目录提供了两个监控脚本，用于实时可视化OpenEvolve的进化过程。

## 📊 监控脚本选择

### 1. evolution_monitor.py - 完整图形界面版本
**推荐用于详细分析**

#### 功能特性：
- 🎨 丰富的图形界面（6个子图表）
- 📈 实时得分趋势图表
- 📊 条形码类型成功率对比
- 🏝️ 岛屿进化状态可视化
- ⏱️ 执行时间分析
- 📱 实时统计面板
- 💾 数据导出功能（CSV格式）

#### 依赖包：
```bash
pip install matplotlib numpy pandas
```

#### 使用方法：
```bash
# 基本使用
python evolution_monitor.py

# 自定义参数
python evolution_monitor.py --output-dir openevolve_output --interval 2.0 --max-points 100 --export

# 参数说明
--output-dir, -o    OpenEvolve输出目录 (默认: openevolve_output)
--interval, -i      图表更新间隔秒数 (默认: 2.0)
--max-points, -m    图表显示的最大数据点 (默认: 100)
--export, -e        程序结束时自动导出数据
```

### 2. simple_monitor.py - 终端版本
**推荐用于简单监控**

#### 功能特性：
- 🖥️ 纯终端界面，无依赖包要求
- 📊 实时文本图表和进度条
- 🏆 当前最佳程序跟踪
- 📈 近期趋势迷你图表
- 🔄 最近活动日志
- 📄 简要总结导出

#### 依赖包：
无（仅使用Python标准库）

#### 使用方法：
```bash
# 基本使用
python simple_monitor.py

# 自定义参数
python simple_monitor.py --output-dir openevolve_output --interval 3.0 --export

# 参数说明
--output-dir, -o    OpenEvolve输出目录 (默认: openevolve_output)
--interval, -i      更新间隔秒数 (默认: 3.0)
--export, -e        退出时导出总结
```

## 🚀 快速开始

### 场景1：首次使用（推荐图形版本）
```bash
# 1. 安装依赖
pip install matplotlib numpy pandas

# 2. 启动监控
python evolution_monitor.py

# 3. 在另一个终端启动OpenEvolve
python ../../openevolve-run.py initial_program.py evaluator.py --config config.yaml
```

### 场景2：服务器环境（推荐终端版本）
```bash
# 1. 启动监控（无需安装额外包）
python simple_monitor.py

# 2. 在另一个终端启动OpenEvolve
python ../../openevolve-run.py initial_program.py evaluator.py --config config.yaml
```

### 场景3：性能分析（使用图形版本）
```bash
# 启动监控并自动导出数据
python evolution_monitor.py --interval 1.0 --max-points 200 --export
```

## 📈 监控界面说明

### 图形界面版本（evolution_monitor.py）

1. **进化得分趋势图** (左上)
   - 蓝线：当前迭代得分
   - 红虚线：历史最佳得分
   - 黄色标注：最佳得分点

2. **成功率变化图** (中左)
   - 绿色曲线：每次迭代的成功率变化

3. **执行时间图** (中右)
   - 橙色曲线：每次迭代的执行时间

4. **条形码类型成功率** (左下)
   - 红线：CODE128识别率
   - 蓝线：CODE39识别率
   - 绿线：QRCODE识别率

5. **岛屿进化状态** (右下)
   - 彩色柱状图：各岛屿最佳得分
   - 标签：每个岛屿的程序数量

6. **实时统计面板** (右侧)
   - 当前最佳程序信息
   - 运行时间统计
   - 峰值记录
   - 岛屿状态详情
   - 最新识别率

### 终端界面版本（simple_monitor.py）

```
🚀 OpenEvolve 条形码预处理进化监控器
=====================================

📊 运行状态
├─ 运行时间: 00:15:32
├─ 监控目录: openevolve_output
├─ 日志文件: openevolve_20250618_130808.log
└─ 更新间隔: 3.0秒

🏆 当前最佳程序
├─ 得分: 96.67
├─ 迭代: #10
├─ 成功率: 71.1%
└─ 发现于: 245秒前

📈 性能统计
├─ 总评估: 25
├─ 峰值得分: 96.67
├─ 峰值迭代: #10
└─ 当前迭代: #23

📱 最新识别率
├─ CODE128: [████████░░░░░░░░░░░░] 53.3%
├─ CODE39: [████████████████░░░░] 80.0%
└─ QRCODE: [████████████████░░░░] 80.0%

🏝️  岛屿状态
├─ 岛屿0: 8个程序 [██████████████░] 85.2
├─ 岛屿1: 6个程序 [████████████░░░] 78.9
└─ 岛屿2: 5个程序 [██████████░░░░░] 67.3

📊 近期得分趋势 (最近25次迭代)
┌─ 最高: 96.7 ─ ████████████████████████████████████████████████████████
│             ██░░░░░░░░░░░░██████████████████████████████████░░░░░░░░░░░░
│             ██░░░░░░░░░░░░██████████████████████████████████░░░░░░░░░░░░
│             ██░░░░░░░░░░░░██████████████████████████████████░░░░░░░░░░░░
│             ██░░░░░░░░░░░░██████████████████████████████████░░░░░░░░░░░░
└─ 最低: 13.3 ─ ████████████████████████████████████████████████████████

🔄 最近活动 (最近5次迭代)
├─ 📊 迭代#23: 45.67分 (42.2%) - 15秒前
├─ 📊 迭代#22: 38.34分 (35.6%) - 55秒前
├─ 🔥 迭代#21: 89.12分 (66.7%) - 95秒前
├─ 📊 迭代#20: 23.45分 (26.7%) - 135秒前
└─ 📊 迭代#19: 34.56分 (31.1%) - 175秒前

💡 提示: 按 Ctrl+C 退出监控
⏰ 下次更新: 3.0秒后
```

## 💾 数据导出

### 图形版本导出（CSV格式）
包含完整的迭代数据：
- iteration: 迭代次数
- score: 得分
- success_rate: 成功率
- execution_time: 执行时间
- throughput: 吞吐量
- best_score: 历史最佳得分
- CODE128_success_rate: CODE128识别率
- CODE39_success_rate: CODE39识别率
- QRCODE_success_rate: QRCODE识别率

### 终端版本导出（TXT格式）
包含监控总结：
- 监控时间统计
- 最佳程序信息
- 最新识别率
- 完整迭代历史

## 🔧 故障排除

### 常见问题

1. **找不到日志文件**
   ```
   ⚠️  未找到日志文件，等待OpenEvolve启动...
   ```
   - 确保OpenEvolve已经启动
   - 检查输出目录路径是否正确

3. **图形界面无法显示**
   ```
   ❌ 缺少依赖包: No module named 'matplotlib'
   ```
   - 安装依赖：`pip install matplotlib numpy pandas`
   - 或使用终端版本：`python simple_monitor.py`

4. **监控数据不更新**
   - 检查日志文件是否在增长
   - 确认OpenEvolve正在运行
   - 尝试重启监控脚本

5. **内存使用过高**
   - 降低max_points参数：`--max-points 50`
   - 增加更新间隔：`--interval 5.0`

### 性能建议

1. **快速监控**：使用simple_monitor.py，更新间隔3-5秒
2. **详细分析**：使用evolution_monitor.py，更新间隔1-2秒
3. **长期运行**：限制数据点数量，定期导出数据
4. **远程监控**：使用简单版本，通过SSH转发X11显示

## 📞 支持

如果遇到问题，请检查：
1. Python版本（推荐3.7+）
2. 依赖包版本
3. OpenEvolve输出目录权限
4. 日志文件是否正常生成

监控脚本会实时解析OpenEvolve的日志文件，提供直观的进化过程可视化，帮助你更好地理解和调优算法进化效果。 