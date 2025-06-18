#!/bin/bash

# 条形码预处理评估器启动脚本
# 自动设置zbar库路径以解决依赖问题

# 设置zbar库路径
export DYLD_LIBRARY_PATH="/opt/homebrew/lib:$DYLD_LIBRARY_PATH"

# 检查参数
if [ $# -eq 0 ]; then
    echo "用法: $0 <program.py> [其他参数]"
    echo "示例: $0 initial_program.py --verbose"
    echo "示例: $0 initial_program.py --max-workers 4 --no-save-failures"
    exit 1
fi

# 运行评估器
echo "🚀 启动条形码预处理评估器..."
echo "📚 目标程序: $1"
echo "⚙️  库路径: $DYLD_LIBRARY_PATH"
echo ""

python evaluator.py "$@" 