#!/usr/bin/env python3
"""
监控脚本功能演示
"""

import simple_monitor

def main():
    print('🚀 监控脚本演示')
    print('='*50)
    
    # 创建监控器实例
    monitor = simple_monitor.SimpleEvolutionMonitor()
    
    # 查找日志文件
    monitor.log_file_path = monitor.find_latest_log()
    if monitor.log_file_path:
        print(f'📄 找到日志文件: {monitor.log_file_path.name}')
        
        # 读取现有数据
        monitor.read_log_updates()
        
        # 显示统计信息
        print(f'📊 已解析迭代数据: {len(monitor.evolution_data)}条')
        print(f'🏆 当前最佳得分: {monitor.current_best["score"]:.2f}')
        print(f'📈 当前最佳成功率: {monitor.current_best["success_rate"]:.1f}%')
        print(f'🎯 当前最佳迭代: #{monitor.current_best["iteration"]}')
        
        if monitor.latest_barcode_stats:
            print('\n📱 最新条形码识别率:')
            for barcode_type, rate in monitor.latest_barcode_stats.items():
                bar = monitor.create_progress_bar(rate, 100, 15)
                print(f'   {barcode_type}: {bar} {rate:.1f}%')
        
        if monitor.island_stats:
            print('\n🏝️  岛屿状态:')
            for island_id, data in monitor.island_stats.items():
                bar = monitor.create_progress_bar(data["best_score"], 100, 10)
                print(f'   岛屿{island_id}: {data["programs"]}个程序 {bar} {data["best_score"]:.1f}')
        
        # 显示最近几次迭代
        if monitor.evolution_data:
            print('\n🔄 最近迭代历史:')
            recent = list(monitor.evolution_data)[-5:]
            for data in recent:
                print(f'   迭代#{data["iteration"]}: {data["score"]:.2f}分 ({data["success_rate"]:.1f}%)')
        
        print('\n✅ 监控脚本工作正常！')
        print('💡 使用 "python simple_monitor.py" 启动实时监控')
        print('💡 使用 "./start_monitor.sh" 启动交互式选择器')
    else:
        print('⚠️  未找到日志文件')
        print('请先运行OpenEvolve生成日志数据')

if __name__ == "__main__":
    main() 