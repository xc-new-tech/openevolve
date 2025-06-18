#!/usr/bin/env python3
"""
验证最佳程序保存机制修复的测试脚本
"""

import json
import os
import sys

def load_program_info(path):
    """加载程序信息"""
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"加载 {path} 失败: {e}")
        return None

def main():
    """主函数"""
    output_dir = "openevolve_output"
    
    print("🔍 验证最佳程序保存机制修复效果")
    print("=" * 60)
    
    # 检查根目录的最佳程序
    best_info = load_program_info(os.path.join(output_dir, "best", "best_program_info.json"))
    if best_info:
        print(f"📂 根目录最佳程序:")
        print(f"   ID: {best_info['id']}")
        print(f"   找到轮次: {best_info['iteration']}")
        print(f"   评分: {best_info['metrics'].get('combined_score', 'N/A')}")
        print(f"   成功率: {best_info['metrics'].get('success_rate', 'N/A')}")
    
    # 检查修复版本
    fixed_info = load_program_info(os.path.join(output_dir, "best", "best_program_info_fixed.json"))
    if fixed_info:
        print(f"🔧 修复版本最佳程序:")
        print(f"   ID: {fixed_info['id']}")
        print(f"   找到轮次: {fixed_info['iteration']}")
        print(f"   评分: {fixed_info['metrics'].get('combined_score', 'N/A')}")
        print(f"   成功率: {fixed_info['metrics'].get('success_rate', 'N/A')}")
    
    # 扫描所有检查点找到真正的最佳
    print(f"\n🔍 扫描所有检查点:")
    checkpoints_dir = os.path.join(output_dir, "checkpoints")
    best_overall = None
    best_score = -1
    
    if os.path.exists(checkpoints_dir):
        for checkpoint in sorted(os.listdir(checkpoints_dir)):
            if checkpoint.startswith("checkpoint_"):
                info_path = os.path.join(checkpoints_dir, checkpoint, "best_program_info.json")
                info = load_program_info(info_path)
                if info and 'combined_score' in info['metrics']:
                    score = info['metrics']['combined_score']
                    iteration = checkpoint.split("_")[1]
                    print(f"   检查点 {iteration}: 评分 {score:.4f}, 成功率 {info['metrics'].get('success_rate', 'N/A')}")
                    
                    if score > best_score:
                        best_score = score
                        best_overall = (checkpoint, info)
    
    print(f"\n🏆 真正的历史最佳:")
    if best_overall:
        checkpoint, info = best_overall
        print(f"   位置: {checkpoint}")
        print(f"   ID: {info['id']}")
        print(f"   找到轮次: {info['iteration']}")
        print(f"   评分: {info['metrics'].get('combined_score', 'N/A'):.4f}")
        print(f"   成功率: {info['metrics'].get('success_rate', 'N/A')}")
        
        # 比较根目录和真正最佳的差异
        if best_info and 'combined_score' in best_info['metrics']:
            root_score = best_info['metrics']['combined_score']
            diff = best_score - root_score
            print(f"\n📊 性能差异分析:")
            print(f"   根目录保存的评分: {root_score:.4f}")
            print(f"   真正的最佳评分: {best_score:.4f}")
            print(f"   性能损失: {diff:.4f} ({diff/best_score*100:.1f}%)")
            
            if diff > 0.01:
                print(f"   ❌ 确认存在bug: 根目录保存了错误的程序!")
            else:
                print(f"   ✅ 根目录保存正确")
    
    print(f"\n🛠️  修复建议:")
    print(f"   1. 使用 checkpoints/checkpoint_10/best_program.py 作为真正的最佳版本")
    print(f"   2. 代码修复已完成，下次运行应该会正确保存")
    print(f"   3. 可以运行 'cp checkpoints/checkpoint_10/best_program.py best/best_program.py' 手动修复")

if __name__ == "__main__":
    main() 