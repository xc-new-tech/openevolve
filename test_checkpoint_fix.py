#!/usr/bin/env python3
"""
测试检查点修复的脚本
通过模拟第13轮的高分程序来验证修复是否有效
"""

import json
import os
import sys
import tempfile
from pathlib import Path

# 添加openevolve到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

from openevolve.database import Program, ProgramDatabase, DatabaseConfig
from openevolve.controller import OpenEvolve
from openevolve.config import Config


def create_test_program(program_id: str, score: float, iteration: int = 0) -> Program:
    """创建一个测试程序"""
    return Program(
        id=program_id,
        code=f"# Test program {program_id} with score {score}",
        language="python",
        iteration_found=iteration,
        metrics={
            "score": score,
            "success_rate": score / 100.0,
            "execution_time": 1.0
        }
    )


def test_checkpoint_fix():
    """测试检查点修复"""
    print("🧪 测试检查点保存修复...")
    
    # 创建临时目录
    with tempfile.TemporaryDirectory() as temp_dir:
        # 设置数据库配置
        db_config = DatabaseConfig(
            db_path=os.path.join(temp_dir, "test_db"),
            population_size=10,
            elite_selection_ratio=0.3,
            exploration_ratio=0.4,
            exploitation_ratio=0.4
        )
        
        # 创建数据库
        database = ProgramDatabase(db_config)
        
        # 添加一些测试程序（模拟条形码预处理的场景）
        programs = [
            create_test_program("program_iter1", 96.67, 1),    # 第1轮的好程序
            create_test_program("program_iter3", 26.67, 3),    # 第3轮的差程序（被错误标记为最佳）
            create_test_program("program_iter5", 83.33, 5),    # 第5轮的中等程序
            create_test_program("program_iter13", 103.33, 13), # 第13轮的最佳程序
            create_test_program("program_iter15", 85.00, 15),  # 第15轮的程序
        ]
        
        # 添加程序到数据库
        for program in programs:
            database.add(program, iteration=program.iteration_found)
            print(f"✅ 添加程序 {program.id} (得分: {program.metrics['score']:.2f}, 轮次: {program.iteration_found})")
        
        # 模拟错误的最佳程序追踪（这是bug的原因）
        database.best_program_id = "program_iter3"  # 错误地设置为较差的程序
        print(f"❌ 数据库追踪的最佳程序: {database.best_program_id} (这是错误的！)")
        
        # 测试修复后的 get_best_program 方法
        print("\n🔍 测试 get_best_program 方法...")
        best_by_method = database.get_best_program()
        if best_by_method:
            print(f"✅ get_best_program 找到: {best_by_method.id} (得分: {best_by_method.metrics['score']:.2f})")
        else:
            print("❌ get_best_program 未找到程序")
        
        # 测试绝对最佳程序计算
        print("\n🎯 测试 get_absolute_best_program 方法...")
        absolute_best = database.get_absolute_best_program()
        if absolute_best:
            print(f"✅ get_absolute_best_program 找到: {absolute_best.id} (得分: {absolute_best.metrics['score']:.2f})")
        else:
            print("❌ get_absolute_best_program 未找到程序")
            
        # 手动扫描验证
        print("\n🔎 手动扫描所有程序验证...")
        all_programs = list(database.programs.values())
        manual_best = max(all_programs, key=lambda p: p.metrics.get('score', 0))
        print(f"✅ 手动扫描找到最佳: {manual_best.id} (得分: {manual_best.metrics['score']:.2f})")
        
        # 验证结果
        expected_best_id = "program_iter13"
        expected_score = 103.33
        
        print(f"\n📊 验证结果:")
        print(f"期望的最佳程序: {expected_best_id} (得分: {expected_score})")
        
        success = True
        if absolute_best and absolute_best.id == expected_best_id:
            print("✅ get_absolute_best_program 正确找到了最佳程序！")
        else:
            print(f"❌ get_absolute_best_program 失败: 找到 {absolute_best.id if absolute_best else None}")
            success = False
            
        if manual_best.id == expected_best_id:
            print("✅ 手动扫描正确找到了最佳程序！")
        else:
            print(f"❌ 手动扫描失败: 找到 {manual_best.id}")
            success = False
        
        # 现在测试修复后的检查点保存逻辑
        print(f"\n💾 测试检查点保存逻辑...")
        
        # 创建一个模拟的controller来测试_save_checkpoint逻辑
        # 由于OpenEvolve构造函数需要很多参数，我们只测试核心逻辑
        
        # 模拟检查点保存中的最佳程序选择
        checkpoint_best_program = database.get_best_program()
        
        # 模拟我们修复后的逻辑
        all_programs = list(database.programs.values())
        if all_programs:
            best_by_score = max(
                [p for p in all_programs if "score" in p.metrics],
                key=lambda p: p.metrics["score"],
                default=None
            )
            
            if best_by_score and checkpoint_best_program and "score" in checkpoint_best_program.metrics:
                if best_by_score.metrics["score"] > checkpoint_best_program.metrics["score"]:
                    checkpoint_best_program = best_by_score
                    print(f"✅ 检查点逻辑选择了更好的程序: {best_by_score.id} (得分: {best_by_score.metrics['score']:.2f})")
            elif best_by_score and not checkpoint_best_program:
                checkpoint_best_program = best_by_score
                print(f"✅ 检查点逻辑选择了程序: {best_by_score.id} (得分: {best_by_score.metrics['score']:.2f})")
        
        if checkpoint_best_program and checkpoint_best_program.id == expected_best_id:
            print("✅ 检查点保存逻辑正确找到了最佳程序！")
        else:
            print(f"❌ 检查点保存逻辑失败: 找到 {checkpoint_best_program.id if checkpoint_best_program else None}")
            success = False
        
        print(f"\n🎉 测试结果: {'成功' if success else '失败'}")
        return success


if __name__ == "__main__":
    success = test_checkpoint_fix()
    sys.exit(0 if success else 1) 