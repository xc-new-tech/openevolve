#!/usr/bin/env python3
"""
Evaluator for symbolic regression C++ example
Compiles and runs C++ programs to test function approximation accuracy
"""

import subprocess
import tempfile
import os
import math
import signal
from typing import Dict, List, Tuple


def evaluate(executable_path: str, source_path: str) -> Dict[str, float]:
    """
    Evaluate a compiled C++ program for symbolic regression
    
    Args:
        executable_path: Path to compiled executable
        source_path: Path to source code (for analysis)
        
    Returns:
        Dictionary with evaluation metrics
    """
    try:
        # Run the executable and capture output
        result = subprocess.run(
            [executable_path],
            capture_output=True,
            text=True,
            timeout=5.0  # 5 second timeout
        )
        
        if result.returncode != 0:
            return {
                "error": 1.0,
                "accuracy": 0.0,
                "compilation": 0.0,
                "stderr": len(result.stderr)
            }
        
        # Parse output to extract function values
        function_values = parse_output(result.stdout)
        
        if not function_values:
            return {
                "error": 1.0,
                "accuracy": 0.0,
                "parsing_error": 1.0
            }
        
        # Calculate accuracy against target function y = x^2 + x
        accuracy = calculate_accuracy(function_values)
        
        # Additional metrics
        code_length = get_code_length(source_path)
        complexity = estimate_complexity(source_path)
        
        return {
            "accuracy": accuracy,
            "error": 1.0 - accuracy,  # Lower is better
            "code_length": code_length,
            "complexity": complexity,
            "runtime_success": 1.0
        }
        
    except subprocess.TimeoutExpired:
        return {
            "error": 1.0,
            "accuracy": 0.0,
            "timeout": 1.0
        }
    except Exception as e:
        return {
            "error": 1.0,
            "accuracy": 0.0,
            "exception": 1.0
        }


def parse_output(output: str) -> List[Tuple[float, float]]:
    """
    Parse program output to extract (x, f(x)) pairs
    Expected format: "f(x) = result"
    """
    function_values = []
    
    for line in output.strip().split('\n'):
        if '=' in line and 'f(' in line:
            try:
                # Extract x and result from "f(x) = result"
                parts = line.split('=')
                if len(parts) == 2:
                    # Extract x from "f(x)"
                    left_part = parts[0].strip()
                    if 'f(' in left_part and ')' in left_part:
                        x_str = left_part[left_part.find('(')+1:left_part.find(')')]
                        x = float(x_str)
                        
                        # Extract result
                        result = float(parts[1].strip())
                        
                        function_values.append((x, result))
            except (ValueError, IndexError):
                continue
    
    return function_values


def calculate_accuracy(function_values: List[Tuple[float, float]]) -> float:
    """
    Calculate accuracy against target function y = x^2 + x
    """
    if not function_values:
        return 0.0
    
    total_error = 0.0
    max_expected = 0.0
    
    for x, actual in function_values:
        expected = x * x + x  # Target function: y = x^2 + x
        error = abs(actual - expected)
        total_error += error
        max_expected = max(max_expected, abs(expected))
    
    # Normalize error
    if max_expected == 0:
        return 1.0 if total_error == 0 else 0.0
    
    # Calculate relative error and convert to accuracy
    mean_error = total_error / len(function_values)
    relative_error = mean_error / max(max_expected, 1.0)
    
    # Convert to accuracy score (0 to 1, higher is better)
    accuracy = max(0.0, 1.0 - relative_error)
    return accuracy


def get_code_length(source_path: str) -> float:
    """Get normalized code length metric"""
    try:
        with open(source_path, 'r') as f:
            content = f.read()
        
        # Count meaningful lines (excluding comments and empty lines)
        lines = [line.strip() for line in content.split('\n')]
        meaningful_lines = [line for line in lines if line and not line.startswith('//')]
        
        # Normalize to 0-1 range (assuming max reasonable length is 100 lines)
        normalized_length = min(1.0, len(meaningful_lines) / 100.0)
        return normalized_length
        
    except Exception:
        return 0.5  # Default value


def estimate_complexity(source_path: str) -> float:
    """Estimate code complexity based on control structures"""
    try:
        with open(source_path, 'r') as f:
            content = f.read()
        
        # Count complexity indicators
        complexity_keywords = ['if', 'else', 'for', 'while', 'switch', 'case']
        complexity_count = 0
        
        for keyword in complexity_keywords:
            complexity_count += content.count(keyword)
        
        # Count mathematical operations
        math_ops = ['+', '-', '*', '/', 'pow', 'sqrt', 'exp', 'log']
        math_count = sum(content.count(op) for op in math_ops)
        
        # Normalize complexity score
        total_complexity = complexity_count + math_count * 0.5
        normalized_complexity = min(1.0, total_complexity / 20.0)
        
        return normalized_complexity
        
    except Exception:
        return 0.5  # Default value


if __name__ == "__main__":
    # Test the evaluator with a sample program
    import sys
    if len(sys.argv) != 3:
        print("Usage: python evaluator.py <executable_path> <source_path>")
        sys.exit(1)
    
    executable_path = sys.argv[1]
    source_path = sys.argv[2]
    
    metrics = evaluate(executable_path, source_path)
    print("Evaluation metrics:")
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}") 