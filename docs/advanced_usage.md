# OpenEvolve 高级用法

本文档面向有经验的用户，介绍 OpenEvolve 的高级功能和复杂使用场景。

## 目录

1. [自定义进化算子](#自定义进化算子)
2. [分布式和并行处理](#分布式和并行处理)
3. [高级配置模式](#高级配置模式)
4. [性能优化](#性能优化)
5. [自定义评估器](#自定义评估器)
6. [集成外部工具](#集成外部工具)
7. [多阶段进化](#多阶段进化)
8. [生产环境部署](#生产环境部署)

## 自定义进化算子

### 创建自定义变异算子

```python
from openevolve.evolution.operators import MutationOperator
from openevolve.core.program import Program
from typing import Dict, Any

class SemanticMutationOperator(MutationOperator):
    """基于语义分析的变异算子"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.semantic_analyzer = self._load_analyzer()
    
    def mutate(self, program: Program, context: Dict[str, Any]) -> str:
        """执行语义感知的变异"""
        # 1. 分析程序语义结构
        ast_tree = self.semantic_analyzer.parse(program.code)
        
        # 2. 识别关键语义节点
        critical_nodes = self.semantic_analyzer.find_critical_nodes(ast_tree)
        
        # 3. 基于上下文选择变异策略
        mutation_strategy = self._select_strategy(context, critical_nodes)
        
        # 4. 执行变异
        mutated_code = mutation_strategy.apply(program.code, critical_nodes)
        
        return mutated_code
    
    def _select_strategy(self, context: Dict, nodes: list) -> 'MutationStrategy':
        """根据上下文选择最佳变异策略"""
        if context.get('performance_focus'):
            return PerformanceMutationStrategy()
        elif context.get('correctness_focus'):
            return CorrectnessMutationStrategy()
        else:
            return BalancedMutationStrategy()

# 注册自定义算子
class CustomEvolutionConfig:
    mutation_operators = [
        {
            'type': 'semantic',
            'class': 'SemanticMutationOperator',
            'weight': 0.4,
            'config': {
                'semantic_model': 'code_bert',
                'mutation_strength': 0.7
            }
        },
        {
            'type': 'random',
            'class': 'RandomMutationOperator', 
            'weight': 0.6
        }
    ]
```

### 自定义选择算子

```python
from openevolve.evolution.selection import SelectionOperator

class MultiObjectiveSelection(SelectionOperator):
    """多目标帕累托前沿选择"""
    
    def select(self, population: List[Program], size: int) -> List[Program]:
        # 1. 计算帕累托前沿
        fronts = self._compute_pareto_fronts(population)
        
        # 2. 按前沿层级选择
        selected = []
        for front in fronts:
            if len(selected) + len(front) <= size:
                selected.extend(front)
            else:
                # 使用拥挤距离选择
                remaining = size - len(selected)
                crowding_distances = self._compute_crowding_distance(front)
                sorted_front = sorted(front, key=lambda p: crowding_distances[p.id], reverse=True)
                selected.extend(sorted_front[:remaining])
                break
        
        return selected
    
    def _compute_pareto_fronts(self, population: List[Program]) -> List[List[Program]]:
        """计算帕累托前沿层级"""
        fronts = []
        remaining = population.copy()
        
        while remaining:
            current_front = []
            dominated = []
            
            for program in remaining:
                is_dominated = False
                for other in remaining:
                    if other != program and self._dominates(other, program):
                        is_dominated = True
                        break
                
                if not is_dominated:
                    current_front.append(program)
                else:
                    dominated.append(program)
            
            fronts.append(current_front)
            remaining = dominated
        
        return fronts
```

## 分布式和并行处理

### 配置分布式评估

```yaml
# distributed_config.yaml
evaluation:
  distributed:
    enabled: true
    strategy: "ray"  # ray, celery, kubernetes
    
    # Ray 配置
    ray:
      head_node: "ray://head-node:10001"
      num_workers: 8
      resources_per_worker:
        num_cpus: 2
        num_gpus: 0.5
    
    # Kubernetes 配置
    kubernetes:
      namespace: "openevolve"
      worker_image: "openevolve:latest"
      worker_replicas: 10
      resource_limits:
        cpu: "2"
        memory: "4Gi"
        gpu: "1"

llm:
  distributed:
    enabled: true
    load_balancer: "round_robin"  # round_robin, least_connections, adaptive
    rate_limiting:
      requests_per_minute: 1000
      tokens_per_minute: 100000
```

### 实现自定义分布式后端

```python
from openevolve.distributed import DistributedBackend
import asyncio
from typing import List, Callable

class CustomDistributedBackend(DistributedBackend):
    """自定义分布式后端实现"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.worker_pool = self._initialize_workers()
    
    async def evaluate_batch(self, 
                           programs: List[Program], 
                           evaluator: Callable) -> List[EvaluationResult]:
        """批量分布式评估"""
        
        # 1. 将程序分组发送到不同的工作节点
        batches = self._create_batches(programs)
        
        # 2. 并行执行评估
        tasks = []
        for batch, worker in zip(batches, self.worker_pool):
            task = self._evaluate_batch_on_worker(batch, evaluator, worker)
            tasks.append(task)
        
        # 3. 收集结果
        batch_results = await asyncio.gather(*tasks)
        
        # 4. 合并结果
        all_results = []
        for results in batch_results:
            all_results.extend(results)
        
        return all_results
    
    async def _evaluate_batch_on_worker(self, 
                                      batch: List[Program], 
                                      evaluator: Callable,
                                      worker: 'Worker') -> List[EvaluationResult]:
        """在特定工作节点上评估批次"""
        try:
            # 发送程序到工作节点
            await worker.send_programs(batch)
            
            # 等待评估完成
            results = await worker.evaluate(evaluator)
            
            return results
            
        except Exception as e:
            # 处理工作节点故障
            self.logger.error(f"Worker {worker.id} failed: {e}")
            
            # 重新分配到其他节点
            backup_worker = self._get_backup_worker()
            return await self._evaluate_batch_on_worker(batch, evaluator, backup_worker)
```

## 高级配置模式

### 动态配置管理

```python
from openevolve.config import ConfigManager
import time

class AdaptiveConfigManager(ConfigManager):
    """自适应配置管理器"""
    
    def __init__(self, base_config: Dict[str, Any]):
        super().__init__(base_config)
        self.performance_history = []
        self.adaptation_rules = self._load_adaptation_rules()
    
    def update_config(self, generation: int, population_stats: Dict) -> Dict[str, Any]:
        """根据进化状态动态更新配置"""
        
        # 记录性能指标
        self.performance_history.append({
            'generation': generation,
            'best_fitness': population_stats['best_fitness'],
            'diversity': population_stats['diversity'],
            'stagnation': population_stats['stagnation_count']
        })
        
        # 应用适应规则
        updated_config = self.config.copy()
        
        for rule in self.adaptation_rules:
            if rule.should_apply(self.performance_history):
                updated_config = rule.apply(updated_config)
                self.logger.info(f"Applied adaptation rule: {rule.name}")
        
        return updated_config

# 配置适应规则
adaptation_rules:
  - name: "increase_mutation_on_stagnation"
    condition: "stagnation_count > 20"
    action:
      mutation_rate: "*= 1.2"
      temperature: "*= 1.1"
  
  - name: "reduce_population_on_convergence"
    condition: "diversity < 0.1"
    action:
      population_size: "//= 2"
      num_islands: "max(1, num_islands - 1)"
  
  - name: "boost_exploration_on_plateau"
    condition: "fitness_improvement_rate < 0.01"
    action:
      exploration_bonus: "+= 0.1"
      novelty_threshold: "*= 0.9"
```

### 多阶段进化配置

```yaml
# multi_stage_config.yaml
stages:
  - name: "exploration"
    iterations: 200
    config:
      llm:
        temperature: 1.0
        models:
          - name: "gpt-4"
            weight: 0.8
          - name: "claude-3"
            weight: 0.2
      evolution:
        mutation_rate: 0.9
        selection_pressure: 0.3
      database:
        population_size: 100
        diversity_threshold: 0.8
  
  - name: "exploitation"
    iterations: 300
    config:
      llm:
        temperature: 0.5
        models:
          - name: "gpt-4"
            weight: 1.0
      evolution:
        mutation_rate: 0.5
        selection_pressure: 0.7
      database:
        population_size: 150
        elitism_rate: 0.2
  
  - name: "refinement"
    iterations: 100
    config:
      llm:
        temperature: 0.2
        models:
          - name: "gpt-4"
            weight: 1.0
      evolution:
        mutation_rate: 0.2
        selection_pressure: 0.9
      database:
        population_size: 50
        elitism_rate: 0.5

# 阶段转换条件
stage_transitions:
  exploration_to_exploitation:
    conditions:
      - "best_fitness > 0.7"
      - "generation > 150"
    operator: "OR"
  
  exploitation_to_refinement:
    conditions:
      - "fitness_improvement < 0.05"
      - "generation > 400"
    operator: "AND"
```

## 性能优化

### 内存优化

```python
from openevolve.optimization import MemoryOptimizer

class AdvancedMemoryOptimizer(MemoryOptimizer):
    """高级内存优化器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.memory_threshold = config.get('memory_threshold', 0.8)
        self.cleanup_strategies = self._initialize_cleanup_strategies()
    
    def optimize_memory_usage(self, database: 'ProgramDatabase') -> None:
        """优化内存使用"""
        
        # 1. 检查内存使用情况
        memory_usage = self._get_memory_usage()
        
        if memory_usage > self.memory_threshold:
            # 2. 应用清理策略
            for strategy in self.cleanup_strategies:
                strategy.apply(database)
                
                # 检查是否已达到目标
                if self._get_memory_usage() < self.memory_threshold:
                    break
    
    def _initialize_cleanup_strategies(self) -> List['CleanupStrategy']:
        """初始化清理策略"""
        return [
            LowFitnessCleanupStrategy(threshold=0.1),
            OldGenerationCleanupStrategy(max_age=50),
            DuplicateRemovalStrategy(),
            ArtifactCompressionStrategy()
        ]

# 使用内存映射存储大型数据
class MemoryMappedDatabase:
    """内存映射数据库实现"""
    
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.mmap_file = None
        self._initialize_mmap()
    
    def store_program(self, program: Program) -> None:
        """存储程序到内存映射文件"""
        serialized = self._serialize_program(program)
        offset = self._allocate_space(len(serialized))
        self.mmap_file[offset:offset+len(serialized)] = serialized
    
    def load_program(self, program_id: str) -> Program:
        """从内存映射文件加载程序"""
        offset, size = self._get_program_location(program_id)
        serialized = self.mmap_file[offset:offset+size]
        return self._deserialize_program(serialized)
```

### LLM 调用优化

```python
from openevolve.llm import LLMOptimizer
import asyncio
from typing import List

class BatchedLLMOptimizer(LLMOptimizer):
    """批量 LLM 调用优化器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.batch_size = config.get('batch_size', 8)
        self.max_concurrent = config.get('max_concurrent', 4)
        self.rate_limiter = self._create_rate_limiter(config)
    
    async def generate_mutations_batch(self, 
                                     programs: List[Program], 
                                     context: Dict[str, Any]) -> List[str]:
        """批量生成变异"""
        
        # 1. 创建批次
        batches = self._create_batches(programs, self.batch_size)
        
        # 2. 并行处理批次
        semaphore = asyncio.Semaphore(self.max_concurrent)
        
        async def process_batch(batch: List[Program]) -> List[str]:
            async with semaphore:
                await self.rate_limiter.acquire()
                
                # 构建批量提示
                batch_prompt = self._build_batch_prompt(batch, context)
                
                # 调用 LLM
                response = await self.llm_client.generate(batch_prompt)
                
                # 解析批量响应
                mutations = self._parse_batch_response(response, len(batch))
                
                return mutations
        
        # 3. 执行所有批次
        tasks = [process_batch(batch) for batch in batches]
        batch_results = await asyncio.gather(*tasks)
        
        # 4. 合并结果
        all_mutations = []
        for results in batch_results:
            all_mutations.extend(results)
        
        return all_mutations
    
    def _build_batch_prompt(self, programs: List[Program], context: Dict) -> str:
        """构建批量处理的提示"""
        prompt = "以下是需要优化的程序列表：\n\n"
        
        for i, program in enumerate(programs):
            prompt += f"程序 {i+1}:\n```python\n{program.code}\n```\n\n"
        
        prompt += "请为每个程序生成一个改进版本，"
        prompt += "按照相同的顺序返回，每个程序之间用 '---PROGRAM_SEPARATOR---' 分隔。"
        
        return prompt
```

## 自定义评估器

### 复合评估器

```python
from openevolve.evaluation import CompositeEvaluator
from typing import List, Dict

class WeightedCompositeEvaluator(CompositeEvaluator):
    """加权复合评估器"""
    
    def __init__(self, evaluators: List[Dict[str, Any]]):
        self.evaluators = []
        self.weights = []
        
        for eval_config in evaluators:
            evaluator = self._load_evaluator(eval_config['class'], eval_config['config'])
            self.evaluators.append(evaluator)
            self.weights.append(eval_config.get('weight', 1.0))
    
    async def evaluate(self, program: Program) -> EvaluationResult:
        """执行加权复合评估"""
        
        # 1. 并行执行所有评估器
        tasks = [evaluator.evaluate(program) for evaluator in self.evaluators]
        results = await asyncio.gather(*tasks)
        
        # 2. 合并指标
        combined_metrics = {}
        combined_artifacts = {}
        
        for i, (result, weight) in enumerate(zip(results, self.weights)):
            # 加权合并指标
            for metric, value in result.metrics.items():
                if metric in combined_metrics:
                    combined_metrics[metric] += value * weight
                else:
                    combined_metrics[metric] = value * weight
            
            # 合并人工制品
            for key, artifact in result.artifacts.items():
                combined_artifacts[f"evaluator_{i}_{key}"] = artifact
        
        # 3. 计算综合得分
        total_weight = sum(self.weights)
        for metric in combined_metrics:
            combined_metrics[metric] /= total_weight
        
        return EvaluationResult(
            metrics=combined_metrics,
            artifacts=combined_artifacts
        )

# 配置示例
composite_evaluator:
  evaluators:
    - class: "PerformanceEvaluator"
      weight: 0.4
      config:
        timeout: 30
        memory_limit: "1GB"
    
    - class: "CorrectnessEvaluator"
      weight: 0.5
      config:
        test_suite: "comprehensive"
        coverage_threshold: 0.9
    
    - class: "CodeQualityEvaluator"
      weight: 0.1
      config:
        metrics: ["complexity", "maintainability", "readability"]
```

### 增量评估器

```python
class IncrementalEvaluator:
    """增量评估器，只评估程序的变化部分"""
    
    def __init__(self, base_evaluator: 'Evaluator'):
        self.base_evaluator = base_evaluator
        self.cache = {}
        self.diff_analyzer = CodeDiffAnalyzer()
    
    async def evaluate(self, program: Program) -> EvaluationResult:
        """增量评估程序"""
        
        # 1. 检查缓存
        cache_key = self._compute_cache_key(program)
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # 2. 查找最相似的已评估程序
        similar_program = self._find_most_similar(program)
        
        if similar_program:
            # 3. 分析差异
            diff = self.diff_analyzer.analyze(similar_program.code, program.code)
            
            if diff.is_minor():
                # 4. 增量评估
                base_result = self.cache[self._compute_cache_key(similar_program)]
                incremental_result = await self._evaluate_diff(diff, base_result)
                
                # 5. 合并结果
                final_result = self._merge_results(base_result, incremental_result)
                
                self.cache[cache_key] = final_result
                return final_result
        
        # 6. 完整评估（如果无法增量评估）
        result = await self.base_evaluator.evaluate(program)
        self.cache[cache_key] = result
        return result
```

## 集成外部工具

### 集成静态分析工具

```python
from openevolve.tools import StaticAnalyzer
import subprocess
import json

class ComprehensiveStaticAnalyzer(StaticAnalyzer):
    """集成多个静态分析工具"""
    
    def __init__(self, config: Dict[str, Any]):
        self.tools = config.get('tools', ['pylint', 'flake8', 'mypy', 'bandit'])
        self.weights = config.get('weights', {})
    
    async def analyze(self, program_code: str) -> Dict[str, Any]:
        """执行静态分析"""
        
        # 1. 保存代码到临时文件
        temp_file = self._save_to_temp_file(program_code)
        
        try:
            # 2. 运行所有分析工具
            results = {}
            
            if 'pylint' in self.tools:
                results['pylint'] = await self._run_pylint(temp_file)
            
            if 'flake8' in self.tools:
                results['flake8'] = await self._run_flake8(temp_file)
            
            if 'mypy' in self.tools:
                results['mypy'] = await self._run_mypy(temp_file)
            
            if 'bandit' in self.tools:
                results['bandit'] = await self._run_bandit(temp_file)
            
            # 3. 合并分析结果
            combined_analysis = self._combine_results(results)
            
            return combined_analysis
            
        finally:
            # 4. 清理临时文件
            self._cleanup_temp_file(temp_file)
    
    async def _run_pylint(self, file_path: str) -> Dict[str, Any]:
        """运行 Pylint 分析"""
        cmd = f"pylint --output-format=json {file_path}"
        result = await self._run_command(cmd)
        
        if result.returncode == 0:
            return json.loads(result.stdout)
        else:
            return {"error": result.stderr, "issues": []}
    
    def _combine_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """合并多个工具的分析结果"""
        combined = {
            "total_issues": 0,
            "severity_breakdown": {"error": 0, "warning": 0, "info": 0},
            "categories": {},
            "overall_score": 0.0,
            "tool_results": results
        }
        
        for tool, tool_result in results.items():
            if "issues" in tool_result:
                combined["total_issues"] += len(tool_result["issues"])
                
                for issue in tool_result["issues"]:
                    severity = issue.get("severity", "info")
                    combined["severity_breakdown"][severity] += 1
                    
                    category = issue.get("category", "general")
                    combined["categories"][category] = combined["categories"].get(category, 0) + 1
        
        # 计算综合评分
        combined["overall_score"] = self._calculate_overall_score(combined)
        
        return combined
```

### 集成测试框架

```python
from openevolve.testing import TestFrameworkIntegration
import pytest
import unittest

class PytestIntegration(TestFrameworkIntegration):
    """Pytest 测试框架集成"""
    
    def __init__(self, config: Dict[str, Any]):
        self.test_directory = config.get('test_directory', 'tests/')
        self.test_patterns = config.get('test_patterns', ['test_*.py'])
        self.coverage_threshold = config.get('coverage_threshold', 0.8)
    
    async def run_tests(self, program: Program) -> Dict[str, Any]:
        """运行测试套件"""
        
        # 1. 准备测试环境
        test_env = await self._prepare_test_environment(program)
        
        try:
            # 2. 运行测试
            test_results = await self._execute_pytest(test_env)
            
            # 3. 生成覆盖率报告
            coverage_report = await self._generate_coverage_report(test_env)
            
            # 4. 分析结果
            analysis = self._analyze_test_results(test_results, coverage_report)
            
            return analysis
            
        finally:
            # 5. 清理测试环境
            await self._cleanup_test_environment(test_env)
    
    async def _execute_pytest(self, test_env: 'TestEnvironment') -> Dict[str, Any]:
        """执行 Pytest"""
        
        # 构建 pytest 命令
        cmd = [
            "python", "-m", "pytest",
            test_env.test_directory,
            "--json-report",
            f"--json-report-file={test_env.results_file}",
            "--cov=.",
            f"--cov-report=json:{test_env.coverage_file}",
            "--tb=short"
        ]
        
        # 执行测试
        result = await self._run_command_async(cmd, cwd=test_env.working_directory)
        
        # 解析结果
        if test_env.results_file.exists():
            with open(test_env.results_file) as f:
                return json.load(f)
        else:
            return {"error": "Test execution failed", "stdout": result.stdout, "stderr": result.stderr}
```

## 多阶段进化

### 阶段管理器

```python
from openevolve.stages import StageManager
from typing import List, Dict, Any

class AdaptiveStageManager(StageManager):
    """自适应阶段管理器"""
    
    def __init__(self, stage_configs: List[Dict[str, Any]]):
        self.stages = [Stage(config) for config in stage_configs]
        self.current_stage = 0
        self.stage_history = []
        self.adaptation_triggers = self._setup_adaptation_triggers()
    
    async def should_advance_stage(self, 
                                 generation: int, 
                                 population_stats: Dict[str, Any]) -> bool:
        """判断是否应该进入下一阶段"""
        
        current_stage = self.stages[self.current_stage]
        
        # 1. 检查硬性条件（最大迭代数）
        if generation >= current_stage.max_iterations:
            return True
        
        # 2. 检查适应性触发条件
        for trigger in self.adaptation_triggers:
            if trigger.should_trigger(current_stage, generation, population_stats):
                self.logger.info(f"Stage advancement triggered by: {trigger.name}")
                return True
        
        return False
    
    async def advance_stage(self) -> bool:
        """推进到下一阶段"""
        if self.current_stage + 1 >= len(self.stages):
            return False  # 已经是最后一个阶段
        
        # 记录当前阶段的统计信息
        current_stats = self._collect_current_stage_stats()
        self.stage_history.append(current_stats)
        
        # 推进到下一阶段
        self.current_stage += 1
        next_stage = self.stages[self.current_stage]
        
        self.logger.info(f"Advanced to stage {self.current_stage}: {next_stage.name}")
        
        # 应用阶段配置
        await self._apply_stage_config(next_stage)
        
        return True
    
    def _setup_adaptation_triggers(self) -> List['AdaptationTrigger']:
        """设置适应性触发器"""
        return [
            FitnessPlateauTrigger(patience=20, threshold=0.01),
            DiversityThresholdTrigger(min_diversity=0.1),
            ConvergenceRateTrigger(max_rate=0.95),
            PerformanceRegressionTrigger(tolerance=0.05)
        ]

class FitnessPlateauTrigger:
    """适应度平台期触发器"""
    
    def __init__(self, patience: int, threshold: float):
        self.patience = patience
        self.threshold = threshold
        self.no_improvement_count = 0
        self.last_best_fitness = None
    
    def should_trigger(self, stage: 'Stage', generation: int, stats: Dict) -> bool:
        current_best = stats['best_fitness']
        
        if self.last_best_fitness is None:
            self.last_best_fitness = current_best
            return False
        
        improvement = current_best - self.last_best_fitness
        
        if improvement < self.threshold:
            self.no_improvement_count += 1
        else:
            self.no_improvement_count = 0
            self.last_best_fitness = current_best
        
        return self.no_improvement_count >= self.patience
```

## 生产环境部署

### 容器化部署

```dockerfile
# Dockerfile.production
FROM python:3.9-slim

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    && rm -rf /var/lib/apt/lists/*

# 创建应用目录
WORKDIR /app

# 复制依赖文件
COPY requirements.txt .
COPY requirements-prod.txt .

# 安装 Python 依赖
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir -r requirements-prod.txt

# 复制应用代码
COPY . .

# 安装 OpenEvolve
RUN pip install -e .

# 创建非 root 用户
RUN useradd --create-home --shell /bin/bash openevolve
RUN chown -R openevolve:openevolve /app
USER openevolve

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "from openevolve import OpenEvolve; print('OK')"

# 启动命令
CMD ["python", "-m", "openevolve.server"]
```

### Kubernetes 部署

```yaml
# k8s-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: openevolve-controller
  labels:
    app: openevolve
    component: controller
spec:
  replicas: 1
  selector:
    matchLabels:
      app: openevolve
      component: controller
  template:
    metadata:
      labels:
        app: openevolve
        component: controller
    spec:
      containers:
      - name: controller
        image: openevolve:latest
        env:
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: openevolve-secrets
              key: openai-api-key
        - name: REDIS_URL
          value: "redis://redis-service:6379"
        - name: DATABASE_URL
          value: "postgresql://postgres:password@postgres-service:5432/openevolve"
        resources:
          requests:
            memory: "2Gi"
            cpu: "1"
          limits:
            memory: "4Gi"
            cpu: "2"
        volumeMounts:
        - name: config-volume
          mountPath: /app/config
        - name: output-volume
          mountPath: /app/output
      volumes:
      - name: config-volume
        configMap:
          name: openevolve-config
      - name: output-volume
        persistentVolumeClaim:
          claimName: openevolve-output-pvc

---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: openevolve-workers
  labels:
    app: openevolve
    component: worker
spec:
  replicas: 5
  selector:
    matchLabels:
      app: openevolve
      component: worker
  template:
    metadata:
      labels:
        app: openevolve
        component: worker
    spec:
      containers:
      - name: worker
        image: openevolve:latest
        command: ["python", "-m", "openevolve.worker"]
        env:
        - name: CONTROLLER_URL
          value: "http://openevolve-controller-service:8080"
        - name: WORKER_ID
          valueFrom:
            fieldRef:
              fieldPath: metadata.name
        resources:
          requests:
            memory: "1Gi"
            cpu: "0.5"
          limits:
            memory: "2Gi"
            cpu: "1"
```

### 监控和日志

```python
from openevolve.monitoring import MetricsCollector, AlertManager
import prometheus_client
from typing import Dict, Any

class ProductionMetricsCollector(MetricsCollector):
    """生产环境指标收集器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.prometheus_registry = prometheus_client.CollectorRegistry()
        self._setup_metrics()
        self.alert_manager = AlertManager(config.get('alerting', {}))
    
    def _setup_metrics(self):
        """设置 Prometheus 指标"""
        
        # 进化指标
        self.generation_counter = prometheus_client.Counter(
            'openevolve_generations_total',
            'Total number of evolution generations',
            registry=self.prometheus_registry
        )
        
        self.fitness_gauge = prometheus_client.Gauge(
            'openevolve_best_fitness',
            'Current best fitness score',
            registry=self.prometheus_registry
        )
        
        self.evaluation_histogram = prometheus_client.Histogram(
            'openevolve_evaluation_duration_seconds',
            'Time spent evaluating programs',
            registry=self.prometheus_registry
        )
        
        # LLM 指标
        self.llm_requests_counter = prometheus_client.Counter(
            'openevolve_llm_requests_total',
            'Total LLM API requests',
            ['model', 'status'],
            registry=self.prometheus_registry
        )
        
        self.llm_tokens_counter = prometheus_client.Counter(
            'openevolve_llm_tokens_total',
            'Total tokens used',
            ['model', 'type'],
            registry=self.prometheus_registry
        )
    
    async def record_generation(self, generation_stats: Dict[str, Any]):
        """记录一代进化的指标"""
        
        # 更新基础指标
        self.generation_counter.inc()
        self.fitness_gauge.set(generation_stats['best_fitness'])
        
        # 检查异常情况并发送告警
        if generation_stats['best_fitness'] < generation_stats.get('previous_best', 0) * 0.9:
            await self.alert_manager.send_alert({
                'type': 'fitness_regression',
                'severity': 'warning',
                'message': 'Significant fitness regression detected',
                'data': generation_stats
            })
        
        # 记录评估时间
        avg_eval_time = generation_stats.get('avg_evaluation_time', 0)
        self.evaluation_histogram.observe(avg_eval_time)
```

---

本文档涵盖了 OpenEvolve 的高级用法，包括自定义组件、分布式部署、性能优化等主题。这些功能适用于生产环境和复杂的研究项目。

如需更多信息，请参考：
- [用户指南](user_guide.md) - 基础使用方法
- [API 文档](api_reference.md) - 详细 API 参考
- [贡献指南](../CONTRIBUTING.md) - 开发和贡献说明 