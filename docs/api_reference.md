# OpenEvolve API 参考

本文档提供 OpenEvolve 的详细 API 参考，包括所有主要类、方法和参数说明。

## 目录

1. [核心类](#核心类)
2. [配置管理](#配置管理)
3. [进化算子](#进化算子)
4. [评估系统](#评估系统)
5. [数据库](#数据库)
6. [LLM 集成](#llm-集成)
7. [实用工具](#实用工具)

## 核心类

### OpenEvolve

主要的进化系统类。

```python
class OpenEvolve:
    """OpenEvolve 主要控制器类"""
    
    def __init__(self, 
                 initial_program_path: str,
                 evaluation_file: str,
                 config_path: Optional[str] = None,
                 config: Optional[Dict[str, Any]] = None):
        """
        初始化 OpenEvolve 系统
        
        Args:
            initial_program_path: 初始程序文件路径
            evaluation_file: 评估器文件路径
            config_path: 配置文件路径（YAML 格式）
            config: 直接传入的配置字典（与 config_path 二选一）
        """
    
    async def run(self, 
                  iterations: int, 
                  checkpoint_dir: Optional[str] = None) -> Program:
        """
        运行进化过程
        
        Args:
            iterations: 进化迭代次数
            checkpoint_dir: 检查点目录（用于恢复）
            
        Returns:
            最佳程序对象
        """
    
    async def run_single_generation(self) -> Dict[str, Any]:
        """
        运行单次进化代
        
        Returns:
            包含进化统计信息的字典
        """
    
    def save_checkpoint(self, checkpoint_dir: str) -> None:
        """
        保存当前状态到检查点
        
        Args:
            checkpoint_dir: 检查点保存目录
        """
    
    def load_checkpoint(self, checkpoint_dir: str) -> None:
        """
        从检查点恢复状态
        
        Args:
            checkpoint_dir: 检查点目录
        """
```

### Program

代表一个程序实例。

```python
class Program:
    """程序对象类"""
    
    def __init__(self, 
                 code: str, 
                 program_id: Optional[str] = None,
                 parent_id: Optional[str] = None,
                 generation: int = 0):
        """
        初始化程序对象
        
        Args:
            code: 程序代码字符串
            program_id: 程序唯一标识符
            parent_id: 父程序标识符
            generation: 所属代数
        """
    
    @property
    def id(self) -> str:
        """获取程序ID"""
    
    @property
    def code(self) -> str:
        """获取程序代码"""
    
    @property
    def metrics(self) -> Dict[str, float]:
        """获取程序评估指标"""
    
    @property
    def artifacts(self) -> Dict[str, Any]:
        """获取程序评估人工制品"""
    
    def set_evaluation_result(self, result: 'EvaluationResult') -> None:
        """
        设置评估结果
        
        Args:
            result: 评估结果对象
        """
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Program':
        """从字典创建程序对象"""
```

### EvaluationResult

评估结果容器。

```python
class EvaluationResult:
    """评估结果类"""
    
    def __init__(self, 
                 metrics: Dict[str, float],
                 artifacts: Optional[Dict[str, Any]] = None,
                 success: bool = True,
                 error_message: Optional[str] = None):
        """
        初始化评估结果
        
        Args:
            metrics: 评估指标字典
            artifacts: 评估人工制品字典
            success: 评估是否成功
            error_message: 错误信息（如果失败）
        """
    
    @property
    def primary_metric(self) -> float:
        """获取主要评估指标"""
    
    def is_better_than(self, other: 'EvaluationResult') -> bool:
        """比较两个评估结果"""
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
```

## 配置管理

### Config

配置管理类。

```python
class Config:
    """配置管理类"""
    
    def __init__(self, config_data: Dict[str, Any]):
        """
        初始化配置
        
        Args:
            config_data: 配置数据字典
        """
    
    @classmethod
    def from_file(cls, config_path: str) -> 'Config':
        """
        从文件加载配置
        
        Args:
            config_path: 配置文件路径
        """
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        获取配置值
        
        Args:
            key: 配置键（支持点符号，如 'llm.temperature'）
            default: 默认值
        """
    
    def set(self, key: str, value: Any) -> None:
        """
        设置配置值
        
        Args:
            key: 配置键
            value: 配置值
        """
    
    def merge(self, other_config: Union['Config', Dict[str, Any]]) -> 'Config':
        """
        合并配置
        
        Args:
            other_config: 要合并的配置
        """
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
```

### 配置参数

主要的配置参数说明：

```yaml
# 基础配置
max_iterations: 1000        # 最大迭代次数
population_size: 100        # 种群大小
random_seed: 42            # 随机种子

# LLM 配置
llm:
  models:                  # 模型列表
    - name: "gpt-4o"
      weight: 0.7
      api_base: "https://api.openai.com/v1"
      api_key: null        # 使用环境变量
    - name: "claude-3"
      weight: 0.3
  temperature: 0.7         # 生成温度
  max_tokens: 4096         # 最大令牌数
  top_p: 1.0              # Top-p 采样

# 进化参数
evolution:
  selection_strategy: "tournament"  # 选择策略
  tournament_size: 3              # 锦标赛大小
  mutation_rate: 0.8              # 变异率
  elitism_rate: 0.1               # 精英保留率
  crossover_rate: 0.2             # 交叉率

# 数据库配置
database:
  num_islands: 5                  # 岛屿数量
  migration_interval: 10          # 迁移间隔
  migration_rate: 0.1             # 迁移率
  map_elites:
    enabled: true
    dimensions:
      - feature: "performance"
        bins: 10
        range: [0.0, 1.0]

# 评估配置
evaluation:
  timeout: 30                     # 评估超时（秒）
  max_concurrent: 10              # 最大并发评估数
  retry_attempts: 3               # 重试次数

# 输出配置
output:
  save_best_program: true         # 保存最佳程序
  save_all_programs: false       # 保存所有程序
  checkpoint_interval: 10         # 检查点间隔
  log_level: "INFO"              # 日志级别
```

## 进化算子

### MutationOperator

变异算子基类。

```python
class MutationOperator(ABC):
    """变异算子基类"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化变异算子
        
        Args:
            config: 算子配置
        """
    
    @abstractmethod
    def mutate(self, program: Program, context: Dict[str, Any]) -> str:
        """
        执行变异操作
        
        Args:
            program: 要变异的程序
            context: 变异上下文
            
        Returns:
            变异后的程序代码
        """
    
    def can_mutate(self, program: Program) -> bool:
        """
        检查是否可以对程序进行变异
        
        Args:
            program: 程序对象
            
        Returns:
            是否可以变异
        """
        return True
```

### SelectionOperator

选择算子基类。

```python
class SelectionOperator(ABC):
    """选择算子基类"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化选择算子
        
        Args:
            config: 算子配置
        """
    
    @abstractmethod
    def select(self, population: List[Program], size: int) -> List[Program]:
        """
        从种群中选择程序
        
        Args:
            population: 候选程序列表
            size: 要选择的程序数量
            
        Returns:
            选中的程序列表
        """
    
    def get_selection_pressure(self) -> float:
        """获取选择压力"""
        return 1.0
```

### 内置算子

#### TournamentSelection

```python
class TournamentSelection(SelectionOperator):
    """锦标赛选择算子"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.tournament_size = config.get('tournament_size', 3)
    
    def select(self, population: List[Program], size: int) -> List[Program]:
        """执行锦标赛选择"""
```

#### LLMMutationOperator

```python
class LLMMutationOperator(MutationOperator):
    """基于 LLM 的变异算子"""
    
    def __init__(self, config: Dict[str, Any], llm_client: 'LLMClient'):
        super().__init__(config)
        self.llm_client = llm_client
        self.prompt_templates = config.get('prompt_templates', [])
    
    async def mutate(self, program: Program, context: Dict[str, Any]) -> str:
        """使用 LLM 执行变异"""
```

## 评估系统

### Evaluator

评估器基类。

```python
class Evaluator(ABC):
    """评估器基类"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化评估器
        
        Args:
            config: 评估器配置
        """
    
    @abstractmethod
    async def evaluate(self, program: Program) -> EvaluationResult:
        """
        评估程序
        
        Args:
            program: 要评估的程序
            
        Returns:
            评估结果
        """
    
    def preprocess(self, program: Program) -> Program:
        """
        预处理程序（可选重写）
        
        Args:
            program: 原程序
            
        Returns:
            预处理后的程序
        """
        return program
    
    def postprocess(self, result: EvaluationResult) -> EvaluationResult:
        """
        后处理评估结果（可选重写）
        
        Args:
            result: 原始评估结果
            
        Returns:
            后处理后的结果
        """
        return result
```

### FunctionEvaluator

基于函数的评估器。

```python
class FunctionEvaluator(Evaluator):
    """基于函数的评估器"""
    
    def __init__(self, evaluation_function: Callable, config: Dict[str, Any]):
        """
        初始化函数评估器
        
        Args:
            evaluation_function: 评估函数
            config: 配置
        """
        super().__init__(config)
        self.evaluation_function = evaluation_function
        self.timeout = config.get('timeout', 30)
    
    async def evaluate(self, program: Program) -> EvaluationResult:
        """使用函数评估程序"""
```

### CompositeeEvaluator

复合评估器。

```python
class CompositeEvaluator(Evaluator):
    """复合评估器"""
    
    def __init__(self, evaluators: List[Evaluator], weights: Optional[List[float]] = None):
        """
        初始化复合评估器
        
        Args:
            evaluators: 子评估器列表
            weights: 权重列表
        """
        self.evaluators = evaluators
        self.weights = weights or [1.0] * len(evaluators)
    
    async def evaluate(self, program: Program) -> EvaluationResult:
        """执行复合评估"""
```

## 数据库

### ProgramDatabase

程序数据库。

```python
class ProgramDatabase:
    """程序数据库类"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化数据库
        
        Args:
            config: 数据库配置
        """
    
    async def store_program(self, program: Program) -> None:
        """
        存储程序
        
        Args:
            program: 程序对象
        """
    
    async def get_program(self, program_id: str) -> Optional[Program]:
        """
        获取程序
        
        Args:
            program_id: 程序ID
            
        Returns:
            程序对象（如果存在）
        """
    
    async def get_population(self, 
                           generation: Optional[int] = None,
                           size: Optional[int] = None) -> List[Program]:
        """
        获取种群
        
        Args:
            generation: 代数过滤
            size: 返回数量限制
            
        Returns:
            程序列表
        """
    
    async def get_best_programs(self, count: int = 10) -> List[Program]:
        """
        获取最佳程序
        
        Args:
            count: 返回数量
            
        Returns:
            最佳程序列表
        """
    
    async def search_programs(self, 
                            criteria: Dict[str, Any]) -> List[Program]:
        """
        搜索程序
        
        Args:
            criteria: 搜索条件
            
        Returns:
            匹配的程序列表
        """
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        获取数据库统计信息
        
        Returns:
            统计信息字典
        """
```

### IslandModel

岛屿模型。

```python
class IslandModel:
    """岛屿模型类"""
    
    def __init__(self, 
                 num_islands: int,
                 migration_interval: int = 10,
                 migration_rate: float = 0.1):
        """
        初始化岛屿模型
        
        Args:
            num_islands: 岛屿数量
            migration_interval: 迁移间隔
            migration_rate: 迁移率
        """
    
    async def migrate(self, generation: int) -> None:
        """
        执行种群迁移
        
        Args:
            generation: 当前代数
        """
    
    def get_island_population(self, island_id: int) -> List[Program]:
        """
        获取岛屿种群
        
        Args:
            island_id: 岛屿ID
            
        Returns:
            岛屿程序列表
        """
    
    def get_island_statistics(self) -> Dict[int, Dict[str, Any]]:
        """
        获取所有岛屿统计信息
        
        Returns:
            岛屿统计信息字典
        """
```

## LLM 集成

### LLMClient

LLM 客户端基类。

```python
class LLMClient(ABC):
    """LLM 客户端基类"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化 LLM 客户端
        
        Args:
            config: 客户端配置
        """
    
    @abstractmethod
    async def generate(self, 
                      prompt: str, 
                      max_tokens: Optional[int] = None,
                      temperature: Optional[float] = None) -> str:
        """
        生成文本
        
        Args:
            prompt: 输入提示
            max_tokens: 最大令牌数
            temperature: 生成温度
            
        Returns:
            生成的文本
        """
    
    async def generate_batch(self, 
                           prompts: List[str],
                           **kwargs) -> List[str]:
        """
        批量生成文本
        
        Args:
            prompts: 提示列表
            **kwargs: 其他参数
            
        Returns:
            生成文本列表
        """
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """
        获取使用统计
        
        Returns:
            使用统计信息
        """
```

### OpenAIClient

OpenAI 客户端实现。

```python
class OpenAIClient(LLMClient):
    """OpenAI 客户端"""
    
    def __init__(self, 
                 api_key: str,
                 model: str = "gpt-4",
                 api_base: str = "https://api.openai.com/v1",
                 **kwargs):
        """
        初始化 OpenAI 客户端
        
        Args:
            api_key: API 密钥
            model: 模型名称
            api_base: API 基础 URL
            **kwargs: 其他配置
        """
    
    async def generate(self, prompt: str, **kwargs) -> str:
        """生成文本"""
```

### LLMEnsemble

LLM 集成。

```python
class LLMEnsemble:
    """LLM 集成类"""
    
    def __init__(self, clients: List[Tuple[LLMClient, float]]):
        """
        初始化 LLM 集成
        
        Args:
            clients: (客户端, 权重) 元组列表
        """
    
    async def generate(self, prompt: str, **kwargs) -> str:
        """
        使用集成生成文本
        
        Args:
            prompt: 输入提示
            **kwargs: 生成参数
            
        Returns:
            生成的文本
        """
    
    async def generate_diverse(self, 
                             prompt: str, 
                             count: int = 3,
                             **kwargs) -> List[str]:
        """
        生成多样化的文本
        
        Args:
            prompt: 输入提示
            count: 生成数量
            **kwargs: 生成参数
            
        Returns:
            生成文本列表
        """
```

## 实用工具

### PromptTemplate

提示模板。

```python
class PromptTemplate:
    """提示模板类"""
    
    def __init__(self, template: str, variables: Optional[List[str]] = None):
        """
        初始化提示模板
        
        Args:
            template: 模板字符串
            variables: 变量列表
        """
    
    def format(self, **kwargs) -> str:
        """
        格式化模板
        
        Args:
            **kwargs: 模板变量
            
        Returns:
            格式化后的提示
        """
    
    def get_variables(self) -> List[str]:
        """
        获取模板变量列表
        
        Returns:
            变量名列表
        """
```

### CodeAnalyzer

代码分析器。

```python
class CodeAnalyzer:
    """代码分析器"""
    
    def __init__(self, language: str = "python"):
        """
        初始化代码分析器
        
        Args:
            language: 编程语言
        """
    
    def parse(self, code: str) -> 'AST':
        """
        解析代码
        
        Args:
            code: 源代码
            
        Returns:
            抽象语法树
        """
    
    def extract_functions(self, code: str) -> List[Dict[str, Any]]:
        """
        提取函数信息
        
        Args:
            code: 源代码
            
        Returns:
            函数信息列表
        """
    
    def calculate_complexity(self, code: str) -> Dict[str, float]:
        """
        计算代码复杂度
        
        Args:
            code: 源代码
            
        Returns:
            复杂度指标字典
        """
    
    def find_similarities(self, code1: str, code2: str) -> float:
        """
        计算代码相似度
        
        Args:
            code1: 第一个代码
            code2: 第二个代码
            
        Returns:
            相似度分数 (0-1)
        """
```

### Logger

日志工具。

```python
class Logger:
    """OpenEvolve 日志器"""
    
    def __init__(self, 
                 name: str = "openevolve",
                 level: str = "INFO",
                 file_path: Optional[str] = None):
        """
        初始化日志器
        
        Args:
            name: 日志器名称
            level: 日志级别
            file_path: 日志文件路径
        """
    
    def debug(self, message: str, **kwargs) -> None:
        """记录调试信息"""
    
    def info(self, message: str, **kwargs) -> None:
        """记录信息"""
    
    def warning(self, message: str, **kwargs) -> None:
        """记录警告"""
    
    def error(self, message: str, **kwargs) -> None:
        """记录错误"""
    
    def log_generation(self, generation: int, stats: Dict[str, Any]) -> None:
        """
        记录进化代信息
        
        Args:
            generation: 代数
            stats: 统计信息
        """
    
    def log_evaluation(self, program: Program, result: EvaluationResult) -> None:
        """
        记录评估信息
        
        Args:
            program: 程序对象
            result: 评估结果
        """
```

## 异常类

### OpenEvolveError

```python
class OpenEvolveError(Exception):
    """OpenEvolve 基础异常类"""
    pass

class ConfigurationError(OpenEvolveError):
    """配置错误"""
    pass

class EvaluationError(OpenEvolveError):
    """评估错误"""
    pass

class LLMError(OpenEvolveError):
    """LLM 相关错误"""
    pass

class DatabaseError(OpenEvolveError):
    """数据库错误"""
    pass
```

## 示例用法

### 基础用法

```python
import asyncio
from openevolve import OpenEvolve

async def main():
    # 初始化系统
    evolve = OpenEvolve(
        initial_program_path="initial_program.py",
        evaluation_file="evaluator.py",
        config_path="config.yaml"
    )
    
    # 运行进化
    best_program = await evolve.run(iterations=100)
    
    print(f"最佳程序指标: {best_program.metrics}")
    print(f"最佳程序代码:\n{best_program.code}")

asyncio.run(main())
```

### 自定义评估器

```python
from openevolve.evaluation import Evaluator, EvaluationResult

class CustomEvaluator(Evaluator):
    async def evaluate(self, program: Program) -> EvaluationResult:
        # 自定义评估逻辑
        score = self.calculate_score(program.code)
        
        return EvaluationResult(
            metrics={"score": score},
            artifacts={"analysis": "详细分析信息"}
        )
    
    def calculate_score(self, code: str) -> float:
        # 实现评估逻辑
        return 0.5

# 使用自定义评估器
evaluator = CustomEvaluator({})
evolve = OpenEvolve(
    initial_program_path="program.py",
    evaluation_file=evaluator  # 直接传入评估器对象
)
```

### 自定义配置

```python
config = {
    "max_iterations": 500,
    "llm": {
        "models": [
            {"name": "gpt-4", "weight": 0.7},
            {"name": "claude-3", "weight": 0.3}
        ],
        "temperature": 0.8
    },
    "evolution": {
        "mutation_rate": 0.9,
        "selection_strategy": "tournament"
    }
}

evolve = OpenEvolve(
    initial_program_path="program.py",
    evaluation_file="evaluator.py",
    config=config
)
```

---

这个 API 参考提供了 OpenEvolve 的主要接口文档。更多详细用法请参考：

- [用户指南](user_guide.md)
- [高级用法](advanced_usage.md)
- [配置指南](api_configuration.md) 