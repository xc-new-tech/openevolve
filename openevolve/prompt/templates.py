"""
Prompt templates for OpenEvolve
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Union

# Base system message template for evolution
BASE_SYSTEM_TEMPLATE = """You are an expert software developer tasked with iteratively improving a codebase.
Your job is to analyze the current program and suggest improvements based on feedback from previous attempts.
Focus on making targeted changes that will increase the program's performance metrics.
"""

BASE_EVALUATOR_SYSTEM_TEMPLATE = """You are an expert code reviewer.
Your job is to analyze the provided code and evaluate it systematically."""

# User message template for diff-based evolution
DIFF_USER_TEMPLATE = """# Current Program Information
- Current performance metrics: {metrics}
- Areas identified for improvement: {improvement_areas}

{artifacts}

# Program Evolution History
{evolution_history}

# Current Program
```{language}
{current_program}
```

# Task
Suggest improvements to the program that will lead to better performance on the specified metrics.

You MUST use the exact SEARCH/REPLACE diff format shown below to indicate changes:

<<<<<<< SEARCH
# Original code to find and replace (must match exactly)
=======
# New replacement code
>>>>>>> REPLACE

Example of valid diff format:
<<<<<<< SEARCH
for i in range(m):
    for j in range(p):
        for k in range(n):
            C[i, j] += A[i, k] * B[k, j]
=======
# Reorder loops for better memory access pattern
for i in range(m):
    for k in range(n):
        for j in range(p):
            C[i, j] += A[i, k] * B[k, j]
>>>>>>> REPLACE

You can suggest multiple changes. Each SEARCH section must exactly match code in the current program.
Be thoughtful about your changes and explain your reasoning thoroughly.

IMPORTANT: Do not rewrite the entire program - focus on targeted improvements.
"""

# User message template for full rewrite
FULL_REWRITE_USER_TEMPLATE = """# Current Program Information
- Current performance metrics: {metrics}
- Areas identified for improvement: {improvement_areas}

{artifacts}

# Program Evolution History
{evolution_history}

# Current Program
```{language}
{current_program}
```

# Task
Rewrite the program to improve its performance on the specified metrics.
Provide the complete new program code.

IMPORTANT: Make sure your rewritten program maintains the same inputs and outputs
as the original program, but with improved internal implementation.

```{language}
# Your rewritten program here
```
"""

# Template for formatting evolution history
EVOLUTION_HISTORY_TEMPLATE = """## Previous Attempts

{previous_attempts}

## Top Performing Programs

{top_programs}
"""

# Template for formatting a previous attempt
PREVIOUS_ATTEMPT_TEMPLATE = """### Attempt {attempt_number}
- Changes: {changes}
- Performance: {performance}
- Outcome: {outcome}
"""

# Template for formatting a top program
TOP_PROGRAM_TEMPLATE = """### Program {program_number} (Score: {score})
```{language}
{program_snippet}
```
Key features: {key_features}
"""

# Template for evaluating a program via an LLM
EVALUATION_TEMPLATE = """Evaluate the following code on a scale of 0.0 to 1.0 for the following metrics:
1. Readability: How easy is the code to read and understand?
2. Maintainability: How easy would the code be to maintain and modify?
3. Efficiency: How efficient is the code in terms of time and space complexity?

For each metric, provide a score between 0.0 and 1.0, where 1.0 is best.

Code to evaluate:
```python
{current_program}
```

Return your evaluation as a JSON object with the following format:
{{
    "readability": [score],
    "maintainability": [score],
    "efficiency": [score],
    "reasoning": "[brief explanation of scores]"
}}
"""

# C/C++ specific templates with safety constraints
CPP_SYSTEM_TEMPLATE = """You are an expert C/C++ developer tasked with iteratively improving C/C++ code.
Your job is to analyze the current program and suggest improvements based on feedback from previous attempts.
Focus on making targeted changes that will increase the program's performance metrics.

CRITICAL SAFETY CONSTRAINTS:
- NO system calls (system(), exec(), etc.)
- NO file I/O operations (fopen, open, etc.)
- NO network operations (socket, etc.)
- NO dynamic memory management issues (avoid memory leaks, buffer overflows)
- NO infinite loops or recursion without proper termination
- Use standard library functions only
- Keep code within safe computational bounds
- All variables must be properly initialized
"""

CPP_DIFF_USER_TEMPLATE = """# Current C/C++ Program Information
- Current performance metrics: {metrics}
- Areas identified for improvement: {improvement_areas}

{artifacts}

# Program Evolution History
{evolution_history}

# Current Program
```{language}
{current_program}
```

# Task
Suggest improvements to the C/C++ program that will lead to better performance on the specified metrics.

SAFETY REQUIREMENTS:
- NO system calls, file I/O, or network operations
- Avoid buffer overflows and memory leaks
- Use only standard library functions
- Ensure proper variable initialization

You MUST use the exact SEARCH/REPLACE diff format shown below to indicate changes:

<<<<<<< SEARCH
// Original code to find and replace (must match exactly)
=======
// New replacement code  
>>>>>>> REPLACE

Example of valid diff format:
<<<<<<< SEARCH
for (int i = 0; i < n; i++) {{
    for (int j = 0; j < m; j++) {{
        result += matrix[i][j];
    }}
}}
=======
// Cache-friendly loop order for better performance
for (int j = 0; j < m; j++) {{
    for (int i = 0; i < n; i++) {{
        result += matrix[i][j];
    }}
}}
>>>>>>> REPLACE

You can suggest multiple changes. Each SEARCH section must exactly match code in the current program.
Be thoughtful about your changes and explain your reasoning thoroughly.

IMPORTANT: Do not rewrite the entire program - focus on targeted improvements.
Ensure all code follows C99 or C++17 standards with safety constraints.
"""

CPP_FULL_REWRITE_USER_TEMPLATE = """# Current C/C++ Program Information
- Current performance metrics: {metrics}
- Areas identified for improvement: {improvement_areas}

{artifacts}

# Program Evolution History
{evolution_history}

# Current Program
```{language}
{current_program}
```

# Task
Rewrite the C/C++ program to improve its performance on the specified metrics.
Provide the complete new program code.

SAFETY REQUIREMENTS:
- NO system calls, file I/O, or network operations
- Avoid buffer overflows and memory leaks
- Use only standard library functions
- Ensure proper variable initialization
- Follow C99 or C++17 standards

IMPORTANT: Make sure your rewritten program maintains the same inputs and outputs
as the original program, but with improved internal implementation.

```{language}
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
// Add other standard headers as needed

// Your rewritten program here
```
"""

CPP_EVALUATION_TEMPLATE = """Evaluate the following C/C++ code on a scale of 0.0 to 1.0 for the following metrics:
1. Readability: How easy is the code to read and understand?
2. Maintainability: How easy would the code be to maintain and modify?
3. Efficiency: How efficient is the code in terms of time and space complexity?
4. Safety: Does the code avoid dangerous operations and follow safety constraints?

For each metric, provide a score between 0.0 and 1.0, where 1.0 is best.

Safety constraints to check:
- No system calls, file I/O, or network operations
- No buffer overflows or memory leaks
- Proper variable initialization
- Use of standard library only

Code to evaluate:
```{language}
{current_program}
```

Return your evaluation as a JSON object with the following format:
{{
    "readability": [score],
    "maintainability": [score], 
    "efficiency": [score],
    "safety": [score],
    "reasoning": "[brief explanation of scores including safety assessment]"
}}
"""

# C specific templates (C99 standard)
C_SYSTEM_TEMPLATE = """You are an expert C developer tasked with iteratively improving C code.
Your job is to analyze the current program and suggest improvements based on feedback from previous attempts.
Focus on making targeted changes that will increase the program's performance metrics.

CRITICAL SAFETY CONSTRAINTS:
- NO system calls (system(), exec(), etc.)
- NO file I/O operations (fopen, open, etc.)
- NO network operations (socket, etc.)
- NO dynamic memory management issues (avoid memory leaks, buffer overflows)
- NO infinite loops or recursion without proper termination
- Use C99 standard library functions only
- Keep code within safe computational bounds
- All variables must be properly initialized
- Use designated initializers where appropriate
"""

# Default templates dictionary
DEFAULT_TEMPLATES = {
    "system_message": BASE_SYSTEM_TEMPLATE,
    "evaluator_system_message": BASE_EVALUATOR_SYSTEM_TEMPLATE,
    "diff_user": DIFF_USER_TEMPLATE,
    "full_rewrite_user": FULL_REWRITE_USER_TEMPLATE,
    "evolution_history": EVOLUTION_HISTORY_TEMPLATE,
    "previous_attempt": PREVIOUS_ATTEMPT_TEMPLATE,
    "top_program": TOP_PROGRAM_TEMPLATE,
    "evaluation": EVALUATION_TEMPLATE,
    # C/C++ specific templates
    "cpp_system_message": CPP_SYSTEM_TEMPLATE,
    "cpp_diff_user": CPP_DIFF_USER_TEMPLATE,
    "cpp_full_rewrite_user": CPP_FULL_REWRITE_USER_TEMPLATE,
    "cpp_evaluation": CPP_EVALUATION_TEMPLATE,
    "c_system_message": C_SYSTEM_TEMPLATE,
}


class TemplateManager:
    """Manages templates for prompt generation"""

    def __init__(self, template_dir: Optional[str] = None):
        self.templates = DEFAULT_TEMPLATES.copy()

        # Load templates from directory if provided
        if template_dir and os.path.isdir(template_dir):
            self._load_templates_from_dir(template_dir)

    def _load_templates_from_dir(self, template_dir: str) -> None:
        """Load templates from a directory"""
        for file_path in Path(template_dir).glob("*.txt"):
            template_name = file_path.stem
            with open(file_path, "r") as f:
                self.templates[template_name] = f.read()

    def get_template(self, template_name: str) -> str:
        """Get a template by name"""
        if template_name not in self.templates:
            raise ValueError(f"Template '{template_name}' not found")
        return self.templates[template_name]

    def add_template(self, template_name: str, template: str) -> None:
        """Add or update a template"""
        self.templates[template_name] = template

    def get_language_template(self, base_template: str, language: str) -> str:
        """Get a language-specific template, falling back to base template if not found"""
        # Normalize language names
        lang_map = {
            "cpp": "cpp",
            "c++": "cpp", 
            "cxx": "cpp",
            "cc": "cpp",
            "c": "c",
            "python": "python",
            "py": "python"
        }
        
        normalized_lang = lang_map.get(language.lower(), language.lower())
        
        # Try language-specific template first
        lang_template_name = f"{normalized_lang}_{base_template}"
        if lang_template_name in self.templates:
            return self.templates[lang_template_name]
        
        # Fall back to base template
        return self.get_template(base_template)

    def get_system_message(self, language: str = "python") -> str:
        """Get system message template for specified language"""
        return self.get_language_template("system_message", language)

    def get_diff_user_template(self, language: str = "python") -> str:
        """Get diff user template for specified language"""
        return self.get_language_template("diff_user", language)

    def get_full_rewrite_template(self, language: str = "python") -> str:
        """Get full rewrite template for specified language"""
        return self.get_language_template("full_rewrite_user", language)

    def get_evaluation_template(self, language: str = "python") -> str:
        """Get evaluation template for specified language"""
        return self.get_language_template("evaluation", language)

    def supports_language(self, language: str) -> bool:
        """Check if language-specific templates are available"""
        normalized_lang = language.lower()
        if normalized_lang in ["cpp", "c++", "cxx", "cc"]:
            normalized_lang = "cpp"
        elif normalized_lang in ["py"]:
            normalized_lang = "python"
        
        return any(key.startswith(f"{normalized_lang}_") for key in self.templates.keys())
