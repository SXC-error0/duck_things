"""aif：域无关的主动推断与决策框架，带三层「自我」能力。

基于 pymdp 1.0.4 扩展。核心入口：

- ``AgentSpec`` / ``Environment``：域无关契约；
- ``GenerativeModel``：A/B/C/D 生成模型；
- ``SelfAwareAgent``：统一闭环（感知 → 推断 → 策略 → 动作）。

示例：``python -m aif.examples.gridworld --steps 50``。
"""

from .agent import SelfAwareAgent, StepResult
from .interfaces import AgentSpec, Environment, Observation, obs_to_list
from .model import GenerativeModel

__version__ = "0.1.0"

__all__ = [
    "AgentSpec",
    "Environment",
    "Observation",
    "obs_to_list",
    "GenerativeModel",
    "SelfAwareAgent",
    "StepResult",
    "__version__",
]
