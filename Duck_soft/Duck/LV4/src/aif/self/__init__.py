"""三层「自我」能力（对自身状态、信念精度、模型结构做二阶推断）。

- ``interoception``：内感受 + 自模型 + allostatic 精度驱动（M2）。
- ``metacognition``：逐模态精度 γ + 策略精度 β 的元认知估计（M3）。
- ``structure``：贝叶斯模型比较 / 结构学习（M4）。
"""

from .interoception import Interoception
from .metacognition import Metacognition
from .structure import StructureLearning

__all__ = ["Interoception", "Metacognition", "StructureLearning"]
