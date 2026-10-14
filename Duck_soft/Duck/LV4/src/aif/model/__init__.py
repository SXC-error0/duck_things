"""生成模型（A/B/C/D）与因子/模态声明。"""

from .factors import FactorSpec, ModalitySpec
from .generative import GenerativeModel

__all__ = ["GenerativeModel", "FactorSpec", "ModalitySpec"]
