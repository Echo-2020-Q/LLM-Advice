"""空间囚徒困境：冯诺依曼邻居、Fermi 更新与可插拔的建议接口。"""

from .llm_advisor import AdviceContext, NeighborObservation, request_advice
from .simulation import SimulationConfig, SpatialPrisonersDilemma, fermi_probability

__all__ = [
    "AdviceContext", "NeighborObservation", "request_advice",
    "SimulationConfig", "SpatialPrisonersDilemma", "fermi_probability",
]
