"""大模型建议接口。此文件目前不读取密钥，也不发起网络请求。

接入时实现 request_advice(context)，或把自定义函数传给仿真器的 advisor。
返回 "C" / "D" 表示采用该策略；返回 None 表示没有建议，回退到 Fermi。
请求异常和非法返回值会终止实验，避免悄悄把失败请求当成有效实验数据。
"""

from dataclasses import dataclass
from typing import Literal, Protocol


Strategy = Literal["C", "D"]
Position = tuple[int, int]


@dataclass(frozen=True)
class NeighborObservation:
    position: Position
    strategy: Strategy
    payoff: float


@dataclass(frozen=True)
class AdviceContext:
    """本次更新前的局部观测；坐标从 0 开始，round 从 1 开始。

    同步更新时，本轮所有观测来自同一旧状态；异步时来自事件发生前的状态。
    payoff 为当时与所有邻居交互的收益之和，不是跨轮累计收益。
    """

    round: int
    position: Position
    strategy: Strategy
    payoff: float
    b: float
    k: float
    neighbors: tuple[NeighborObservation, ...]

    @property
    def cooperating_neighbors(self) -> int:
        return sum(neighbor.strategy == "C" for neighbor in self.neighbors)

    @property
    def defecting_neighbors(self) -> int:
        return len(self.neighbors) - self.cooperating_neighbors


class Advisor(Protocol):
    def __call__(self, context: AdviceContext) -> Strategy | None:
        ...


def request_advice(context: AdviceContext) -> Strategy | None:
    """预留接口：将 context 转成提示词，并返回经过严格校验的 C 或 D。

    当前不调用任何模型。允许显式返回 None，但不要把任意错误输出转换为 D。
    """
    return None
