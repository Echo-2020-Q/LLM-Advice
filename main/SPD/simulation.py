"""仅依赖 Python 标准库的空间弱囚徒困境仿真（Python >= 3.10）。

收益矩阵（行为自身策略、列为邻居策略，顺序 C、D）：[[1, 0], [b, 0]]。
个体不与自己博弈；收益取所有冯诺依曼邻居的收益之和。
每次策略更新以概率 lambda_prob 请求并直接采用建议，否则执行 Fermi 模仿。
建议为 None 时也执行 Fermi；lambda_prob 是每次更新的概率，不是固定个体比例。

随机异步：一个 Monte Carlo 步包含 L*L 次有放回抽样，即时修改策略和重算收益。
同步：每人根据同一旧状态更新一次，最后统一替换；一轮记为一个时间步。
"""

import math
import random
from dataclasses import dataclass
from typing import Sequence

from .llm_advisor import (
    AdviceContext, Advisor, NeighborObservation, Position, Strategy, request_advice,
)


@dataclass(frozen=True)
class SimulationConfig:
    size: int = 50
    b: float = 1.5
    k: float = 0.1
    lambda_prob: float = 0.1
    initial_cooperation: float = 0.5
    seed: int = 42
    boundary: str = "periodic"
    update_mode: str = "asynchronous"

    def __post_init__(self):
        if type(self.size) is not int or self.size < 3:
            raise ValueError("size 必须是至少为 3 的整数，避免周期边界的重复邻居。")
        if not math.isfinite(self.b) or self.b <= 1 or not math.isfinite(4 * self.b):
            raise ValueError("b 必须大于 1，且最大个体收益 4*b 必须有限。")
        if not math.isfinite(self.k) or self.k <= 0:
            raise ValueError("k 必须是有限正数。")
        for name in ("lambda_prob", "initial_cooperation"):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} 必须在 [0, 1] 内。")
        if type(self.seed) is not int:
            raise ValueError("seed 必须是整数。")
        if self.boundary not in {"periodic", "open"}:
            raise ValueError("boundary 必须为 periodic 或 open。")
        if self.update_mode not in {"asynchronous", "synchronous"}:
            raise ValueError("update_mode 必须为 asynchronous 或 synchronous。")


@dataclass(frozen=True)
class RoundStats:
    """轮末状态及本轮计数（计数不跨轮累加）。"""

    round: int
    cooperation_rate: float
    average_payoff: float
    advice_requests: int = 0
    advice_used: int = 0
    fermi_updates: int = 0
    strategy_changes: int = 0


def fermi_probability(own_payoff: float, neighbor_payoff: float, k: float) -> float:
    """模仿邻居的概率 1 / (1 + exp((Pi - Pj) / K))，使用稳定计算。"""
    if not math.isfinite(k) or k <= 0:
        raise ValueError("k 必须是有限正数。")
    if not math.isfinite(own_payoff) or not math.isfinite(neighbor_payoff):
        raise ValueError("收益必须有限。")
    difference = (own_payoff - neighbor_payoff) / k
    # 指数始终非正；即使极小 K 使 difference 成为 inf，也不会溢出。
    weight = math.exp(-abs(difference))
    return weight / (1 + weight) if difference >= 0 else 1 / (1 + weight)


class SpatialPrisonersDilemma:
    def __init__(
        self,
        config: SimulationConfig,
        advisor: Advisor = request_advice,
        initial_grid: Sequence[Sequence[Strategy]] | None = None,
    ):
        self.config = config
        self.advisor = advisor
        self.round = 0
        self._population = config.size ** 2
        self._rng = random.Random(config.seed)
        # 单独的建议触发随机流，使空接口不会改变同种子的纯 Fermi 演化轨迹。
        self._advice_rng = random.Random(f"advice:{config.seed}")
        self._neighbors = tuple(self._build_neighbors(i) for i in range(self._population))
        if initial_grid is None:
            self._strategies: list[Strategy] = [
                "C" if self._rng.random() < config.initial_cooperation else "D"
                for _ in range(self._population)
            ]
        else:
            if len(initial_grid) != config.size or any(
                len(row) != config.size for row in initial_grid
            ):
                raise ValueError("initial_grid 必须是 size × size 的二维网格。")
            self._strategies = [strategy for row in initial_grid for strategy in row]
            if any(strategy not in ("C", "D") for strategy in self._strategies):
                raise ValueError("initial_grid 中只能包含字符串 C 或 D。")

    def _build_neighbors(self, index: int) -> tuple[int, ...]:
        size = self.config.size
        row, col = divmod(index, size)
        positions = ((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1))
        if self.config.boundary == "periodic":
            return tuple((r % size) * size + c % size for r, c in positions)
        return tuple(r * size + c for r, c in positions if 0 <= r < size and 0 <= c < size)

    def _index(self, row: int, col: int) -> int:
        if not (0 <= row < self.config.size and 0 <= col < self.config.size):
            raise IndexError("网格坐标越界。")
        return row * self.config.size + col

    @property
    def grid(self) -> tuple[tuple[Strategy, ...], ...]:
        """返回不可变快照；修改外部初始网格也不会影响仿真。"""
        size = self.config.size
        return tuple(tuple(self._strategies[i:i + size]) for i in range(0, self._population, size))

    def neighbors(self, row: int, col: int) -> tuple[Position, ...]:
        """返回上、下、左、右邻居；开放边界省略越界方向。"""
        return tuple(divmod(i, self.config.size) for i in self._neighbors[self._index(row, col)])

    def _payoff(self, index: int, state: Sequence[Strategy]) -> float:
        cooperating = sum(state[j] == "C" for j in self._neighbors[index])
        return float(cooperating) if state[index] == "C" else self.config.b * cooperating

    def payoff(self, row: int, col: int) -> float:
        return self._payoff(self._index(row, col), self._strategies)

    def _context(
        self, index: int, state: Sequence[Strategy], payoffs: list[float] | None,
    ) -> AdviceContext:
        def current_payoff(i):
            return payoffs[i] if payoffs is not None else self._payoff(i, state)

        return AdviceContext(
            round=self.round + 1,
            position=divmod(index, self.config.size),
            strategy=state[index],
            payoff=current_payoff(index),
            b=self.config.b,
            k=self.config.k,
            neighbors=tuple(
                NeighborObservation(divmod(j, self.config.size), state[j], current_payoff(j))
                for j in self._neighbors[index]
            ),
        )

    def _choose_strategy(
        self, index: int, state: Sequence[Strategy], payoffs: list[float] | None,
    ) -> tuple[Strategy, int, int]:
        requested = int(
            self.config.lambda_prob > 0
            and self._advice_rng.random() < self.config.lambda_prob
        )
        if requested:
            advice = self.advisor(self._context(index, state, payoffs))
            if advice is not None:
                if not isinstance(advice, str) or advice not in ("C", "D"):
                    raise ValueError("建议接口必须返回 C、D 或 None。")
                return advice, requested, 1

        neighbor = self._rng.choice(self._neighbors[index])
        own_payoff = payoffs[index] if payoffs is not None else self._payoff(index, state)
        neighbor_payoff = payoffs[neighbor] if payoffs is not None else self._payoff(neighbor, state)
        probability = fermi_probability(own_payoff, neighbor_payoff, self.config.k)
        strategy = state[neighbor] if self._rng.random() < probability else state[index]
        return strategy, requested, 0

    def measure(self) -> RoundStats:
        """只测量当前状态，更新次数为零；平均收益按当前策略重新计算。"""
        # 先除以人口再求和，避免大 b 时总收益超出浮点范围。
        average_payoff = math.fsum(
            self._payoff(i, self._strategies) / self._population
            for i in range(self._population)
        )
        return RoundStats(
            round=self.round,
            cooperation_rate=self._strategies.count("C") / self._population,
            average_payoff=average_payoff,
        )

    def step(self) -> RoundStats:
        """执行一轮更新。异步轮内同一个人可能被抽中多次，也可能未被抽中。"""
        synchronous = self.config.update_mode == "synchronous"
        state = self._strategies
        payoffs = [self._payoff(i, state) for i in range(self._population)] if synchronous else None
        new_state = state.copy() if synchronous else state
        requests = used = changes = 0
        for event in range(self._population):
            index = event if synchronous else self._rng.randrange(self._population)
            strategy, requested, adopted = self._choose_strategy(index, state, payoffs)
            requests += requested
            used += adopted
            changes += strategy != state[index]
            new_state[index] = strategy
        self._strategies = new_state
        self.round += 1
        measurement = self.measure()
        return RoundStats(
            round=self.round,
            cooperation_rate=measurement.cooperation_rate,
            average_payoff=measurement.average_payoff,
            advice_requests=requests,
            advice_used=used,
            fermi_updates=self._population - used,
            strategy_changes=changes,
        )

    def run(self, steps: int) -> list[RoundStats]:
        """返回当前状态及之后 steps 轮的统计；初次运行包含第 0 轮。"""
        if type(steps) is not int or steps < 0:
            raise ValueError("steps 必须是非负整数。")
        history = [self.measure()]
        for _ in range(steps):
            history.append(self.step())
        return history
