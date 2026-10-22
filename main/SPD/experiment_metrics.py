"""按固定 LLM 资格组记录轮末收益，不改变博弈收益、策略或随机数状态。

累计福利仅包含完整更新后的 round=1..T，每人每轮计入一次当轮空间收益。
round=0 展示初始瞬时收益并参与全 D 状态检测，但不计入福利或时间平均值。
全 D 首次出现不代表永久灭绝；建议仍可能使合作在后续轮次重新出现。
"""

from collections import deque
import math


GROUPS = ("population", "eligible", "ineligible")

HISTORY_FIELDS = tuple(
    group + "_" + field
    for group in GROUPS
    for field in ("mean_payoff", "cumulative_mean_payoff")
)

SUMMARY_FIELDS = tuple(
    group + "_" + field
    for group in GROUPS
    for field in (
        "count", "cumulative_total_payoff", "cumulative_mean_payoff",
        "time_mean_payoff", "early_mean_payoff", "tail_mean_payoff",
    )
) + (
    "observed_steps", "early_observed_steps", "tail_observed_steps",
    "time_mean_cooperation_rate", "first_extinction_round", "extinction_observed",
    "extinction_events", "cooperation_revivals", "final_extinction_start_round",
    "extinct_at_end",
)


class PayoffTracker:
    """从第 0 轮开始连续观察；固定资格只在创建时读取一次。

early_window_steps 和 tail_window_steps 均为正整数。短实验的前后窗口可以
重叠，实际长度由 summary 中的 early/tail_observed_steps 给出。
空组的累计总收益为 0，人均收益为 None；没有更新轮时，各时间均值为 None。
"""

    def __init__(self, model, early_window_steps, tail_window_steps):
        for name, value in (
            ("early_window_steps", early_window_steps),
            ("tail_window_steps", tail_window_steps),
        ):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} 必须为正整数。")
        self._model = model
        access = model.llm_access_grid
        positions = tuple((row, col) for row, line in enumerate(access) for col in range(len(line)))
        self._positions = {
            "population": positions,
            "eligible": tuple(position for position in positions if access[position[0]][position[1]]),
            "ineligible": tuple(position for position in positions if not access[position[0]][position[1]]),
        }
        self._counts = {group: len(self._positions[group]) for group in GROUPS}
        self._cumulative_totals = dict.fromkeys(GROUPS, 0.0)
        self._early_totals = dict.fromkeys(GROUPS, 0.0)
        self._early_window_steps = early_window_steps
        self._tail = deque(maxlen=tail_window_steps)
        self._observed_steps = 0
        self._last_round = -1
        self._cooperation_sum = 0.0
        self._first_extinction_round = None
        self._extinction_events = 0
        self._cooperation_revivals = 0
        self._final_extinction_start_round = None
        self._extinct_at_end = False

    def observe(self, round_stats):
        """在该轮更新完成后调用，返回按 HISTORY_FIELDS 排列的独立字典。"""
        round_number = round_stats.round
        if type(round_number) is not int or round_number != self._last_round + 1:
            raise ValueError("observe 必须从 round=0 开始连续逐轮调用，不能重复或跳过轮次。")
        if round_number != self._model.round:
            raise ValueError("round_stats 与模型当前轮次不一致；请在每轮更新后立即观察。")
        cooperation = round_stats.cooperation_rate
        if not math.isfinite(cooperation) or not 0 <= cooperation <= 1:
            raise ValueError("cooperation_rate 必须是 [0, 1] 范围内的有限数值。")

        # 每个个体仅查询一次公开收益接口，不使用累计福利参与 Fermi 更新。
        payoffs = {
            position: self._model.payoff(*position)
            for position in self._positions["population"]
        }
        totals = {
            group: math.fsum(payoffs[position] for position in self._positions[group])
            for group in GROUPS
        }
        if round_number > 0:
            self._observed_steps += 1
            for group in GROUPS:
                self._cumulative_totals[group] = math.fsum((self._cumulative_totals[group], totals[group]))
                if self._observed_steps <= self._early_window_steps:
                    self._early_totals[group] = math.fsum((self._early_totals[group], totals[group]))
            self._tail.append(totals)
            self._cooperation_sum = math.fsum((self._cooperation_sum, cooperation))

        extinct = cooperation == 0
        if extinct and not self._extinct_at_end:
            self._extinction_events += 1
            self._final_extinction_start_round = round_number
            if self._first_extinction_round is None:
                self._first_extinction_round = round_number
        elif not extinct and self._extinct_at_end:
            self._cooperation_revivals += 1
            self._final_extinction_start_round = None
        self._extinct_at_end = extinct
        self._last_round = round_number

        history = {}
        for group in GROUPS:
            count = self._counts[group]
            history[group + "_mean_payoff"] = totals[group] / count if count else None
            history[group + "_cumulative_mean_payoff"] = (
                self._cumulative_totals[group] / count if count else None
            )
        return history

    def summary(self):
        """返回当前汇总快照；调用本方法不会新增观测或消耗随机数。"""
        early_steps = min(self._observed_steps, self._early_window_steps)
        tail_steps = len(self._tail)
        result = {}
        for group in GROUPS:
            count = self._counts[group]
            total = self._cumulative_totals[group]
            result[group + "_count"] = count
            result[group + "_cumulative_total_payoff"] = total
            result[group + "_cumulative_mean_payoff"] = total / count if count else None
            result[group + "_time_mean_payoff"] = (
                total / count / self._observed_steps if count and self._observed_steps else None
            )
            result[group + "_early_mean_payoff"] = (
                self._early_totals[group] / count / early_steps if count and early_steps else None
            )
            result[group + "_tail_mean_payoff"] = (
                math.fsum(item[group] / count / tail_steps for item in self._tail)
                if count and tail_steps else None
            )
        result.update(
            observed_steps=self._observed_steps,
            early_observed_steps=early_steps,
            tail_observed_steps=tail_steps,
            time_mean_cooperation_rate=(
                self._cooperation_sum / self._observed_steps if self._observed_steps else None
            ),
            first_extinction_round=self._first_extinction_round,
            extinction_observed=self._first_extinction_round is not None,
            extinction_events=self._extinction_events,
            cooperation_revivals=self._cooperation_revivals,
            final_extinction_start_round=self._final_extinction_start_round,
            extinct_at_end=self._extinct_at_end,
        )
        return result
