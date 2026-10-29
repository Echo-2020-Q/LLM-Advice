"""汇总资格者信任与采纳反馈；观测不修改模型或随机流。"""

from collections import deque
import math


TRUST_POINT_FIELDS = (
    "eligible_trust_mean", "eligible_trust_std", "eligible_trust_min", "eligible_trust_max",
    "eligible_low_trust_fraction", "eligible_high_trust_fraction", "eligible_aspiration_mean",
    "eligible_feedback_payoff_mean",
) + tuple(f"eligible_trust_bin_{index}" for index in range(10))
TRUST_FEEDBACK_FIELDS = (
    "trust_positive_feedback", "trust_negative_feedback", "trust_zero_feedback",
)
TRUST_HISTORY_FIELDS = TRUST_POINT_FIELDS + ("eligible_advice_rate",) + TRUST_FEEDBACK_FIELDS
TRUST_SUMMARY_FIELDS = (
    "initial_eligible_trust_mean",
) + tuple("final_" + field for field in TRUST_POINT_FIELDS) + (
    "time_mean_eligible_trust", "early_mean_eligible_trust", "tail_mean_eligible_trust",
    "time_mean_eligible_advice_rate", "early_mean_eligible_advice_rate", "tail_mean_eligible_advice_rate",
) + tuple("total_" + field for field in TRUST_FEEDBACK_FIELDS)


class TrustTracker:
    """round0只记初态；时间平均使用round1..T，窗口与现有福利统计相同。"""

    def __init__(self, simulation, early_steps, tail_steps):
        self.simulation = simulation
        self.early_steps = early_steps
        self.tail = deque(maxlen=tail_steps)
        self.initial = None
        self.final = None
        self._last_round = -1
        self._sums = {"eligible_trust_mean": 0.0, "eligible_advice_rate": 0.0}
        self._counts = dict.fromkeys(self._sums, 0)
        self._early_sums = dict(self._sums)
        self._early_counts = dict(self._counts)
        self.totals = dict.fromkeys(TRUST_FEEDBACK_FIELDS, 0)

    def observe(self, round_number):
        if round_number != self._last_round + 1:
            raise ValueError("信任统计必须从round0开始，逐轮观测且不能重复。")
        values = self.simulation.trust_statistics()
        self._last_round = round_number
        if round_number == 0:
            self.initial = dict(values)
        else:
            self.tail.append({field: values[field] for field in self._sums})
            for field in self._sums:
                value = values[field]
                if value is not None:
                    self._sums[field] += value
                    self._counts[field] += 1
                    if round_number <= self.early_steps:
                        self._early_sums[field] += value
                        self._early_counts[field] += 1
            for field in self.totals:
                self.totals[field] += values[field]
        self.final = dict(values)
        return values

    def summary(self):
        if self.initial is None:
            raise ValueError("汇总前必须观测round0。")
        result = {"initial_eligible_trust_mean": self.initial["eligible_trust_mean"]}
        result.update({"final_" + field: self.final[field] for field in TRUST_POINT_FIELDS})
        for field, suffix in (("eligible_trust_mean", "eligible_trust"),
                              ("eligible_advice_rate", "eligible_advice_rate")):
            count = self._counts[field]
            early_count = self._early_counts[field]
            result["time_mean_" + suffix] = self._sums[field] / count if count else None
            result["early_mean_" + suffix] = (
                self._early_sums[field] / early_count if early_count else None
            )
            tail_values = [item[field] for item in self.tail if item[field] is not None]
            result["tail_mean_" + suffix] = (
                math.fsum(value / len(tail_values) for value in tail_values) if tail_values else None
            )
        result.update({"total_" + field: value for field, value in self.totals.items()})
        return result
