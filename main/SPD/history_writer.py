"""仅控制历史 CSV 的列与采样间隔，不参与逐轮统计或模型更新。"""

import csv
from pathlib import Path


COMPACT_HISTORY_FIELDS = (
    "round",
    "cooperation_rate",
    "average_payoff",
    "eligible_mean_payoff",
    "ineligible_mean_payoff",
    "control_cooperation_rate",
    "control_average_payoff",
    "control_eligible_mean_payoff",
    "control_ineligible_mean_payoff",
    "eligible_trust_mean",
    "eligible_trust_std",
    "eligible_advice_rate",
    "eligible_feedback_reference_mean",
    "eligible_aspiration_mean",
    "advice_used",
    "trust_positive_feedback",
    "trust_negative_feedback",
    "trust_zero_feedback",
)


class HistoryWriter:
    """按 full/compact/none 模式保存指定轮次，始终保留初始和最终轮。

    调用方仍应每轮计算统计；every 仅影响文件采样，不能用来跳过模型观测。
    full 保留首条已写入记录的全部字段，compact 使用固定核心字段，none
    不创建文件。对象应通过 with 使用，fields 和 rows_written 可用于输出元数据。
    """

    def __init__(self, path, mode="full", every=1, total_steps=0):
        if mode not in ("full", "compact", "none"):
            raise ValueError("history mode 必须为 full、compact 或 none。")
        if type(every) is not int or every <= 0:
            raise ValueError("history every 必须为正整数。")
        if type(total_steps) is not int or total_steps < 0:
            raise ValueError("history total_steps 必须为非负整数。")
        self.path = Path(path)
        self.mode = mode
        self.every = every
        self.total_steps = total_steps
        self.fields = list(COMPACT_HISTORY_FIELDS) if mode == "compact" else []
        self.rows_written = 0
        self._file = None
        self._writer = None
        self._active = False
        self._used = False

    def __enter__(self):
        if self._used:
            raise RuntimeError("HistoryWriter 不能重复进入上下文。")
        if self.mode != "none":
            self._file = self.path.open("w", newline="", encoding="utf-8")
        self._active = True
        self._used = True
        return self

    def should_write(self, round_number):
        """返回该轮是否采样；初始、最终轮不受采样间隔限制。"""
        if type(round_number) is not int or round_number < 0:
            raise ValueError("history round 必须为非负整数。")
        return self.mode != "none" and (
            round_number == 0
            or round_number == self.total_steps
            or round_number % self.every == 0
        )

    def write(self, row):
        """写入采样轮，返回是否实际写入；compact 缺少必需字段时直接报错。"""
        if not self._active:
            raise RuntimeError("请在 HistoryWriter 的 with 上下文中写入。")
        if not self.should_write(row["round"]):
            return False
        if self.mode == "compact":
            missing = [field for field in self.fields if field not in row]
            if missing:
                raise ValueError("compact history 缺少字段：" + ", ".join(missing))
            selected_row = {field: row[field] for field in self.fields}
        else:
            selected_row = row
        if self._writer is None:
            if self.mode == "full":
                self.fields = list(row)
            self._writer = csv.DictWriter(self._file, fieldnames=self.fields)
            self._writer.writeheader()
        self._writer.writerow(selected_row)
        self.rows_written += 1
        return True

    def flush(self):
        """刷新已缓冲的数据；none 模式下不执行文件操作。"""
        if self._file is not None:
            self._file.flush()

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            if self._file is not None:
                self._file.close()
        finally:
            self._active = False
        return False
