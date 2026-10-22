"""大模型建议接口及本地已采集回答的抽样器；不读取密钥或发起网络请求。

EmpiricalAdvisor 自动识别完整 3×3 布局表和旧的四邻居计数表，有放回抽样原始回答。
也可以实现 request_advice(context)，或把自定义函数传给仿真器的 advisor。
返回 "C" / "D" 表示采用该策略；返回 None 表示没有建议，回退到 Fermi。
请求异常和非法返回值会终止实验，避免悄悄把失败请求当成有效实验数据。
"""

import csv
import hashlib
import io
import json
import math
import random
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Literal, Optional, Protocol


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
    grid_rows 为以本人居中的 3×3 观察窗口，北到南三行，每行西到东三个 C/D。
    四个角仅提供观察信息；neighbors 仍只含直接博弈的四邻居。
    """

    round: int
    position: Position
    strategy: Strategy
    payoff: float
    b: float
    k: float
    neighbors: tuple[NeighborObservation, ...]
    grid_rows: Optional[tuple[str, str, str]] = None

    @property
    def grid_key(self) -> Optional[str]:
        return "".join(self.grid_rows) if self.grid_rows is not None else None

    @property
    def cooperating_neighbors(self) -> int:
        return sum(neighbor.strategy == "C" for neighbor in self.neighbors)

    @property
    def defecting_neighbors(self) -> int:
        return len(self.neighbors) - self.cooperating_neighbors


class Advisor(Protocol):
    def __call__(self, context: AdviceContext) -> Optional[Strategy]:
        ...


class EmpiricalAdvisor:
    """从完成的 LLM 批次中抽样；每组保留全部 n 条回答及其经验概率。

    仿真中的一次咨询独立、有放回地抽取一条既有回答，不消耗 API 额度。
    grid_3x3 按完整有方向的九格布局匹配；neighbor_counts 按自身策略和四邻居计数匹配。
    b、payoff、坐标、轮数不参与匹配；不合并旋转、镜像或相同计数的不同布局。
    旧批次的 b 仅保留为数据来源记录；新批次可省略 b，仿真收益仍由实验的 b 决定。
    数据不完整或局面不匹配时直接报错，不返回 None，也不回退到 Fermi。
    """

    _REQUIRED_COLUMNS = {
        "request_index", "current_action", "cooperating_neighbors",
        "defecting_neighbors", "status", "recommendation",
    }

    def __init__(self, batch_dir, seed: int = 42):
        if type(seed) is not int:
            raise ValueError("LLM 抽样 seed 必须是整数。")
        self.batch_dir = Path(batch_dir).expanduser().resolve()
        self.seed = seed
        source_bytes = {}
        for filename in ("config.json", "results.csv"):
            path = self.batch_dir / filename
            try:
                source_bytes[filename] = path.read_bytes()
            except OSError as exc:
                raise ValueError(f"无法读取本地 LLM 数据文件：{path}") from exc
        try:
            config = json.loads(source_bytes["config.json"].decode("utf-8-sig"))
        except (ValueError, UnicodeError) as exc:
            raise ValueError("LLM 批次 config.json 不是有效的 UTF-8 JSON。") from exc
        if not isinstance(config, dict):
            raise ValueError("LLM 批次 config.json 必须是对象。")
        self.b = self._parse_b(config["b"]) if config.get("b") is not None else None
        self.observation_format = config.get("observation_format", "neighbor_counts")
        if self.observation_format not in ("neighbor_counts", "grid_3x3"):
            raise ValueError("LLM observation_format 必须为 neighbor_counts 或 grid_3x3。")
        if self.observation_format == "grid_3x3":
            self._keys = tuple("".join(cells) for cells in product("CD", repeat=9))
        else:
            self._keys = tuple((action, c, 4 - c) for action in ("C", "D") for c in range(5))
        self.case_count = len(self._keys)
        case_count = config.get("case_count", self.case_count)
        if type(case_count) is not int or case_count != self.case_count:
            raise ValueError(f"LLM 批次 case_count 必须为 {self.case_count}。")
        self.n = config.get("n")
        if type(self.n) is not int or self.n < 1:
            raise ValueError("LLM 批次 config.json 的 n 必须是正整数。")
        planned = config.get("planned_requests", self.case_count * self.n)
        if type(planned) is not int or planned != self.case_count * self.n:
            raise ValueError(f"LLM 批次 planned_requests 必须等于 {self.case_count}*n。")

        if "cases" in config:
            cases = config["cases"]
            if not isinstance(cases, list) or len(cases) != self.case_count:
                raise ValueError(f"LLM 批次 config.json 的 cases 必须包含 {self.case_count} 种组合。")
            if {self._case_key(case) for case in cases} != set(self._keys):
                raise ValueError(f"LLM 批次 config.json 的 cases 必须完整覆盖 {self.case_count} 种组合。")

        samples = {key: [] for key in self._keys}
        seen_indices = set()
        try:
            reader = csv.DictReader(
                io.StringIO(source_bytes["results.csv"].decode("utf-8-sig"), newline=""),
                strict=True,
            )
            columns = reader.fieldnames or []
            required = self._REQUIRED_COLUMNS
            if self.observation_format == "grid_3x3":
                required = required | {"observation_format", "grid_key", "grid_rows"}
            if len(set(columns)) != len(columns) or not required <= set(columns):
                raise ValueError("LLM results.csv 缺少必要列或存在重复列名。")
            for line_number, row in enumerate(reader, start=2):
                if None in row:
                    raise ValueError(f"LLM results.csv 第 {line_number} 行列数不正确。")
                index = self._parse_int(row.get("request_index"), "request_index")
                if index in seen_indices:
                    raise ValueError(f"LLM results.csv 的 request_index={index} 重复。")
                if not 1 <= index <= self.case_count * self.n:
                    raise ValueError(f"LLM results.csv 的 request_index={index} 超出批次范围。")
                seen_indices.add(index)
                if row.get("status") != "success":
                    raise ValueError(f"LLM 请求 {index} 尚未成功；请先补齐该批次再接入仿真。")
                action = row.get("recommendation")
                if action not in ("C", "D"):
                    raise ValueError(f"LLM 请求 {index} 的 recommendation 必须为 C 或 D。")
                if "output_text" in row and (row["output_text"] or "").strip() != action:
                    raise ValueError(f"LLM 请求 {index} 的原始输出与 recommendation 不一致。")
                samples[self._case_key(row)].append((index, action))
        except (csv.Error, UnicodeError) as exc:
            raise ValueError("LLM results.csv 不是有效的 UTF-8 CSV。") from exc

        self._samples = {}
        for key, answers in samples.items():
            if len(answers) != self.n:
                raise ValueError(
                    f"LLM 组合 {key} "
                    f"需要 {self.n} 条成功回答，实际 {len(answers)} 条。"
                )
            # 按请求序号排序，使 CSV 行的排列变化不影响同种子的抽样序列。
            self._samples[key] = tuple(action for _, action in sorted(answers))
        self._rng = random.Random(f"empirical-advisor:{seed}")
        self._usage = {key: {"C": 0, "D": 0} for key in self._keys}
        self._model = config.get("model")
        self._batch_id = config.get("batch_id", self.batch_dir.name)
        self._file_hashes = {
            name: hashlib.sha256(content).hexdigest() for name, content in source_bytes.items()
        }
        prompt = config.get("system_prompt")
        self._prompt_hash = (
            hashlib.sha256(prompt.encode("utf-8")).hexdigest() if isinstance(prompt, str) else None
        )

    @staticmethod
    def _parse_b(value) -> float:
        try:
            b = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("LLM 数据的 b 必须是大于 1 的有限数值。") from exc
        if isinstance(value, bool) or not math.isfinite(b) or b <= 1:
            raise ValueError("LLM 数据的 b 必须是大于 1 的有限数值。")
        return b

    @staticmethod
    def _parse_int(value, name) -> int:
        if type(value) is int:
            return value
        if isinstance(value, str):
            try:
                return int(value)
            except ValueError:
                pass
        raise ValueError(f"LLM 数据的 {name} 必须是整数。")

    @staticmethod
    def _grid_key(grid_rows) -> str:
        """配置使用三行列表，CSV 使用斜杠分隔；统一成保留方位的九字符键。"""
        if isinstance(grid_rows, str):
            grid_rows = grid_rows.split("/")
        if (not isinstance(grid_rows, (list, tuple)) or len(grid_rows) != 3
                or any(not isinstance(row, str) or len(row) != 3 or set(row) - {"C", "D"}
                       for row in grid_rows)):
            raise ValueError("LLM 3×3 grid_rows 必须为三行，每行恰好三个 C/D。")
        return "".join(grid_rows)

    def _case_key(self, case):
        if not isinstance(case, dict):
            raise ValueError("LLM 组合必须是包含策略和邻居数的对象。")
        if "observation_format" in case and case["observation_format"] != self.observation_format:
            raise ValueError("LLM 结果中的 observation_format 与批次 config.json 不一致。")
        # 兼容含 b 的旧数据，仍检查其内部一致性；b 不用于匹配仿真局面。
        if (self.b is not None and case.get("b") not in (None, "")
                and self._parse_b(case["b"]) != self.b):
            raise ValueError("LLM 结果中的 b 与批次 config.json 不一致。")
        action = case.get("current_action")
        c = self._parse_int(case.get("cooperating_neighbors"), "cooperating_neighbors")
        d = self._parse_int(case.get("defecting_neighbors"), "defecting_neighbors")
        if action not in ("C", "D") or not 0 <= c <= 4 or not 0 <= d <= 4 or c + d != 4:
            raise ValueError("LLM 组合必须为当前策略 C/D，且合作、背叛邻居数为 0～4、总和为 4。")
        if self.observation_format == "grid_3x3":
            key = self._grid_key(case.get("grid_rows"))
            if case.get("grid_key") != key:
                raise ValueError("LLM 结果中的 grid_key 与 grid_rows 不一致。")
            if key[4] != action or sum(key[i] == "C" for i in (1, 3, 5, 7)) != c:
                raise ValueError("LLM 3×3 布局与中心策略或四邻居计数不一致。")
            return key
        if case.get("grid_key") or case.get("grid_rows"):
            raise ValueError("包含3×3布局的数据必须使用 grid_3x3 observation_format。")
        return action, c, d

    def validate_config(self, b: float, boundary: str) -> None:
        """校验周期边界；保留 b 入参以兼容已有调用，但不限制其值。"""
        if boundary != "periodic":
            raise ValueError("本地 LLM 策略表目前仅支持 periodic 周期边界。")

    def __call__(self, context: AdviceContext) -> Strategy:
        if context.strategy not in ("C", "D") or len(context.neighbors) != 4:
            raise ValueError("本地 LLM 建议要求个体策略为 C/D，且恰好有四个邻居。")
        if any(neighbor.strategy not in ("C", "D") for neighbor in context.neighbors):
            raise ValueError("当前局面的邻居策略只能为 C 或 D。")
        if self.observation_format == "grid_3x3":
            key = self._grid_key(context.grid_rows)
            if (key[4] != context.strategy
                    or sum(key[i] == "C" for i in (1, 3, 5, 7)) != context.cooperating_neighbors):
                raise ValueError("当前3×3观察与自身策略或直接邻居计数不一致。")
        else:
            key = (context.strategy, context.cooperating_neighbors, context.defecting_neighbors)
        action = self._rng.choice(self._samples[key])
        self._usage[key][action] += 1
        return action

    def metadata(self) -> dict:
        """返回来源和抽样配置；b 是源批次记录（可为空），不是匹配条件。"""
        return {
            "source": "local_llm_batch", "batch_dir": str(self.batch_dir),
            "batch_id": self._batch_id, "model": self._model, "b": self.b,
            "n": self.n, "seed": self.seed, "sampling": "uniform_with_replacement",
            "observation_format": self.observation_format, "case_count": self.case_count,
            "matching_fields": (["grid_key"] if self.observation_format == "grid_3x3"
                                else ["current_action", "cooperating_neighbors", "defecting_neighbors"]),
            "grid_layout": ("row_major_north_to_south_west_to_east; center_index=4; no_symmetry_merging"
                            if self.observation_format == "grid_3x3" else None),
            "interaction_neighborhood": "von_neumann_4",
            "system_prompt_sha256": self._prompt_hash,
            "source_files": {
                name: {"path": str(self.batch_dir / name), "sha256": digest}
                for name, digest in self._file_hashes.items()
            },
        }

    def table_rows(self) -> list[dict]:
        """原始回答的经验分布；b 仅记录源批次值，每次返回新的字典。"""
        rows = []
        for key in self._keys:
            if self.observation_format == "grid_3x3":
                action = key[4]
                c = sum(key[i] == "C" for i in (1, 3, 5, 7))
                d = 4 - c
            else:
                action, c, d = key
            answers = self._samples[key]
            row = {
                "b": self.b, "current_action": action,
                "cooperating_neighbors": c, "defecting_neighbors": d,
                "samples": len(answers), "C": answers.count("C"), "D": answers.count("D"),
                "p_C": answers.count("C") / len(answers),
            }
            if self.observation_format == "grid_3x3":
                row.update(observation_format=self.observation_format,
                           grid_key=key, grid_rows="/".join((key[:3], key[3:6], key[6:])))
            rows.append(row)
        return rows

    def usage_rows(self) -> list[dict]:
        """本次仿真每组咨询次数及抽到的 C/D 次数，不累计保存事件明细。"""
        rows = self.table_rows()
        for row, key in zip(rows, self._keys):
            counts = self._usage[key]
            row.update(draws=sum(counts.values()), selected_C=counts["C"], selected_D=counts["D"])
        return rows


def request_advice(context: AdviceContext) -> Optional[Strategy]:
    """预留接口：将 context 转成提示词，并返回经过严格校验的 C 或 D。

    当前不调用任何模型。允许显式返回 None，但不要把任意错误输出转换为 D。
    """
    return None
