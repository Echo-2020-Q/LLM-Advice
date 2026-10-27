"""批次 CSV 快照保存；正式文件被占用时保存完整备用快照。"""

import csv
from pathlib import Path
import sys
import time
from uuid import uuid4


class SummaryWriter:
    """仅由主进程调用；成功返回表示完整快照已保存到 latest_path。"""

    def __init__(self, output, fieldnames):
        self.output = Path(output)
        self.fieldnames = list(fieldnames)
        self.primary_path = self.output / "summary.csv"
        self.fallback_path = self.output / "summary.pending.csv"
        self.latest_path = self.primary_path
        self.pending = False

    def save(self, rows):
        temporary = self.output / "summary.csv.tmp"
        # 写入失败必须抛出；不能截断之前成功保存的正式或备用快照。
        with temporary.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=self.fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        # 初次遇到占用短暂重试一次；持续占用时每个快照只尝试一次，避免拖慢长扫描。
        attempts = 1 if self.pending else 2
        for attempt in range(attempts):
            try:
                temporary.replace(self.primary_path)
            except PermissionError as exc:
                replace_error = exc
                if attempt + 1 < attempts:
                    time.sleep(0.1)
            else:
                was_pending = self.pending
                self.latest_path = self.primary_path
                self.pending = False
                if was_pending:
                    print(f"汇总文件已恢复更新：{self.primary_path}", flush=True)
                return self.latest_path

        # 正式文件可能被 Excel/WPS 等程序占用，也可能缺少替换权限。
        # 只有完整备用快照成功落盘，才允许批次继续。
        fallback = self.fallback_path
        try:
            temporary.replace(fallback)
        except PermissionError:
            # 备用 CSV 也被打开时使用新名称，不覆盖仍被占用的旧快照。
            fallback = self.output / f"summary.pending.{uuid4().hex}.csv"
            temporary.replace(fallback)

        announce = not self.pending or fallback != self.latest_path
        self.fallback_path = fallback
        self.latest_path = fallback
        self.pending = True
        if announce:
            print(
                f"暂时无法替换汇总文件 {self.primary_path}：{replace_error}\n"
                f"最新完整汇总已保存到：{fallback}；实验继续。\n"
                "请关闭占用汇总文件的程序或检查文件权限；后续保存会再次尝试更新 summary.csv。"
                "备用期间 summary.csv 可能不是最新结果。",
                file=sys.stderr, flush=True,
            )
        return self.latest_path
