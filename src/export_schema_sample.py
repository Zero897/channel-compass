from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


FILES = (
    "销售流水.csv",
    "业务回款明细.csv",
    "应收快照_月末24期.csv",
    "库龄快照_季末8期.csv",
    "增值合同签约明细.csv",
    "展期记录.csv",
    "客户授信.csv",
)


def export_schema_sample(source: Path, output: Path, rows: int = 20) -> None:
    output.mkdir(parents=True, exist_ok=True)
    for filename in FILES:
        frame = pd.read_csv(source / filename, dtype="string", nrows=rows)
        frame.to_csv(output / filename, index=False, encoding="utf-8-sig")
    (output / "README.md").write_text(
        "# 企业字段结构样例\n\n"
        "本目录每张表仅保留前20行，用于查看字段、编码和日期格式，不用于训练、模型评价或业务结论。"
        "完整企业脱敏模拟数据不进入提交压缩包。\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="导出七表字段结构小样")
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--rows", type=int, default=20)
    args = parser.parse_args()
    export_schema_sample(args.source, args.output, args.rows)
    print(f"字段结构样例已导出到：{args.output.resolve()}")


if __name__ == "__main__":
    main()
