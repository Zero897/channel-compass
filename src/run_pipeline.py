from __future__ import annotations

from build_features import build
from demo_flow import run_demo
from export_feishu import export
from generate_data import generate


def main() -> None:
    generate()
    build()
    export()
    run_demo()
    print("渠智罗盘 MVP 已生成：3个客户、5个SKU、10条风险事件、1条C003本地闭环。")


if __name__ == "__main__":
    main()
