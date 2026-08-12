from __future__ import annotations

from common import MOCK_ENTERPRISE_DIR, PROJECT_ROOT, REPORTS_DIR
from build_distribution_features import build_distribution_features
from generate_enterprise_mock import generate_mock_enterprise_tables
from profile_data import profile_directory
from run_pipeline import main as run_demo_pipeline
from validate_distribution_data import validate


def main() -> None:
    run_demo_pipeline()
    generate_mock_enterprise_tables()
    build_distribution_features()
    profile_directory(MOCK_ENTERPRISE_DIR, REPORTS_DIR / "mock_data_profile.json")
    report = validate(
        MOCK_ENTERPRISE_DIR,
        PROJECT_ROOT / "config" / "distribution_schema.json",
        REPORTS_DIR / "distribution_validation.json",
    )
    if not report["valid"]:
        raise ValueError("8月4日三表关联校验失败，请查看distribution_validation.json")
    print("8月4日数据接入底座已完成：三张模拟企业表、数据体检、字段及关联校验均已生成。")


if __name__ == "__main__":
    main()
