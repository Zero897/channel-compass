from __future__ import annotations

import argparse
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from build_submission_package import _package_files
from project_paths import PROJECT_ROOT, portable_path, sha256_file


ENTERPRISE_DATA_DIR = PROJECT_ROOT / "data" / "company" / "AFFT模拟数据集"


def collaborator_files() -> list[Path]:
    files = _package_files()
    if not ENTERPRISE_DATA_DIR.exists():
        raise FileNotFoundError(f"协作者包缺少企业数据目录：{portable_path(ENTERPRISE_DATA_DIR)}")
    files.extend(path for path in ENTERPRISE_DATA_DIR.rglob("*") if path.is_file())
    unique = sorted(set(files), key=portable_path)
    forbidden = {".env", "data/feedback/task_feedback.csv"}
    leaked = [portable_path(path) for path in unique if portable_path(path) in forbidden]
    if leaked:
        raise ValueError(f"协作者包不得包含真实密钥或反馈文件：{', '.join(leaked)}")
    return unique


def build_collaborator_package(output_path: Path) -> dict[str, object]:
    files = collaborator_files()
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "sharing_scope": "仅限参赛团队内部使用，不得公开上传",
        "contains_enterprise_deidentified_simulated_data": True,
        "contains_real_credentials": False,
        "file_count": len(files),
        "files": [
            {
                "path": portable_path(path),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in files
        ],
        "quick_start": {
            "create_environment": "python -m venv .venv",
            "install": "& .\\.venv\\Scripts\\python.exe -m pip install -r requirements-all.txt",
            "frozen_inference": "& .\\.venv\\Scripts\\python.exe src\\run_frozen_company_pipeline.py",
            "portable_demo": "& .\\.venv\\Scripts\\python.exe src\\run_portable_demo.py",
            "backend": "复制.env.example为.env并安全填写凭据后：& .\\.venv\\Scripts\\python.exe src\\backend_api.py",
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(
        output_path,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=6,
        allowZip64=True,
    ) as archive:
        for path in files:
            archive.write(path, portable_path(path))
        archive.writestr(
            "COLLABORATOR_MANIFEST.json",
            json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf-8"),
        )
    return {
        "status": "pass",
        "output": portable_path(output_path),
        "file_count": len(files) + 1,
        "size_bytes": output_path.stat().st_size,
        "sha256": sha256_file(output_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="构建仅限团队内部传递的渠智罗盘完整运行包")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "dist" / "渠智罗盘_协作者可运行包_20260813.zip",
    )
    args = parser.parse_args()
    result = build_collaborator_package(args.output)
    print(
        f"协作者包完成：{result['file_count']}个文件，"
        f"{result['size_bytes'] / 1024 / 1024:.2f}MB，SHA256={result['sha256']}"
    )


if __name__ == "__main__":
    main()
