"""
加载 .env 文件。纯 Python 实现，不依赖外部包。
从当前目录向上查找 .env 文件，只设置尚未存在的环境变量。
"""

import os
from pathlib import Path


def load_dotenv(start_dir: str = None) -> str | None:
    """从 start_dir（默认当前目录）向上查找 .env 文件并加载。
    返回找到的 .env 文件路径，未找到返回 None。"""
    search_dir = Path(start_dir) if start_dir else Path.cwd()

    for directory in [search_dir] + list(search_dir.parents):
        env_file = directory / ".env"
        if env_file.exists():
            _parse_env_file(env_file)
            return str(env_file)
    return None


def _parse_env_file(filepath: Path):
    """解析 .env 文件。不覆盖已有环境变量。"""
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value
