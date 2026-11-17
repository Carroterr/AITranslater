import hashlib
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from time import monotonic

import httpx


MODELS_DIR = Path(__file__).resolve().parent / "models"


@dataclass(frozen=True)
class ModelAssets:
    label: str
    repository: str
    revision: str
    files: dict[str, tuple[int, str]]

    @property
    def directory(self):
        return MODELS_DIR / self.repository.rsplit("/", 1)[-1]

    def prepare(self, stop, status):
        missing = []
        total = sum(size for size, _ in self.files.values())
        completed = total
        for name, (size, expected) in self.files.items():
            path = self.directory / name
            if not path.is_file() or path.stat().st_size != size:
                missing.append((name, path, size, expected))
                completed -= size
        if not missing or stop.is_set():
            return

        self.directory.mkdir(parents=True, exist_ok=True)
        base_url = f"https://huggingface.co/{self.repository}/resolve/{self.revision}"
        last_update = 0.0
        with httpx.Client(follow_redirects=True, timeout=20) as client:
            for name, path, size, expected in missing:
                if stop.is_set():
                    return
                temp_path = None
                received = 0
                # Hugging Face reports Git blob IDs for small files and SHA-256 for LFS files.
                digest = hashlib.sha1(f"blob {size}\0".encode()) if len(expected) == 40 else hashlib.sha256()
                try:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    status(f"正在下载{self.label} · {completed * 100 // total}%")
                    with client.stream("GET", f"{base_url}/{name}") as response:
                        response.raise_for_status()
                        with NamedTemporaryFile(dir=path.parent, suffix=".part", delete=False) as temp:
                            temp_path = Path(temp.name)
                            for chunk in response.iter_bytes(chunk_size=128 * 1024):
                                if stop.is_set():
                                    return
                                received += len(chunk)
                                if received > size:
                                    raise ValueError(f"{path.name} 文件大小异常")
                                temp.write(chunk)
                                digest.update(chunk)
                                now = monotonic()
                                if now - last_update >= 0.25:
                                    percent = (completed + received) * 100 // total
                                    status(f"正在下载{self.label} · {percent}%")
                                    last_update = now
                    if received != size or digest.hexdigest() != expected:
                        raise ValueError(f"{path.name} 文件校验失败")
                    temp_path.replace(path)
                except (httpx.HTTPError, OSError, ValueError) as exc:
                    raise RuntimeError(f"{self.label}下载失败，请检查网络后重试。{exc}") from exc
                finally:
                    if temp_path is not None:
                        temp_path.unlink(missing_ok=True)
                completed += size
