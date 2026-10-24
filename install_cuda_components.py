#!/usr/bin/env python3
"""Install the APXInf CUDA 12.8 development components from NVIDIA archives."""
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import subprocess

base = Path(os.environ.get("APX_RUNTIME", str(Path.home() / "apxinf-runtime")))
cache = base / "bootstrap" / "cuda-downloads"
prefix = base / "cuda-12.8"
cache.mkdir(parents=True, exist_ok=True)
prefix.mkdir(parents=True, exist_ok=True)
manifest_file = base / "bootstrap" / "cuda-redistrib-12.8.1.json"
root = "https://developer.download.nvidia.com/compute/cuda/redist/"
subprocess.run(["curl", "-fsSL", "--retry", "3", "--connect-timeout", "15", "--max-time", "120",
                root + "redistrib_12.8.1.json", "-o", str(manifest_file)], check=True)
manifest = json.loads(manifest_file.read_text())
parts = ["cuda_nvcc", "cuda_cudart", "cuda_cccl", "cuda_nvtx", "libcublas"]

def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def download(name):
    meta = manifest[name]["linux-x86_64"]
    path = cache / Path(meta["relative_path"]).name
    if not path.exists() or path.stat().st_size != int(meta["size"]) or sha256(path) != meta["sha256"]:
        print("DOWNLOAD", name, meta["size"], flush=True)
        subprocess.run(["curl", "-fsSL", "--retry", "3", "--connect-timeout", "15", "--max-time", "1800",
                        root + meta["relative_path"], "-o", str(path)], check=True)
    if sha256(path) != meta["sha256"]:
        raise RuntimeError("SHA256 mismatch: " + name)
    print("SHA256_OK", name, flush=True)
    return name, path

with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
    downloads = list(pool.map(download, parts))
for name, path in downloads:
    subprocess.run(["tar", "-xJf", str(path), "--strip-components=1", "-C", str(prefix)], check=True)
    print("INSTALLED", name, manifest[name]["version"], flush=True)
if not (prefix / "lib64").exists():
    (prefix / "lib64").symlink_to("lib", target_is_directory=True)
print("CUDA_COMPONENTS_OK", prefix, flush=True)

