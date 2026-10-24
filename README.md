# APXInf RTX 4090 教程配套脚本

配合 APXInf / π0.5 本地部署教程使用，适配 [RLinf/APXinf-robo](https://github.com/RLinf/APXinf-robo) 及其 `apxinf/` 推理引擎子模块。提供 CUDA 开发组件安装、模型资产清单、最小推理示例和推理耗时测量。这些是教程辅助脚本。

目标环境：Linux x86_64、RTX 4090 24GB、Python 3.11、CUDA 12.8，使用 π0.5 LIBERO 权重和 BF16 精度。运行推理前，请按教程或[上游构建说明](https://github.com/RLinf/APXinf-robo#build-apxinf-robo)安装驱动、Rust、构建工具，并完成 APXInf 的编译安装和 GPU 检查。本仓库只包含下列配套文件，源码构建及可能需要的 SM89 兼容补丁请参照对应版本教程。

## 文件说明

| 文件 | 用途 |
| --- | --- |
| `install_cuda_components.py` | 从 NVIDIA 官方归档下载并校验 CUDA 12.8.1 开发组件，安装到 `$APX_RUNTIME/cuda-12.8`；驱动需另行安装 |
| `model-assets.lock.json` | 记录四项模型资产的来源、固定 revision、大小和 SHA256 |
| `quickstart_real.py` | 最小真实权重调用示例，检查动作输出并保存 `actions.npy`、`summary.json` |
| `first_infer.py` | 记录模型加载、首次调用、预热与多次测量，保存动作、逐次耗时及汇总 |

教程对应版本：APXinf-robo [`835f68a`](https://github.com/RLinf/APXinf-robo/tree/835f68af3d59d48d6bf4a724191127d4434d8cbb)，引擎 `7f37642dd6c5896bca66d7640d9944c2a3fe04b4`；模型 [`lerobot/pi05_libero_base`](https://huggingface.co/lerobot/pi05_libero_base/tree/5fcf06fc42b6717e22ab885fd381a096574b12c7)，revision `5fcf06fc42b6717e22ab885fd381a096574b12c7`。使用其他版本时，请核对接口及资产兼容性。

## 1. 下载到哪个目录

建议把本仓库下载到 `~/apxinf-runtime/bootstrap`。在该目录尚不存在或为空时执行：

```bash
mkdir -p "$HOME/apxinf-runtime"
git clone https://github.com/zxw1992/APXinf-scripts-on4090.git \
  "$HOME/apxinf-runtime/bootstrap"
export APX_RUNTIME="$HOME/apxinf-runtime"
```

如果 `bootstrap` 已有文件，可以下载本仓库 ZIP，解压后将上表中的四个文件直接放入 `bootstrap`。目录布局建议如下：

```text
~/apxinf-runtime/
├── bootstrap/           # 本仓库配套文件
├── APXinf-robo/         # 单独克隆的上游源码，含 apxinf/ 子模块
├── venv/                # 教程中创建的 Python 虚拟环境
├── models/              # 模型权重和附属资产
└── results/             # 每次推理的输出
```

后续命令在同一终端执行。重新打开终端时，重新设置 `APX_RUNTIME`、激活安装了 APXInf 的虚拟环境，并恢复 CUDA 环境变量。

## 2. 可选：安装 CUDA 开发组件

已有可用 CUDA 12.8 开发环境可跳过。选择本脚本安装时，需要 Python 3、`curl`、`tar` 和 `xz`：

```bash
python3 "$APX_RUNTIME/bootstrap/install_cuda_components.py"
export CUDA_HOME="$APX_RUNTIME/cuda-12.8"
export CUDA_PATH="$CUDA_HOME"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
nvcc --version
```

完成标志为 `CUDA_COMPONENTS_OK`。安装目录默认为 `~/apxinf-runtime/cuda-12.8`，也可通过 `APX_RUNTIME` 更改；随后使用这个路径编译 APXInf。

## 3. 准备并校验模型

以下操作假定 APXInf 已完成编译安装。激活教程中创建的环境，下载固定版本资产；权重约 14.5 GB，需预留足够磁盘空间：

```bash
source "$APX_RUNTIME/venv/bin/activate"
export APX_MODEL_DIR="$APX_RUNTIME/models/pi05_libero_base-5fcf06f"
mkdir -p "$APX_MODEL_DIR"
APX_MODEL_URL='https://huggingface.co/lerobot/pi05_libero_base/resolve/5fcf06fc42b6717e22ab885fd381a096574b12c7'
curl -fL --retry 3 "$APX_MODEL_URL/config.json" -o "$APX_MODEL_DIR/config.json"
curl -fL --retry 3 -C - "$APX_MODEL_URL/model.safetensors" -o "$APX_MODEL_DIR/model.safetensors"
curl -fL --retry 3 https://storage.googleapis.com/openpi-assets/checkpoints/pi05_libero/assets/physical-intelligence/libero/norm_stats.json \
  -o "$APX_MODEL_DIR/norm_stats.json"
curl -fL --retry 3 https://storage.googleapis.com/big_vision/paligemma_tokenizer.model \
  -o "$APX_MODEL_DIR/tokenizer.model"
```

按清单校验四个文件；全部通过后生成 `first_infer.py` 所需的 `verified-assets.json`：

```bash
python - <<'PY'
import hashlib
import json
import os
from pathlib import Path

model_dir = Path(os.environ["APX_MODEL_DIR"])
lock_path = Path(os.environ["APX_RUNTIME"]) / "bootstrap/model-assets.lock.json"
lock = json.loads(lock_path.read_text(encoding="utf-8"))
for asset in lock["assets"]:
    path = model_dir / asset["name"]
    if not path.is_file() or path.stat().st_size != asset["bytes"]:
        raise SystemExit(f"文件缺失或大小不符：{path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != asset["sha256"]:
        raise SystemExit(f"SHA256 不符：{path}")
    print("SHA256_OK", path.name)
(model_dir / "verified-assets.json").write_text(
    json.dumps(lock, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
)
print("REAL_WEIGHTS_VERIFIED")
PY
```

## 4. 运行推理

先运行最小示例：

```bash
python -u "$APX_RUNTIME/bootstrap/quickstart_real.py" \
  --model-dir "$APX_MODEL_DIR" \
  --out-dir "$APX_RUNTIME/results/quickstart-$(date -u +%Y%m%d-%H%M%S)"
```

看到 `QUICKSTART_REAL_WEIGHTS_OK`，并生成 `actions.npy` 和 `summary.json` 即通过。输出为 `(10, 7)`、`float32` 且数值有限的动作数组。

需要记录加载耗时、首次调用和稳定运行耗时，可再运行测量脚本：

```bash
python -u "$APX_RUNTIME/bootstrap/first_infer.py" \
  --model-dir "$APX_MODEL_DIR" \
  --out-dir "$APX_RUNTIME/results/first-infer-$(date -u +%Y%m%d-%H%M%S)" \
  --warmup 10 --samples 30
```

完成标志为 `REAL_CHECKPOINT_FIRST_INFER_VERIFIED`，结果目录包含 `first_actions.npy`、`first-call.json`、`measurements.jsonl`、`summary.json` 等。每次运行必须指定一个尚不存在的新输出目录。

两个示例均使用真实权重、黑色图像、零状态和固定文本指令，用于检查推理接口与计时。机器人任务效果需另用真实观察或仿真评测。

## 许可证

本仓库配套脚本采用 [MIT License](LICENSE)。APXinf-robo、APXInf、模型权重及 NVIDIA 组件分别遵循各自的许可证。
