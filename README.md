# APXInf RTX 4090 教程配套脚本

配合 APXInf / π0.5 本地部署教程使用，适配 [RLinf/APXinf-robo](https://github.com/RLinf/APXinf-robo) 及其 `apxinf/` 推理引擎子模块。提供 CUDA 开发组件安装、SM89 兼容补丁、模型资产清单、最小推理示例和推理耗时测量。这些是教程辅助文件。

目标环境：Linux x86_64、RTX 4090 24GB、Python 3.11、CUDA 12.8，使用 π0.5 LIBERO 权重和 BF16 精度。运行推理前，请按教程或[上游构建说明](https://github.com/RLinf/APXinf-robo#build-apxinf-robo)安装驱动、Rust、构建工具，并完成 APXInf 的编译安装和 GPU 检查。SM89 补丁及应用脚本已包含在本仓库，遇到下述特定导入错误时，按第 3 节处理。

## 文件说明

| 文件 | 用途 |
| --- | --- |
| `install_cuda_components.py` | 从 NVIDIA 官方归档下载并校验 CUDA 12.8.1 开发组件，安装到 `$APX_RUNTIME/cuda-12.8`；驱动需另行安装 |
| `sm89-gdn-compat.patch` | 固定引擎版本的 SM89 局部兼容补丁，处理 `apxinf_flashinfer_gdn_launch` 符号缺失 |
| `apply_sm89_compat.sh` | 检查 GPU 计算能力及引擎提交，应用上述补丁，并向 `env.sh` 追加编译所需的环境变量 |
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

如果 `bootstrap` 已有文件，可以下载本仓库 ZIP，解压后将上表中的六个文件直接放入 `bootstrap`。如果该目录已是本仓库的 Git 克隆，可用 `git -C "$HOME/apxinf-runtime/bootstrap" pull --ff-only` 更新。目录布局建议如下：

```text
~/apxinf-runtime/
├── bootstrap/           # 本仓库配套文件
├── env.sh               # 按教程创建的环境配置，本仓库未提供
├── APXinf-robo/         # 单独克隆的上游源码，含 apxinf/ 子模块
├── venv/                # 教程中创建的 Python 虚拟环境
├── models/              # 模型权重和附属资产
└── results/             # 每次推理的输出
```

后续命令在同一终端执行。重新打开终端时，重新设置 `APX_RUNTIME`、激活安装了 APXInf 的虚拟环境，并恢复 CUDA 环境变量。如已按教程创建 `env.sh`，可执行 `source "$HOME/apxinf-runtime/env.sh"` 加载；第 3 节的补丁应用脚本依赖这个文件。

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

## 3. 可选：处理 SM89 原生扩展导入错误

完成编译安装后，若 Python 导入检查已输出 `IMPORT_OK`，可直接继续准备模型。若遇到：

```text
undefined symbol: apxinf_flashinfer_gdn_launch
```

且 GPU 计算能力为 `8.9`、引擎提交为 `7f37642dd6c5896bca66d7640d9944c2a3fe04b4`，可使用本仓库中的两个补丁文件。其他版本或错误需另行排查。

先按教程创建 `~/apxinf-runtime/env.sh`，其中需设置 `APX_RUNTIME`、`APX_PROJECT`、`CARGO_TARGET_DIR`、CUDA 环境及已安装 APXInf 的 Python 虚拟环境。应用脚本固定从 `~/apxinf-runtime/env.sh` 加载配置，并在 `$APX_RUNTIME/bootstrap` 查找补丁；请将两个文件都放在该目录。

```bash
source "$HOME/apxinf-runtime/env.sh"
bash "$APX_RUNTIME/bootstrap/apply_sm89_compat.sh"
```

检查通过并完成后会输出 `SM89_COMPAT_READY`；若已应用过补丁，还会输出 `SM89_PATCH_ALREADY_APPLIED`。如果 GPU、版本或补丁检查失败，请先核对环境和源码，暂时不要继续编译。

脚本会自动向 `$APX_RUNTIME/env.sh` 追加 `export APXINF_SM89_GDN_COMPAT=1`，无需提前手工加入。由于脚本在子进程中运行，当前终端需重新加载配置，再重新编译、安装原生扩展并检查导入：

```bash
source "$APX_RUNTIME/env.sh"
cd "$APX_PROJECT"

maturin build --release --locked --features cuda --auditwheel skip \
  -m apxinf/crates/apxinf-py/Cargo.toml
python -m pip install --force-reinstall \
  "$CARGO_TARGET_DIR"/wheels/apxinf_py-*.whl

python -c 'import apxinf_py, apxinf, apxinf_robo; print("IMPORT_OK", apxinf_py.__version__, apxinf_robo.__version__)'
```

看到 `IMPORT_OK` 后，继续教程中的随机权重 GPU 检查，通过后再进入真实模型推理。这份补丁仅为上述固定版本的 SM89 构建补齐 C ABI 入口，对 GDN 调用返回 `801`（不支持）；本次 π0.5 推理不使用该路径。

## 4. 准备并校验模型

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

## 5. 运行推理

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
