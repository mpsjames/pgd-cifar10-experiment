# ⚔️ PGD Adversarial Attack & Defense — A Multi-Architecture Study on CIFAR-10

> Một nghiên cứu **có thể tái lập đầy đủ** về tấn công đối kháng PGD/APGD, adversarial
> training, và transferability — chạy chéo qua hai họ kiến trúc tiêu biểu (CNN ↔
> Transformer) trên CIFAR-10.
>
> *A fully reproducible study of PGD/APGD adversarial attacks, adversarial training,
> and cross-architecture transferability on CIFAR-10.*

---

## 🇻🇳 Phiên bản tiếng Việt

> *English version below — cuộn xuống phần "🇬🇧 English version".*

### Giới thiệu

Dự án này triển khai một thí nghiệm robustness hoàn chỉnh trên CIFAR-10:

- **Tấn công**: FGSM · BIM · PGD-{10, 40, 100} · APGD-CE-{10, 100} · Square (black-box query-only).
- **Phòng thủ**: APGD adversarial training (baseline Madry et al., ICLR 2018).
- **Threat model**: white-box · gray-box (same-arch) · cross-arch · black-box — đầy đủ bốn ngữ cảnh.
- **Kiến trúc**: `resnet18` (CNN, locality bias) và `vit_tiny` (Transformer, global attention).

Kho mã được tổ chức theo nguyên tắc production: `src/` chứa lõi, `scripts/` là CLI mỏng,
notebook chỉ đóng vai trò *báo cáo* — không lặp lại logic.

### Điểm nổi bật

| Hạng mục | Chi tiết |
| --- | --- |
| **Reproducibility** | Single seed `42` cho toàn bộ phase, deterministic loaders, frozen dataclass configs |
| **Tracking** | MLflow HTTP server + JSON mirror + rotating `experiment.log` — fail-safe khi sink lỗi |
| **Honest reporting** | Notebook helper emit `full-campaign-pending` khi thiếu checkpoint thay vì bịa số |
| **Threat models** | white-box / gray-box / cross-arch / black-box bao quát đủ phổ kiến thức của attacker |
| **Architecture coverage** | CNN (ResNet-18) và Transformer (ViT-Tiny) — so sánh hai họ inductive bias |
| **Hardware-aware** | Preset `gpu_default`, `cpu`, `p40` cho local dev và cloud rental |
| **Smoke-friendly** | Mọi script/notebook chạy được trên CPU với `--smoke` (~vài phút) |

### Bắt đầu nhanh

```bash
git clone https://github.com/<user>/pgd-cifar10-experiment.git
cd pgd-cifar10-experiment

python -m venv .venv && source .venv/bin/activate
pip install -e '.[dev,notebooks]'

pytest tests/ -q
bash scripts/reproduce.sh --smoke       # auto-detect GPU, fallback CPU
```

### Tái lập đầy đủ

```bash
bash scripts/reproduce.sh
```

Pipeline đầy đủ: huấn luyện checkpoint sạch cho cả hai kiến trúc với `seed=42` →
adversarial training → white-box + transfer evaluation → epsilon sweep → render
notebook báo cáo.

**Phần cứng mục tiêu**: NVIDIA A1000 4 GB VRAM.

### Bản đồ kiến trúc

```text
scripts/  ─►  src/cli/           ─►  src/experiments/runner.py
                                 ─►  src/training/{CleanTrainer, AdversarialTrainer}
                                 ─►  src/evaluation/AttackEvaluator
                                 ─►  src/{attacks, models, data, tracking}

notebooks/ ─►  src/reporting/{attack, constants, loaders, registry}
```

Workflow có state quan trọng dùng service object (`ExperimentRunner`, các trainer,
`AttackEvaluator`); transformation thuần và serialization vẫn là function.

### Hợp đồng kỹ thuật cốt lõi

| Hợp đồng | Ý nghĩa |
| --- | --- |
| **Reproducibility** | Mọi entry point gọi `set_all_seeds(seed)`; tracking ghi metadata Git + môi trường |
| **Raw-input attack** | Attacker thao tác ảnh `[0, 1]`; `NormalizedModel` chuẩn hoá CIFAR-10 *bên trong* `forward` |
| **Fail-fast** | Batch adversarial được kiểm `verify_perturbation` trước khi tin số liệu |
| **Frozen configs** | Experiment / training / model / attack config là dataclass immutable sau khi load |
| **Linf-only** | Toàn bộ codebase scope về `Linf` perturbation norm |
| **Honest reports** | Thiếu checkpoint → `full-campaign-pending`, không fabricate kết quả |

### Giao diện dòng lệnh (CLI)

```text
scripts/train_clean.py          --arch {resnet18, vit_tiny} --seed N [--epochs E] [--smoke]
scripts/train_adversarial.py    --arch ... --seed N [--smoke]
scripts/run_white_box.py        --arch ... [--attack ATTACK] [--seed N] [--smoke]
scripts/run_transfer.py         --mode {cross_arch, gray_box} [--attack ATTACK] [--max-pairs K] [--smoke]
scripts/run_epsilon_sweep.py    [--arch ARCH] [--seed N] [--epsilon EPS] [--smoke]
scripts/run_black_box_square.py --arch ... [--variant clean|adv] [--num-queries N] [--smoke]
scripts/reproduce.sh            [--smoke]
```

Ví dụ thường dùng:

```bash
python scripts/train_clean.py       --arch resnet18 --seed 42 --epochs 100
python scripts/train_adversarial.py --arch resnet18 --seed 42 --epochs 100
python scripts/run_white_box.py     --arch resnet18 --seed 42
python scripts/run_transfer.py      --mode cross_arch --seed 42
python scripts/run_epsilon_sweep.py --arch resnet18 --seed 42
```

### Cấu hình

Config gốc: [configs/config.yaml](configs/config.yaml).

```yaml
defaults:
  - base: default
  - architecture: resnet18
  - attack: pgd_10
  - training: clean
  - _self_
```

Các fragment YAML chuyên biệt nằm dưới `configs/{architecture, attack, training, transfer, sweeps}/`,
được `load_experiment_config`, `load_attack_config`, `load_training_config` resolve
thành dataclass immutable.

#### Hardware preset

| Preset | Mục đích |
| --- | --- |
| `gpu_default` *(mặc định)* | Bảo thủ, deterministic, AMP theo training config |
| `cpu` | Test cục bộ không GPU |
| `p40` | Tesla P40 cloud (8 workers, persistent, cudnn benchmark, AMP off) |

Ví dụ: `python scripts/train_clean.py --arch resnet18 --hardware p40 --batch-size 512`.

### Theo dõi thực nghiệm

- **MLflow HTTP**: `http://127.0.0.1:5000`
- **JSON mirror**: `results/logs/<run-name>.json`
- **Rotating log**: `results/logs/experiment.log`

Nếu JSON sink fail, MLflow run được giữ + gắn tag `json_sink_failed=true` thay vì abort.

```bash
bash scripts/mlflow.sh        # khởi động server, mở UI ở http://127.0.0.1:5000
```

Cho smoke run cục bộ hoặc CI không cần server, dùng `--no-mlflow` — JSON + file log vẫn chạy đầy đủ.

### Phạm vi & ngoài phạm vi

- ✅ FGSM / BIM / PGD / APGD-CE white-box, transfer attacks, epsilon sweep, Square black-box, APGD adversarial training, CIFAR-10.
- ❌ L2 / L1 / L0 norm attacks, targeted attacks, AutoAttack baseline đầy đủ, TRADES / MART defense, mở rộng ImageNet.

### Trích dẫn

```bibtex
@inproceedings{madry2018towards,
  title     = {Towards Deep Learning Models Resistant to Adversarial Attacks},
  author    = {Madry, Aleksander and Makelov, Aleksandar and Schmidt, Ludwig
               and Tsipras, Dimitris and Vladu, Adrian},
  booktitle = {ICLR},
  year      = {2018},
}
```

### Giấy phép

Xem [`LICENSE`](LICENSE).

---

## 🇬🇧 English version

### What this is

A reproducible CIFAR-10 study covering the full robustness pipeline:

- **Attacks**: FGSM · BIM · PGD-{10, 40, 100} · APGD-CE-{10, 100} · Square (black-box, query-only).
- **Defense**: APGD adversarial training (Madry et al., ICLR 2018 baseline).
- **Threat models**: white-box · gray-box (same-arch) · cross-arch · black-box — all four contexts evaluated.
- **Architectures**: `resnet18` (CNN, locality bias) and `vit_tiny` (Transformer, global attention).

The repo is organised as production code under `src/`, thin CLI entry points under
`scripts/`, and a report notebook under `notebooks/` that consumes — never re-implements —
the same `src/` helpers.

### Highlights

| Aspect | Details |
| --- | --- |
| **Reproducibility** | Single seed `42` across every phase, deterministic data loaders, frozen-dataclass configs |
| **Tracking** | MLflow HTTP server + JSON mirror + rotating `experiment.log` — fail-safe on sink errors |
| **Honest reporting** | Notebook helpers emit `full-campaign-pending` instead of fabricating missing numbers |
| **Threat models** | white-box / gray-box / cross-arch / black-box, spanning the attacker-knowledge spectrum |
| **Architectures** | CNN (ResNet-18) vs Transformer (ViT-Tiny) — compares two families of inductive bias |
| **Hardware-aware** | `gpu_default`, `cpu`, and `p40` presets covering local dev and cloud rentals |
| **Smoke-friendly** | Every script/notebook runs on CPU with `--smoke` in minutes |

### Quick start

```bash
git clone https://github.com/<user>/pgd-cifar10-experiment.git
cd pgd-cifar10-experiment

python -m venv .venv && source .venv/bin/activate
pip install -e '.[dev,notebooks]'

pytest tests/ -q
bash scripts/reproduce.sh --smoke       # auto-detects GPU; falls back to CPU
```

### Full reproduction

```bash
bash scripts/reproduce.sh
```

The full campaign trains clean checkpoints for both architectures with `seed=42`,
runs adversarial training, executes white-box and transfer evaluation, performs
the epsilon sweep, and renders the report notebook in place.

**Hardware target**: NVIDIA A1000 4 GB VRAM.

### Architecture map

```text
scripts/  ─►  src/cli/           ─►  src/experiments/runner.py
                                 ─►  src/training/{CleanTrainer, AdversarialTrainer}
                                 ─►  src/evaluation/AttackEvaluator
                                 ─►  src/{attacks, models, data, tracking}

notebooks/ ─►  src/reporting/{attack, constants, loaders, registry}
```

Stateful workflows use service objects (`ExperimentRunner`, the trainers, and
`AttackEvaluator`). Pure transformations and serialization helpers stay as
functions. Notebook code imports only from `src.reporting`; scripts share
bootstrap, checkpoint, and smoke behaviour through `src.cli`.

### Key contracts

| Contract | Meaning |
| --- | --- |
| **Reproducibility** | Every entry point calls `set_all_seeds(seed)`; tracking records Git + environment metadata |
| **Raw-input attacks** | Attacks consume `[0, 1]` images; `NormalizedModel` performs CIFAR-10 normalization inside `forward` |
| **Fail-fast verification** | Adversarial batches go through `verify_perturbation` before metrics are trusted |
| **Frozen configs** | Experiment / training / model / attack configs are immutable dataclasses after loading |
| **Linf only** | All attack/evaluation code assumes the project's scoped perturbation norm |
| **Honest notebooks** | Missing checkpoints surface as `full-campaign-pending`, never as invented numbers |

### Command-line interface

```text
scripts/train_clean.py          --arch {resnet18, vit_tiny} --seed N [--epochs E] [--smoke]
scripts/train_adversarial.py    --arch ... --seed N [--smoke]
scripts/run_white_box.py        --arch ... [--attack ATTACK] [--seed N] [--smoke]
scripts/run_transfer.py         --mode {cross_arch, gray_box} [--attack ATTACK] [--max-pairs K] [--smoke]
scripts/run_epsilon_sweep.py    [--arch ARCH] [--seed N] [--epsilon EPS] [--smoke]
scripts/run_black_box_square.py --arch ... [--variant clean|adv] [--num-queries N] [--smoke]
scripts/reproduce.sh            [--smoke]
```

Common examples:

```bash
python scripts/train_clean.py       --arch resnet18 --seed 42 --epochs 100
python scripts/train_adversarial.py --arch resnet18 --seed 42 --epochs 100
python scripts/run_white_box.py     --arch resnet18 --seed 42
python scripts/run_transfer.py      --mode cross_arch --seed 42
python scripts/run_epsilon_sweep.py --arch resnet18 --seed 42
```

### Configuration

Root config: [configs/config.yaml](configs/config.yaml).

```yaml
defaults:
  - base: default
  - architecture: resnet18
  - attack: pgd_10
  - training: clean
  - _self_
```

YAML fragments under `configs/{architecture, attack, training, transfer, sweeps}/`
are resolved by `load_experiment_config`, `load_attack_config`, and
`load_training_config` into immutable dataclasses.

#### Hardware presets

| Preset | Purpose |
| --- | --- |
| `gpu_default` *(default)* | Conservative, deterministic, AMP per training config |
| `cpu` | Local testing without a GPU |
| `p40` | Tesla P40 cloud rentals (8 workers, persistent, cudnn benchmark, AMP off) |

Example: `python scripts/train_clean.py --arch resnet18 --hardware p40 --batch-size 512`.

### Experiment tracking

- **MLflow HTTP API**: `http://127.0.0.1:5000`
- **JSON mirror**: `results/logs/<run-name>.json`
- **Rotating log**: `results/logs/experiment.log`

If the JSON sink fails, the MLflow run is preserved and tagged with
`json_sink_failed=true` instead of aborting the experiment.

```bash
bash scripts/mlflow.sh          # starts the server, UI at http://127.0.0.1:5000
```

For one-off local or CI smoke runs without the server, pass `--no-mlflow`;
JSON and file logs keep running.

### Scope & non-goals

- ✅ FGSM / BIM / PGD / APGD-CE white-box, transfer, epsilon sweeps, Square black-box, APGD adversarial training on CIFAR-10.
- ❌ L2 / L1 / L0 attacks, targeted attacks, full AutoAttack baseline, TRADES / MART defenses, ImageNet expansion.

### Citation

```bibtex
@inproceedings{madry2018towards,
  title     = {Towards Deep Learning Models Resistant to Adversarial Attacks},
  author    = {Madry, Aleksander and Makelov, Aleksandar and Schmidt, Ludwig
               and Tsipras, Dimitris and Vladu, Adrian},
  booktitle = {ICLR},
  year      = {2018},
}
```

### License

See [`LICENSE`](LICENSE).
