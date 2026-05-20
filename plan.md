# Codebase Audit and Refactor Plan

## 1. Tổng quan codebase

Project là một experiment harness nghiên cứu adversarial attack/defense trên CIFAR-10, bao phủ FGSM/BIM/PGD/APGD-CE/Square cho ba kiến trúc (`resnet18`, `wrn_34_10`, `vit_tiny`), single-seed (`SEED=42`) sau khi refactor từ workflow 5-seed.

Cấu trúc:

```text
configs/                       YAML fragments (architecture / attack / training / transfer / sweeps)
scripts/                       Thin CLI entrypoints
src/
  attacks/                     BaseAttack + FGSM/PGD/APGD/Square (+ verify, factory)
  cli/                         Shared CLI scaffolding (runner, sweep, transfer, attack_configs, loader)
  data/                        CIFAR-10 + smoke loaders + tensor validation
  evaluation/                  AttackEvaluator + per-sample/aggregate metrics
  experiments/                 Frozen config dataclasses + loader + ExperimentRunner + checkpoint paths
  models/                      Normalizer + builders + ResNet/WRN/ViT + grad-cam target registry
  reporting/                   nb01..nb11 helpers + MLflow queries + io + gates + model registry
  tracking/                    ExperimentTracker + run logger + env metadata
  training/                    BaseTrainer + CleanTrainer + AdversarialTrainer
  utils/                       seed, timing, logger
  visualize/                   gradcam, perturbation_panels, style
tests/                         Mirrors src layout
notebooks/                     01..11 thin notebooks that import src.reporting helpers
```

Logic flow chính:

- `scripts/*.py` → `src.cli.runner.bootstrap()` (seeds + load config) → `ExperimentRunner` orchestrates training/eval → trainers/AttackEvaluator → attacks/models/data → `ExperimentTracker` mirrors to MLflow + JSON.
- Notebooks gọi helpers trong `src/reporting/nb*_*.py`; helpers tự discovery checkpoint (clean/adv) qua `ReportingModelPair` và emit `full-campaign-pending` khi thiếu artifact.
- Mọi attack consume raw `[0, 1]` images; `Normalizer` thực hiện chuẩn hóa trong forward; sau mỗi attack có `verify_perturbation` chạy fail-fast.

## 2. Danh sách file đã đọc

Toàn bộ tree không-binary đã được đọc trực tiếp hoặc qua subagent (test files), không bỏ sót:

Configs (16):
- configs/config.yaml, configs/base/default.yaml
- configs/architecture/resnet18.yaml, vit_tiny.yaml, wrn_34_10.yaml
- configs/attack/apgd_ce_10.yaml, apgd_ce_100.yaml, bim_10.yaml, fgsm.yaml, pgd_10.yaml, pgd_40.yaml, pgd_100.yaml, square_5000.yaml
- configs/training/clean.yaml, apgd_at.yaml
- configs/transfer/cross_seed_pairs.yaml, gray_box_pairs.yaml, transfer_pairs.yaml
- configs/sweeps/pgd_epsilon_sweep.yaml

Scripts (7): `__init__.py`, train_clean.py, train_adversarial.py, run_white_box.py, run_transfer.py, run_epsilon_sweep.py, run_black_box_square.py, reproduce.sh, mlflow.sh

src/ (toàn bộ ~70 file): attacks (10 file), cli (6), data (4), evaluation (3), experiments (5), models (7), reporting (21), tracking (5), training (4), utils (4), visualize (4), `src/__init__.py`.

tests/ (toàn bộ): conftest.py + tất cả `test_*.py` đã được đọc qua subagent (xem trace).

Notebooks (13): NB01..NB11 đã được dump cells và đọc nội dung markdown/code.

Other: README.md, pyproject.toml, pgd_cifar10_experiment.egg-info/SOURCES.txt.

Không có file source nào bị bỏ qua. `.git/`, `mlruns/`, `outputs/`, build cache không nằm trong scope (đúng yêu cầu).

## 3. Lỗi logic và bugs

### Bug: train_adversarial.py truyền training="adversarial" trong khi YAML là apgd_at.yaml

- File: [scripts/train_adversarial.py:17](scripts/train_adversarial.py#L17)
- Class/function/method: `main()`
- Mô tả: Script gọi `bootstrap(args, arch=args.arch, training="adversarial")`. `load_training_config` resolve sang `configs/training/adversarial.yaml`, nhưng file đó không tồn tại — chỉ có `configs/training/apgd_at.yaml` và `clean.yaml`.
- Vì sao là lỗi: OmegaConf sẽ raise `FileNotFoundError` ngay khi script chạy với checkpoint thật (smoke test có thể vẫn fail sớm vì cùng path). Toàn bộ entry point adversarial training bị hỏng.
- Ảnh hưởng: Không thể chạy `python scripts/train_adversarial.py --arch ... --seed ...` (đường dẫn được README quảng cáo). Reproducibility plan và `scripts/reproduce.sh` không thể train AT.
- Hướng sửa: Đổi `training="adversarial"` thành `training="apgd_at"` (chuẩn nhất, ăn khớp tên file). Hoặc rename file YAML thành `adversarial.yaml` (kém preferred vì xóa tín hiệu "APGD"). Cập nhật `scripts/train_adversarial.py` ưu tiên hơn.

### Bug: nb04_main_results CSV schema không ăn khớp với figure rendering

- File: [src/reporting/nb04_main_results.py](src/reporting/nb04_main_results.py)
- Function: `nb04_main_results`, `_render_main_figure`, `_render_time_vs_asr`
- Mô tả: Sau refactor về single-seed, `nb04_main_results()` ghi CSV với keys `asr`, `robust_acc`, `time_per_image_ms`. Nhưng `_render_main_figure` lọc bằng `str(r.get("asr_mean", "")) != ""` và `_render_time_vs_asr` đọc `time_per_image_ms_mean`/`asr_mean`/`asr_std`. Không row nào có những key `*_mean/*_std` đó nên `visible` luôn rỗng.
- Vì sao là lỗi: Cả hai figure đều luôn được render với placeholder "full-campaign-pending" ngay cả khi đã có checkpoints. Hoàn toàn không match với điều kiện "có data thì plot bar chart".
- Ảnh hưởng: Figure NB04 (`results/figures/04_main.png`, `04_time_vs_asr.png`) không bao giờ thực sự render data dù chạy đầy đủ campaign.
- Hướng sửa: Cập nhật `_render_main_figure` và `_render_time_vs_asr` để đọc trực tiếp `asr` và `time_per_image_ms` (single-seed); bỏ branch `_std`; hoặc khôi phục thêm cột `asr_mean = asr, asr_std = ""` trong row khi single-seed. Đồng thời cập nhật read-side: `square_rows` filter ở dòng 50-54 cũng dùng `asr_mean`/`time_per_image_ms_mean` — phải nhất quán với schema NB08 (square table vẫn dùng `_mean` vì có thể nhiều run).

### Bug: nb09 transfer analysis empty-list mean fallback không hoạt động

- File: [src/reporting/nb09_transfer_analysis.py:57-60](src/reporting/nb09_transfer_analysis.py#L57-L60)
- Function: `_render_gray_box`
- Mô tả: `float(np.mean([...]) or 0.0)` — khi list rỗng, `np.mean([])` trả về NaN (kèm RuntimeWarning), NaN là truthy trong Python (`bool(float("nan")) == True`), nên `or 0.0` không bao giờ kích hoạt. Khi list không rỗng, mean là numpy scalar (cũng truthy ngoại trừ giá trị 0).
- Vì sao là lỗi: Khi một (arch, variant) combination không có sample, hàm gán giá trị NaN vào height của bar chart → matplotlib có thể vẽ bar trống / cảnh báo. Intent là 0.0 nhưng implementation bị broken.
- Ảnh hưởng: Bar chart gray-box có thể bị NaN khi data partial. Vô hại với matplotlib nhưng misleading.
- Hướng sửa: Thay bằng explicit check:
  ```python
  matches = [float(r["asr_mean"]) for r in rows if r["arch"] == arch and r["victim_variant"] == variant]
  means.append(float(np.mean(matches)) if matches else 0.0)
  ```

### Bug: PAIR_FILES không có "cross_seed" trong khi config và README còn nhắc

- File: [src/cli/transfer.py:13-16](src/cli/transfer.py#L13-L16), [README.md:82](README.md#L82), [notebooks/09_transfer_attack_analysis.ipynb](notebooks/09_transfer_attack_analysis.ipynb), [configs/transfer/cross_seed_pairs.yaml](configs/transfer/cross_seed_pairs.yaml)
- Mô tả: `PAIR_FILES = {"cross_arch": ..., "gray_box": ...}` — không có entry cho `cross_seed`. Tuy nhiên: (1) `configs/transfer/cross_seed_pairs.yaml` vẫn tồn tại, (2) README liệt kê `--mode {cross_arch,cross_seed,gray_box}`, (3) NB09 markdown nói "Modes: `cross_arch` and `cross_seed`", (4) `src/reporting/mlflow_queries.read_transfer_mlflow_runs` có code `if mode == "cross_seed": continue  # cross_seed mode removed`.
- Vì sao là lỗi: User chạy `--mode cross_seed` sẽ bị argparse reject. Documentation và config không đồng bộ với code đã được dọn dẹp.
- Ảnh hưởng: Confusion, dead file, README và NB09 lie về capability.
- Hướng sửa: Xóa hẳn `configs/transfer/cross_seed_pairs.yaml`, sửa README `--mode {cross_arch,gray_box}`, sửa NB09 markdown bỏ "cross_seed". Hoặc khôi phục mode (project memory ghi nhận removed — confirmed remove).

### Bug: README chỉ tới mlflow_server.sh và download_robustbench_wrn.py không tồn tại

- File: [README.md:85](README.md#L85), [README.md:135](README.md#L135)
- Mô tả: README mention `scripts/download_robustbench_wrn.py` (đã bị xóa, xem `git status`) và `bash scripts/mlflow_server.sh` (file thật tên là `scripts/mlflow.sh`).
- Ảnh hưởng: User copy-paste commands sẽ gặp "No such file" lỗi.
- Hướng sửa: Xóa dòng `scripts/download_robustbench_wrn.py` trong CLI block; đổi `mlflow_server.sh` thành `mlflow.sh` ở "Browse the UI" block.

### Bug: README documents `--resume PATH` cho train_adversarial nhưng flag không tồn tại

- File: [README.md:80](README.md#L80), [scripts/train_adversarial.py](scripts/train_adversarial.py), [configs/training/apgd_at.yaml](configs/training/apgd_at.yaml)
- Mô tả: README mô tả `scripts/train_adversarial.py [--resume PATH]`, và YAML `apgd_at.yaml` có `resume_from: null`. Nhưng script không define `--resume`, `TrainingConfig` không có field `resume_from`, và `_load_training_config` không đọc key đó (silently ignored).
- Ảnh hưởng: Resume capability claim không khả dụng. `save_every_epochs: 5` trong YAML cũng bị silently ignore (không có field).
- Hướng sửa: Quyết định một trong hai: (a) implement resume + periodic save (đáng kể hơn refactor), hoặc (b) xóa các key thừa khỏi `apgd_at.yaml` và xóa mô tả trong README.

### Bug: CleanTrainer và AdversarialTrainer nhận val_loader nhưng không dùng

- File: [src/training/clean.py:19-46](src/training/clean.py#L19-L46), [src/training/adversarial.py:33-59](src/training/adversarial.py#L33-L59), [src/training/base.py:33-50](src/training/base.py#L33-L50)
- Mô tả: `BaseTrainer.__init__` lưu `self.val_loader = val_loader` nhưng `fit()` của hai subclass không gọi val pass nào. `best_metric` được tính từ training accuracy chứ không phải validation. `ExperimentRunner.train_clean` / `train_adversarial` truyền cùng test loader vào cả train và val slot.
- Vì sao là lỗi: Tham số `val_loader` là API noise: caller có thể tưởng có validation thực sự xảy ra; `best_metric` mang ý nghĩa khác với tiêu chuẩn (training acc thay vì val acc).
- Ảnh hưởng: Reports nói "best_metric" mà thực ra là cuối-epoch train acc; không có early stopping/model selection thật.
- Hướng sửa: Hoặc (a) thêm pass `evaluate()` ở cuối mỗi epoch và đặt `best_metric` thành val acc, hoặc (b) xóa `val_loader` khỏi `BaseTrainer.__init__` và `_loaders()` chỉ trả 1 loader cho training. Lựa chọn (b) khả thi vì ít invasion nhất.

### Bug: nb01_protocol mutates frozen config in production code path

- File: [src/reporting/nb01_protocol.py:14-20](src/reporting/nb01_protocol.py#L14-L20)
- Mô tả: Code thử `config.seed = 7` bên trong try để xác định frozen. Đây là một "self-test" được nhúng giữa code production của notebook helper.
- Vì sao là anti-pattern (sát biên giới bug): Production-path side effect; nếu config được cache (như `tracking_config` ở mlflow_queries) thì lần gọi sau có thể ảnh hưởng.
- Hướng sửa: Thay bằng kiểm tra metaclass-level: `frozen = config.__dataclass_params__.frozen` rồi assert.

### Bug: ScriptContext.build_tracker đặt args.tracking_uri trên Namespace chứa "tracking-uri"

- File: [src/cli/runner.py:16-22](src/cli/runner.py#L16-L22)
- Mô tả: argparse convert `--tracking-uri` → `args.tracking_uri`. ScriptContext.build_tracker đọc `self.args.tracking_uri` — OK. Không phải bug, đã xác thực.
- (Loại khỏi bug list — chỉ note để loại trừ false positive.)

### Bug: SOURCES.txt egg-info chứa file không tồn tại

- File: [pgd_cifar10_experiment.egg-info/SOURCES.txt:28](pgd_cifar10_experiment.egg-info/SOURCES.txt#L28)
- Mô tả: SOURCES.txt liệt kê `src/cli/adversarial_train.py` nhưng file đó không tồn tại trong src/cli/.
- Ảnh hưởng: Stale build metadata, không ảnh hưởng runtime; pip install -e có thể warning.
- Hướng sửa: Rebuild egg-info, hoặc gitignore `*.egg-info/`.

### Bug: scripts/reproduce.sh "full" mode chỉ chạy 5 epochs

- File: [scripts/reproduce.sh:24, 28](scripts/reproduce.sh#L24)
- Mô tả: README quảng cáo full campaign 100 epochs (119-141 GPU-hours). `reproduce.sh` ở full mode chạy `--epochs 5` cho cả clean và adversarial.
- Vì sao là lỗi: Reproduce nói một đằng, code làm một nẻo. Một full run sẽ ra checkpoint kém chất lượng, gates trong NB02/NB07 (≥0.93/0.95/0.85 clean, ≥0.80/0.83/0.75 AT) sẽ fail vì model under-trained.
- Ảnh hưởng: Reproducibility claim sai lệch.
- Hướng sửa: Đổi `--epochs 5` thành `--epochs 100` cho cả hai phase, hoặc thêm flag `--quick` / sửa README để khớp với mức epoch thật sự.

### Bug: notebook markdown vẫn mô tả 5-seed protocol đã bị refactor đi

- File: [notebooks/01_research_protocol.ipynb](notebooks/01_research_protocol.ipynb), [notebooks/02_baseline_clean_models.ipynb](notebooks/02_baseline_clean_models.ipynb), [notebooks/04_main_quantitative_results.ipynb](notebooks/04_main_quantitative_results.ipynb), [notebooks/05_vulnerability_analysis.ipynb](notebooks/05_vulnerability_analysis.ipynb), [notebooks/09_transfer_attack_analysis.ipynb](notebooks/09_transfer_attack_analysis.ipynb), [notebooks/10_architecture_robustness_comparison.ipynb](notebooks/10_architecture_robustness_comparison.ipynb)
- Mô tả: NB01 nói `Canonical seeds: {42, 123, 456, 789, 1024}`; NB02 nói "mean +- std when at least three runs exist"; NB04 nói "Error bars: ±1 std across clean-model seeds"; NB05 nói "Seeds: {42, 123, 456} for the sweep"; NB09 nói "Modes: cross_arch and cross_seed"; NB10 nói "Statistical protocol: one-sided Welch test". Tất cả đều phản ánh code cũ trước single-seed refactor.
- Vì sao là lỗi: Vi phạm nguyên tắc "honest notebooks" được README claim — markdown nói multi-seed Welch test nhưng code chỉ load 1 seed và không chạy Welch. Reader bị misled.
- Ảnh hưởng: Báo cáo cuối có thể chứa text Methodology không khớp với artifact thực.
- Hướng sửa: Cập nhật từng markdown cell cho khớp với code đang chạy: single-seed, không Welch, không ±std, modes={cross_arch, gray_box}.

### Bug: nb07 disclosure string khi WRN-34-10 không có AT checkpoint vẫn nói "RobustBench fallback documented if fallback_triggered=true"

- File: [src/reporting/nb07_adversarial_training.py:43-47](src/reporting/nb07_adversarial_training.py#L43-L47)
- Mô tả: Khi `not models.has_adversarial`, code đặt disclosure="single-seed APGD AT; RobustBench fallback documented if fallback_triggered=true". Nhưng nếu chưa có checkpoint thì không có fallback nào được trigger; câu này dễ gây hiểu nhầm rằng có fallback artifact nhưng pending.
- Ảnh hưởng: Misleading message trong CSV.
- Hướng sửa: Đơn giản hóa thành `"single-seed APGD AT; full training pending"` cho WRN khi missing.

### Bug: src/data/cifar10.py:_seed_worker double-applies worker_id

- File: [src/data/cifar10.py:15-17](src/data/cifar10.py#L15-L17)
- Mô tả: `worker_seed = torch.initial_seed() % 2**32` đã là worker-specific (PyTorch khởi tạo `torch.initial_seed()` khác nhau cho mỗi worker). Hàm rồi `np.random.seed(worker_seed + worker_id)` — cộng thêm worker_id một lần nữa.
- Vì sao là lỗi (mild): Không gây trùng seed (vì base đã khác), nhưng quan hệ giữa torch seed và numpy seed bị offset. Worker không deterministic theo cách "đối xứng".
- Ảnh hưởng: Reproducibility vẫn được giữ vì seed chain deterministic, chỉ là double-applied. Không phải bug nghiêm trọng nhưng vi phạm convention chuẩn (`np.random.seed(worker_seed)` là pattern PyTorch official).
- Hướng sửa: `np.random.seed(worker_seed)` thay vì cộng worker_id. Cần kiểm chứng thêm với test seed-determinism.

### Bug: AttackEvaluator.run() và ExperimentRunner._run_transfer_evaluation chạy clean forward pass trên model wrong cho transfer

- File: [src/experiments/runner.py:202-217](src/experiments/runner.py#L202-L217)
- Mô tả: `_run_transfer_evaluation` đo `clean_conf` từ **victim** model (đúng), nhưng `verify_perturbation` (line 206) chạy ngay sau khi perturb từ surrogate. Logic ok. Tuy nhiên `linfs/l2s/psnrs/ssims` được tính trên perturbations `x_adv vs x` — đúng. (No bug here — false positive, removed.)
- Note: confirmed transfer flow correct.

### Bug: PAIR_FILES dùng path string tuyệt đối từ CWD

- File: [src/cli/transfer.py:13-16](src/cli/transfer.py#L13-L16)
- Mô tả: `PAIR_FILES["cross_arch"] = "configs/transfer/transfer_pairs.yaml"` — relative path. `OmegaConf.load(Path(...))` sẽ resolve theo CWD của process. Nếu chạy từ thư mục khác repo, fail.
- Vì sao là lỗi nhẹ: Inconsistent với `CONFIG_ROOT` được dùng trong `config_loader.py` (resolve theo `__file__`).
- Hướng sửa: Dùng `Path(__file__).resolve().parents[2] / "configs/transfer/..."` giống pattern `CONFIG_ROOT`.

### Bug: read_csv trả về dict[str, str] nhưng float() được gọi không-defensive

- File: [src/reporting/nb04_main_results.py:138-141](src/reporting/nb04_main_results.py#L138-L141), [src/reporting/nb04_main_results.py:50-54](src/reporting/nb04_main_results.py#L50-L54)
- Mô tả: `read_csv` đọc CSV theo `csv.DictReader` → all strings. Khi `_render_time_vs_asr` filter rồi `float(r["time_per_image_ms_mean"])`, nếu `r["time_per_image_ms_mean"] == ""` filter đã loại. OK. Nhưng kết hợp với bug schema mismatch ở trên, filter expression không hoạt động đúng. Liên đới với bug NB04 schema.

## 4. Anti-pattern code

### Anti-pattern: Logic duplication giữa AttackEvaluator.run và ExperimentRunner._run_transfer_evaluation

- File: [src/evaluation/runner.py:92-161](src/evaluation/runner.py#L92-L161), [src/experiments/runner.py:184-238](src/experiments/runner.py#L184-L238)
- Vấn đề: Hai vòng lặp xấp xỉ ~100% giống nhau (clean conf gather → perturb → verify → adv conf → metrics → aggregate). Khác duy nhất: transfer dùng surrogate cho perturb + victim cho predict, single-model dùng cùng một.
- Vì sao chưa tốt: DRY violation, hai chỗ cùng logic. Bất kỳ fix metric nào (vd. thêm `keep_per_sample`, đổi confidence drop formula) phải sync giữa hai file. `_run_transfer_evaluation` không hỗ trợ `keep_per_sample`.
- Hướng refactor: Mở rộng `AttackEvaluator` với hai model — `perturb_model` và `eval_model` (default cùng object). `ExperimentRunner.evaluate_transfer` chỉ cần khởi tạo `AttackEvaluator(perturb_model=surrogate, eval_model=victim, ...)` thay vì duplicate vòng lặp.

### Anti-pattern: Duplicated checkpoint-or-smoke loader

- File: [src/cli/loader.py:14-29](src/cli/loader.py#L14-L29), [src/experiments/runner.py:167-182](src/experiments/runner.py#L167-L182)
- Vấn đề: `load_checkpoint_or_smoke` (cli) và `_load_checkpoint_or_smoke_model` (experiments) giống nhau y hệt: build canonical path → load if exists → fallback smoke. Project memory `feedback_principles_audit.md` đã có rule "no copy-paste helpers" — chính vi phạm mục #6 trong audit cũ.
- Hướng refactor: Xoá `_load_checkpoint_or_smoke_model` (private) trong `ExperimentRunner`, thay bằng import `load_checkpoint_or_smoke` từ `src/cli/loader.py`. Hoặc đưa helper lên `src/models/builders.py` (cùng layer với `load_model_from_checkpoint`) và import từ cả CLI và Experiment.

### Anti-pattern: ExperimentRunner._log_evaluation_result rớt PSNR/SSIM/confidence_drop

- File: [src/experiments/runner.py:240-250](src/experiments/runner.py#L240-L250)
- Vấn đề: Function log tracker chỉ với `{asr, robust_acc, linf_mean, l2_mean, time_per_image_ms, n_samples}` — bỏ `psnr_mean`, `ssim_mean`, `confidence_drop_mean` mặc dù chúng vẫn được compute. Inconsistent với `EvaluationResult` định nghĩa đầy đủ.
- Hướng refactor: Bổ sung psnr/ssim/confidence_drop vào dict log. Hoặc helper `EvaluationResult.as_metric_dict()` để tránh skew.

### Anti-pattern: Inline list comprehension khó đọc trong nb09_transfer_analysis._render_gray_box

- File: [src/reporting/nb09_transfer_analysis.py:57-60](src/reporting/nb09_transfer_analysis.py#L57-L60)
- Vấn đề: List comprehension lồng nhau với `np.mean(...) or 0.0` (bug đã ghi ở mục 3). Khó đọc.
- Hướng refactor: Tách helper `_mean_asr(rows, arch, variant) -> float`.

### Anti-pattern: Mutating frozen AttackConfig via object.__setattr__ trong tests

- File: tests/test_attacks/test_apgd.py, tests/test_attacks/test_base_attack.py, tests/test_attacks/test_square.py
- Vấn đề: Tests dùng `object.__setattr__(cfg, "norm", "L2")` để test reject; pattern này bypass `frozen=True`. Hiện diện 3+ lần. Test cũng dùng `try/except ValueError; else: raise AssertionError(...)` thay vì `pytest.raises` — không nhất quán với test khác cùng file.
- Hướng refactor: Tạo fixture builder `make_attack_config(**overrides)` chấp nhận `norm` parameter; dùng `pytest.raises(ValueError)` để check.

### Anti-pattern: Bare try/except pattern thay cho pytest.raises

- File: tests/test_attacks/test_apgd.py, tests/test_attacks/test_square.py, tests/test_scripts/test_scripts_smoke.py, tests/test_tracking/test_mlflow_logger.py
- Vấn đề: Một số tests dùng `try: ...; except X: pass; else: raise AssertionError(...)`. Verbose, lệch với phần còn lại của test suite.
- Hướng refactor: Chuyển sang `with pytest.raises(X):`.

### Anti-pattern: AdversarialTrainer.__init__ chấp nhận cả inner_attack kwarg và self.training_config.inner_attack

- File: [src/training/adversarial.py:22-31](src/training/adversarial.py#L22-L31)
- Vấn đề: Có hai source-of-truth: caller có thể truyền `inner_attack=`, hoặc lấy từ config. Trong production duy nhất caller (`ExperimentRunner.train_adversarial`) không truyền, chỉ dựa config. Code này chỉ phục vụ cho test mode + adds confusion.
- Hướng refactor: Xoá keyword `inner_attack` overriding. Test có thể dùng `dataclasses.replace(training_config, inner_attack=fake_attack)` để inject (đã có sẵn pattern trong test_adversarial_loop.py).

### Anti-pattern: AttackFactory._registry case handling mixed

- File: [src/attacks/factory.py:20-26](src/attacks/factory.py#L20-L26), [configs/attack/square_5000.yaml:1](configs/attack/square_5000.yaml#L1)
- Vấn đề: Registry keys uppercase (`"FGSM"`, `"PGD"`, `"APGD-CE"`, `"SQUARE"`). Lookup do `config.name.upper()`. YAML thì `name: Square` (Title), `name: FGSM` (upper), `name: APGD-CE` (upper-dash) — không nhất quán. Tests thì hardcode "Square" (title case in conftest fixture).
- Hướng refactor: Standardize tất cả YAML `name:` field về uppercase. Hoặc canonicalize registry case-insensitively cũng OK (đã có) nhưng config nên đồng nhất.

### Anti-pattern: `nb01_protocol` dùng try-except để test frozen state trong production

- File: [src/reporting/nb01_protocol.py:14-21](src/reporting/nb01_protocol.py#L14-L21)
- Vấn đề: Production code path attempting mutation rồi assert. Đây là test-thinking trong production module.
- Hướng refactor: `frozen = config.__dataclass_params__.frozen` rồi `assert frozen`.

### Anti-pattern: Square Attack patch_candidate loop từng sample trong Python

- File: [src/attacks/square.py:77-95](src/attacks/square.py#L77-L95)
- Vấn đề: `for sample_idx in range(B)` để gắn patch — chậm với batch lớn (b=64+).
- Hướng refactor: Vectorize bằng cách build mask `(B, 1, H, W)` rồi `candidate = torch.where(mask, x_orig + sign, x_adv)` và project_linf. Trade-off: memory cao hơn cho batch lớn, có thể giữ loop dạng generic nhưng dùng `vmap` cho speed-up.

### Anti-pattern: tracker.tracking._config flatten dùng asdict + flatten — sửa giữa dict và dataclass

- File: [src/tracking/tracker.py:86-93, 186-196](src/tracking/tracker.py#L186-L196)
- Vấn đề: `asdict(config)` flatten dataclass thành dict, rồi `_flatten_dict` lại kiểm tra `is_dataclass(value)` lần nữa — redundant guard vì `asdict` đã unwrap. Code path khó đọc.
- Hướng refactor: Bỏ branch `if is_dataclass(value)` trong `_flatten_dict` (đã được `asdict` xử lý).

### Anti-pattern: src/utils/logger.py gọi logging.basicConfig mỗi lần get_logger

- File: [src/utils/logger.py:21-23](src/utils/logger.py#L21-L23)
- Vấn đề: `logging.basicConfig` chỉ có hiệu lực lần đầu được gọi trong process. Tài liệu kèm theo của module cũng cảnh báo điều này. Side-effect implicit dễ confused.
- Hướng refactor: Thay `get_logger` thành thin wrapper `return logging.getLogger(name)`; nếu cần root config, tách thành `configure_root_logger()` được gọi explicit từ entry point.

### Anti-pattern: Stale egg-info file pointer tới adversarial_train.py không tồn tại

- File: [pgd_cifar10_experiment.egg-info/SOURCES.txt:28](pgd_cifar10_experiment.egg-info/SOURCES.txt#L28)
- Vấn đề: Build artifact bị commit (có thể) và ghi tham chiếu file dead.
- Hướng refactor: Thêm `*.egg-info/` vào `.gitignore`; xóa thư mục `pgd_cifar10_experiment.egg-info/` khỏi tracking.

### Anti-pattern: `evaluation_loader` and `evaluation_inputs` đều hardcode `batch=128 if cuda else 32`

- File: [src/reporting/data_loaders.py:39-67](src/reporting/data_loaders.py#L39-L67)
- Vấn đề: Magic numbers nằm trong helper, không khớp với batch_size từ `build_common_parser` default 256 hay từ YAML attack default. Loader build inside reporting helpers độc lập với CLI batch flag.
- Hướng refactor: Cho phép caller truyền `batch_size`; hoặc đặt constant `EVAL_BATCH_GPU`/`EVAL_BATCH_CPU` trong `src/reporting/constants.py`.

### Anti-pattern: `lru_cache(maxsize=1)` trên tracking_config trong mlflow_queries

- File: [src/reporting/mlflow_queries.py:106-108](src/reporting/mlflow_queries.py#L106-L108)
- Vấn đề: Cached process-wide; tests cần clear nhưng không có public API. Tăng coupling cho test isolation.
- Hướng refactor: Bỏ `lru_cache` (rẻ — chỉ load 1 YAML); hoặc expose `tracking_config.cache_clear()` cho test.

## 5. Flow hoặc chức năng tương tự nhưng chưa unify

### Ununified logic: load checkpoint or smoke fallback

- File/vị trí liên quan: [src/cli/loader.py:14-29](src/cli/loader.py#L14-L29), [src/experiments/runner.py:167-182](src/experiments/runner.py#L167-L182)
- Logic tương tự: Build canonical path từ (arch, seed, variant), nếu exists thì load checkpoint, nếu smoke thì warn + build fresh, ngược lại raise FileNotFoundError.
- Điểm chưa thống nhất: Hai file định nghĩa cùng helper với chữ ký tương đương; CLI là module-level function, ExperimentRunner là private method.
- Có nên unify: **Có**. Đã có project memory feedback (#6) cấm copy-paste helpers.
- Hướng unify: Đặt logic ở `src/cli/loader.py::load_checkpoint_or_smoke` (giữ nguyên), `ExperimentRunner._load_checkpoint_or_smoke_model` chỉ delegate.

### Ununified logic: evaluation loop (single-model vs transfer)

- File/vị trí liên quan: [src/evaluation/runner.py:92-161](src/evaluation/runner.py#L92-L161), [src/experiments/runner.py:184-238](src/experiments/runner.py#L184-L238)
- Logic tương tự: Vòng lặp batch → clean conf → perturb → verify → adv conf → metrics → aggregate.
- Điểm chưa thống nhất: Transfer cần hai model khác nhau (surrogate perturb, victim eval); single-model dùng cùng. Đồng thời transfer không hỗ trợ `keep_per_sample`.
- Có nên unify: **Có**.
- Hướng unify: Mở rộng `AttackEvaluator` để hỗ trợ optional `perturb_model: Normalizer | None = None` (default = `self.model`). Logic chính chỉ dùng `perturb_model or self.model` để gọi `attack.perturb()`.

### Ununified logic: build attack config khởi tạo cho notebook vs script

- File/vị trí liên quan: [src/reporting/attack.py:12-15](src/reporting/attack.py#L12-L15), [src/cli/attack_configs.py:12-21](src/cli/attack_configs.py#L12-L21), [src/attacks/factory.py:28-36](src/attacks/factory.py#L28-L36)
- Logic tương tự: Load attack config, optionally replace fields, build attack.
- Điểm chưa thống nhất: Reporting có hàm `build_attack_for_report` (= load+build), CLI có `square_config` (chỉ load+replace), factory chỉ build. Reporting cũng có `pgd_at_epsilon` riêng cho epsilon sweep.
- Có nên unify: Một phần. Reporting/notebook layer có lý do riêng để bọc convenience, nhưng `build_attack_for_report` chỉ là 1-line alias cho `AttackFactory.build(load_attack_config(name))` — có thể inline hoặc đổi tên rõ hơn.
- Hướng unify: Loại bỏ `build_attack_for_report`, dùng `AttackFactory.build(load_attack_config(name))` trực tiếp; giữ `pgd_at_epsilon` vì có replace logic riêng.

### Ununified logic: epsilon sweep replace pattern

- File/vị trí liên quan: [src/cli/sweep.py:38-42](src/cli/sweep.py#L38-L42), [src/reporting/attack.py:24-31](src/reporting/attack.py#L24-L31)
- Logic tương tự: Replace `(epsilon, alpha=min(base.alpha, epsilon) if epsilon>0 else 0.0)` trên `AttackConfig`.
- Điểm chưa thống nhất: `attack.py::pgd_at_epsilon` cũng đặt `random_start = base.random_start and epsilon > 0`; `sweep.py::run_sweep_point` không.
- Có nên unify: **Có** — share một helper `replace_attack_epsilon(cfg, epsilon)`.
- Hướng unify: Đặt helper trong `src/attacks/factory.py` hoặc `src/experiments/config_loader.py`.

### Ununified logic: load_training_config inner_attack vs load_attack_config

- File/vị trí liên quan: [src/experiments/config_loader.py:125-185](src/experiments/config_loader.py#L125-L185), [src/experiments/config_loader.py:72-122](src/experiments/config_loader.py#L72-L122)
- Logic tương tự: Build `AttackConfig` dataclass từ dict.
- Điểm chưa thống nhất: `load_attack_config` validate required/optional keys; `load_training_config` chỉ silently lấy keys. `load_training_config` không hỗ trợ `p_init`/`loss` cho inner_attack (rare nhưng inconsistent).
- Có nên unify: **Có** — share constructor.
- Hướng unify: Tách `_build_attack_config_from_dict(resolved: dict, source: str)` được dùng bởi cả hai loader.

### Ununified logic: tracker context boilerplate trong sweep.py vs CLI scripts

- File/vị trí liên quan: [src/cli/sweep.py:51-66](src/cli/sweep.py#L51-L66), [src/cli/runner.py:32-41](src/cli/runner.py#L32-L41), [scripts/run_white_box.py:36-44](scripts/run_white_box.py#L36-L44)
- Logic tương tự: Mở `ExperimentTracker` với tracking_uri/json_dir/enable.
- Điểm chưa thống nhất: Scripts dùng `ctx.build_tracker(run_name, tags)`; sweep manually construct `ExperimentTracker(...)` không dùng ScriptContext.
- Có nên unify: **Có**.
- Hướng unify: Pass `ScriptContext` xuống `run_sweep`, sweep dùng `ctx.build_tracker(run_name, tags)`.

### Ununified logic: synthetic batch fallback ở data/smoke.py vs reporting/data_loaders.py

- File/vị trí liên quan: [src/data/smoke.py:9-14](src/data/smoke.py#L9-L14), [src/reporting/data_loaders.py:33-36](src/reporting/data_loaders.py#L33-L36)
- Logic tương tự: Tạo `(x, y)` random shape (n, 3, 32, 32) + label.
- Điểm chưa thống nhất: data/smoke trả `DataLoader`, reporting/data_loaders.synthetic_batch trả tensor pair.
- Có nên unify: Không bắt buộc nhưng có thể share `synthetic_batch` rồi wrap.
- Hướng unify: `make_smoke_loader` dùng `synthetic_batch` từ reporting hoặc đặt cả hai vào `src/data/smoke.py`.

### Ununified logic: variant ↔ checkpoint path string ("clean", "adv", "adversarial")

- File/vị trí liên quan: [src/experiments/checkpoint_paths.py:18-24](src/experiments/checkpoint_paths.py#L18-L24), [src/reporting/model_registry.py:13](src/reporting/model_registry.py#L13), [src/experiments/config.py:153](src/experiments/config.py#L153), [scripts/run_white_box.py:23](scripts/run_white_box.py#L23)
- Logic tương tự: Đại diện "trained with adversarial loop" có ba representation: `"adv"` (CLI flag + path stem), `"adversarial"` (TrainingConfig.mode literal + CheckpointVariant literal trong reporting/model_registry.py).
- Điểm chưa thống nhất: TrainingConfig.mode = "adversarial", checkpoint_paths.variant_checkpoint_path = "adv", ReportingCheckpoint.variant = "adversarial" (CheckpointVariant literal).
- Có nên unify: **Có**.
- Hướng unify: Chọn 1 nhãn duy nhất ("adv" gọn hơn cho path stem). Đổi `CheckpointVariant = Literal["clean", "adv"]` cho `model_registry.py`; sửa `TrainingConfig.mode` literal nếu cần. Hoặc add explicit mapping `MODE_TO_VARIANT = {"clean": "clean", "adversarial": "adv"}`.

## 6. Naming issues

| Tên hiện tại | Vị trí | Vấn đề | Tên đề xuất | Lý do |
|---|---|---|---|---|
| `tests/test_models/test_normalize_wrapper.py` | tests/test_models/ | File reference module đã rename: `normalize_wrapper.py` → `normalizer.py` (deleted in git status). | `test_normalizer.py` | Khớp tên module hiện tại. |
| `test_normalize_wrapper_*` (test functions) | tests/test_models/test_normalize_wrapper.py | Function names dính tên class cũ `NormalizedModel`/module cũ. | `test_normalizer_*` | Khớp class hiện tại `Normalizer`. |
| `test_global_log_rotates` | tests/test_tracking/test_mlflow_logger.py | Test không thực sự test rotation — chỉ assert hai run log vào cùng file. | `test_global_log_appends_across_runs` | Mô tả đúng hành vi đang assert. |
| `test_amp_does_not_break_determinism_within_run` | tests/test_reproducibility/test_seed_determinism.py | Body chỉ chạy 2 forward pass deterministic; AMP không thực sự load-bearing. | `test_eval_forward_pass_is_deterministic` | Tên hiện tại implies hơn so với assertion thực. |
| `"adv"` vs `"adversarial"` | xuyên codebase (TrainingConfig.mode, CheckpointVariant, CLI --variant, checkpoint path stem) | 3 representation cho cùng concept. | Chọn `"adv"` (path stem) hoặc `"adversarial"` (full word) toàn bộ. | DRY/least-surprise. |
| `NormalizedModel` (chỉ trong docstring) | src/attacks/base.py:5, src/experiments/config.py:115-117 | Class thật là `Normalizer`. Docstring cũ chưa rename. | `Normalizer` | Khớp code. |
| `build_normalized_model` | src/models/builders.py:70 | Trả về `Normalizer` — tên hợp lý nhưng `NormalizedModel` không còn tồn tại. | OK as is, hoặc `build_with_normalizer` | Optional rename để bám lớp hiện tại. |
| `build_attack_for_report` | src/reporting/attack.py:12 | Chỉ là alias 1-line cho `AttackFactory.build(load_attack_config(name))`. Tên không hé lộ gì khác. | Xóa hoặc đổi thành `load_and_build_attack` | Giảm thừa. |
| `PAIR_FILES` | src/cli/transfer.py:13 | Tên chung; thực ra là registry mode→YAML path. | `TRANSFER_MODE_CONFIGS` | Rõ scope. |
| `run_pair` | src/cli/transfer.py:24 | Hàm chạy 1 transfer evaluation pair — tên OK nhưng cùng module có `pair_spec` / `_cross_arch_spec` không đồng đều. | `run_transfer_pair` | Rõ ràng hơn ở entry-point. |
| `run_sweep` / `run_sweep_point` | src/cli/sweep.py | OK. Nhưng `base` (parameter ở run_sweep_point) ambiguous. | `base_attack_config` | base có thể là nhiều thứ. |
| `_optional_int`, `_optional_float` | src/experiments/config_loader.py:301-308 | Helper dùng `or` fallback bị bug với value 0; tên không hint điều đó. | `_get_or_none_int` | Tên hiện tại không lệch nghĩa nhưng chỉnh nhẹ rõ hơn. |
| `nb01_protocol`, `nb02_clean_models`, ... | src/reporting/nb*.py | Prefix `nb01_` lặp tên file/notebook. OK với scaffolding test, nhưng tên function lệch convention Python (số ở đầu). | Giữ as-is nếu test phụ thuộc; ngược lại `report_protocol`, `report_clean_models`. | Test scaffold hardcode prefix nên có rủi ro break. |
| `evaluation_inputs` vs `evaluation_loader` | src/reporting/data_loaders.py | Tên không phân biệt rõ "1 batch" vs "loader". | `fixed_eval_batch`, `subset_eval_loader` | Rõ semantic. |
| `_run_transfer_evaluation` | src/experiments/runner.py:184 | Logic gần như identical với `AttackEvaluator.run()`; tên private không phản ánh việc duplicate. | (Xóa khi unify) | Hoặc đổi thành `_transfer_loop`, ngắn hơn. |
| `runner.py` ở src/cli/ và src/experiments/ và src/evaluation/ | nhiều thư mục | Ba file cùng tên `runner.py` khác trách nhiệm — confusion khi grep. | `cli_bootstrap.py`, `experiment_runner.py`, `attack_evaluator.py` | Rõ ngữ cảnh. |
| `_log_evaluation_result` | src/experiments/runner.py:240 | Function thiếu psnr/ssim/confidence_drop trong log; tên ngầm hiểu là log đầy đủ. | `_log_core_eval_metrics` | Nếu giữ partial logging, đặt tên đúng phạm vi. |
| `apgd_at.yaml` | configs/training/apgd_at.yaml | Hàm bootstrap truyền key `"adversarial"` — không khớp. | Đổi thành `adversarial.yaml` (sync với code) **hoặc** sửa script gọi `training="apgd_at"`. | Bắt buộc fix vì script đang gãy. |
| `cross_seed_pairs.yaml` | configs/transfer/cross_seed_pairs.yaml | Mode đã bị remove khỏi `PAIR_FILES` nhưng file vẫn còn — orphan config. | Xóa file. | Cleanup. |
| `mlflow.sh` | scripts/mlflow.sh | README chỉ tới `mlflow_server.sh`. | Đổi thành `mlflow_server.sh` hoặc sửa README. | README claim sai. |
| `save_every_epochs`, `resume_from` | configs/training/apgd_at.yaml | Keys tồn tại YAML nhưng không có field tương ứng trong `TrainingConfig`; silently ignored. | Xóa nếu không implement. | Dead config. |
| `CheckpointVariant` | src/reporting/model_registry.py:13 | Literal["clean", "adversarial"] nhưng `variant_checkpoint_path` dùng `"clean"/"adv"`. | `Literal["clean", "adv"]` (sync) | Sync với rest of codebase. |
| `ReportingCheckpoint`, `ReportingModelPair` | src/reporting/model_registry.py | Prefix `Reporting` không tên gọi sai — nhưng dài. | `Checkpoint`, `ModelPair` (trong reporting namespace) | Namespace đã rõ. |
| `last_query_count` | src/attacks/square.py:22 | Field public, đặt tên thiếu prefix; trông như private state. | OK, nhưng có thể chuyển thành property trên result. | Optional. |
| `Square` (in YAML name field) vs "SQUARE" (registry key) | configs/attack/square_5000.yaml | Case mixing — registry uppercase, YAML title. | Chuẩn hóa toàn bộ về uppercase trong YAML. | Consistency. |

## 7. Plan refactor

Plan được sắp xếp theo thứ tự ưu tiên: BUGS critical → unify duplications → naming.

### Step 1: Fix scripts/train_adversarial.py training-config-name bug

- Mục tiêu: Khôi phục entry point adversarial training.
- File cần sửa: [scripts/train_adversarial.py](scripts/train_adversarial.py)
- Việc cần làm: Đổi `bootstrap(args, arch=args.arch, training="adversarial")` thành `bootstrap(args, arch=args.arch, training="apgd_at")`.
- Vấn đề được fix: Bug #1 (FileNotFoundError khi training).
- Ghi chú: Kiểm tra reproduce.sh không phụ thuộc string `"adversarial"` ở nơi khác; tests/test_scripts_smoke có import script này (chỉ import test) — không break.

### Step 2: Fix NB04 schema mismatch giữa CSV writer và figure renderer

- Mục tiêu: Figure NB04 render được data khi có checkpoint.
- File cần sửa: [src/reporting/nb04_main_results.py](src/reporting/nb04_main_results.py)
- Việc cần làm: Cập nhật `_render_main_figure` và `_render_time_vs_asr` đọc keys `asr` / `time_per_image_ms` thay vì `_mean`/`_std`. Filter `square_rows` ở dòng 50-54 vẫn đúng vì NB08 (multi-run square) thực sự dùng `_mean` schema — không đổi. Bỏ `yerr=stds` trong bar plot (single-seed không có std).
- Vấn đề được fix: Bug NB04 figure pending-forever.
- Ghi chú: Kiểm tra test `test_notebook_report_bugs.py::nb04_main_results_csv_schema` để confirm CSV schema không thay đổi — chỉ rendering thay đổi.

### Step 3: Fix scripts/run_transfer.py & PAIR_FILES, xóa cross_seed orphan

- Mục tiêu: Đồng bộ README, NB09, code, config.
- File cần sửa: [README.md](README.md), [notebooks/09_transfer_attack_analysis.ipynb](notebooks/09_transfer_attack_analysis.ipynb), [configs/transfer/cross_seed_pairs.yaml](configs/transfer/cross_seed_pairs.yaml), [src/cli/transfer.py](src/cli/transfer.py), [src/reporting/mlflow_queries.py](src/reporting/mlflow_queries.py)
- Việc cần làm: Xóa file `cross_seed_pairs.yaml`. Bỏ branch `if mode == "cross_seed"` trong `read_transfer_mlflow_runs`. Sửa README block CLI và markdown NB09 cell 2 để chỉ list `cross_arch` + `gray_box`. 
- Vấn đề được fix: Stale config, README claim, NB09 markdown.

### Step 4: Fix `np.mean(...) or 0.0` empty-list bug trong nb09

- Mục tiêu: Bar chart gray-box gán 0.0 thay vì NaN.
- File cần sửa: [src/reporting/nb09_transfer_analysis.py:57-60](src/reporting/nb09_transfer_analysis.py#L57-L60)
- Việc cần làm: Tách helper `_mean_asr(rows, arch, variant)` trả về `0.0` khi list rỗng.
- Vấn đề được fix: NaN trong bar height.

### Step 5: Cập nhật README các path file đã bị xóa hoặc đổi tên

- Mục tiêu: README chỉ tới file thật.
- File cần sửa: [README.md](README.md)
- Việc cần làm: 
  - Xóa dòng `scripts/download_robustbench_wrn.py`
  - Đổi `bash scripts/mlflow_server.sh` thành `bash scripts/mlflow.sh` (hoặc rename script)
  - Xóa `[--resume PATH]` khỏi train_adversarial CLI block (cho tới khi implement)
  - Update CLI block reflect actual choices
- Vấn đề được fix: Tài liệu sai.

### Step 6: Cleanup config keys dead

- Mục tiêu: Bỏ key YAML không được loader đọc.
- File cần sửa: [configs/training/apgd_at.yaml](configs/training/apgd_at.yaml)
- Việc cần làm: Xóa `save_every_epochs: 5` và `resume_from: null` (không có field tương ứng).
- Vấn đề được fix: Misleading config (giả vờ có feature).

### Step 7: Cập nhật notebook markdown để khớp single-seed reality

- Mục tiêu: Notebooks honest theo claim README.
- File cần sửa: [notebooks/01_research_protocol.ipynb](notebooks/01_research_protocol.ipynb), [notebooks/02_baseline_clean_models.ipynb](notebooks/02_baseline_clean_models.ipynb), [notebooks/04_main_quantitative_results.ipynb](notebooks/04_main_quantitative_results.ipynb), [notebooks/05_vulnerability_analysis.ipynb](notebooks/05_vulnerability_analysis.ipynb), [notebooks/09_transfer_attack_analysis.ipynb](notebooks/09_transfer_attack_analysis.ipynb), [notebooks/10_architecture_robustness_comparison.ipynb](notebooks/10_architecture_robustness_comparison.ipynb)
- Việc cần làm: Cập nhật markdown cells: seed = `{42}`, bỏ "mean ± std", "Welch test", "5 seeds", "cross_seed". NB07 disclosure đã ok.
- Vấn đề được fix: Honest-notebooks principle.
- Ghi chú/rủi ro: test `test_notebook_scaffold.py` check substring "WRN RobustBench fallback" trong NB07b/NB11 — không động vào.

### Step 8: Fix scripts/reproduce.sh epoch count

- Mục tiêu: Reproduce thực sự reproduce.
- File cần sửa: [scripts/reproduce.sh](scripts/reproduce.sh)
- Việc cần làm: Đổi `--epochs 5` thành `--epochs 100` cho cả train_clean và train_adversarial loops. Hoặc thêm flag `--quick` để giữ behavior nhanh và phân biệt với full.
- Vấn đề được fix: 119-141 GPU-hours claim trong README.

### Step 9: Unify "checkpoint or smoke" helper

- Mục tiêu: 1 source of truth.
- File cần sửa: [src/cli/loader.py](src/cli/loader.py), [src/experiments/runner.py](src/experiments/runner.py)
- Việc cần làm: Xóa `ExperimentRunner._load_checkpoint_or_smoke_model`. Trong các call site, import và gọi `load_checkpoint_or_smoke(...)`. Cập nhật signature nếu cần (đổi `model_config` keyword first).
- Vấn đề được fix: Anti-pattern duplication, vi phạm rule #6 audit cũ.

### Step 10: Unify AttackEvaluator + transfer evaluation

- Mục tiêu: Loại bỏ vòng lặp trùng lặp giữa single-model và transfer.
- File cần sửa: [src/evaluation/runner.py](src/evaluation/runner.py), [src/experiments/runner.py](src/experiments/runner.py)
- Việc cần làm: Thêm tham số `perturb_model: Normalizer | None = None` vào `AttackEvaluator`. Vòng lặp dùng `(perturb_model or self.model)` cho `attack.perturb()`, `self.model` cho dự đoán/metrics. `ExperimentRunner.evaluate_transfer` chỉ cần `AttackEvaluator(victim, attack, loader, device, perturb_model=surrogate).run()` rồi log như cũ.
- Vấn đề được fix: Bug "transfer không có keep_per_sample", anti-pattern duplication.
- Ghi chú: Test transfer hiện tại không cover `keep_per_sample`, nhưng helper trở nên hữu dụng cho NB09 sau này.

### Step 11: Unify attack-config builder giữa load_attack_config và load_training_config

- Mục tiêu: Cùng validation + cùng schema cho cả root-attack và inner-attack.
- File cần sửa: [src/experiments/config_loader.py](src/experiments/config_loader.py)
- Việc cần làm: Tách `_build_attack_from_dict(resolved: dict, source: str) -> AttackConfig` được dùng bởi cả `load_attack_config` và inner_attack branch trong `load_training_config`. 
- Vấn đề được fix: Inconsistency giữa hai loader.

### Step 12: Unify epsilon-replace helper

- Mục tiêu: Một helper duy nhất `replace_attack_epsilon`.
- File cần sửa: [src/reporting/attack.py](src/reporting/attack.py), [src/cli/sweep.py](src/cli/sweep.py)
- Việc cần làm: Đặt helper `replace_attack_epsilon(cfg, epsilon) -> AttackConfig` trong `src/attacks/factory.py` (hoặc utility module mới). Cả `pgd_at_epsilon` và `run_sweep_point` import.
- Vấn đề được fix: Logic giống nhau, hai cách viết khác.

### Step 13: Đồng bộ naming "adv" vs "adversarial"

- Mục tiêu: 1 nhãn duy nhất cho variant.
- File cần sửa: [src/reporting/model_registry.py](src/reporting/model_registry.py), [src/experiments/config.py](src/experiments/config.py) (optional), [src/attacks/base.py](src/attacks/base.py) (docstring), [src/experiments/config.py](src/experiments/config.py) (docstring), notebooks (nếu có)
- Việc cần làm: Đổi `CheckpointVariant = Literal["clean", "adv"]`. Update `ReportingCheckpoint.variant` value site (line 74) thành `"adv"`. Tests check `variant` ở pair_spec dùng `"adv"` đã (gray_box_pairs.yaml). Update các docstring nói `NormalizedModel` → `Normalizer`.
- Vấn đề được fix: 3-representation confusion.
- Ghi chú: TrainingConfig.mode literal có thể giữ "adversarial" vì nó là phase mode (semantically khác); chỉ unify variant.

### Step 14: Đổi tên 3 file `runner.py` để giảm collision

- Mục tiêu: Tăng readability khi grep.
- File cần sửa: rename [src/cli/runner.py](src/cli/runner.py) → `src/cli/bootstrap.py`; [src/experiments/runner.py](src/experiments/runner.py) → `src/experiments/experiment_runner.py`; [src/evaluation/runner.py](src/evaluation/runner.py) → `src/evaluation/attack_evaluator.py` (nội dung đã là `AttackEvaluator`).
- Việc cần làm: Mass rename + cập nhật mọi import (chỉ trong src + scripts + tests).
- Vấn đề được fix: Naming collision.
- Ghi chú/rủi ro: Touches nhiều file; cần grep/replace cẩn thận. Test có thể có hardcode path import — kiểm chứng từng test file.

### Step 15: Đổi tên test_normalize_wrapper.py → test_normalizer.py

- Mục tiêu: Khớp module-under-test.
- File cần sửa: rename [tests/test_models/test_normalize_wrapper.py](tests/test_models/test_normalize_wrapper.py) → `test_normalizer.py`, đổi tên các function `test_normalize_wrapper_*` → `test_normalizer_*`.
- Vấn đề được fix: Stale test naming.

### Step 16: Đổi tên test mislabeled

- Mục tiêu: Test name phản ánh đúng assertion.
- File cần sửa: [tests/test_tracking/test_mlflow_logger.py](tests/test_tracking/test_mlflow_logger.py), [tests/test_reproducibility/test_seed_determinism.py](tests/test_reproducibility/test_seed_determinism.py)
- Việc cần làm: `test_global_log_rotates` → `test_global_log_appends_across_runs`. `test_amp_does_not_break_determinism_within_run` → `test_eval_forward_pass_is_deterministic`.

### Step 17: Refactor anti-pattern test patterns (optional)

- Mục tiêu: Nhất quán pytest style.
- File cần sửa: [tests/test_attacks/test_apgd.py](tests/test_attacks/test_apgd.py), [tests/test_attacks/test_square.py](tests/test_attacks/test_square.py), [tests/test_scripts/test_scripts_smoke.py](tests/test_scripts/test_scripts_smoke.py), [tests/test_tracking/test_mlflow_logger.py](tests/test_tracking/test_mlflow_logger.py)
- Việc cần làm: Thay bare try/except + `else: raise AssertionError` bằng `pytest.raises`. Thay `object.__setattr__(cfg, ...)` bằng fixture-based config builder.
- Ghi chú: Optional cleanup, không fix bug.

### Step 18: Cleanup egg-info stale entry

- Mục tiêu: Bỏ artifact build dirty.
- File cần sửa: [.gitignore](.gitignore) (thêm), xóa thư mục `pgd_cifar10_experiment.egg-info/`.
- Việc cần làm: Add `*.egg-info/` vào .gitignore; `git rm -r --cached pgd_cifar10_experiment.egg-info`.
- Vấn đề được fix: Stale build artifact tracking.

### Step 19: Bổ sung PSNR/SSIM/confidence_drop vào ExperimentRunner._log_evaluation_result

- Mục tiêu: Log đầy đủ EvaluationResult.
- File cần sửa: [src/experiments/runner.py:240-250](src/experiments/runner.py#L240-L250)
- Việc cần làm: Thêm `"psnr_mean"`, `"ssim_mean"`, `"confidence_drop_mean"` vào metrics dict.
- Vấn đề được fix: Partial metric logging.

### Step 20: Cleanup CleanTrainer/AdversarialTrainer unused val_loader

- Mục tiêu: Loại bỏ tham số chết.
- File cần sửa: [src/training/base.py](src/training/base.py), [src/training/clean.py](src/training/clean.py), [src/training/adversarial.py](src/training/adversarial.py), [src/experiments/runner.py](src/experiments/runner.py)
- Việc cần làm: Xóa `val_loader` khỏi `BaseTrainer.__init__`, `_loaders()` trả 1 loader; hoặc thực sự dùng val_loader để compute val metric mỗi epoch (preferred if epochs > 1).
- Vấn đề được fix: API noise + sai semantic của `best_metric`.

## 8. Checklist sau refactor

- [ ] `scripts/train_adversarial.py` chạy được end-to-end với checkpoint thật (bug #1).
- [ ] NB04 figure render bar chart khi `results/tables/main_results.csv` có data (bug #2).
- [ ] `scripts/run_transfer.py --mode cross_seed` raise rõ ràng, không có file orphan `cross_seed_pairs.yaml` (bug #4).
- [ ] `_render_gray_box` không sinh NaN khi (arch, variant) rỗng (bug #3).
- [ ] README + NB09 markdown đồng bộ về modes thật và file thật (bugs #4, #5).
- [ ] `configs/training/apgd_at.yaml` không còn key dead (`save_every_epochs`, `resume_from`).
- [ ] Notebook markdown phản ánh single-seed reality (NB01/02/04/05/09/10).
- [ ] `scripts/reproduce.sh` thực sự chạy 100 epochs ở full mode (hoặc README cập nhật để khớp với 5 epochs).
- [ ] `load_checkpoint_or_smoke` chỉ tồn tại ở duy nhất một nơi; ExperimentRunner delegate.
- [ ] `AttackEvaluator.run()` cover được cả single-model lẫn transfer (thông qua `perturb_model`).
- [ ] `load_attack_config` và inner_attack trong `load_training_config` dùng cùng builder.
- [ ] Single epsilon-replace helper được dùng bởi cả epsilon sweep và reporting helper.
- [ ] Variant nhãn `"clean"` / `"adv"` thống nhất ở mọi điểm dùng path stem (CLI variant, ReportingCheckpoint, CheckpointVariant Literal).
- [ ] Test file rename `test_normalize_wrapper.py` → `test_normalizer.py`.
- [ ] Tests mislabeled (`test_global_log_rotates`, `test_amp_does_not_break_determinism_within_run`) đã đổi tên.
- [ ] (Optional) 3 `runner.py` được rename để tránh collision khi grep.
- [ ] PSNR/SSIM/confidence_drop được log trong tracker.
- [ ] `val_loader` được dùng thật hoặc xóa khỏi trainer API.
- [ ] `pgd_cifar10_experiment.egg-info/` không còn track bởi git; `.gitignore` cập nhật.
- [ ] Toàn bộ test suite (`pytest tests/ -q`) pass.
- [ ] Reproducer smoke `bash scripts/reproduce.sh --smoke` chạy không lỗi.
- [ ] Code dễ đọc, naming nhất quán, không còn duplication critical (`load_checkpoint_or_smoke`, evaluation loop, attack config builder).
- [ ] Không file nào trong codebase bị bỏ sót trong audit (đã liệt kê đủ ở mục 2).
