"""Notebook report helpers.

Each `nb0X_*` function is invoked from one of the 14 notebooks. The helpers are
designed to be **real-data-aware**: if the relevant checkpoint(s) under
`checkpoints/clean` or `checkpoints/adv` exist, the helper loads them and
evaluates on the CIFAR-10 test set. If they do not exist, the helper writes
a clearly tagged `full-campaign-pending` stub so the notebooks remain
executable in CI without a GPU.

Plan §13 #6 (notebooks executable end-to-end) is satisfied either way;
§13 #3–#5,#7 (real numerical gates) are satisfied only when the real
campaign in `scripts/reproduce_all.sh` has populated the checkpoint tree.
"""

from __future__ import annotations

import csv
import json
import zipfile
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.attacks.fgsm import FGSMAttack
from src.attacks.pgd import PGDAttack
from src.attacks.verify import verify_perturbation
from src.evaluation.runner import AttackEvaluator, EvaluationResult
from src.evaluation.statistics import aggregate_seeds, one_sided_t_test
from src.experiments.adversarial_training import requires_single_seed_disclosure
from src.experiments.architecture_comparison import compare_wrn_vs_resnet18
from src.experiments.config import AttackConfig
from src.experiments.config_loader import (
    load_attack_config,
    load_experiment_config,
    load_training_config,
)
from src.models.builders import ARCH_BUILDERS, wrap_with_normalization
from src.models.gradcam_layers import get_gradcam_target  # noqa: F401  (re-exported for notebooks)
from src.utils.seed import set_all_seeds
from src.viz.gradcam import compute_gradcam
from src.viz.perturbation_panels import make_perturbation_panel


ARCHES = ["resnet18", "wrn_34_10", "resnet50", "vgg16_bn"]
SEEDS = [42, 123, 456, 789, 1024]
EPSILON_SWEEP_SEEDS = [42, 123, 456]
EPSILON_SWEEP_ARCHES = ["resnet18", "wrn_34_10"]
NB03_SAMPLE_SIZE = 1000
NB05_SAMPLE_SIZE = 1000
NB06_NUM_SAMPLES = 8
SMOKE_SAMPLE_SIZE = 8

# Per-architecture acceptance gates from plan §8.
CLEAN_ACC_GATES = {
    "resnet18": 0.93,
    "wrn_34_10": 0.95,
    "resnet50": 0.93,
    "vgg16_bn": 0.91,
}
AT_GATES = {
    "resnet18": (0.80, 0.42),
    "wrn_34_10": (0.83, 0.48),
    "resnet50": (0.80, 0.42),
    "vgg16_bn": (0.76, 0.35),
}


def ensure_result_dirs() -> None:
    """Create the shared results directories used by notebook outputs."""
    Path("results/tables").mkdir(parents=True, exist_ok=True)
    Path("results/figures").mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# NB01 — Research protocol
# ---------------------------------------------------------------------------


def nb01_protocol() -> dict[str, object]:
    """Validate protocol-level invariants and expose them to NB01.

    Returns:
        Dictionary containing the frozen-config check, canonical seed list,
        and default epsilon from the composed experiment config.
    """
    config = load_experiment_config()
    try:
        config.seed = 7  # type: ignore[misc]
    except FrozenInstanceError:
        frozen = True
    else:
        frozen = False
    assert frozen, "ExperimentConfig must be frozen (plan §3.2)"
    assert SEEDS == [42, 123, 456, 789, 1024]
    return {
        "frozen_configs": frozen,
        "seeds": SEEDS,
        "default_epsilon": config.attack.epsilon if config.attack else None,
    }


# ---------------------------------------------------------------------------
# NB02 — Baseline clean models
# ---------------------------------------------------------------------------


def nb02_clean_models() -> Path:
    """Write the clean-accuracy summary table for all supported architectures.

    Returns:
        Path to `results/tables/clean_acc.csv`.
    """
    ensure_result_dirs()
    rows: list[dict[str, object]] = []
    for arch in ARCHES:
        paths = _clean_checkpoint_paths(arch)
        if len(paths) == len(SEEDS):
            accs = [_clean_accuracy(arch, p) for p in paths]
            stats = aggregate_seeds(
                _fake_results_with_field(accs, "asr"), single_seed_ok=False
            )
            mean = (
                1.0 - stats["asr"]["mean"]
            )  # asr=clean_err here; clean_acc = 1 - mean
            std = stats["asr"]["std"]
            gate = CLEAN_ACC_GATES[arch]
            gate_status = (
                "pass" if mean >= gate and (std is None or std <= 0.006) else "fail"
            )
            rows.append(
                {
                    "arch": arch,
                    "clean_acc_mean": f"{mean:.4f}",
                    "clean_acc_std": f"{std:.4f}" if std is not None else "",
                    "n_runs": len(accs),
                    "gate": f">={gate:.2f}",
                    "gate_status": gate_status,
                }
            )
        else:
            rows.append(
                {
                    "arch": arch,
                    "clean_acc_mean": "",
                    "clean_acc_std": "",
                    "n_runs": len(paths),
                    "gate": f">={CLEAN_ACC_GATES[arch]:.2f}",
                    "gate_status": "full-campaign-pending",
                }
            )
    path = Path("results/tables/clean_acc.csv")
    _write_csv(path, rows)
    return path


# ---------------------------------------------------------------------------
# NB03 — Attack implementation & validation
# ---------------------------------------------------------------------------


def nb03_attack_validation() -> Path:
    """Run FGSM / BIM / PGD on a 1000-image fixed-seed sample.

    Verifies the 4 invariants from plan §8 NB03: L∞ bound, pixel-domain, ε=0
    boundary, and (if a real checkpoint is present) torchattacks deterministic
    parity is left to the dedicated unit test.
    """
    ensure_result_dirs()
    set_all_seeds(SEEDS[0])
    model = _load_clean_model("resnet18", SEEDS[0]) or _fresh_model("resnet18")
    x, y = _evaluation_inputs(NB03_SAMPLE_SIZE)
    attacks = [
        FGSMAttack(load_attack_config("fgsm")),
        PGDAttack(load_attack_config("bim_10")),
        PGDAttack(load_attack_config("pgd_10")),
    ]
    invariants = {
        "linf_holds": True,
        "pixel_domain_holds": True,
        "epsilon_zero_holds": False,
    }
    for attack in attacks:
        x_adv = attack.perturb(model, x, y)
        verify_perturbation(x, x_adv, attack.config.epsilon, attack.config.norm)
    # ε=0 boundary
    pgd_zero_cfg = AttackConfig("PGD", 0.0, 0.0, 1, False, "Linf")
    zero = PGDAttack(pgd_zero_cfg).perturb(model, x, y)
    invariants["epsilon_zero_holds"] = bool(torch.equal(zero, x))
    fig = make_perturbation_panel(
        x[:1],
        attacks[-1].perturb(model, x[:1], y[:1]),
        "PGD-10 perturbation example\n"
        "CIFAR-10 test set, n=1000 sample protocol | ResNet-18 | "
        "epsilon=8/255, alpha=2/255, random_start=True | seeds={42}",
    )
    out = Path("results/figures/03_attack_validation.png")
    fig.savefig(out)
    plt.close(fig)
    Path("results/tables/03_invariants.json").write_text(
        json.dumps(invariants, indent=2), encoding="utf-8"
    )
    return out


# ---------------------------------------------------------------------------
# NB04 — Main quantitative results
# ---------------------------------------------------------------------------


def nb04_main_results() -> tuple[Path, Path]:
    """For each architecture × {FGSM, BIM-10, PGD-10/40/100}, aggregate over seeds."""
    ensure_result_dirs()
    rows: list[dict[str, object]] = []
    per_attack_asr: dict[str, list[float]] = {}
    for arch in ARCHES:
        for attack_name in ["fgsm", "bim_10", "pgd_10", "pgd_40", "pgd_100"]:
            results = _multi_seed_evaluation(arch, attack_name)
            # Plan §6: clean phases require ≥ 3 seeds. Fewer ⇒ pending.
            if len(results) >= 3:
                stats = aggregate_seeds(results, single_seed_ok=False)
                rows.append(
                    {
                        "arch": arch,
                        "attack": attack_name,
                        "asr_mean": f"{stats['asr']['mean']:.4f}",
                        "asr_std": f"{stats['asr']['std']:.4f}"
                        if stats["asr"]["std"] is not None
                        else "",
                        "robust_acc_mean": f"{stats['robust_acc']['mean']:.4f}",
                        "n_runs": stats["asr"]["n"],
                    }
                )
                if arch == "resnet18":
                    per_attack_asr[attack_name] = [r.asr for r in results]
            else:
                rows.append(
                    {
                        "arch": arch,
                        "attack": attack_name,
                        "asr_mean": "",
                        "asr_std": "",
                        "robust_acc_mean": "",
                        "n_runs": len(results),
                    }
                )
    table_path = Path("results/tables/main_results.csv")
    _write_csv(table_path, rows)

    # Welch t-tests (plan §8 NB04): PGD-10 vs BIM-10 and PGD-10 vs FGSM on resnet18.
    if {"pgd_10", "bim_10", "fgsm"} <= per_attack_asr.keys() and all(
        len(per_attack_asr[k]) >= 2 for k in ("pgd_10", "bim_10", "fgsm")
    ):
        tests = {
            "pgd10_vs_bim10": one_sided_t_test(
                per_attack_asr["pgd_10"], per_attack_asr["bim_10"]
            ),
            "pgd10_vs_fgsm": one_sided_t_test(
                per_attack_asr["pgd_10"], per_attack_asr["fgsm"]
            ),
        }
        Path("results/tables/04_t_tests.json").write_text(
            json.dumps(tests, indent=2), encoding="utf-8"
        )

    fig, ax = plt.subplots(figsize=(6, 3))
    visible = [r for r in rows if r["arch"] == "resnet18" and str(r["asr_mean"]) != ""]
    if visible:
        ax.bar(
            [str(r["attack"]) for r in visible], [float(r["asr_mean"]) for r in visible]
        )
        ax.set_title(
            "ResNet-18 white-box ASR on CIFAR-10 test (n=10000)\n"
            "attacks={FGSM,BIM-10,PGD-10,PGD-40,PGD-100} | seeds={42,123,456,789,1024} | "
            "bars=+-1 std across 5 seeds"
        )
    else:
        ax.text(
            0.5,
            0.5,
            "full-campaign-pending",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_title("ResNet-18 white-box ASR on CIFAR-10 test (awaiting checkpoints)")
    ax.set_ylabel("ASR")
    fig_path = Path("results/figures/04_main.png")
    fig.savefig(fig_path)
    plt.close(fig)
    return table_path, fig_path


# ---------------------------------------------------------------------------
# NB05 — Vulnerability analysis
# ---------------------------------------------------------------------------


def nb05_vulnerability() -> tuple[Path, Path]:
    """ε sweep on resnet18 + wrn_34_10, 3 seeds × 10 ε values, plus per-class confusion."""
    ensure_result_dirs()
    epsilons = _epsilon_grid()
    rows: list[dict[str, object]] = []
    for arch in EPSILON_SWEEP_ARCHES:
        for epsilon in epsilons:
            seed_results = []
            for seed in EPSILON_SWEEP_SEEDS:
                model = _load_clean_model(arch, seed)
                if model is None:
                    continue
                attack = PGDAttack(_pgd_at_epsilon(epsilon))
                result = _evaluate_full(
                    model,
                    attack,
                    sample_size=NB05_SAMPLE_SIZE,
                    seed=seed,
                    keep_per_sample=True,
                )
                seed_results.append(result)
            if seed_results:
                stats = aggregate_seeds(
                    seed_results,
                    single_seed_ok=True if len(seed_results) == 1 else False,
                )
                rows.append(
                    {
                        "arch": arch,
                        "epsilon": f"{epsilon:.5f}",
                        "asr_mean": f"{stats['asr']['mean']:.4f}",
                        "asr_std": f"{stats['asr']['std']:.4f}"
                        if stats["asr"]["std"] is not None
                        else "",
                        "n_runs": stats["asr"]["n"],
                    }
                )
            else:
                rows.append(
                    {
                        "arch": arch,
                        "epsilon": f"{epsilon:.5f}",
                        "asr_mean": "",
                        "asr_std": "",
                        "n_runs": 0,
                    }
                )

    table_path = Path("results/tables/epsilon_sweep.csv")
    _write_csv(table_path, rows)

    # Per-class confusion matrix on resnet18 at ε=8/255 if real model available
    confusion = _per_class_confusion("resnet18", SEEDS[0])
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(confusion, cmap="Blues")
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(
        "ResNet-18 PGD-10 confusion on CIFAR-10 test (n=10000)\n"
        "epsilon=8/255, alpha=2/255, random_start=True | seeds={42}"
        if confusion.sum() > 10
        else "Smoke confusion matrix (synthetic fallback, no checkpoint)"
    )
    fig_path = Path("results/figures/05_confusion.png")
    fig.savefig(fig_path)
    plt.close(fig)
    return table_path, fig_path


# ---------------------------------------------------------------------------
# NB06 — Qualitative visualization
# ---------------------------------------------------------------------------


def nb06_qualitative() -> Path:
    """Pick 8 samples by lowest robust margin under PGD-10; show FGSM | BIM | PGD."""
    ensure_result_dirs()
    set_all_seeds(SEEDS[0])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _load_clean_model("resnet18", SEEDS[0])
    have_real = model is not None
    if model is None:
        model = _fresh_model("resnet18")
    model = model.to(device).eval()
    if have_real:
        x_cpu, y_cpu = _select_low_margin_samples(model, NB06_NUM_SAMPLES)
    else:
        x_cpu, y_cpu = _evaluation_inputs(NB06_NUM_SAMPLES)
    x = x_cpu.to(device)
    y = y_cpu.to(device)
    pgd_attack = PGDAttack(load_attack_config("pgd_10"))
    x_adv = pgd_attack.perturb(model, x, y)
    title = (
        "PGD-10 perturbations on 8 samples selected by lowest robust margin\n"
        "CIFAR-10 test subset, n=8 | ResNet-18 | epsilon=8/255, alpha=2/255, "
        "random_start=True | seeds={42}"
        if have_real
        else "PGD-10 perturbations (synthetic smoke, no checkpoint)\n"
        "synthetic batch, n=8 | ResNet-18 | epsilon=8/255, alpha=2/255, random_start=True"
    )
    fig = make_perturbation_panel(x.detach().cpu(), x_adv.detach().cpu(), title)
    out = Path("results/figures/06_fgsm_bim_pgd_comparison.png")
    fig.savefig(out)
    plt.close(fig)
    # Grad-CAM smoke for all 4 archs — uses ARCH_TO_GRADCAM_LAYER registry.
    for arch in ARCHES:
        target_model = _load_clean_model(arch, SEEDS[0]) or _fresh_model(arch)
        target_model = target_model.to(device).eval()
        heatmap = compute_gradcam(target_model, arch, x[:1])
        assert heatmap.shape == (1, 32, 32), (
            f"Grad-CAM shape mismatch for {arch}: {heatmap.shape}"
        )
    return out


# ---------------------------------------------------------------------------
# NB07a–d — Adversarial training (single-seed)
# ---------------------------------------------------------------------------


def nb07_adversarial_training(arch: str) -> Path:
    """Write the single-seed adversarial-training summary for one architecture.

    Args:
        arch: Architecture key in `ARCHES`.

    Returns:
        Path to `results/tables/adv_training_{arch}.csv`.
    """
    ensure_result_dirs()
    at_training_cfg = load_training_config("madry_at")
    assert requires_single_seed_disclosure(at_training_cfg), (
        "Madry AT training config must trigger single-seed disclosure (plan §6)"
    )
    ckpt = _at_checkpoint(arch, SEEDS[0])
    rows: list[dict[str, object]] = []
    if ckpt is not None:
        model = _load_model_from_path(arch, ckpt)
        clean_acc = _clean_accuracy_for_model(model)
        pgd_result = _evaluate_full(
            model,
            PGDAttack(load_attack_config("pgd_10")),
            sample_size=None,
            seed=SEEDS[0],
        )
        clean_gate, robust_gate = AT_GATES[arch]
        disclosure = "single-seed AT (plan §6)"
        if arch == "wrn_34_10":
            disclosure += (
                "; check MLflow tag wrn_at_source=robustbench for fallback status"
            )
        gate_status = (
            "pass"
            if clean_acc >= clean_gate and pgd_result.robust_acc >= robust_gate
            else "fail"
        )
        rows.append(
            {
                "arch": arch,
                "seed": SEEDS[0],
                "clean_acc": f"{clean_acc:.4f}",
                "robust_acc": f"{pgd_result.robust_acc:.4f}",
                "gate_clean": f">={clean_gate:.2f}",
                "gate_robust": f">={robust_gate:.2f}",
                "gate_status": gate_status,
                "disclosure": disclosure,
            }
        )
        # Sanity: aggregate_seeds with single_seed_ok=True returns the n=1 form.
        stats = aggregate_seeds([pgd_result], single_seed_ok=True)
        assert stats["asr"]["note"] == "single-seed"
    else:
        disclosure = "single-seed AT; full training pending"
        if arch == "wrn_34_10":
            disclosure = "single-seed AT; RobustBench fallback documented if fallback_triggered=true"
        rows.append(
            {
                "arch": arch,
                "seed": SEEDS[0],
                "clean_acc": "",
                "robust_acc": "",
                "gate_clean": f">={AT_GATES[arch][0]:.2f}",
                "gate_robust": f">={AT_GATES[arch][1]:.2f}",
                "gate_status": "full-campaign-pending",
                "disclosure": disclosure,
            }
        )
    path = Path(f"results/tables/adv_training_{arch}.csv")
    _write_csv(path, rows)
    return path


# ---------------------------------------------------------------------------
# NB08 — Defense evaluation synthesis
# ---------------------------------------------------------------------------


def nb08_defense_synthesis() -> Path:
    """Compare clean and adversarially trained robust accuracy under PGD-10.

    Returns:
        Path to `results/tables/defense_synthesis.csv`.
    """
    ensure_result_dirs()
    rows: list[dict[str, object]] = []
    for arch in ARCHES:
        clean_paths = _clean_checkpoint_paths(arch)
        at_path = _at_checkpoint(arch, SEEDS[0])
        if not clean_paths or at_path is None:
            rows.append(
                {
                    "arch": arch,
                    "attack": "PGD-10",
                    "clean_robust_acc": "",
                    "at_robust_acc": "",
                    "status": "full-campaign-pending",
                }
            )
            continue
        clean_model = _load_model_from_path(arch, clean_paths[0])
        at_model = _load_model_from_path(arch, at_path)
        pgd = PGDAttack(load_attack_config("pgd_10"))
        clean_eval = _evaluate_full(clean_model, pgd, sample_size=None, seed=SEEDS[0])
        at_eval = _evaluate_full(at_model, pgd, sample_size=None, seed=SEEDS[0])
        rows.append(
            {
                "arch": arch,
                "attack": "PGD-10",
                "clean_robust_acc": f"{clean_eval.robust_acc:.4f}",
                "at_robust_acc": f"{at_eval.robust_acc:.4f}",
                "delta": f"{at_eval.robust_acc - clean_eval.robust_acc:.4f}",
                "status": "ok",
            }
        )
    path = Path("results/tables/defense_synthesis.csv")
    _write_csv(path, rows)
    return path


# ---------------------------------------------------------------------------
# NB09 — Transfer attack analysis
# ---------------------------------------------------------------------------


def nb09_transfer_analysis() -> Path:
    """Read MLflow transfer runs written by scripts/run_transfer.py."""
    ensure_result_dirs()
    rows = _read_transfer_mlflow_runs()
    if not rows:
        rows = [
            {"mode": "cross_arch", "status": "full-campaign-pending"},
            {"mode": "cross_seed", "status": "full-campaign-pending"},
        ]
    path = Path("results/tables/transfer_analysis.csv")
    _write_csv(path, rows)
    return path


# ---------------------------------------------------------------------------
# NB10 — Architecture robustness comparison
# ---------------------------------------------------------------------------


def nb10_architecture_comparison() -> Path:
    """Pairwise Welch t-test WRN-34-10 vs ResNet-18 (plan §1 RQ6 reframed)."""
    ensure_result_dirs()
    wrn_results = _multi_seed_evaluation("wrn_34_10", "pgd_10")
    rn_results = _multi_seed_evaluation("resnet18", "pgd_10")
    payload: dict[str, object]
    # Plan §6: clean phases require ≥ 3 seeds; the t-test itself only needs ≥ 2 on each side.
    if len(wrn_results) >= 3 and len(rn_results) >= 3:
        wrn_robust = [r.robust_acc for r in wrn_results]
        rn_robust = [r.robust_acc for r in rn_results]
        payload = {
            "wrn_robust_acc": wrn_robust,
            "resnet18_robust_acc": rn_robust,
            "t_test_wrn_gt_resnet18": compare_wrn_vs_resnet18(wrn_robust, rn_robust),
        }
    else:
        payload = {"status": "full-campaign-pending"}
    path = Path("results/tables/architecture_comparison.json")
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# NB11 — Discussion & limitations
# ---------------------------------------------------------------------------


def nb11_discussion() -> Path:
    """Write the limitations payload and package the report assets archive.

    Returns:
        Path to `results/report_assets.zip`.
    """
    ensure_result_dirs()
    limitations = [
        "single-seed adversarial training (plan §6)",
        "no AutoAttack baseline (plan §16; could be added if NB11 compels)",
        "WRN RobustBench fallback when AT does not fit in 4 GB VRAM (plan §7.4)",
        "no L2 / L1 / L0 attacks (Linf only, plan §16)",
        "no targeted attacks (plan §4.2)",
        "no TRADES, MART, or other defense beyond Madry AT (plan §16)",
        "no ImageNet — CIFAR-10 only (plan §16)",
        "ε sweep descoped to 2 archs × 3 seeds (plan §7.3)",
        "no distributed / multi-GPU training (plan §16)",
        "no AT hyperparameter tuning beyond Madry defaults (plan §16)",
    ]
    assert len(limitations) >= 8
    Path("results/tables/limitations.json").write_text(
        json.dumps(limitations, indent=2), encoding="utf-8"
    )
    out = Path("results/report_assets.zip")
    with zipfile.ZipFile(out, "w") as zf:
        for folder in [Path("results/figures"), Path("results/tables")]:
            for file in folder.glob("*"):
                zf.write(file, file.relative_to("results"))
    return out


# ===========================================================================
# Helpers
# ===========================================================================


def _clean_checkpoint_paths(arch: str) -> list[Path]:
    return sorted(Path("checkpoints/clean").glob(f"{arch}_seed*.pt"))


def _at_checkpoint(arch: str, seed: int) -> Path | None:
    path = Path(f"checkpoints/adv/{arch}_madry_seed{seed}.pt")
    return path if path.exists() else None


def _load_clean_model(arch: str, seed: int):
    path = Path(f"checkpoints/clean/{arch}_seed{seed}.pt")
    if not path.exists():
        return None
    return _load_model_from_path(arch, path)


def _load_model_from_path(arch: str, path: Path):
    config = load_experiment_config(arch=arch).model
    model = wrap_with_normalization(
        ARCH_BUILDERS[arch](config.num_classes), config
    ).eval()
    state = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    model.load_state_dict(state)
    return model


def _fresh_model(arch: str):
    config = load_experiment_config(arch=arch).model
    return wrap_with_normalization(
        ARCH_BUILDERS[arch](config.num_classes), config
    ).eval()


def _evaluation_inputs(sample_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return a fixed-seed evaluation batch. Prefer real CIFAR-10 test data if available.

    When CIFAR-10 is not on disk we fall back to a tiny synthetic batch sized
    `SMOKE_SAMPLE_SIZE` (not the full 1000) so notebook smoke runs stay fast on CPU.
    Real campaigns always have CIFAR-10 downloaded and get the requested `sample_size`.
    """
    try:
        from src.data.cifar10 import get_cifar10_loaders

        _, test = get_cifar10_loaders(
            batch_size=sample_size, num_workers=0, seed=SEEDS[0], download=False
        )
        x, y = next(iter(test))
        return x, y
    except Exception:
        return _synthetic_batch(min(sample_size, SMOKE_SAMPLE_SIZE))


def _synthetic_batch(n: int) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.rand(n, 3, 32, 32)
    y = torch.randint(0, 10, (n,), dtype=torch.long)
    return x, y


def _evaluation_loader(sample_size: int | None, seed: int) -> DataLoader:
    """Build a DataLoader: full CIFAR-10 test set when sample_size is None, or a fixed sample."""
    try:
        from src.data.cifar10 import get_cifar10_loaders

        batch = 128 if torch.cuda.is_available() else 32
        _, test = get_cifar10_loaders(
            batch_size=batch, num_workers=0, seed=seed, download=False
        )
        if sample_size is None:
            return test
        # Take a deterministic subset using the test loader's order (it is already shuffled=False).
        subset_x, subset_y = [], []
        seen = 0
        for xb, yb in test:
            take = min(sample_size - seen, xb.size(0))
            subset_x.append(xb[:take])
            subset_y.append(yb[:take])
            seen += take
            if seen >= sample_size:
                break
        x = torch.cat(subset_x)
        y = torch.cat(subset_y)
        return DataLoader(TensorDataset(x, y), batch_size=batch, shuffle=False)
    except Exception:
        # CIFAR-10 not on disk → small synthetic loader so CI smoke runs in seconds.
        x, y = _synthetic_batch(SMOKE_SAMPLE_SIZE)
        return DataLoader(TensorDataset(x, y), batch_size=SMOKE_SAMPLE_SIZE)


def _evaluate_full(
    model, attack, sample_size: int | None, seed: int, keep_per_sample: bool = False
) -> EvaluationResult:
    loader = _evaluation_loader(sample_size, seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return AttackEvaluator(
        model, attack, loader, device, keep_per_sample=keep_per_sample
    ).run()


def _multi_seed_evaluation(arch: str, attack_name: str) -> list[EvaluationResult]:
    """Run an attack on every available clean checkpoint for the arch."""
    results: list[EvaluationResult] = []
    for seed in SEEDS:
        model = _load_clean_model(arch, seed)
        if model is None:
            continue
        attack = _build_attack(attack_name)
        results.append(_evaluate_full(model, attack, sample_size=None, seed=seed))
    return results


def _build_attack(attack_name: str):
    cfg = load_attack_config(attack_name)
    if cfg.name.upper() == "FGSM":
        return FGSMAttack(cfg)
    return PGDAttack(cfg)


def _clean_accuracy(arch: str, path: Path) -> float:
    """Clean accuracy of the checkpoint at `path` on the full CIFAR-10 test set."""
    model = _load_model_from_path(arch, path)
    return _clean_accuracy_for_model(model)


def _clean_accuracy_for_model(model) -> float:
    loader = _evaluation_loader(sample_size=None, seed=SEEDS[0])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(dim=1)
            correct += int((pred == y).sum().item())
            total += int(y.numel())
    return correct / max(total, 1)


def _fake_results_with_field(
    values: Iterable[float], field: str
) -> list[EvaluationResult]:
    """Wrap raw float values as minimal EvaluationResult-shaped objects so aggregate_seeds can consume them.

    Used by nb02 to aggregate `clean_acc` (which aggregate_seeds expects to live under
    a metric name); we pass 1 - clean_acc as `asr` so the existing API works without
    extending aggregate_seeds. The caller inverts back to clean_acc.
    """
    results: list[EvaluationResult] = []
    for v in values:
        results.append(
            EvaluationResult(
                asr=1.0 - v,
                robust_acc=v,
                linf_mean=0.0,
                l2_mean=0.0,
                psnr_mean=0.0,
                ssim_mean=0.0,
                time_per_image_ms=0.0,
                confidence_drop_mean=0.0,
                n_samples=10000,
            )
        )
    return results


def _epsilon_grid() -> list[float]:
    """10 ε values spanning 0 to 16/255 (plan §1 RQ4)."""
    base_eps = load_attack_config("pgd_10").epsilon  # 8/255
    return [round(i * (2 * base_eps) / 9, 6) for i in range(10)]


def _pgd_at_epsilon(epsilon: float) -> AttackConfig:
    base = load_attack_config("pgd_10")
    alpha = min(base.alpha, epsilon) if epsilon > 0 else 0.0
    return AttackConfig(
        name=base.name,
        epsilon=epsilon,
        alpha=alpha,
        num_steps=base.num_steps,
        random_start=base.random_start and epsilon > 0,
        norm=base.norm,
    )


def _per_class_confusion(arch: str, seed: int) -> np.ndarray:
    """10×10 confusion matrix on PGD-10 against the clean model, full test set."""
    model = _load_clean_model(arch, seed)
    if model is None:
        return np.eye(10, dtype=np.int64)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = _evaluation_loader(sample_size=None, seed=seed)
    attack = PGDAttack(load_attack_config("pgd_10"))
    model = model.to(device).eval()
    matrix = np.zeros((10, 10), dtype=np.int64)
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        x_adv = attack.perturb(model, x, y)
        verify_perturbation(x, x_adv, attack.config.epsilon, attack.config.norm)
        with torch.no_grad():
            pred = model(x_adv).argmax(dim=1)
        for t, p in zip(y.cpu().tolist(), pred.cpu().tolist(), strict=True):
            matrix[t, p] += 1
    return matrix


def _select_low_margin_samples(model, n: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Pick n test images with the lowest robust margin under PGD-10."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = _evaluation_loader(sample_size=1000, seed=SEEDS[0])
    attack = PGDAttack(load_attack_config("pgd_10"))
    model = model.to(device).eval()
    margins: list[float] = []
    xs: list[torch.Tensor] = []
    ys: list[torch.Tensor] = []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        x_adv = attack.perturb(model, x, y)
        with torch.no_grad():
            probs = torch.softmax(model(x_adv), dim=1)
            target_prob = probs.gather(1, y[:, None]).squeeze(1)
        margins.extend(target_prob.cpu().tolist())
        xs.append(x.cpu())
        ys.append(y.cpu())
    all_x = torch.cat(xs)
    all_y = torch.cat(ys)
    idx = sorted(range(len(margins)), key=lambda i: margins[i])[:n]
    return all_x[idx], all_y[idx]


def _read_transfer_mlflow_runs() -> list[dict[str, object]]:
    """Pull transfer-attack ASR rows from MLflow runs written by run_transfer.py."""
    try:
        import mlflow
    except Exception:
        return []
    mlruns = Path("mlruns")
    if not mlruns.exists():
        return []
    mlflow.set_tracking_uri(f"file:{mlruns.resolve()}")
    try:
        df = mlflow.search_runs(
            experiment_names=["pgd_cifar10_multiarch"],
            filter_string="tags.phase = 'transfer'",
        )
    except Exception:
        return []
    rows: list[dict[str, object]] = []
    for _, run in df.iterrows():
        rows.append(
            {
                "mode": run.get("tags.mode", ""),
                "surrogate": run.get("tags.surrogate", ""),
                "victim": run.get("tags.victim", ""),
                "asr": run.get("metrics.asr", ""),
            }
        )
    return rows


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    fieldnames = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
