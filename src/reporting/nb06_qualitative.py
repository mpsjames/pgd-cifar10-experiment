"""Notebook helper for qualitative visualization and Grad-CAM (NB06)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import torch

from src.reporting.attack import build_attack_for_report
from src.reporting.constants import ARCHES, NB06_NUM_SAMPLES, SEED
from src.reporting.data_loaders import evaluation_inputs, evaluation_loader
from src.reporting.io import ensure_result_dirs
from src.reporting.model_registry import build_fresh_model, reporting_model_pair
from src.utils.seed import set_all_seeds
from src.visualize.gradcam import compute_gradcam
from src.visualize.perturbation_panels import make_perturbation_panel


def nb06_qualitative() -> Path:
    """Create perturbation panel and run Grad-CAM smoke checks for all archs."""
    ensure_result_dirs()
    set_all_seeds(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = reporting_model_pair("resnet18", SEED)
    model = models.clean.load_or_none()
    have_real = model is not None
    if model is None:
        model = build_fresh_model("resnet18")
    model = model.to(device).eval()
    x_cpu, y_cpu = (
        _select_low_margin_pgd_samples(model, NB06_NUM_SAMPLES)
        if have_real
        else evaluation_inputs(NB06_NUM_SAMPLES)
    )
    x = x_cpu.to(device)
    y = y_cpu.to(device)
    attack = build_attack_for_report("pgd_10")
    x_adv = attack.perturb(model, x, y)
    title = (
        f"PGD-10 perturbations on 8 samples selected by lowest robust margin\n"
        f"CIFAR-10 test subset, n=8 | ResNet-18 | epsilon=8/255, alpha=2/255, "
        f"random_start=True | seed={SEED}"
        if have_real
        else "PGD-10 perturbations (synthetic smoke, no checkpoint)\n"
        "synthetic batch, n=8 | ResNet-18 | epsilon=8/255, alpha=2/255, random_start=True"
    )
    fig = make_perturbation_panel(x.detach().cpu(), x_adv.detach().cpu(), title)
    out = Path("results/figures/06_fgsm_bim_pgd_comparison.png")
    fig.savefig(out)
    plt.close(fig)
    for arch in ARCHES:
        target_model = reporting_model_pair(arch, SEED).clean.load_or_none()
        if target_model is None:
            target_model = build_fresh_model(arch)
        target_model = target_model.to(device).eval()
        heatmap = compute_gradcam(target_model, arch, x[:1])
        assert heatmap.shape == (1, 32, 32), f"Grad-CAM shape mismatch: {arch}"
    return out


def _select_low_margin_pgd_samples(model, n: int) -> tuple[torch.Tensor, torch.Tensor]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = evaluation_loader(sample_size=1000, seed=SEED)
    attack = build_attack_for_report("pgd_10")
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
