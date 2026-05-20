"""Build deterministic CIFAR-10 data loaders for training and evaluation."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from src.utils.seed import get_generator


def _seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed + worker_id)


def get_cifar10_loaders(
    batch_size: int,
    num_workers: int = 2,
    augment_train: bool = True,
    seed: int | None = None,
    root: str = "data/cifar10",
    download: bool = True,
    pin_memory: bool | None = None,
    persistent_workers: bool = False,
    prefetch_factor: int = 2,
) -> tuple[DataLoader, DataLoader]:
    """Return CIFAR-10 train/test loaders with raw `[0, 1]` image tensors.

    Args:
        batch_size: Batch size for both loaders. Must be at least 1.
        num_workers: Number of DataLoader workers. When `seed` is provided,
            worker RNGs are derived deterministically from the torch worker
            seed.
        augment_train: When True, apply standard CIFAR-10 crop/flip
            augmentation to the training split only.
        seed: Optional reproducibility seed for shuffling and worker
            initialization.
        root: Dataset cache directory.
        download: When True, allow torchvision to download CIFAR-10 into
            `root` if it is missing.
        pin_memory: Override page-locked transfer. When `None`, fall back to
            `torch.cuda.is_available()` (legacy behavior).
        persistent_workers: When True and `num_workers > 0`, keep worker
            processes alive between epochs.
        prefetch_factor: Batches each worker pre-fetches. Only forwarded when
            `num_workers > 0` (PyTorch rejects it otherwise).

    Returns:
        `(train_loader, test_loader)` where both yield `(x, y)` batches with
        `x.shape == (B, 3, 32, 32)`, `x.dtype == float32`, `x in [0, 1]`, and
        `y.dtype == long`.
    """
    train_transforms: list[Callable] = []
    if augment_train:
        train_transforms.extend(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
            ]
        )
    train_transforms.append(transforms.ToTensor())

    test_transform = transforms.ToTensor()
    train_set = datasets.CIFAR10(
        root=root,
        train=True,
        transform=transforms.Compose(train_transforms),
        download=download,
    )
    test_set = datasets.CIFAR10(root=root, train=False, transform=test_transform, download=download)

    generator = get_generator(seed) if seed is not None else None
    pin = torch.cuda.is_available() if pin_memory is None else pin_memory
    # PyTorch rejects persistent_workers/prefetch_factor when num_workers == 0.
    effective_persistent = persistent_workers and num_workers > 0
    loader_kwargs: dict[str, object] = {}
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = prefetch_factor

    return (
        DataLoader(
            train_set,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            worker_init_fn=_seed_worker if seed is not None else None,
            generator=generator,
            pin_memory=pin,
            persistent_workers=effective_persistent,
            **loader_kwargs,
        ),
        DataLoader(
            test_set,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin,
            persistent_workers=effective_persistent,
            **loader_kwargs,
        ),
    )
