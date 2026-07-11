"""Gradient-based digital attacks for YOLOv8-style detectors."""

from __future__ import annotations

from typing import Iterable

import cv2
import numpy as np
import torch


def normalize_torch_device(device: str | int | torch.device | None = None) -> torch.device:
    """Accept Ultralytics-style numeric device ids in raw PyTorch attack code."""

    if device is None:
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if isinstance(device, torch.device):
        return device
    if isinstance(device, int):
        return torch.device(f"cuda:{device}" if torch.cuda.is_available() else "cpu")
    device_text = str(device).strip()
    if device_text.isdigit():
        return torch.device(f"cuda:{device_text}" if torch.cuda.is_available() else "cpu")
    return torch.device(device_text)


def refresh_detection_head_tensors(model: torch.nn.Module) -> None:
    """Convert cached Ultralytics inference tensors back to normal tensors.

    YOLOv8 predict/track calls run under inference mode and may cache anchors or
    strides on the Detect head. Gradient attacks reuse the same module with
    autograd enabled, so those cached tensors must be cloned before backward.
    """

    for module in model.modules():
        for name in ("anchors", "strides"):
            value = getattr(module, name, None)
            if isinstance(value, torch.Tensor):
                setattr(module, name, value.detach().clone())


def _bgr_to_tensor(image: np.ndarray, size: int | tuple[int, int], device: str | torch.device) -> torch.Tensor:
    if isinstance(size, int):
        resized = cv2.resize(image, (size, size), interpolation=cv2.INTER_LINEAR)
    else:
        resized = cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(rgb).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    return tensor.to(device)


def _tensor_to_bgr(tensor: torch.Tensor, original_shape: tuple[int, int]) -> np.ndarray:
    tensor = tensor.detach().clamp(0, 1).squeeze(0).permute(1, 2, 0).cpu().numpy()
    rgb = (tensor * 255.0).round().astype(np.uint8)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return cv2.resize(bgr, (original_shape[1], original_shape[0]), interpolation=cv2.INTER_LINEAR)


def detector_confidence_loss(preds, target_classes: Iterable[int] | None = None) -> torch.Tensor:
    """Differentiable confidence objective for YOLOv8 raw predictions.

    YOLOv8 inference tensors are commonly shaped [B, 4 + C, N], where the first
    four channels encode boxes and the remaining channels encode class
    confidence. This loss returns the maximum class confidence over all anchors.
    Minimizing it suppresses detections; maximizing it can create false alarms.
    """

    if isinstance(preds, (tuple, list)):
        preds = preds[0]
    if isinstance(preds, (tuple, list)):
        preds = preds[0]
    if preds.ndim != 3:
        raise ValueError(f"Unsupported YOLO prediction shape: {tuple(preds.shape)}")

    if preds.shape[1] >= preds.shape[2]:
        class_scores = preds[:, 4:, :]
    else:
        class_scores = preds[:, :, 4:].transpose(1, 2)

    if target_classes is not None:
        target_classes = list(target_classes)
        class_scores = class_scores[:, target_classes, :]
    return class_scores.sigmoid().amax(dim=1).amax(dim=1).mean()


def fgsm_attack(
    model: torch.nn.Module,
    image: np.ndarray,
    eps: float = 8 / 255,
    image_size: int = 640,
    target_classes: Iterable[int] | None = None,
    minimize_confidence: bool = True,
    device: str | torch.device | None = None,
) -> np.ndarray:
    """Fast Gradient Sign Method attack against YOLO confidence."""

    device = normalize_torch_device(device)
    model = model.to(device).eval()
    refresh_detection_head_tensors(model)
    x = _bgr_to_tensor(image, image_size, device).requires_grad_(True)
    preds = model(x)
    loss = detector_confidence_loss(preds, target_classes)
    signed_grad = torch.autograd.grad(loss, x, retain_graph=False, create_graph=False)[0].sign()
    direction = -1.0 if minimize_confidence else 1.0
    adv = (x + direction * eps * signed_grad).clamp(0.0, 1.0)
    return _tensor_to_bgr(adv, image.shape[:2])


def pgd_attack(
    model: torch.nn.Module,
    image: np.ndarray,
    eps: float = 8 / 255,
    alpha: float = 2 / 255,
    steps: int = 10,
    image_size: int = 640,
    target_classes: Iterable[int] | None = None,
    minimize_confidence: bool = True,
    random_start: bool = True,
    device: str | torch.device | None = None,
) -> np.ndarray:
    """Projected Gradient Descent attack against YOLO confidence."""

    device = normalize_torch_device(device)
    model = model.to(device).eval()
    refresh_detection_head_tensors(model)
    x0 = _bgr_to_tensor(image, image_size, device)
    if random_start:
        x_adv = (x0 + torch.empty_like(x0).uniform_(-eps, eps)).clamp(0.0, 1.0)
    else:
        x_adv = x0.clone()

    direction = -1.0 if minimize_confidence else 1.0
    for _ in range(steps):
        x_adv = x_adv.detach().requires_grad_(True)
        refresh_detection_head_tensors(model)
        loss = detector_confidence_loss(model(x_adv), target_classes)
        grad = torch.autograd.grad(loss, x_adv, retain_graph=False, create_graph=False)[0].sign()
        x_adv = x_adv + direction * alpha * grad
        x_adv = torch.max(torch.min(x_adv, x0 + eps), x0 - eps).clamp(0.0, 1.0)

    return _tensor_to_bgr(x_adv, image.shape[:2])
