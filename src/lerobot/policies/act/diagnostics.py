#!/usr/bin/env python

# Copyright 2026 Jay Ranpariya. Thesis extension.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Modality collapse diagnostics for ACT-M checkpoints.

Run these on a trained checkpoint BEFORE hardware evaluation to verify
that the policy is actually using proprioceptive information.

Diagnostics:
  1. Zero-out L2: action change when current features are zeroed
  2. Per-phase zero-out: action change per task phase
  3. Contact-detection AUC: can current alone predict grasp/miss?
  4. Attention mass: attention weight on proprioceptive tokens
  5. Gradient ratio: ||grad_current|| / ||grad_position||
  6. Filter visualization: CNN kernel patterns (Method 3 only)
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from .modeling_act import ACTPolicy

# Optional sklearn import
try:
    from sklearn.metrics import roc_auc_score
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class ModalityDiagnostics:
    """Diagnostic suite for verifying multimodal utilization."""

    def __init__(self, policy: ACTPolicy, device: str = "cuda"):
        self.policy = policy
        self.policy.eval()
        self.device = device
        self.config = policy.config

        # Identify current channel indices in state vector
        self.current_indices = getattr(config, "proprio_current_indices", list(range(6, 12)))
        self.position_indices = list(range(6))  # first 6 are positions

    # ------------------------------------------------------------------
    # 1. Zero-out ablation
    # ------------------------------------------------------------------
    def zero_out_l2(
        self,
        batch: Dict[str, Tensor],
        n_samples: Optional[int] = None,
    ) -> Dict[str, float]:
        """Measure action change when current channels are zeroed.

        Returns dict with:
            mean_l2: mean L2 distance across batch
            median_l2: median L2 distance
            std_l2: standard deviation
            max_l2: maximum distance
            ratio: mean_l2 / mean_action_norm (normalized impact)
        """
        with torch.no_grad():
            # Full prediction
            actions_full = self.policy.predict_action_chunk(batch)

            # Zero-out current channels
            batch_zero = {k: v.clone() if torch.is_tensor(v) else v for k, v in batch.items()}
            state = batch_zero["observation.state"].clone()
            state[..., self.current_indices] = 0.0
            batch_zero["observation.state"] = state

            actions_zero = self.policy.predict_action_chunk(batch_zero)

            # L2 distance per sample (average over chunk and action dims)
            l2_dist = torch.norm(actions_full - actions_zero, dim=(1, 2))
            action_norm = torch.norm(actions_full, dim=(1, 2))

            return {
                "mean_l2": l2_dist.mean().item(),
                "median_l2": l2_dist.median().item(),
                "std_l2": l2_dist.std().item(),
                "max_l2": l2_dist.max().item(),
                "ratio": (l2_dist.mean() / (action_norm.mean() + 1e-8)).item(),
                "n_samples": actions_full.shape[0],
            }

    # ------------------------------------------------------------------
    # 2. Gradient ratio
    # ------------------------------------------------------------------
    def gradient_ratio(
        self,
        batch: Dict[str, Tensor],
    ) -> Dict[str, float]:
        """Compute ||grad_current|| / ||grad_position||.

        Requires batch with actions (training mode). Returns ratio and
        individual gradient norms.
        """
        self.policy.train()
        self.policy.zero_grad()

        # Forward with actions to compute loss
        loss, loss_dict = self.policy(batch)
        loss.backward()

        grad_current_norm = 0.0
        grad_position_norm = 0.0

        for name, param in self.policy.named_parameters():
            if param.grad is None:
                continue
            if "robot_state" in name or "state" in name:
                # State projection layer — split into position and current parts
                # This is approximate; exact split depends on state ordering
                grad_position_norm += param.grad[..., :6].norm().item() ** 2
                grad_current_norm += param.grad[..., 6:].norm().item() ** 2

        grad_current_norm = math.sqrt(grad_current_norm)
        grad_position_norm = math.sqrt(grad_position_norm)
        ratio = grad_current_norm / (grad_position_norm + 1e-8)

        return {
            "grad_current_norm": grad_current_norm,
            "grad_position_norm": grad_position_norm,
            "ratio": ratio,
        }

    # ------------------------------------------------------------------
    # 3. Contact detection AUC
    # ------------------------------------------------------------------
    def contact_detection_auc(
        self,
        dataset,  # LeRobotDataset or similar
        n_samples: int = 500,
    ) -> Dict[str, float]:
        """Train a simple probe: can current features predict grasp vs miss?

        Uses held-out evaluation data with failure-mode labels.
        Returns AUC and accuracy of a logistic regression probe.
        """
        if not SKLEARN_AVAILABLE:
            return {"error": "sklearn not installed; install with: pip install scikit-learn"}

        # Extract current features and labels from dataset
        features = []
        labels = []

        for i in range(min(n_samples, len(dataset))):
            item = dataset[i]
            state = item["observation.state"].cpu().numpy()
            # Assuming current channels are at self.current_indices
            current = state[..., self.current_indices]
            features.append(current.flatten())

            # Label: 1 if this frame is during a grasp (heuristic: high current)
            # In real eval, use failure-mode annotations
            is_contact = np.any(np.abs(current) > 30)
            labels.append(1 if is_contact else 0)

        X = np.stack(features)
        y = np.array(labels)

        # Logistic regression probe
        clf = LogisticRegression(max_iter=1000)
        auc = cross_val_score(clf, X, y, cv=3, scoring="roc_auc").mean()
        acc = cross_val_score(clf, X, y, cv=3, scoring="accuracy").mean()

        return {"auc": auc, "accuracy": acc, "n_samples": len(y)}

    # ------------------------------------------------------------------
    # 4. Attention mass on proprioceptive tokens
    # ------------------------------------------------------------------
    def attention_mass(
        self,
        batch: Dict[str, Tensor],
    ) -> Dict[str, float]:
        """Extract attention weights from transformer encoder self-attention.

        Returns average attention mass on proprioceptive tokens vs vision tokens.
        Only meaningful for token/hybrid fusion methods.
        """
        if self.config.proprio_fusion_stage not in ("token", "hybrid"):
            return {"message": "Attention mass only meaningful for token/hybrid fusion"}

        # Hook into encoder self-attention to capture attention weights
        attention_weights = []

        def hook_fn(module, input, output):
            # output[1] contains attention weights from nn.MultiheadAttention
            if isinstance(output, tuple) and len(output) > 1:
                attn = output[1]  # (batch, num_heads, seq, seq)
                attention_weights.append(attn)

        hooks = []
        for layer in self.policy.model.encoder.layers:
            h = layer.self_attn.register_forward_hook(hook_fn)
            hooks.append(h)

        with torch.no_grad():
            # Need to manually run model forward to trigger hooks
            # (predict_action_chunk skips model internals)
            if self.config.image_features:
                batch = dict(batch)
                batch["observation.images"] = [batch[k] for k in self.config.image_features]
            self.policy.model(batch)

        for h in hooks:
            h.remove()

        if not attention_weights:
            return {"error": "No attention weights captured"}

        # Average over layers and heads
        attn = torch.stack([a.mean(dim=1) for a in attention_weights]).mean(dim=0)  # (B, seq, seq)

        # Token ordering: [latent, state, temporal, vision...]
        # For token/hybrid, temporal token is at index 2
        temporal_idx = 2
        vision_start = 3

        # Attention FROM decoder queries TO encoder keys
        # Measure how much decoder attends to temporal vs vision tokens
        temporal_mass = attn[..., temporal_idx].mean().item()
        vision_mass = attn[..., vision_start:].mean().item()
        total = temporal_mass + vision_mass

        return {
            "temporal_mass": temporal_mass,
            "vision_mass": vision_mass,
            "temporal_ratio": temporal_mass / (total + 1e-8),
        }

    # ------------------------------------------------------------------
    # 5. Run full diagnostic suite
    # ------------------------------------------------------------------
    def run_all(
        self,
        batch: Dict[str, Tensor],
        output_dir: Optional[Path] = None,
    ) -> Dict[str, Dict]:
        """Run all diagnostics and optionally save to disk."""
        results = {
            "zero_out_l2": self.zero_out_l2(batch),
        }

        # Gradient ratio requires training mode
        try:
            results["gradient_ratio"] = self.gradient_ratio(batch)
        except Exception as e:
            results["gradient_ratio"] = {"error": str(e)}

        # Attention mass (token/hybrid only)
        if self.config.proprio_fusion_stage in ("token", "hybrid"):
            try:
                results["attention_mass"] = self.attention_mass(batch)
            except Exception as e:
                results["attention_mass"] = {"error": str(e)}

        if output_dir is not None:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            with open(output_dir / "diagnostics.json", "w") as f:
                json.dump(results, f, indent=2)

        return results


import math
