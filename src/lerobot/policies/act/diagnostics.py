#!/usr/bin/env python

# Copyright 2026 Jay Ranpariya. Thesis extension.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Modality-collapse diagnostic suite for ACT-M checkpoints (ADR-0012).

This module is the SINGLE source of truth for modality-use diagnostics.
The CLI in experiments/scripts/modality_analysis.py is a thin loader+wrapper
that calls ModalityDiagnostics.report(); no diagnostic logic lives there.

Two earlier scripts are SUPERSEDED by this module + the CLI:
  - experiments/_archive/phase-5-scripts/zero_out_ablation.py
  - experiments/_archive/phase-5-scripts/attention_analysis.py

=== Framework (ADR-0012): four-tier triangulation ===

Each tier answers a distinct question; the PATTERN across tiers
identifies the collapse mechanism and selects the next method (M1-M6).

  Tier 1  Information availability  (on the data, no model):
          - contact_detection_auc            [operational]
          - mutual_information              [confirmatory, stub]
  Tier 2  Utilisation               (on the trained model):
          - relative_zero_out               [operational]  (ADR-0011 gate)
          - per_phase_zero_out              [operational, core-4]
          - attention_mass                  [operational, core-4] (M3/M4 only)
  Tier 3  Mechanism                 (the navigational core):
          - embedding_linear_probe          [operational, core-4]
          - gradient_flow_trajectory        [operational, core-4] (needs act target)
          - input_gradient_saliency         [confirmatory, stub]
          - representation_similarity_cka   [confirmatory, stub]
  Tier 4  Counterfactual            (would a fix help?):
          - probe_gap_upper_bound           [operational, derived]
          - finetune_probe_counterfactual   [confirmatory, stub]

Core-4 (navigation-critical): per_phase_zero_out, embedding_linear_probe,
attention_mass, gradient_flow_trajectory (+ the existing relative_zero_out
gate). The confirmatory-4 are stubbed and labelled as such; implement only
if a Tier-3 tie must be broken.
"""

import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from .modeling_act import ACTPolicy

# Optional sklearn (Tier-1 AUC, Tier-3 embedding probe)
try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class ModalityDiagnostics:
    """Four-tier modality-collapse triangulation suite (ADR-0012).

    Construct with a trained ``ACTPolicy``; call :meth:`report` once
    per checkpoint to get the full tiered JSON (``verdict`` +
    ``recommended_next_method`` drawn from the navigation table).
    """

    # ADR-0012 navigation table: (predicate-on-report) -> recommended method.
    # Evaluated in order; first match wins. See Ch6 §6.9 / tab:diagnostic-navigation.
    _NAVIGATION = [
        # pattern key -> (mechanism, method)
    ]

    def __init__(self, policy: ACTPolicy, device: str = "cuda"):
        self.policy = policy
        self.policy.eval()
        self.device = device
        self.config = policy.config
        self.current_indices = getattr(
            self.config, "proprio_current_indices", list(range(6, 12))
        )
        # report() is the single entry point; subclasses must not mutate policy state.

    # ==================================================================
    # Tier 2: utilisation
    # ==================================================================
    def _zero_out(self, batch: Dict[str, Tensor], which: str) -> Tensor:
        """Return actions with `which` channels zeroed. which in {"current","position"}."""
        with torch.no_grad():
            actions_full = self.policy.predict_action_chunk(batch)
            b = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in batch.items()}
            idx = (
                self.current_indices
                if which == "current"
                else [i for i in range(b["observation.state"].shape[-1])
                      if i not in self.current_indices]
            )
            if "observation.state" in b:
                s = b["observation.state"].clone()
                s[..., idx] = 0.0
                b["observation.state"] = s
            if "observation.state_window" in b:
                w = b["observation.state_window"].clone()
                n_cur = len(self.current_indices)
                kp1 = w.shape[-1] // n_cur
                if which == "current":
                    for k in range(kp1):
                        w[..., k * n_cur:(k + 1) * n_cur] = 0.0
                b["observation.state_window"] = w
            actions_zero = self.policy.predict_action_chunk(b)
            return actions_full, actions_zero

    def relative_zero_out(self, batch: Dict[str, Tensor], ratio_threshold: float = 20.0) -> Dict:
        """ADR-0011 gate: z_curr vs z_pos control. Used iff z_curr > z_pos/r."""
        af, ac = self._zero_out(batch, "current")
        _, ap = self._zero_out(batch, "position")
        z_curr = (torch.norm(af - ac, dim=(1, 2)).mean() /
                  (torch.norm(af, dim=(1, 2)).mean() + 1e-8)).item()
        z_pos = (torch.norm(af - ap, dim=(1, 2)).mean() /
                 (torch.norm(af, dim=(1, 2)).mean() + 1e-8)).item()
        rel = z_curr / (z_pos + 1e-8)
        return {
            "z_curr": z_curr, "z_pos": z_pos, "relative_curr": rel,
            "used": bool(rel > 1.0 / ratio_threshold),
            "threshold_ratio": 1.0 / ratio_threshold,
            "n_samples": int(af.shape[0]),
            "tier": 2, "probe": "relative_zero_out",
        }

    def per_phase_zero_out(self, batch: Dict[str, Tensor],
                           phase_labels: Optional[List[str]] = None) -> Dict:
        """CORE-4. Split z_curr by task phase using frame phase labels.

        phase_labels: list aligned with batch dim, each in
        {"approach","grasp","insert","release"} or None. If None or all
        same phase, reports the overall only with a note.
        """
        af, ac = self._zero_out(batch, "current")
        l2 = torch.norm(af - ac, dim=(1, 2))  # (B,)
        norm = torch.norm(af, dim=(1, 2))
        per = {}
        if phase_labels is None:
            phase_labels = ["unknown"] * int(af.shape[0])
        for ph in sorted(set(phase_labels)):
            mask = [i for i, p in enumerate(phase_labels) if p == ph]
            if not mask:
                continue
            m = torch.tensor(mask, device=l2.device)
            per[ph] = {
                "z_curr_mean": (l2[m].mean() / (norm[m].mean() + 1e-8)).item(),
                "n": len(mask),
            }
        return {"per_phase": per, "tier": 2, "probe": "per_phase_zero_out",
                "note": "phase labels from rim-contact frame annotations; "
                        "n small if few annotated episodes in batch"}

    def attention_mass(self, batch: Dict[str, Tensor]) -> Dict:
        """CORE-4 (M3/M4 only). Decoder cross-attention onto the proprio token.

        Uses forward hooks on encoder self-attention; reports temporal-token
        attention share vs vision tokens. Only meaningful for token/hybrid fusion.
        """
        if self.config.proprio_fusion_stage not in ("token", "hybrid"):
            return {"message": "attention_mass only meaningful for token/hybrid fusion (M3/M4)",
                    "tier": 2, "probe": "attention_mass"}
        captured: List[Tensor] = []

        def hook(_m, _i, o):
            if isinstance(o, tuple) and len(o) > 1 and o[1] is not None:
                captured.append(o[1])  # (B, heads, S, S)

        hooks = [layer.self_attn.register_forward_hook(hook)
                 for layer in self.policy.model.encoder.layers]
        try:
            with torch.no_grad():
                b = dict(batch)
                if self.config.image_features:
                    b["observation.images"] = [b[k] for k in self.config.image_features]
                self.policy.model(b)
        finally:
            for h in hooks:
                h.remove()
        if not captured:
            return {"error": "no attention weights captured (need need_weights=True)",
                    "tier": 2, "probe": "attention_mass"}
        attn = torch.stack([a.mean(dim=1) for a in captured]).mean(dim=0)  # (B, S, S)
        temporal_idx, vision_start = 2, 3
        t_mass = attn[..., temporal_idx].mean().item()
        v_mass = attn[..., vision_start:].mean().item()
        return {"temporal_mass": t_mass, "vision_mass": v_mass,
                "temporal_ratio": t_mass / (t_mass + v_mass + 1e-8),
                "tier": 2, "probe": "attention_mass"}

    # ==================================================================
    # Tier 3: mechanism (the navigational core)
    # ==================================================================
    def embedding_linear_probe(self, batch: Dict[str, Tensor],
                               labels: Optional[np.ndarray] = None,
                               dataset=None, n_samples: int = 200) -> Dict:
        """CORE-4. Freeze encoder, probe proprio embedding for task info.

        High probe + low z_curr => routing failure (-> M3/M5). Low probe +
        low z_curr => representation failure (-> M1/M2/M3).
        Labels: if `dataset` given, pseudo-label from current > threshold
        (contact-vs-free); if `labels` given directly, use them.
        """
        if not SKLEARN_AVAILABLE:
            return {"error": "sklearn not installed", "tier": 3,
                    "probe": "embedding_linear_probe"}
        # Extract proprio embeddings via a forward hook on the state projection.
        feats: List[np.ndarray] = []
        lbls: List[int] = []
        hook = []
        proj = getattr(self.policy.model, "encoder_robot_state_input_proj", None)
        if proj is None:
            return {"error": "no encoder_robot_state_input_proj on model", "tier": 3,
                    "probe": "embedding_linear_probe"}
        out_box: Dict[str, Tensor] = {}

        def cap(_m, _i, o):
            out_box["e"] = o.detach().cpu()

        hook.append(proj.register_forward_hook(cap))
        try:
            self.policy.eval()
            with torch.no_grad():
                items = ([dataset[i] for i in range(min(n_samples, len(dataset)))]
                         if dataset is not None else None)
                if items is None:
                    for _ in range(min(n_samples, batch["observation.state"].shape[0])):
                        self.policy.predict_action_chunk(batch)
                        if "e" in out_box:
                            feats.append(out_box["e"].numpy())
                            cur = batch["observation.state"][..., self.current_indices]
                            lbls.append(int(cur.abs().max() > 30))
                else:
                    # spread samples across the whole dataset, not the first N frames
                    # of one episode (which are all free-space and give one class).
                    # Pseudo-label per frame: contact iff the ELBOW current
                    # (joint 3, the cleanest insertion signal) is elevated, NOT the
                    # max over all channels (gripper closure saturates most frames).
                    step = max(1, len(items) // n_samples) if len(items) > n_samples else 1
                    sampled = items[::step][:n_samples]
                    elbow_local = 2  # joints are shoulder,shoulder,elbow,wrist,wrist,gripper
                    for it in sampled:
                        b = {k: (v.unsqueeze(0).to(self.device) if torch.is_tensor(v) else v)
                             for k, v in it.items()}
                        if "observation.images.top" in b and self.config.image_features:
                            b["observation.images.top"] = b["observation.images.top"]
                        cur_all = it["observation.state"][..., self.current_indices]
                        # elbow current value in this frame
                        elbow_cur = float(cur_all[..., elbow_local].abs().max()) \
                            if cur_all[..., elbow_local].numel() else 0.0
                        lbls.append(int(elbow_cur > 10))  # elbow threshold from Phase-1 (29x noise, 92% prev)
                        try:
                            self.policy.predict_action_chunk(b)
                            if "e" in out_box:
                                feats.append(out_box["e"].squeeze(0).numpy())
                        except Exception:
                            continue
        finally:
            for h in hook:
                h.remove()
        if len(feats) < 10 or len(set(lbls)) < 2:
            return {"error": f"insufficient labelled data (n={len(feats)}, "
                             f"classes={len(set(lbls))})", "tier": 3,
                    "probe": "embedding_linear_probe"}
        X = np.stack(feats)
        y = np.array(lbls)
        clf = LogisticRegression(max_iter=1000)
        try:
            auc = cross_val_score(clf, X, y, cv=min(3, len(y)), scoring="roc_auc").mean()
            acc = cross_val_score(clf, X, y, cv=min(3, len(y)), scoring="accuracy").mean()
        except Exception as e:
            return {"error": f"cv failed: {e}", "tier": 3,
                    "probe": "embedding_linear_probe"}
        return {"embedding_probe_auc": float(auc), "embedding_probe_acc": float(acc),
                "n_samples": len(y), "tier": 3, "probe": "embedding_linear_probe"}

    def gradient_flow_trajectory(self, batch: Dict[str, Tensor]) -> Dict:
        """CORE-4. ||grad_current|| / ||grad_position|| at the current step.

        NOTE: only the *final* ratio is available from a checkpoint (trajectories
        over training require per-step logging during training; this is the
        checkpoint-time version). Requires the action target in `batch` because
        the VAE objective (use_vae=True) needs it to compute loss.
        """
        if "action" not in batch:
            return {"error": "gradient_flow_trajectory needs 'action' in batch "
                             "(use_vae=True requires the action target)",
                    "tier": 3, "probe": "gradient_flow_trajectory"}
        self.policy.train()
        self.policy.zero_grad()
        loss, _ = self.policy(batch)
        loss.backward()
        gc2 = gp2 = 0.0
        for name, p in self.policy.named_parameters():
            if p.grad is None:
                continue
            # Only the 2D LINEAR weights of the state projections (skip biases,
            # conv1d weights, embeddings -- they mismatch the pos/cur split).
            if p.grad.dim() != 2:
                continue
            if (name.endswith("encoder_robot_state_input_proj.weight") or
                name.endswith("vae_encoder_robot_state_input_proj.weight")):
                g = p.grad
                in_dim = g.shape[-1]
                n_cur = len(self.current_indices)
                gp2 += g[..., :in_dim - n_cur].norm().item() ** 2
                gc2 += g[..., in_dim - n_cur:].norm().item() ** 2
        self.policy.eval()
        gc = math.sqrt(gc2); gp = math.sqrt(gp2)
        return {"grad_current_norm": gc, "grad_position_norm": gp,
                "ratio": gc / (gp + 1e-8), "tier": 3,
                "probe": "gradient_flow_trajectory",
                "note": "final-step ratio only; trajectory-over-training requires "
                        "per-step W&B logging during training"}

    def input_gradient_saliency(self, batch: Dict[str, Tensor]) -> Dict:
        """CONFIRMATORY (stub). Integrated gradients on the current channels."""
        return {"status": "confirmatory_stub",
                "implementation": "captum.attr.IntegratedGradients on the current "
                                   "input channels; implement only if Tier-3 tie "
                                   "must be broken (ADR-0012).",
                "tier": 3, "probe": "input_gradient_saliency"}

    def representation_similarity_cka(self, batch: Dict[str, Tensor]) -> Dict:
        """CONFIRMATORY (stub). CKA between proprio and vision branch activations."""
        return {"status": "confirmatory_stub",
                "implementation": "linear CKA between the proprio projection output "
                                   "and the ResNet feature map (flattened); implement "
                                   "only if Tier-3 tie must be broken.",
                "tier": 3, "probe": "representation_similarity_cka"}

    # ==================================================================
    # Tier 1: information availability (on the data, no model)
    # ==================================================================
    def contact_detection_auc(self, dataset=None, batch: Optional[Dict[str, Tensor]] = None,
                              n_samples: int = 500) -> Dict:
        """Logistic probe on RAW currents -> contact-vs-free pseudo-labels."""
        if not SKLEARN_AVAILABLE:
            return {"error": "sklearn not installed", "tier": 1,
                    "probe": "contact_detection_auc"}
        feats: List[np.ndarray] = []
        lbls: List[int] = []
        if dataset is not None:
            n = len(dataset)
            step = max(1, n // n_samples)
            for i in range(0, n, step):
                if len(feats) >= n_samples:
                    break
                it = dataset[i]
                cur = it["observation.state"][..., self.current_indices]
                feats.append(cur.flatten().numpy())
                lbls.append(int(cur.abs().max() > 30))
        elif batch is not None:
            cur = batch["observation.state"][..., self.current_indices]
            for i in range(cur.shape[0]):
                feats.append(cur[i].flatten().cpu().numpy())
                lbls.append(int(cur[i].abs().max() > 30))
        if len(feats) < 10 or len(set(lbls)) < 2:
            return {"error": f"insufficient labels (n={len(feats)}, "
                             f"classes={len(set(lbls))})", "tier": 1,
                    "probe": "contact_detection_auc",
                    "note": "pseudo-labels from current>30; gold labels require "
                            "rim-contact frame annotations"}
        X = np.stack(feats); y = np.array(lbls)
        clf = LogisticRegression(max_iter=1000)
        try:
            auc = cross_val_score(clf, X, y, cv=min(3, len(y)),
                                  scoring="roc_auc").mean()
            acc = cross_val_score(clf, X, y, cv=min(3, len(y)),
                                 scoring="accuracy").mean()
        except Exception as e:
            return {"error": f"cv failed: {e}", "tier": 1,
                    "probe": "contact_detection_auc"}
        return {"auc": float(auc), "accuracy": float(acc), "n_samples": len(y),
                "tier": 1, "probe": "contact_detection_auc",
                "label_basis": "pseudo (current>30); gold = rim-contact annotations"}

    def mutual_information(self, dataset=None, n_samples: int = 500) -> Dict:
        """CONFIRMATORY (stub). MI/CCA between currents and labels."""
        return {"status": "confirmatory_stub",
                "implementation": "sklearn.feature_selection.mutual_info_classif "
                                   "on currents vs contact labels; implement only "
                                   "if Tier-1 AUC is ambiguous.",
                "tier": 1, "probe": "mutual_information"}

    # ==================================================================
    # Tier 4: counterfactual
    # ==================================================================
    def probe_gap_upper_bound(self, tier1_auc: Optional[float],
                             tier3_probe_auc: Optional[float]) -> Dict:
        """Derived: data ceiling - model ceiling = max recoverable by a fix."""
        if tier1_auc is None or tier3_probe_auc is None:
            return {"error": "requires both Tier-1 AUC and Tier-3 embedding probe",
                    "tier": 4, "probe": "probe_gap_upper_bound"}
        return {"data_ceiling_auc": tier1_auc, "model_ceiling_auc": tier3_probe_auc,
                "recoverable_gap": tier1_auc - tier3_probe_auc,
                "interpretation": ("a fix can recover at most this gap in AUC; "
                                   "small gap => architecture already near-optimal, "
                                   "collapses is a routing problem"),
                "tier": 4, "probe": "probe_gap_upper_bound"}

    def finetune_probe_counterfactual(self) -> Dict:
        """CONFIRMATORY (stub). Unfreeze proprio path, fine-tune 1k steps."""
        return {"status": "confirmatory_stub",
                "implementation": "load collapsed checkpoint, set requires_grad=True "
                                   "only on proprio projection + temporal_encoder, "
                                   "fine-tune ~1k steps, re-run relative_zero_out; "
                                   "implement only if Tier-3 is ambiguous.",
                "tier": 4, "probe": "finetune_probe_counterfactual"}

    # ==================================================================
    # Navigation + report
    # ==================================================================
    def _navigate(self, report: Dict) -> Tuple[str, str]:
        """Apply the ADR-0012 navigation table => (mechanism, recommended_method)."""
        t1 = report.get("tier1", {})
        t2 = report.get("tier2", {})
        t3 = report.get("tier3", {})
        auc = (t1.get("contact_detection_auc") or {}).get("auc")
        zc = (t2.get("relative_zero_out") or {}).get("z_curr")
        used = (t2.get("relative_zero_out") or {}).get("used")
        probe_auc = (t3.get("embedding_linear_probe") or {}).get("embedding_probe_auc")
        grad = (t3.get("gradient_flow_trajectory") or {}).get("ratio")
        per_phase = (t2.get("per_phase_zero_out") or {}).get("per_phase", {})

        if used:
            return "no collapse", "none (deploy; primary hardware comparison)"
        # collapse confirmed -> triangulate mechanism
        if auc is not None and auc < 0.6:
            return "signal inadequate", "none: Tier-0 report (rethink sensing)"
        if probe_auc is not None and probe_auc >= 0.7:
            return "routing / suppression failure", "M3 token, M5 FiLM, M4 trigger"
        if probe_auc is not None and probe_auc < 0.6:
            return "representation failure", "M1 history, M2 explicit, M3 CNN"
        # phase-conditional?
        ins = per_phase.get("insert", {})
        if ins and ins.get("z_curr_mean", 1) < 0.05 and used is False:
            app = per_phase.get("approach", {}).get("z_curr_mean", 0)
            if app < 0.05:
                pass  # ignored everywhere
            else:
                return "phase-conditional failure", "M4 trigger (phase-gated)"
        if grad is not None and grad < 0.05:
            return "generalisation-rate mismatch", "M6 OGM-GE / FACTR curriculum"
        return "collapse mechanism undetermined", "run confirmatory probes (saliency/CKA/finetune)"

    def report(self, batch: Dict[str, Tensor], dataset=None,
               phase_labels: Optional[List[str]] = None,
               ratio_threshold: float = 20.0,
               include_tier1: bool = True,
               include_tier3: bool = True,
               n_probe_samples: int = 200) -> Dict:
        """Run the full four-tier triangulation and emit verdict + recommended method.

        This is the single entry point. `batch` must contain observation.state and
        observation.images.<cam>; optionally 'action' (enables the gradient probe
        for use_vae=True checkpoints). `dataset` (optional) enables Tier-1 AUC
        and the Tier-3 embedding probe over more samples. `phase_labels` (optional,
        aligned with batch) enables per-phase zero-out.
        """
        rep: Dict = {"checkpoint_variant": {
            "temporal_encoder": self.config.proprio_temporal_encoder,
            "fusion_stage": self.config.proprio_fusion_stage,
        }}
        # Tier 2
        rep["tier2"] = {}
        rep["tier2"]["relative_zero_out"] = self.relative_zero_out(batch, ratio_threshold)
        rep["tier2"]["per_phase_zero_out"] = self.per_phase_zero_out(batch, phase_labels)
        try:
            rep["tier2"]["attention_mass"] = self.attention_mass(batch)
        except Exception as e:
            rep["tier2"]["attention_mass"] = {"error": str(e)}
        # Tier 1 (data, no model)
        if include_tier1 and dataset is not None:
            rep["tier1"] = {}
            try:
                rep["tier1"]["contact_detection_auc"] = self.contact_detection_auc(dataset=dataset)
            except Exception as e:
                rep["tier1"]["contact_detection_auc"] = {"error": str(e)}
            rep["tier1"]["mutual_information"] = self.mutual_information(dataset=dataset)
        # Tier 3 (mechanism)
        if include_tier3:
            rep["tier3"] = {}
            try:
                rep["tier3"]["embedding_linear_probe"] = self.embedding_linear_probe(
                    batch, dataset=dataset, n_samples=n_probe_samples)
            except Exception as e:
                rep["tier3"]["embedding_linear_probe"] = {"error": str(e)}
            try:
                rep["tier3"]["gradient_flow_trajectory"] = self.gradient_flow_trajectory(batch)
            except Exception as e:
                rep["tier3"]["gradient_flow_trajectory"] = {"error": str(e)}
            rep["tier3"]["input_gradient_saliency"] = self.input_gradient_saliency(batch)
            rep["tier3"]["representation_similarity_cka"] = self.representation_similarity_cka(batch)
        # Tier 4 (counterfactual / derived)
        t1_auc = ((rep.get("tier1", {}).get("contact_detection_auc") or {}).get("auc"))
        t3_auc = ((rep.get("tier3", {}).get("embedding_linear_probe") or {}).get("embedding_probe_auc"))
        rep["tier4"] = {}
        try:
            rep["tier4"]["probe_gap_upper_bound"] = self.probe_gap_upper_bound(t1_auc, t3_auc)
        except Exception as e:
            rep["tier4"]["probe_gap_upper_bound"] = {"error": str(e)}
        rep["tier4"]["finetune_probe_counterfactual"] = self.finetune_probe_counterfactual()
        # Navigation
        mechanism, method = self._navigate(rep)
        rep["verdict"] = "USED" if (rep["tier2"]["relative_zero_out"].get("used")) else "IGNORED"
        rep["mechanism"] = mechanism
        rep["recommended_next_method"] = method
        return rep
