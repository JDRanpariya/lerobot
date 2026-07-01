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

=== Framework (ADR-0012): four-tier triangulation ===

Each tier answers a distinct question; the PATTERN across tiers identifies
the collapse mechanism and selects the next method (M1-M6).

  Tier 1  Information availability  (on the DATA, computed ONCE in Phase-1;
          reported in Ch4. NOT recomputed per-checkpoint):
          - contact_detection_auc   -> Phase-1 prevalence script
          - mutual_information      [confirmatory, stub]
          Passed into report() as `tier1_auc` (the Phase-1 value).
  Tier 2  Utilisation               (on the trained model, PER PHASE):
          - relative_zero_out       [operational]  (ADR-0011 gate, per-phase)
          - attention_mass          [operational] (M3/M4 only)
  Tier 3  Mechanism                 (the navigational core, PER PHASE):
          - embedding_linear_probe  [operational, core-4]  -- hooks the
            post-encoder proprio token (idx 1 for early/film, 2 for token/hybrid),
            NOT the input projection. A linear classifier on the contextualised
            representation => representation-vs-routing discriminator.
          - gradient_flow_trajectory[operational, core-4] (weight-only; needs action)
          - input_gradient_saliency [confirmatory, stub]
          - representation_similarity_cka [confirmatory, stub]
  Tier 4  Counterfactual            (would a fix help?):
          - probe_gap_upper_bound   [operational, derived from T1+T3]

Core-4 (navigation-critical): per_phase_zero_out, embedding_linear_probe,
attention_mass, gradient_flow_trajectory (+ the existing relative_zero_out
gate). Confirmatory probes stubbed and labelled; implement only if a Tier-3
tie must be broken.
"""

import json
import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from .modeling_act import ACTPolicy

# Phase labels helper (experiments/scripts/phase_labels.py mirror)
def _proprio_token_index(fusion_stage: str) -> int:
    """Token index in the post-encoder sequence carrying proprio info.

    early/film:   [latent(0), state(1), vision(2..)]       -> 1
    token/hybrid: [latent(0), state(1), temporal(2), vis..] -> 2
    """
    return 2 if fusion_stage in ("token", "hybrid") else 1

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class ModalityDiagnostics:
    """Four-tier modality-collapse triangulation suite (ADR-0012)."""

    def __init__(self, policy: ACTPolicy, device: str = "cuda"):
        self.policy = policy
        self.policy.eval()
        self.device = device
        self.config = policy.config
        self.current_indices = getattr(
            self.config, "proprio_current_indices", list(range(6, 12))
        )

    # ==============================================================
    # shared zero-out
    # ==============================================================
    def _zero_out(self, batch: Dict[str, Tensor], which: str) -> Tuple[Tensor, Tensor]:
        """Return (actions_full, actions_with_`which`-channels-zeroed)."""
        with torch.no_grad():
            af = self.policy.predict_action_chunk(batch)
            b = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in batch.items()}
            idx = (self.current_indices if which == "current"
                   else [i for i in range(b["observation.state"].shape[-1])
                         if i not in self.current_indices])
            if "observation.state" in b:
                s = b["observation.state"].clone(); s[..., idx] = 0.0; b["observation.state"] = s
            if "observation.state_window" in b:
                w = b["observation.state_window"].clone()
                n_cur = len(self.current_indices)
                kp1 = w.shape[-1] // n_cur
                if which == "current":
                    for k in range(kp1):
                        w[..., k * n_cur:(k + 1) * n_cur] = 0.0
                b["observation.state_window"] = w
            az = self.policy.predict_action_chunk(b)
            return af, az

    # ==============================================================
    # Tier 2: utilisation
    # ==============================================================
    def relative_zero_out(self, batch: Dict[str, Tensor],
                          ratio_threshold: float = 20.0) -> Dict:
        """ADR-0011 gate: z_curr vs z_pos control. Used iff z_curr > z_pos/r."""
        af, ac = self._zero_out(batch, "current")
        _, ap = self._zero_out(batch, "position")
        l2c = torch.norm(af - ac, dim=(1, 2)); l2p = torch.norm(af - ap, dim=(1, 2))
        norm = torch.norm(af, dim=(1, 2))
        z_curr = (l2c.mean() / (norm.mean() + 1e-8)).item()
        z_pos = (l2p.mean() / (norm.mean() + 1e-8)).item()
        rel = z_curr / (z_pos + 1e-8)
        # per-sample z_curr for per-phase splitting
        per_sample_z_curr = (l2c / (norm + 1e-8)).cpu().numpy().tolist()
        return {
            "z_curr": z_curr, "z_pos": z_pos, "relative_curr": rel,
            "used": bool(rel > 1.0 / ratio_threshold),
            "threshold_ratio": 1.0 / ratio_threshold,
            "n_samples": int(af.shape[0]),
            "per_sample_z_curr": per_sample_z_curr,
            "tier": 2, "probe": "relative_zero_out",
        }

    def per_phase_zero_out(self, batch: Dict[str, Tensor],
                           phase_labels: Optional[List[str]] = None) -> Dict:
        """Per-phase z_curr split. `phase_labels` aligned with batch dim."""
        af, ac = self._zero_out(batch, "current")
        l2c = torch.norm(af - ac, dim=(1, 2))
        norm = torch.norm(af, dim=(1, 2))
        per_sample = (l2c / (norm + 1e-8)).cpu().numpy()
        if phase_labels is None:
            phase_labels = ["unknown"] * int(af.shape[0])
        per: Dict[str, Dict] = {}
        for ph in sorted(set(phase_labels)):
            mask = [i for i, p in enumerate(phase_labels) if p == ph]
            if not mask:
                continue
            ixs = torch.tensor(mask, device=l2c.device)
            per[ph] = {
                "z_curr_mean": float(per_sample[mask].mean()),
                "n": len(mask),
            }
        # consolidated phase-specific collapse pattern (Phase-1 insert should matter)
        insert = per.get("insert", {}).get("z_curr_mean")
        approach = per.get("approach", {}).get("z_curr_mean")
        pattern = {}
        if insert is not None and approach is not None:
            if insert < 0.05 and approach < 0.05:
                pattern["phase_profile"] = "ignored_everywhere"
            elif insert < 0.05 and approach >= 0.05:
                pattern["phase_profile"] = "insert_ignored_only"  # phase-conditional
        return {"per_phase": per, "phase_pattern": pattern,
                "tier": 2, "probe": "per_phase_zero_out"}

    def attention_mass(self, batch: Dict[str, Tensor]) -> Dict:
        """CORE-4 (M3/M4 only). Decoder cross-attn onto proprio token."""
        if self.config.proprio_fusion_stage not in ("token", "hybrid"):
            return {"message": "attention_mass only meaningful for token/hybrid (M3/M4)",
                    "tier": 2, "probe": "attention_mass"}
        captured: List[Tensor] = []

        def hook(_m, _i, o):
            if isinstance(o, tuple) and len(o) > 1 and o[1] is not None:
                captured.append(o[1])

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
            return {"error": "no attention weights captured (need_weights=True)",
                    "tier": 2, "probe": "attention_mass"}
        attn = torch.stack([a.mean(dim=1) for a in captured]).mean(dim=0)  # (B, S, S)
        pidx = _proprio_token_index(self.config.proprio_fusion_stage)
        vstart = pidx + 1
        pmass = attn[..., pidx].mean().item()
        vmass = attn[..., vstart:].mean().item()
        return {"proprio_mass": pmass, "vision_mass": vmass,
                "proprio_ratio": pmass / (pmass + vmass + 1e-8),
                "proprio_token_index": pidx, "tier": 2, "probe": "attention_mass"}

    # ==============================================================
    # Tier 3: mechanism (navigational core)
    # ==============================================================
    def embedding_linear_probe(self, dataset=None, batch: Optional[Dict[str, Tensor]] = None,
                               phase_labels: Optional[List[str]] = None,
                               n_samples: int = 200) -> Dict:
        """CORE-4. Frozen linear probe on the POST-ENCODER proprio token.

        Hooks self.policy.model.encoder (output (S, B, D)), indexes the
        proprio token (1 for early/film, 2 for token/hybrid) -- the
        contextualised, post-self-attention representation. A linear
        classifier from that D-vector to the contact label discriminates
        representation failure (probe low -> encoder never learned it ->
        M1/M2/M3) from routing failure (probe high, zero-out low ->
        M3-token/M5/M4).
        """
        if not SKLEARN_AVAILABLE:
            return {"error": "sklearn not installed", "tier": 3,
                    "probe": "embedding_linear_probe"}
        if dataset is None:
            return {"error": "embedding_linear_probe needs the dataset for samples",
                    "tier": 3, "probe": "embedding_linear_probe"}

        pidx = _proprio_token_index(self.config.proprio_fusion_stage)
        out_box: Dict[str, Tensor] = {}
        enc = self.policy.model.encoder

        def cap(_m, _i, o):
            # ACTEncoder.forward returns (S, B, D) tensor (no tuple)
            t = o[0] if isinstance(o, tuple) else o
            out_box["enc"] = t.detach().cpu()

        h = enc.register_forward_hook(cap)
        feats: List[np.ndarray] = []
        lbls_phase: List[str] = []
        n = len(dataset)
        step = max(1, n // n_samples)
        sample_idxs = list(range(0, n, step))[:n_samples]
        try:
            self.policy.eval()
            with torch.no_grad():
                for i in sample_idxs:
                    it = dataset[i]
                    b = {k: (v.unsqueeze(0).to(self.device) if torch.is_tensor(v) else v)
                         for k, v in it.items()}
                    # pseudo-label: contact iff elbow current elevated (joint 3)
                    cur = it["observation.state"][..., self.current_indices]
                    elbow = float(cur[..., 2].abs().max()) if cur[..., 2].numel() else 0.0
                    lbls_phase.append("contact" if elbow > 10 else "free")
                    try:
                        self.policy.predict_action_chunk(b)
                        if "enc" in out_box and out_box["enc"].shape[0] > pidx:
                            feats.append(out_box["enc"][pidx, 0, :].numpy())
                    except Exception:
                        continue
        finally:
            h.remove()

        if len(feats) < 10 or len(set(lbls_phase)) < 2:
            return {"error": f"insufficient 2-class data (n={len(feats)}, "
                             f"classes={len(set(lbls_phase))}); "
                             f"raise probe_samples or lower threshold",
                    "tier": 3, "probe": "embedding_linear_probe",
                    "proprio_token_index": pidx}
        X = np.stack(feats)
        y = np.array([1 if l == "contact" else 0 for l in lbls_phase])
        clf = LogisticRegression(max_iter=1000)
        cv = min(3, len(y))
        try:
            auc = float(cross_val_score(clf, X, y, cv=cv, scoring="roc_auc").mean())
            acc = float(cross_val_score(clf, X, y, cv=cv, scoring="accuracy").mean())
        except Exception as e:
            return {"error": f"cv failed: {e}", "tier": 3,
                    "probe": "embedding_linear_probe"}
        res = {"embedding_probe_auc": auc, "embedding_probe_acc": acc,
               "n_samples": len(y), "proprio_token_index": pidx,
               "hook": "encoder_output (post-self-attention)",
               "tier": 3, "probe": "embedding_linear_probe"}
        # optional per-phase split if labels given (aligned with sample_idxs)
        if phase_labels is not None and len(phase_labels) == len(sample_idxs):
            per_ph = {}
            for ph in sorted(set(phase_labels)):
                m_ = np.array([p == ph for p in phase_labels])
                if m_.sum() >= 5 and len(np.unique(y[m_])) > 1:
                    try:
                        a = float(cross_val_score(
                            clf, X[m_], y[m_], cv=min(3, m_.sum()), scoring="roc_auc").mean())
                        per_ph[ph] = {"auc": a, "n": int(m_.sum())}
                    except Exception:
                        pass
            if per_ph:
                res["per_phase_auc"] = per_ph
        return res

    def gradient_flow_trajectory(self, batch: Dict[str, Tensor]) -> Dict:
        """CORE-4. ||grad_current|| / ||grad_position|| at current step
        on the 2D Linear weights of the two state projections.

        NOTE: checkpoint-time only; trajectory-over-training needs per-step
        W&B logging. Requires `action` in batch (use_vae=True needs the target).
        """
        if "action" not in batch:
            return {"error": "needs 'action' in batch (use_vae=True target)",
                    "tier": 3, "probe": "gradient_flow_trajectory"}
        self.policy.train(); self.policy.zero_grad()
        loss, _ = self.policy(batch); loss.backward()
        gc2 = gp2 = 0.0
        for name, p in self.policy.named_parameters():
            if p.grad is None or p.grad.dim() != 2:
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
                "note": "final-step ratio only"}

    def input_gradient_saliency(self) -> Dict:
        return {"status": "confirmatory_stub", "tier": 3,
                "probe": "input_gradient_saliency",
                "implementation": "captum IntegratedGradients on current channels; "
                                   "implement only if Tier-3 tie must be broken"}

    def representation_similarity_cka(self) -> Dict:
        return {"status": "confirmatory_stub", "tier": 3,
                "probe": "representation_similarity_cka",
                "implementation": "linear CKA proprio-projection vs ResNet features; "
                                   "implement only if Tier-3 tie must be broken"}

    # ==============================================================
    # Tier 4: counterfactual
    # ==============================================================
    def probe_gap_upper_bound(self, t1_auc: Optional[float],
                             t3_probe_auc: Optional[float]) -> Dict:
        if t1_auc is None or t3_probe_auc is None:
            return {"error": "requires Tier-1 AUC (Phase-1) and Tier-3 embedding probe",
                    "tier": 4, "probe": "probe_gap_upper_bound"}
        return {"data_ceiling_auc": t1_auc, "model_ceiling_auc": t3_probe_auc,
                "recoverable_gap": t1_auc - t3_probe_auc,
                "interpretation": "small/negative gap => encoder already represents the info; "
                                  "collapse is routing, not representation (-> M3/M5/M4). "
                                  "large gap => representation failure (-> M1/M2/M3).",
                "tier": 4, "probe": "probe_gap_upper_bound"}

    def finetune_probe_counterfactual(self) -> Dict:
        return {"status": "confirmatory_stub", "tier": 4,
                "probe": "finetune_probe_counterfactual",
                "implementation": "unfreeze only proprio path, fine-tune ~1k steps, re-run z_curr"}

    # ==============================================================
    # navigation + report
    # ==============================================================
    def _navigate(self, rep: Dict, tier1_auc: Optional[float]) -> Tuple[str, str]:
        rzo = rep.get("tier2", {}).get("relative_zero_out", {})
        pp = rep.get("tier2", {}).get("per_phase_zero_out", {}).get("phase_pattern", {})
        probe_auc = (rep.get("tier3", {}).get("embedding_linear_probe", {})
                     .get("embedding_probe_auc"))
        grad = (rep.get("tier3", {}).get("gradient_flow_trajectory", {})
                .get("ratio"))
        used = rzo.get("used")

        if used:
            return "no collapse", "none (deploy; primary hardware comparison)"
        if tier1_auc is not None and tier1_auc < 0.6:
            return "signal inadequate (Tier-1)", "none: Tier-0 report"
        # phase-conditional first (per-phase insert ignored but approach used)
        if pp.get("phase_profile") == "insert_ignored_only":
            return "phase-conditional failure", "M4 trigger (phase-gated)"
        if probe_auc is not None and probe_auc >= 0.7:
            return "routing / suppression failure", "M3 token, M5 FiLM, M4 trigger"
        if probe_auc is not None and probe_auc < 0.6:
            return "representation failure", "M1 history, M2 explicit, M3 CNN"
        if grad is not None and grad < 0.05:
            return "generalisation-rate mismatch", "M6 OGM-GE / FACTR curriculum"
        return "collapse mechanism undetermined", "run confirmatory probes (saliency/CKA/finetune)"

    def report(self, batch: Dict[str, Tensor], phase_labels: Optional[List[str]] = None,
               dataset=None, tier1_auc: Optional[float] = None,
               ratio_threshold: float = 20.0, include_tier3: bool = True,
               probe_phase_labels: Optional[List[str]] = None,
               n_probe_samples: int = 200) -> Dict:
        """Single entry point. emit tiered JSON + verdict + recommended_next_method.

        `tier1_auc` is the Phase-1 value (Ch4) -- NOT recomputed here.
        `phase_labels` aligned with batch for per-phase T2.
        `probe_phase_labels` aligned with the embedding probe's sampled frames.
        """
        rep: Dict = {"checkpoint_variant": {
            "temporal_encoder": self.config.proprio_temporal_encoder,
            "fusion_stage": self.config.proprio_fusion_stage,
        }, "tier1_reference_auc": tier1_auc,
           "tier1_note": "Tier-1 is data-level (Phase-1 / Ch4), not recomputed per checkpoint"}
        # Tier 2
        rep["tier2"] = {}
        rep["tier2"]["relative_zero_out"] = self.relative_zero_out(batch, ratio_threshold)
        rep["tier2"]["per_phase_zero_out"] = self.per_phase_zero_out(batch, phase_labels)
        try:
            rep["tier2"]["attention_mass"] = self.attention_mass(batch)
        except Exception as e:
            rep["tier2"]["attention_mass"] = {"error": str(e)}
        rep["tier2"]["attention_mass"]["tier"] = 2
        # Tier 3
        if include_tier3:
            rep["tier3"] = {}
            try:
                rep["tier3"]["embedding_linear_probe"] = self.embedding_linear_probe(
                    dataset=dataset, batch=batch,
                    phase_labels=probe_phase_labels, n_samples=n_probe_samples)
            except Exception as e:
                rep["tier3"]["embedding_linear_probe"] = {"error": str(e)}
            try:
                rep["tier3"]["gradient_flow_trajectory"] = self.gradient_flow_trajectory(batch)
            except Exception as e:
                rep["tier3"]["gradient_flow_trajectory"] = {"error": str(e)}
            rep["tier3"]["input_gradient_saliency"] = self.input_gradient_saliency()
            rep["tier3"]["representation_similarity_cka"] = self.representation_similarity_cka()
        # Tier 4
        t3_auc = ((rep.get("tier3", {}).get("embedding_linear_probe") or {})
                  .get("embedding_probe_auc"))
        rep["tier4"] = {}
        try:
            rep["tier4"]["probe_gap_upper_bound"] = self.probe_gap_upper_bound(tier1_auc, t3_auc)
        except Exception as e:
            rep["tier4"]["probe_gap_upper_bound"] = {"error": str(e)}
        rep["tier4"]["finetune_probe_counterfactual"] = self.finetune_probe_counterfactual()
        # navigation
        mech, method = self._navigate(rep, tier1_auc)
        rep["verdict"] = "USED" if rep["tier2"]["relative_zero_out"].get("used") else "IGNORED"
        rep["mechanism"] = mech
        rep["recommended_next_method"] = method
        return rep
