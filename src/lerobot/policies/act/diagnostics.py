#!/usr/bin/env python

# Copyright 2026 Jay Ranpariya. Thesis extension.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Modality-collapse diagnostic suite for ACT-M checkpoints (ADR-0012).

SINGLE source of truth for modality-use diagnostics. The CLI in
experiments/scripts/modality_analysis.py is a thin loader+wrapper around
ModalityDiagnostics.report(); no diagnostic logic lives there.

=== Framework (ADR-0012): four-tier triangulation, 7 probes ===

Each tier answers a distinct question; the PATTERN across tiers identifies
the collapse mechanism and selects the next method (M1-M6). All 7 probes
below are implemented (the finetune counterfactual was dropped: it's slow,
mutates the model, and the navigation reaches a verdict without it on actm).

=== Data requirements (honest) ===

  Model-only (no dataset; a single observation batch suffices):
    - relative_zero_out        [T2]   (also needs an action chunk for the
                                       gradient probe, but that's a forward pass)
    - attention_mass           [T2]   (M3/M4 only)
    - gradient_flow_trajectory [T3]

  Dataset + pseudo-labels (the contact-vs-free label is derived from the
  current signal itself; defensible because contact IS defined by elevated
  current, but NOTE this is for the contact-vs-free PROBE target, not for
  the task-phase labels -- those come from MANUAL annotation in phases.csv):
    - embedding_linear_probe   [T3]   (stratified by manual phase labels)
    - mutual_information       [T3]
    - input_gradient_saliency  [T3]

  Dataset, no labels:
    - representation_similarity_cka [T3]

  Derived (T1 + T3):
    - probe_gap_upper_bound    [T4]

=== Phase labels (no longer self-referential) ===

Task-phase labels (approach/grasp/insert/release) come from MANUAL annotation
in experiments/datasets/peg-tight-vertical-100/phase-annotations/phases.csv
(see annotate_phases.py). The loader falls back to a current-signal heuristic
for un-annotated episodes and reports how many are gold vs heuristic, so the
diagnostic can note label confidence. This is the SAME infrastructure the
exposé (line 113) requires for phase-conditioned eval-trial annotation.

=== Performance ===

Phase labels + MI + CKA read state from PARQUET (~0.03 ms/frame) not via
dataset[i] (~12 ms/video-decode). Full 7-probe run on actm: ~30 s, not 400 s.
"""

import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from .modeling_act import ACTPolicy

PHASES = ["approach", "grasp", "insert", "release"]

# locate the phase-annotations module (sibling of the lerobot package)
_ANNOTATION_PATH = (Path(__file__).resolve().parents[5]
                    / "experiments" / "scripts")
if str(_ANNOTATION_PATH) not in sys.path:
    sys.path.insert(0, str(_ANNOTATION_PATH))
try:
    from phase_annotations import load_phase_labels, proprio_token_index as _pti
    _HAS_ANNOTATIONS = True
except ImportError:
    _HAS_ANNOTATIONS = False

    def _pti(fusion_stage: str) -> int:
        return 2 if fusion_stage in ("token", "hybrid") else 1


def _proprio_token_index(fusion_stage: str) -> int:
    return _pti(fusion_stage)


def _build_action_chunk(dataset, idx: int, chunk_size: int) -> Tensor:
    """Assemble a (chunk_size, action_dim) action chunk for frame idx.
    The dataset stores one action per frame; ACT.forward needs a horizon.
    Collects chunk_size consecutive actions, clamping at episode boundaries.
    """
    it = dataset[idx]
    a0 = it["action"]
    if a0.dim() == 1:
        a0 = a0.unsqueeze(0)
    if a0.shape[0] >= chunk_size:
        return a0[:chunk_size]
    chunk = [a0]
    ep = int(it["episode_index"].item()) if "episode_index" in it else None
    j = idx + 1
    while len(chunk) < chunk_size and j < len(dataset):
        nit = dataset[j]
        if ep is not None and int(nit["episode_index"].item()) != ep:
            break
        na = nit["action"]
        chunk.append(na.unsqueeze(0) if na.dim() == 1 else na)
        j += 1
    while len(chunk) < chunk_size:
        chunk.append(chunk[-1])
    return torch.cat(chunk, dim=0)[:chunk_size]


try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class ModalityDiagnostics:
    """Four-tier modality-collapse triangulation suite (ADR-0012). 7 probes."""

    def __init__(self, policy, device: str = "cuda",
                 parquet_path: Optional[Path] = None):
        # Policy-agnostic: accepts ACTPolicy or DiffusionPolicy.
        self.policy = policy
        self.policy.eval()
        self.device = device
        self.config = policy.config
        # detect policy family for architecture-specific hookpoints (ADR-0013 D4)
        self.policy_type = getattr(self.config, "type", "act")
        self.is_diffusion = (self.policy_type == "diffusion")
        # current channels: for ACT this is proprio_current_indices (default 6:12);
        # for DP there is no such field, but the state layout is the same (pos 0:6, cur 6:12)
        # so we use the same range. For DP-V (act-v view, state dim 6) there are no currents.
        self.current_indices = list(getattr(
            self.config, "proprio_current_indices", list(range(6, 12))))
        # chunk/horizon: ACT uses chunk_size, DP uses horizon
        self.chunk_size = int(getattr(self.config, "chunk_size",
                                     getattr(self.config, "horizon", 100)))
        self.parquet_path = parquet_path
        # cache manual phase labels (loaded lazily, from parquet)
        self._phase_labels: Optional[List[str]] = None
        self._phase_meta: Dict = {}

    # ----- phase labels (manual annotation, parquet-fast) -----
    def _load_phase_labels(self, dataset) -> None:
        if self._phase_labels is not None or not _HAS_ANNOTATIONS:
            return
        # find the parquet backing this dataset
        pq = self.parquet_path
        if pq is None:
            # walk the dataset's root to find data/chunk-000/file-000.parquet
            root = getattr(dataset, "root", None)
            if root is None and hasattr(dataset, "base"):
                root = getattr(dataset.base, "root", None)
            if root is not None:
                cand = Path(root) / "data" / "chunk-000" / "file-000.parquet"
                if cand.exists():
                    pq = cand
            if pq is None or not Path(pq).exists():
                self._phase_meta = {"error": "parquet not found; cannot load phase labels"}
                return
        try:
            labels, meta = load_phase_labels(Path(pq))
            self._phase_labels = labels
            self._phase_meta = meta
        except Exception as e:
            self._phase_meta = {"error": str(e)}

    def _phase_of_frame(self, dataset, frame_idx: int) -> str:
        """Phase label for a dataset frame, using the cached manual labels."""
        self._load_phase_labels(dataset)
        if self._phase_labels and frame_idx < len(self._phase_labels):
            return self._phase_labels[frame_idx]
        return "unknown"

    # ----- shared zero-out -----
    def _predict_chunk(self, batch: Dict[str, Tensor]) -> Tensor:
        """Policy-agnostic action-chunk prediction. ACT uses predict_action_chunk(batch);
        DP's predict_action_chunk reads queues (ignores batch), so we call
        diffusion.generate_actions(batch) directly with a (B, n_obs_steps, ...) batch."""
        if self.is_diffusion:
            return self.policy.diffusion.generate_actions(batch)
        return self.policy.predict_action_chunk(batch)

    def _zero_out(self, batch: Dict[str, Tensor], which: str) -> Tuple[Tensor, Tensor]:
        with torch.no_grad():
            af = self._predict_chunk(batch)
            b = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in batch.items()}
            state_dim = b["observation.state"].shape[-1]
            # if this is a vision-only model (state_dim <= max(current_indices)), there are
            # no current channels to zero; zero the FULL state as a no-op control (z_curr ~ 0)
            if which == "current" and state_dim <= max(self.current_indices):
                # no proprio channel in this checkpoint (e.g. DP-V vision-only baseline);
                # return unchanged -> z_curr will be 0, verdict IGNORED trivially (correct:
                # a vision-only policy has no modality to use)
                az = af.clone()
                return af, az
            idx = (self.current_indices if which == "current"
                   else [i for i in range(state_dim)
                         if i not in self.current_indices])
            if "observation.state" in b:
                s = b["observation.state"].clone(); s[..., idx] = 0.0; b["observation.state"] = s
            # observation.state_window is ACT-only (temporal encoders); DP has no window
            if not self.is_diffusion and "observation.state_window" in b:
                w = b["observation.state_window"].clone()
                n_cur = len(self.current_indices)
                kp1 = w.shape[-1] // n_cur
                if which == "current":
                    for k in range(kp1):
                        w[..., k * n_cur:(k + 1) * n_cur] = 0.0
                b["observation.state_window"] = w
            az = self._predict_chunk(b)
            return af, az

    # ==============================================================
    # Tier 2: utilisation
    # ==============================================================
    def relative_zero_out(self, batch: Dict[str, Tensor],
                          ratio_threshold: float = 20.0) -> Dict:
        af, ac = self._zero_out(batch, "current")
        _, ap = self._zero_out(batch, "position")
        l2c = torch.norm(af - ac, dim=(1, 2)); l2p = torch.norm(af - ap, dim=(1, 2))
        norm = torch.norm(af, dim=(1, 2))
        z_curr = (l2c.mean() / (norm.mean() + 1e-8)).item()
        z_pos = (l2p.mean() / (norm.mean() + 1e-8)).item()
        rel = z_curr / (z_pos + 1e-8)
        per_sample = (l2c / (norm + 1e-8)).cpu().numpy().tolist()
        return {"z_curr": z_curr, "z_pos": z_pos, "relative_curr": rel,
                "used": bool(rel > 1.0 / ratio_threshold),
                "threshold_ratio": 1.0 / ratio_threshold,
                "n_samples": int(af.shape[0]),
                "per_sample_z_curr": per_sample,
                "tier": 2, "probe": "relative_zero_out"}

    def per_phase_zero_out(self, batch: Dict[str, Tensor], dataset=None,
                           batch_indices: Optional[List[int]] = None) -> Dict:
        """Per-phase z_curr split, using MANUAL phase labels from phases.csv."""
        af, ac = self._zero_out(batch, "current")
        l2c = torch.norm(af - ac, dim=(1, 2))
        norm = torch.norm(af, dim=(1, 2))
        per_sample = (l2c / (norm + 1e-8)).cpu().numpy()
        # resolve phase per sample
        phase_labels: List[str] = []
        if dataset is not None and batch_indices is not None:
            for i in batch_indices:
                phase_labels.append(self._phase_of_frame(dataset, i))
        else:
            phase_labels = ["unknown"] * int(af.shape[0])
        per: Dict[str, Dict] = {}
        for ph in sorted(set(phase_labels)):
            ixs = np.array([i for i, p in enumerate(phase_labels) if p == ph])
            if len(ixs) == 0:
                continue
            per[ph] = {"z_curr_mean": float(per_sample[ixs].mean()),
                       "n": int(len(ixs))}
        pattern = {}
        insert = per.get("insert", {}).get("z_curr_mean")
        approach = per.get("approach", {}).get("z_curr_mean")
        grasp = per.get("grasp", {}).get("z_curr_mean")
        if insert is not None and approach is not None:
            if insert < 0.05 and approach < 0.05:
                pattern["phase_profile"] = "ignored_everywhere"
            elif insert < 0.05 and (approach >= 0.05 or (grasp is not None and grasp >= 0.05)):
                pattern["phase_profile"] = "insert_ignored_only"  # -> M4
        return {"per_phase": per, "phase_pattern": pattern,
                "phase_label_meta": self._phase_meta,
                "tier": 2, "probe": "per_phase_zero_out"}

    def attention_weight_analysis(self, batch: Dict[str, Tensor],
                                   save_figure: Optional[Path] = None) -> Dict:
        """Per-layer, per-head attention-weight analysis (the exposé's named
diagnostic). Returns the full attention distribution, not a single ratio.

SAFETY (verified): the monkey-patch passes `need_weights=True` to
nn.MultiheadAttention, which ONLY additionally materialises the attention
weight matrix that the forward pass already computes internally. The
attn_output (and hence the encoder output) is bit-for-bit identical
(max abs diff = 0.0 on the actm checkpoint). The patch is applied only
during this diagnostic, on a loaded checkpoint, in eval mode under
no_grad, and restored in a finally block. Training, the checkpoint file,
and all subsequent inference are completely unaffected.

STRUCTURAL LIMITATION: for early/film fusion the 6 current channels are
merged into the state token (index 1) before the encoder — there is no
separate proprio token to attend to, so attention weights cannot
decompose current-vs-position utilisation. The exposé explicitly allows
"an equivalent ablation-based diagnostic" (zero-out) for this case, which
the suite provides (relative_zero_out). This probe activates only for
token/hybrid fusion (M3/M4), where proprio enters as a separate token."""
        import types
        fs = getattr(self.config, "proprio_fusion_stage", None) if not self.is_diffusion else None
        if self.is_diffusion:
            return {"status": "structurally_impossible",
                    "message": ("Diffusion Policy has no attention matrix; the U-Net "
                                "conditions on global_cond via FiLM (not self-attention). "
                                "The suite relies on the ablation-based zero-out probe for "
                                "DP, which the exposé permits as 'an equivalent "
                                "ablation-based diagnostic' (ADR-0013 D4)."),
                    "tier": 2, "probe": "attention_weight_analysis"}
        if fs not in ("token", "hybrid"):
            return {"status": "structurally_impossible",
                    "message": (f"fusion_stage={fs}: current channels are merged into "
                                f"the state token before the encoder; there is no separate "
                                f"proprio token to attend to. Attention weights cannot "
                                f"decompose current-vs-position use. The suite relies on "
                                f"the ablation-based diagnostic (relative_zero_out) for "
                                f"early/film fusion, which the exposé (line 111) explicitly "
                                f"permits as 'an equivalent ablation-based diagnostic'."),
                    "tier": 2, "probe": "attention_weight_analysis"}

        captured: List[Tensor] = []
        orig_forwards = []
        layers = self.policy.model.encoder.layers
        # monkey-patch each encoder layer's forward to pass need_weights=True
        for layer in layers:
            orig = layer.forward
            orig_forwards.append(orig)

            def patched(self, x, pos_embed=None, key_padding_mask=None, _orig=orig):
                skip = x
                if self.pre_norm:
                    x = self.norm1(x)
                q = k = x if pos_embed is None else x + pos_embed
                out, attn = self.self_attn(q, k, value=x,
                                          key_padding_mask=key_padding_mask,
                                          need_weights=True,
                                          average_attn_weights=False)
                captured.append(attn.detach().clone())
                x = out[0] if isinstance(out, tuple) else out
                x = skip + self.dropout1(x)
                if self.pre_norm:
                    skip = x; x = self.norm2(x)
                else:
                    x = self.norm1(x); skip = x
                x = self.linear2(self.dropout(self.activation(self.linear1(x))))
                x = skip + self.dropout2(x)
                if not self.pre_norm:
                    x = self.norm2(x)
                return x
            layer.forward = types.MethodType(patched, layer)

        try:
            with torch.no_grad():
                b = dict(batch)
                # ACT.forward needs observation.images as a list + env_state
                if self.config.image_features:
                    b["observation.images"] = [b[k] for k in self.config.image_features]
                    for k in self.config.image_features:
                        b.pop(k, None)
                if "observation.environment_state" not in b:
                    b["observation.environment_state"] = b["observation.state"]
                self.policy.model(b)
        finally:
            for layer, of in zip(layers, orig_forwards):
                layer.forward = of

        if not captured:
            return {"error": "no attention weights captured", "tier": 2,
                    "probe": "attention_weight_analysis"}

        pidx = _proprio_token_index(fs)
        # token layout: [latent(0), state(1), temporal(2), vision(3..)]
        n_layers = len(captured)
        n_heads = captured[0].shape[1]
        # aggregate: for each layer, mean attention TO the proprio token
        # (averaged over heads and query positions)
        per_layer = []
        for li, attn in enumerate(captured):
            # attn: (B, H, S, S); attn[..., :, pidx] = attention TO proprio token
            to_proprio = attn[..., :, pidx].mean().item()
            to_latent = attn[..., :, 0].mean().item()
            to_state = attn[..., :, 1].mean().item()
            to_vision = attn[..., :, 3:].mean().item() if attn.shape[-1] > 3 else 0.0
            # also: how much does the DECODER attend to proprio (the [CLS] query)?
            # encoder self-attn is symmetric; report the mass distribution.
            per_layer.append({
                "layer": li, "to_latent": to_latent, "to_state": to_state,
                "to_proprio": to_proprio, "to_vision": to_vision,
                "proprio_ratio": to_proprio / (to_proprio + to_vision + 1e-8),
            })
        # head-level variation (does any head specialise in proprio?)
        last_layer = captured[-1]  # (B, H, S, S)
        per_head_proprio = [float(last_layer[0, h, :, pidx].mean())
                            for h in range(n_heads)]
        overall_proprio = float(np.mean([pl["to_proprio"] for pl in per_layer]))
        overall_vision = float(np.mean([pl["to_vision"] for pl in per_layer]))
        overall_latent = float(np.mean([pl["to_latent"] for pl in per_layer]))
        overall_state = float(np.mean([pl["to_state"] for pl in per_layer]))
        total = overall_latent + overall_state + overall_proprio + overall_vision + 1e-8

        result = {
            "n_layers": n_layers, "n_heads": n_heads,
            "proprio_token_index": pidx,
            # full attention distribution (all four token classes, normalised to ~1)
            "attention_distribution": {
                "latent_cls": round(overall_latent / total, 3),
                "state_pos": round(overall_state / total, 3),
                "proprio_temporal": round(overall_proprio / total, 3),
                "vision": round(overall_vision / total, 3),
            },
            "overall_proprio_mass": overall_proprio,
            "overall_vision_mass": overall_vision,
            "proprio_vs_vision_ratio": overall_proprio / (overall_vision + 1e-8),
            "overall_proprio_ratio": overall_proprio / (overall_proprio + overall_vision + 1e-8),
            "per_layer": per_layer,
            "per_head_proprio_mass": {f"head{h}": per_head_proprio[h]
                                       for h in range(n_heads)},
            "max_head_proprio": float(max(per_head_proprio)),
            "interpretation": ("proprio_ratio < 0.1 with no head specialising "
                               "=> proprio token is being ignored (M3/M5); "
                               "one head concentrating > 0.3 => routed but "
                               "under-amplified (M5 FiLM); "
                               "overall > 0.2 => actively attended"),
            "tier": 2, "probe": "attention_weight_analysis",
        }
        if save_figure is not None:
            try:
                import matplotlib.pyplot as plt
                fig, axes = plt.subplots(1, 2, figsize=(11, 4))
                # left: per-layer mass to each token type
                ll = [pl["layer"] for pl in per_layer]
                axes[0].plot(ll, [pl["to_latent"] for pl in per_layer], "o-", label="latent")
                axes[0].plot(ll, [pl["to_state"] for pl in per_layer], "s-", label="state")
                axes[0].plot(ll, [pl["to_proprio"] for pl in per_layer], "^-", label="proprio (temporal)", color="red")
                axes[0].plot(ll, [pl["to_vision"] for pl in per_layer], "d-", label="vision", alpha=0.6)
                axes[0].set_xlabel("encoder layer"); axes[0].set_ylabel("mean attention mass")
                axes[0].set_title("Per-layer attention distribution")
                axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)
                # right: per-head proprio mass (last layer)
                axes[1].bar(range(n_heads), per_head_proprio, color="red", alpha=0.7)
                axes[1].set_xlabel("attention head (last layer)")
                axes[1].set_ylabel("mean attention to proprio token")
                axes[1].set_title("Per-head proprio specialisation")
                axes[1].grid(alpha=0.3, axis="y")
                plt.tight_layout()
                save_figure.parent.mkdir(parents=True, exist_ok=True)
                fig.savefig(save_figure, dpi=150, bbox_inches="tight")
                plt.close()
                result["figure"] = str(save_figure)
            except Exception as e:
                result["figure_error"] = str(e)
        return result

    # ==============================================================
    # Tier 3: mechanism (5 probes; finetune dropped)
    # ==============================================================
    def embedding_linear_probe(self, dataset=None, n_samples: int = 150) -> Dict:
        """Linear probe on the POST-ENCODER proprio token (architecture-aware).
        Stratified by manual phase labels so per-phase AUC is real."""
        if not SKLEARN_AVAILABLE:
            return {"error": "sklearn not installed", "tier": 3,
                    "probe": "embedding_linear_probe"}
        if dataset is None:
            return {"error": "needs dataset", "tier": 3, "probe": "embedding_linear_probe"}
        # DP has no proprio token (conditions via global_cond + FiLM); structurally impossible.
        if self.is_diffusion:
            return {"status": "structurally_impossible",
                    "message": ("Diffusion Policy has no proprio token to probe; the U-Net "
                                "conditions on global_cond (state+vision) via FiLM. The "
                                "ablation-based zero-out probe is the DP-equivalent diagnostic "
                                "(ADR-0013 D4)."),
                    "tier": 3, "probe": "embedding_linear_probe"}

        self._load_phase_labels(dataset)
        pidx = _proprio_token_index(self.config.proprio_fusion_stage)
        out_box: Dict[str, Tensor] = {}

        def cap(_m, _i, o):
            t = o[0] if isinstance(o, tuple) else o
            out_box["enc"] = t.detach().cpu()

        h = self.policy.model.encoder.register_forward_hook(cap)

        # stratified sampling by phase (uses manual labels; falls back to
        # "approach" if labels missing for that frame)
        n = len(dataset)
        per_phase_budget = max(1, n_samples // len(PHASES))
        sample_idxs: List[int] = []
        sample_phases: List[str] = []
        if self._phase_labels and len(self._phase_labels) == n:
            for ph in PHASES:
                ixs = [i for i, p in enumerate(self._phase_labels) if p == ph]
                step_p = max(1, len(ixs) // per_phase_budget)
                picked = ixs[::step_p][:per_phase_budget]
                sample_idxs.extend(picked)
                sample_phases.extend([ph] * len(picked))
        else:
            # unlabelled: even sample
            step = max(1, n // n_samples)
            sample_idxs = list(range(0, n, step))[:n_samples]
            sample_phases = ["unknown"] * len(sample_idxs)

        feats: List[np.ndarray] = []
        lbls: List[int] = []  # contact-vs-free (current-threshold pseudo-label)
        phs: List[str] = []
        try:
            self.policy.eval()
            with torch.no_grad():
                for i in sample_idxs:
                    it = dataset[i]
                    b = {k: (v.unsqueeze(0).to(self.device) if torch.is_tensor(v) else v)
                         for k, v in it.items()}
                    cur = it["observation.state"][..., self.current_indices]
                    elbow = float(cur[..., 2].abs().max()) if cur[..., 2].numel() else 0.0
                    lbls.append(int(elbow > 10))
                    try:
                        self.policy.predict_action_chunk(b)
                        if "enc" in out_box and out_box["enc"].shape[0] > pidx:
                            feats.append(out_box["enc"][pidx, 0, :].numpy())
                            phs.append(self._phase_of_frame(dataset, i))
                    except Exception:
                        continue
        finally:
            h.remove()

        if len(feats) < 10 or len(set(lbls)) < 2:
            return {"error": f"insufficient 2-class data (n={len(feats)})",
                    "tier": 3, "probe": "embedding_linear_probe",
                    "proprio_token_index": pidx}
        X = np.stack(feats); y = np.array(lbls[:len(feats)])
        clf = LogisticRegression(max_iter=1000)
        cv = min(5, len(y))
        try:
            auc = float(cross_val_score(clf, X, y, cv=cv, scoring="roc_auc").mean())
            acc = float(cross_val_score(clf, X, y, cv=cv, scoring="accuracy").mean())
        except Exception as e:
            return {"error": f"cv failed: {e}", "tier": 3, "probe": "embedding_linear_probe"}
        res = {"embedding_probe_auc": auc, "embedding_probe_acc": acc,
               "n_samples": len(y), "proprio_token_index": pidx,
               "hook": "encoder_output (post-self-attention)",
               "contact_label_source": "current-threshold pseudo-label (elbow > 10)",
               "phase_label_meta": self._phase_meta,
               "tier": 3, "probe": "embedding_linear_probe"}
        # per-phase AUC (real, because stratified)
        per_ph = {}
        for ph in sorted(set(phs)):
            m_ = np.array([p == ph for p in phs])
            if m_.sum() >= 5 and len(np.unique(y[m_])) > 1:
                try:
                    a = float(cross_val_score(
                        clf, X[m_], y[m_], cv=min(3, int(m_.sum())),
                        scoring="roc_auc").mean())
                    per_ph[ph] = {"auc": a, "n": int(m_.sum())}
                except Exception:
                    pass
        if per_ph:
            res["per_phase_auc"] = per_ph
        return res

    def gradient_flow_trajectory(self, batch: Dict[str, Tensor]) -> Dict:
        """||grad_current|| / ||grad_position|| on the state-input-projection
        Linear weights, at the final training step."""
        # DP has no dedicated state-projection Linear; the state is raw-concatenated
        # into global_cond and each UNet block's cond_encoder Linear mixes proprio
        # + vision + timestep, so a clean current-vs-position gradient ratio is not
        # computable. (ADR-0013 D4: structurally transformer-specific in spirit.)
        if self.is_diffusion:
            return {"status": "structurally_impossible",
                    "message": ("Diffusion Policy has no dedicated state-input projection; "
                                "the state is concatenated into global_cond and the U-Net "
                                "block cond_encoder Linears mix proprio+vision+timestep, so "
                                "a clean current-vs-position gradient ratio is not "
                                "computable. The ablation-based zero-out probe is the "
                                "DP-equivalent diagnostic (ADR-0013 D4)."),
                    "tier": 3, "probe": "gradient_flow_trajectory"}
        if "action" not in batch or batch["action"].dim() != 3:
            return {"error": "needs 'action' as (B, chunk_size, act_dim). "
                             "build_batch(include_action=True) assembles it.",
                    "tier": 3, "probe": "gradient_flow_trajectory"}
        self.policy.train(); self.policy.zero_grad()
        loss, _ = self.policy(batch); loss.backward()
        gc2 = gp2 = 0.0
        n_cur = len(self.current_indices)
        for name, p in self.policy.named_parameters():
            if p.grad is None or p.grad.dim() != 2:
                continue
            if (name.endswith("encoder_robot_state_input_proj.weight") or
                name.endswith("vae_encoder_robot_state_input_proj.weight")):
                g = p.grad
                in_dim = g.shape[-1]
                gp2 += float(g[..., :in_dim - n_cur].norm().item()) ** 2
                gc2 += float(g[..., in_dim - n_cur:].norm().item()) ** 2
        self.policy.eval()
        gc = math.sqrt(gc2); gp = math.sqrt(gp2)
        return {"grad_current_norm": gc, "grad_position_norm": gp,
                "ratio": gc / (gp + 1e-8), "tier": 3,
                "probe": "gradient_flow_trajectory",
                "note": "final-step ratio; trajectory-over-training needs W&B logs"}

    def mutual_information(self, parquet_path: Optional[Path] = None,
                            n_samples: int = 1000) -> Dict:
        """MI between each raw current channel and the contact-vs-free label.
        Reads state from PARQUET (fast)."""
        pq = parquet_path or self.parquet_path
        if pq is None or not Path(pq).exists():
            return {"error": "needs parquet_path (set in __init__ or pass)", "tier": 3,
                    "probe": "mutual_information"}
        import pandas as pd
        df = pd.read_parquet(pq)
        states = np.stack(df["observation.state"].values)
        cur = states[:, self.current_indices]  # (N, 6)
        # contact label: elbow (col 2) > 10
        elbow = cur[:, 2]
        y = (elbow > 10).astype(int)
        if len(set(y)) < 2:
            return {"error": "single-class labels", "tier": 3, "probe": "mutual_information"}
        # subsample
        step = max(1, len(cur) // n_samples)
        X = cur[::step]; ys = y[::step]

        def _mi_1d(x, yb, n_bins=3):
            x_range = float(np.ptp(x))
            x = (x - x.min()) / (x_range + 1e-9)
            xb = np.clip((x * n_bins).astype(int), 0, n_bins - 1)
            eps = 1e-12
            pxy = np.histogram2d(xb, yb, bins=n_bins)[0] + eps
            pxy /= pxy.sum()
            px = pxy.sum(1, keepdims=True); py = pxy.sum(0, keepdims=True)
            return float((pxy * np.log(pxy / (px @ py))).sum())
        per_ch = [_mi_1d(X[:, j], ys) for j in range(X.shape[1])]
        return {"per_channel_mi": {f"J{j+1}": per_ch[j] for j in range(len(per_ch))},
                "total_mi": float(sum(per_ch)), "n_samples": len(ys),
                "tier": 3, "probe": "mutual_information",
                "label_source": "current-threshold pseudo-label (elbow > 10)"}

    def input_gradient_saliency(self, dataset=None, n_samples: int = 50) -> Dict:
        """Saliency of current channels on the VAE reconstruction loss.
        Calls model() (NOT predict_action_chunk, which runs no_grad)."""
        if dataset is None:
            return {"error": "needs dataset", "tier": 3, "probe": "input_gradient_saliency"}
        # DP loss-path (diffusion.compute_loss) expects a different batch structure
        # than the ACT VAE path; saliency adaptation is non-trivial and the probe is
        # confirmatory (Tier-3). Mark structurally-limited for DP (ADR-0013 D4).
        if self.is_diffusion:
            return {"status": "structurally_limited",
                    "message": ("Diffusion Policy's loss path (diffusion.compute_loss) "
                                "expects a batch structure incompatible with the ACT-VAE "
                                "saliency hookpoint; adaptation is non-trivial. The "
                                "ablation-based zero-out probe is the DP-equivalent "
                                "diagnostic (ADR-0013 D4)."),
                    "tier": 3, "probe": "input_gradient_saliency"}
        sal_cur = []; sal_pos = []
        n = len(dataset); step = max(1, n // n_samples)
        self.policy.train()
        for i in range(0, n, step):
            it = dataset[i]
            b = {k: (v.unsqueeze(0).to(self.device) if torch.is_tensor(v) else v)
                 for k, v in it.items()}
            chunk = _build_action_chunk(dataset, i, self.chunk_size).to(self.device)
            b["action"] = chunk.unsqueeze(0)
            b["action_is_pad"] = torch.zeros(1, self.chunk_size, dtype=torch.bool,
                                             device=self.device)
            s = b["observation.state"]
            if not s.requires_grad:
                s.requires_grad_(True)
            b["observation.state"] = s
            try:
                with torch.enable_grad():
                    loss, _ = self.policy(b)
                if loss.grad_fn is None:
                    continue
                loss.backward()
                if s.grad is not None:
                    g = s.grad[0]
                    sal_cur.append(float(g[..., self.current_indices].abs().sum()))
                    pos_idx = [j for j in range(s.shape[-1]) if j not in self.current_indices]
                    sal_pos.append(float(g[..., pos_idx].abs().sum()))
            except Exception:
                continue
        self.policy.eval()
        if not sal_cur:
            return {"error": "no saliency captured", "tier": 3, "probe": "input_gradient_saliency"}
        sc = float(np.mean(sal_cur)); sp = float(np.mean(sal_pos))
        return {"saliency_current": sc, "saliency_position": sp,
                "ratio": sc / (sp + 1e-8), "n_samples": len(sal_cur),
                "tier": 3, "probe": "input_gradient_saliency",
                "interpretation": "ratio<0.1 with concentrated spatial pattern -> M5 FiLM"}

    def representation_similarity_cka(self, dataset=None, n_samples: int = 100) -> Dict:
        """Linear CKA between the proprio and vision post-encoder tokens.
        High CKA = modalities collapsed to the same representation -> M3/M5."""
        if dataset is None:
            return {"error": "needs dataset", "tier": 3, "probe": "representation_similarity_cka"}
        # DP branch: CKA between the proprio portion of global_cond and the rgb_encoder output.
        # DP has no separate proprio/vision tokens; the global_cond concatenates state + img_features.
        # We hook rgb_encoder to get vision features, and read the proprio from observation.state.
        if self.is_diffusion:
            return self._cka_diffusion(dataset, n_samples)
        pidx = _proprio_token_index(self.config.proprio_fusion_stage)
        vidx = pidx + 1
        out_box: Dict[str, Tensor] = {}

        def cap(_m, _i, o):
            t = o[0] if isinstance(o, tuple) else o
            out_box["enc"] = t.detach()

        h = self.policy.model.encoder.register_forward_hook(cap)
        proprio_feats: List[np.ndarray] = []; vision_feats: List[np.ndarray] = []
        n = len(dataset); step = max(1, n // n_samples)
        try:
            self.policy.eval()
            with torch.no_grad():
                for i in range(0, n, step):
                    it = dataset[i]
                    b = {k: (v.unsqueeze(0).to(self.device) if torch.is_tensor(v) else v)
                         for k, v in it.items()}
                    try:
                        self.policy.predict_action_chunk(b)
                        enc = out_box.get("enc")
                        if enc is not None and enc.shape[0] > vidx:
                            proprio_feats.append(enc[pidx, 0, :].cpu().numpy())
                            vision_feats.append(enc[vidx, 0, :].cpu().numpy())
                    except Exception:
                        continue
        finally:
            h.remove()
        if len(proprio_feats) < 10:
            return {"error": "insufficient samples", "tier": 3, "probe": "representation_similarity_cka"}
        X = np.stack(proprio_feats); Y = np.stack(vision_feats)

        def _cka(A, B):
            A = A - A.mean(0); B = B - B.mean(0)
            num = float(np.linalg.norm(A.T @ B) ** 2)
            den = (float(np.linalg.norm(A.T @ A) ** 2) *
                   float(np.linalg.norm(B.T @ B) ** 2)) ** 0.5
            return num / (den + 1e-12)
        cka = _cka(X, Y)
        return {"cka": cka, "n_samples": len(X),
                "proprio_token_index": pidx, "vision_token_index": vidx,
                "tier": 3, "probe": "representation_similarity_cka",
                "interpretation": "high CKA (>0.8) = modality collapse -> M3/M5"}

    def _cka_diffusion(self, dataset, n_samples: int) -> Dict:
        """DP-adapted CKA: similarity between the proprio portion of global_cond
        (the raw current channels) and the rgb_encoder output (vision features).
        High CKA means the conditioning has collapsed the two modalities together."""
        out_box: Dict[str, Tensor] = {}
        def cap(_m, _i, o):
            out_box["vis"] = o.detach()
        h = self.policy.diffusion.rgb_encoder.register_forward_hook(cap)
        proprio_feats: List[np.ndarray] = []; vision_feats: List[np.ndarray] = []
        n = len(dataset); step = max(1, n // n_samples)
        try:
            self.policy.eval()
            with torch.no_grad():
                for i in range(0, n, step):
                    it = dataset[i]
                    b = {k: (v.unsqueeze(0).to(self.device) if torch.is_tensor(v) else v)
                         for k, v in it.items()}
                    # DP expects (B, n_obs_steps, ...) for state; replicate the single frame
                    if b["observation.state"].dim() == 2:
                        b["observation.state"] = b["observation.state"].unsqueeze(0)
                    img = b.get("observation.images.top")
                    if img is None or img.dim() < 4:
                        continue
                    # run the encoder directly on one frame
                    try:
                        self.policy.diffusion.rgb_encoder(img if img.dim()==4 else img.unsqueeze(0))
                        if "vis" in out_box:
                            vis = out_box["vis"].flatten().cpu().numpy()
                            # proprio = the current channels from the state
                            st = b["observation.state"].flatten().cpu().numpy()
                            cur = st[self.current_indices] if len(st) > max(self.current_indices) else st
                            # pad/truncate to match vision feat length for CKA (needs same N across samples)
                            proprio_feats.append(cur[:len(vis)] if len(cur) >= len(vis) else
                                                 np.pad(cur, (0, len(vis) - len(cur))))
                            vision_feats.append(vis[:len(proprio_feats[-1])])
                    except Exception:
                        continue
        finally:
            h.remove()
        if len(proprio_feats) < 10:
            return {"error": f"insufficient samples (n={len(proprio_feats)})", "tier": 3,
                    "probe": "representation_similarity_cka"}
        X = np.stack(proprio_feats); Y = np.stack(vision_feats)
        def _cka(A, B):
            A = A - A.mean(0); B = B - B.mean(0)
            num = float(np.linalg.norm(A.T @ B) ** 2)
            den = (float(np.linalg.norm(A.T @ A) ** 2) *
                   float(np.linalg.norm(B.T @ B) ** 2)) ** 0.5
            return num / (den + 1e-12)
        cka = _cka(X, Y)
        return {"cka": cka, "n_samples": len(X),
                "note": "DP-adapted: CKA between proprio channels and rgb_encoder output",
                "tier": 3, "probe": "representation_similarity_cka",
                "interpretation": "high CKA (>0.8) = modality collapse -> M3/M5"}

    # ==============================================================
    # Tier 4: derived
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

    # ==============================================================
    # navigation + report
    # ==============================================================
    def _navigate(self, rep: Dict, tier1_auc: Optional[float]) -> Tuple[str, str]:
        rzo = rep.get("tier2", {}).get("relative_zero_out", {})
        pp = rep.get("tier2", {}).get("per_phase_zero_out", {}).get("phase_pattern", {})
        awa = rep.get("tier2", {}).get("attention_weight_analysis", {})
        probe_auc = (rep.get("tier3", {}).get("embedding_linear_probe", {})
                     .get("embedding_probe_auc"))
        grad = (rep.get("tier3", {}).get("gradient_flow_trajectory", {}).get("ratio"))
        cka = (rep.get("tier3", {}).get("representation_similarity_cka", {}).get("cka"))
        saliency = (rep.get("tier3", {}).get("input_gradient_saliency", {}).get("ratio"))
        used = rzo.get("used")
        if used:
            return "no collapse", "none (deploy; primary hardware comparison)"
        if tier1_auc is not None and tier1_auc < 0.6:
            return "signal inadequate (Tier-1)", "none: Tier-0 report"
        if pp.get("phase_profile") == "insert_ignored_only":
            return "phase-conditional failure", "M4 trigger (phase-gated)"
        if cka is not None and cka > 0.8:
            return "modal representation collapse (high CKA)", "M3 separate encoder, M5 FiLM"
        # attention-weight analysis (only available for token/hybrid fusion)
        if awa.get("overall_proprio_ratio") is not None:
            pr = awa["overall_proprio_ratio"]
            if pr < 0.05 and awa.get("max_head_proprio", 0) < 0.1:
                return "proprio token unattended (attention-weight analysis)", "M3 token routing, M5 FiLM"
            if pr < 0.1 and awa.get("max_head_proprio", 0) >= 0.3:
                return "proprio routed but under-amplified", "M5 FiLM (amplification)"
        if probe_auc is not None and probe_auc >= 0.7:
            return "routing / suppression failure", "M3 token, M5 FiLM, M4 trigger"
        if probe_auc is not None and probe_auc < 0.6:
            return "representation failure", "M1 history, M2 explicit, M3 CNN"
        if grad is not None and grad < 0.05:
            return "generalisation-rate mismatch", "M6 OGM-GE / FACTR curriculum"
        if saliency is not None and saliency < 0.1:
            return "signal-too-quiet (saliency concentrated but small)", "M5 FiLM (amplification)"
        return "collapse mechanism undetermined", "run confirmatory probes (saliency/CKA)"

    def report(self, batch: Dict[str, Tensor], dataset=None,
               batch_indices: Optional[List[int]] = None,
               tier1_auc: Optional[float] = None,
               ratio_threshold: float = 20.0, include_tier3: bool = True,
               n_probe_samples: int = 150,
               attention_figure: Optional[Path] = None) -> Dict:
        """Single entry point. tier1_auc is the Phase-1 value (Ch4), NOT recomputed."""
        rep: Dict = {"checkpoint_variant": {
            "policy_type": self.policy_type,
            "temporal_encoder": getattr(self.config, "proprio_temporal_encoder", "none"),
            "fusion_stage": getattr(self.config, "proprio_fusion_stage", "none"),
        }, "tier1_reference_auc": tier1_auc,
           "tier1_note": "Tier-1 is data-level (Phase-1 / Ch4), not recomputed per checkpoint",
           "n_probes": 7}
        # Tier 2
        rep["tier2"] = {}
        rep["tier2"]["relative_zero_out"] = self.relative_zero_out(batch, ratio_threshold)
        rep["tier2"]["per_phase_zero_out"] = self.per_phase_zero_out(
            batch, dataset=dataset, batch_indices=batch_indices)
        try:
            rep["tier2"]["attention_weight_analysis"] = self.attention_weight_analysis(
                batch, save_figure=attention_figure)
        except Exception as e:
            rep["tier2"]["attention_weight_analysis"] = {"error": str(e)}
        # Tier 3
        if include_tier3:
            rep["tier3"] = {}
            try:
                rep["tier3"]["embedding_linear_probe"] = self.embedding_linear_probe(
                    dataset=dataset, n_samples=n_probe_samples)
            except Exception as e:
                rep["tier3"]["embedding_linear_probe"] = {"error": str(e)}
            try:
                rep["tier3"]["gradient_flow_trajectory"] = self.gradient_flow_trajectory(batch)
            except Exception as e:
                rep["tier3"]["gradient_flow_trajectory"] = {"error": str(e)}
            if dataset is not None and self.parquet_path is not None:
                try:
                    rep["tier3"]["mutual_information"] = self.mutual_information(
                        parquet_path=self.parquet_path)
                except Exception as e:
                    rep["tier3"]["mutual_information"] = {"error": str(e)}
            try:
                rep["tier3"]["input_gradient_saliency"] = self.input_gradient_saliency(
                    dataset=dataset, n_samples=50)
            except Exception as e:
                rep["tier3"]["input_gradient_saliency"] = {"error": str(e)}
            try:
                rep["tier3"]["representation_similarity_cka"] = self.representation_similarity_cka(
                    dataset=dataset, n_samples=100)
            except Exception as e:
                rep["tier3"]["representation_similarity_cka"] = {"error": str(e)}
        # Tier 4
        t3_auc = ((rep.get("tier3", {}).get("embedding_linear_probe") or {})
                  .get("embedding_probe_auc"))
        rep["tier4"] = {}
        try:
            rep["tier4"]["probe_gap_upper_bound"] = self.probe_gap_upper_bound(tier1_auc, t3_auc)
        except Exception as e:
            rep["tier4"]["probe_gap_upper_bound"] = {"error": str(e)}
        # navigation
        mech, method = self._navigate(rep, tier1_auc)
        rep["verdict"] = "USED" if rep["tier2"]["relative_zero_out"].get("used") else "IGNORED"
        rep["mechanism"] = mech
        rep["recommended_next_method"] = method
        return rep
