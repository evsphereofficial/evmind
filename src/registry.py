"""Task Registry -- per-parameter importance footprints and region marks.

After learning each task, the registry captures:
  1. Per-parameter footprint: |grad| * |weight| at task completion
     (which parameters were critical for this task)
  2. Region-level marks: mean importance per fixed region
     (which regions of the model were "claimed" by this task)

During subsequent tasks, the governor receives these as additional
features so it can protect old-task territory.

Design: replaces H_MEM + RAIRAW with a simple, interpretable memory
that directly answers "which weights belong to which task?"
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass(frozen=True)
class Region:
    """A contiguous chunk of the flattened weight space."""
    region_id: int
    group_name: str
    start: int
    stop: int

    @property
    def size(self) -> int:
        return self.stop - self.start


def make_regions(groups: list, region_size: int = 1000) -> list[Region]:
    """Partition every module's flattened weights into contiguous regions.

    Regions never span modules. A 17,249-weight model with region_size=1000
    yields 18 regions (17 x 1000 + 249).
    """
    regions: list[Region] = []
    rid = 0
    for g in groups:
        n = g.size
        for start in range(0, n, region_size):
            stop = min(start + region_size, n)
            regions.append(Region(region_id=rid, group_name=g.name,
                                  start=start, stop=stop))
            rid += 1
    return regions


def compute_footprint(
    params_flat: torch.Tensor,
    grads_flat: torch.Tensor,
    method: str = "grad_x_weight",
) -> torch.Tensor:
    """Compute per-parameter importance footprint for one task.

    Args:
        params_flat: flattened parameter values at task completion
        grads_flat: flattened gradient magnitudes at task completion
        method: footprint computation method
            "grad_x_weight": |grad| * |weight|
            "grad_only": |grad|
            "weight_only": |weight|
            "binary": 1.0 where |grad| > threshold, 0 otherwise

    Returns:
        importance: (N,) non-negative importance per parameter
    """
    if method == "grad_x_weight":
        return grads_flat.abs() * params_flat.abs()
    elif method == "grad_only":
        return grads_flat.abs()
    elif method == "weight_only":
        return params_flat.abs()
    elif method == "binary":
        threshold = grads_flat.abs().mean() + grads_flat.abs().std()
        return (grads_flat.abs() > threshold).float()
    else:
        raise ValueError(f"unknown footprint method: {method!r}")


class ShadowRecognizer(nn.Module):
    """Tiny learnable registry: shadow -> protection.

    Shared per-weight MLP: (1,) -> hidden -> 1, ReLU, clamp >=0.
    Task-ID'd via per-task storage: each task's shadow is transformed
    independently, so the module learns how to sharpen/scale occupancy
    into a protection signal the governor can use.
    """
    def __init__(self, hidden: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1))
        # init to near-identity: small weights so early training matches raw shadow
        with torch.no_grad():
            self.net[-1].weight.mul_(0.1)
            self.net[-1].bias.zero_()

    def forward(self, shadow_norm: torch.Tensor) -> torch.Tensor:
        # shadow_norm: (N,) mean 1
        y = self.net(shadow_norm.unsqueeze(-1)).squeeze(-1)
        # residual + non-negative: shadow * (1 + delta) keeps identity prior
        y = shadow_norm + y
        return y.clamp_min(0)


class TaskRegistry:
    """Stores per-task weight importance footprints and region marks.

    After each task's training phase finishes, call capture() to record
    which parameters mattered. During subsequent tasks, call
    get_protection() to get the combined protection signal.

    Two levels of tracking:
      Per-parameter: (N,) importance vector per task -- fine-grained
      Region-level: (R,) importance per region -- coarse, stable

    The governor receives both as additional input features.
    Shadow tracker: if use_shadow, accumulate |grad| online per batch,
    then per-task footprint is shadow-derived (task-ID'd via separate
    footprints[t]). If learnable, a ShadowRecognizer MLP transforms shadow
    -> footprint and is trained jointly with the governor.
    """

    def __init__(
        self,
        groups: list,
        region_size: int = 1000,
        footprint_method: str = "grad_x_weight",
        normalize: bool = True,
        use_shadow: bool = False,
        shadow_alpha: float = 0.0,
        learnable: bool = False,
        recognizer_hidden: int = 16,
    ):
        self.groups = groups
        self.regions = make_regions(groups, region_size)
        self.footprint_method = footprint_method
        self.normalize = normalize
        self.use_shadow = use_shadow
        self.shadow_alpha = shadow_alpha  # 0 = sum, >0 = EMA mixing
        self.learnable = learnable

        self.footprints: list[torch.Tensor] = []
        self.region_marks: list[torch.Tensor] = []
        self.task_names: list[str] = []
        self.n = 0

        # Shadow tracker: non-trainable per-weight occupancy accumulator for
        # the CURRENT task. Updated online via accumulate() each backward,
        # consumed at capture(). Never participates in forward/loss.
        self._shadow: torch.Tensor | None = None
        self._shadow_device: torch.device | None = None

        # Learnable recognizer: shadow_norm -> footprint (per-weight, shared)
        self.recognizer: ShadowRecognizer | None = None
        if self.learnable:
            self.recognizer = ShadowRecognizer(hidden=recognizer_hidden)

    # -- shadow tracker API (non-trainable) -----------------------------------
    @torch.no_grad()
    def begin_task(self, device: torch.device | None = None) -> None:
        """Start a fresh occupancy accumulation for the next task."""
        total = sum(g.size for g in self.groups)
        dev = device or self._shadow_device or torch.device("cpu")
        self._shadow = torch.zeros(total, device=dev)
        self._shadow_device = dev

    @torch.no_grad()
    def accumulate(self, gradients: list[torch.Tensor]) -> None:
        """Online update: add |grad| for this batch to the current tracker.

        Call after every loss.backward() during the task. Detached, no graph.
        """
        if self._shadow is None:
            # lazy init on first use
            total = sum(g.size for g in self.groups)
            dev = gradients[0].device if gradients else torch.device("cpu")
            self._shadow = torch.zeros(total, device=dev)
            self._shadow_device = dev
        gflat = torch.cat([g.detach().flatten().abs() for g in gradients])
        if gflat.device != self._shadow.device:
            self._shadow = self._shadow.to(gflat.device)
            self._shadow_device = gflat.device
        if self.shadow_alpha and self.shadow_alpha > 0:
            # EMA: keeps scale bounded for long phases
            self._shadow.mul_(self.shadow_alpha).add_(gflat, alpha=1 - self.shadow_alpha)
        else:
            self._shadow.add_(gflat)

    def has_shadow(self) -> bool:
        return self._shadow is not None and bool((self._shadow != 0).any())

    def capture(
        self,
        model: nn.Module,
        gradients: list[torch.Tensor] | None = None,
        task_name: str = "",
        differentiable: bool = False,
    ) -> None:
        """Capture one task's footprint after its training phase.

        Args:
            model: model whose current params are the task's footprint
            gradients: per-group gradient tensors (from last backward).
                If None, uses |weight| as a fallback.
            task_name: optional label for logging
            differentiable: if True and learnable recognizer exists, keep graph
                for recognizer (meta-training). Else detached (live eval).

        Shadow mode: if use_shadow and accumulate() was called, the shadow
        accumulator is used as grads_flat (usage over the whole phase) instead
        of the single final gradient. Task-ID'd via separate footprints[t].
        Learnable mode transforms shadow_norm through recognizer.
        Falls back to grad_x_weight if shadow empty.
        """
        params_flat = torch.cat([
            p.detach().flatten() for p in model.parameters()
            if p.requires_grad
        ])

        if self.use_shadow and self.has_shadow():
            grads_flat = self._shadow.detach().clone()
            shadow_norm = grads_flat / (grads_flat.mean() + 1e-12)
            if self.learnable and self.recognizer is not None:
                if differentiable:
                    footprint = self.recognizer(shadow_norm)
                else:
                    with torch.no_grad():
                        footprint = self.recognizer(shadow_norm)
            else:
                footprint = shadow_norm  # pure usage; keep normalized
        elif gradients is not None:
            grads_flat = torch.cat([g.detach().flatten() for g in gradients])
            footprint = compute_footprint(
                params_flat, grads_flat, method=self.footprint_method)
        else:
            grads_flat = torch.ones_like(params_flat)
            footprint = compute_footprint(
                params_flat, grads_flat, method=self.footprint_method)

        if self.normalize:
            # recognizer already outputs mean ~1 residual; re-normalize for stability
            footprint = footprint / (footprint.mean() + 1e-12)
            if not differentiable:
                footprint = footprint.detach()
        else:
            if not differentiable:
                footprint = footprint.detach()

        self.footprints.append(footprint)
        self.task_names.append(task_name)
        self.n += 1

        with torch.no_grad():
            region_imp = self._compute_region_importance(footprint.detach())
        self.region_marks.append(region_imp)

        # reset shadow for next task
        if self._shadow is not None:
            self._shadow.zero_()

    def _compute_region_importance(self, footprint: torch.Tensor) -> torch.Tensor:
        """Aggregate per-parameter footprint into per-region importance.

        Regions store LOCAL start/stop within their module; the footprint
        is GLOBAL (all modules concatenated). Map via per-group offsets.
        """
        region_imp = torch.zeros(len(self.regions), device=footprint.device)
        # global offset per group name
        group_offset: dict[str, int] = {}
        off = 0
        for g in self.groups:
            group_offset[g.name] = off
            off += g.size
        for r in self.regions:
            base = group_offset.get(r.group_name, 0)
            gs, ge = base + r.start, base + r.stop
            # clamp defensively (footprint length must be total weights)
            gs = max(0, min(gs, footprint.numel()))
            ge = max(gs + 1, min(ge, footprint.numel()))
            region_imp[r.region_id] = footprint[gs:ge].mean()
        return region_imp

    def get_protection(
        self,
        up_to_task: int | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Get combined protection signal from all captured tasks.

        Args:
            up_to_task: only include tasks before this index (exclusive).
                None = include all tasks.

        Returns:
            param_protection: (N,) summed importance across old tasks
            region_protection: (R,) max importance across old tasks
        """
        if self.n == 0:
            return None, None

        idx = up_to_task if up_to_task is not None else self.n
        if idx == 0:
            return None, None

        fps = self.footprints[:idx]
        rms = self.region_marks[:idx]

        param_protection = torch.stack(fps).sum(dim=0)
        region_protection = torch.stack(rms).max(dim=0).values

        return param_protection, region_protection

    def get_task_footprint(self, task_idx: int) -> torch.Tensor | None:
        """Get a single task's footprint."""
        if task_idx < 0 or task_idx >= self.n:
            return None
        return self.footprints[task_idx]

    def get_task_region_marks(self, task_idx: int) -> torch.Tensor | None:
        """Get a single task's region marks."""
        if task_idx < 0 or task_idx >= self.n:
            return None
        return self.region_marks[task_idx]

    def parameters(self):
        """Yield learnable parameters (recognizer) if any."""
        if self.recognizer is not None:
            yield from self.recognizer.parameters()

    def to(self, device: torch.device) -> None:
        """Move all stored footprints to device."""
        self.footprints = [fp.to(device) for fp in self.footprints]
        self.region_marks = [rm.to(device) for rm in self.region_marks]
        if self._shadow is not None:
            self._shadow = self._shadow.to(device)
            self._shadow_device = device
        if self.recognizer is not None:
            self.recognizer.to(device)

    def summary(self) -> dict:
        """Summary statistics for logging."""
        if self.n == 0:
            return {"n_tasks": 0}
        all_fps = torch.stack(self.footprints)
        all_rms = torch.stack(self.region_marks)
        return {
            "n_tasks": self.n,
            "task_names": self.task_names.copy(),
            "footprint_mean": float(all_fps.mean()),
            "footprint_std": float(all_fps.std()),
            "footprint_max": float(all_fps.max()),
            "region_mean": float(all_rms.mean()),
            "region_std": float(all_rms.std()),
        }

    @property
    def total_params(self) -> int:
        if self.n == 0:
            return 0
        return self.footprints[0].numel()

    @property
    def num_regions(self) -> int:
        return len(self.regions)

    def region_weights_per_task(self) -> list[dict[str, float]]:
        """Per-task region ownership breakdown (for visualization)."""
        results = []
        for t in range(self.n):
            rm = self.region_marks[t]
            top_k = torch.topk(rm, min(5, len(rm)))
            ownership = {}
            for idx, val in zip(top_k.indices.tolist(), top_k.values.tolist()):
                r = self.regions[idx]
                ownership[f"region_{idx}({r.group_name})"] = round(val, 4)
            results.append(ownership)
        return results
