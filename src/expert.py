"""ExpertRegistry — disjoint experts with shared live registry + conditional router.

Each expert is a binary mask over the 17K global weight pool (disjoint, labelled,
tracked via shared occupied/ownership). Training is isolated: only the expert's
weights are updated. Inference is conditional: router g(x) selects expert(s) per
input, still live (no task_id, single stream).

Shared registry: TaskRegistry's occupied/ownership already tracks which global
weights belong to which expert/task, visible to future tasks.
"""

from __future__ import annotations
import torch
import torch.nn as nn
from .registry import TaskRegistry, predict_required_weights

class ConditionalRouter(nn.Module):
    """Input-conditional expert router — live, no task_id.

    g(x) = softmax( MLP( global_feats(x) ) ),  global_feats = 9 dims from
    compute_global_features (loss, grad mean, etc.) + x stats. Small, trainable.
    """
    def __init__(self, num_experts: int = 8, hidden: int = 16, global_feat_dim: int = 9):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(global_feat_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, num_experts))
        self.num_experts = num_experts

    def forward(self, global_feats: torch.Tensor) -> torch.Tensor:
        # global_feats: (9,) or (B,9) -> logits
        if global_feats.dim() == 1:
            global_feats = global_feats.unsqueeze(0)
        logits = self.net(global_feats)
        return torch.softmax(logits, dim=-1)

class ExpertRegistry:
    """Disjoint expert pool — true isolation via separate model instances.

    Each expert is a separate TinyNumericTransformer (isolated weights, no
    shared forward). Shared live registry (TaskRegistry occupied) still tracks
    which expert owns which task for conditional routing, but isolation is via
    separate params, so no cross-talk via LayerNorm/attention.
    """
    def __init__(self, task_registry: TaskRegistry, max_experts: int = 8, model_fn=None, device=None):
        self.registry = task_registry
        self.max_experts = max_experts
        self.model_fn = model_fn
        self.device = device
        self.experts: list[torch.nn.Module] = []
        self.expert_task: list[str] = []
        self.expert_k: list[int] = []

    def num_experts(self) -> int:
        return len(self.experts)

    def allocate_expert(self, task_name: str, target_acc: float = 98.0) -> int:
        k_alloc = predict_required_weights(task_name, target_acc=target_acc)
        if len(self.experts) >= self.max_experts:
            return 0
        # create isolated expert model
        model = self.model_fn().to(self.device) if self.model_fn else None
        # also mark occupied in shared registry for tracking (counts as k_alloc)
        # we simulate footprint as ones for allocation
        if self.registry is not None:
            dummy = torch.ones(sum(g.size for g in self.registry.groups), device=self.device)
            try:
                self.registry._ensure_occupied(device=self.device)
                # directly mark k_alloc as occupied without needing footprint sorting
                occ = self.registry.get_occupied()
                free_idx = torch.where(~occ)[0]
                chosen = free_idx[:k_alloc] if free_idx.numel() >= k_alloc else torch.arange(min(k_alloc, occ.numel()), device=occ.device)
                occ[chosen] = True
                self.registry._allocations[task_name] = {"k_pred": k_alloc, "indices": chosen.cpu(), "task_id": len(self.experts)}
            except Exception:
                pass
        self.experts.append(model)
        self.expert_task.append(task_name)
        self.expert_k.append(k_alloc)
        return len(self.experts) - 1

    def get_expert(self, expert_id: int):
        if 0 <= expert_id < len(self.experts):
            return self.experts[expert_id]
        return None

    def expert_for_task(self, task_name: str):
        for i, t in enumerate(self.expert_task):
            if t == task_name:
                return self.experts[i], i
        return None, -1

    def summary(self) -> dict:
        occ = int(self.registry.get_occupied().sum().item()) if hasattr(self.registry, "_occupied") and self.registry._occupied is not None else 0
        return {
            "num_experts": len(self.experts),
            "per_expert": [{"task": t, "k": k} for t, k in zip(self.expert_task, self.expert_k)],
            "occupied": occ,
        }
