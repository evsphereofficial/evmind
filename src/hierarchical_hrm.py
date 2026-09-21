"""Hierarchical HRM: main governor routes to per-expert tiny governors.

Main: ConditionalRouter g(x) -> expert_id (global, 8 experts)
Per-expert: HRMIntentGovernorTiny (hidden 12) per expert, gates its expert's weights.
Registry 17K pool, v2 law sizes, shared occupied.
Trained live: main via support-set, per-expert via masked gradients.
"""
import torch
import torch.nn as nn
from .hrm import HRMIntentGovernor, build_module_groups, compute_global_features

class PerExpertGovernor(nn.Module):
    def __init__(self, num_groups, hidden=12):
        super().__init__()
        # tiny per-expert gate: same as HRMIntentGovernor but smaller
        self.gov = HRMIntentGovernor(num_groups=num_groups, granularity="weight",
                                     hidden_dim=hidden, refine_steps=1, init_mask=0.5,
                                     per_weight_feat_dim=8, registry_feat_dim=0)
    def forward(self, *args, **kwargs):
        return self.gov.gate(*args, **kwargs)

class HierarchicalHRM(nn.Module):
    def __init__(self, num_experts=5, num_groups=29, hidden_main=16, hidden_per=12):
        super().__init__()
        self.router = nn.Sequential(
            nn.Linear(9, hidden_main), nn.ReLU(),
            nn.Linear(hidden_main, hidden_main), nn.ReLU(),
            nn.Linear(hidden_main, num_experts)
        )
        self.per_expert_govs = nn.ModuleList(
            [PerExpertGovernor(num_groups, hidden=hidden_per) for _ in range(num_experts)]
        )
        self.num_experts=num_experts
    def route(self, global_feats):
        logits=self.router(global_feats)
        return torch.softmax(logits, dim=-1)
