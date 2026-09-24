"""EvAGI LLM components: neuron register, experts, governors, support router.

Maps the v2 capacity law onto GPT-Neo FFN neurons (c_fc rows / c_proj cols):
  Register  — per-layer occupancy of the 12288-neuron FFN pool
  Expert    — disjoint neuron set sized by predict_required_weights()
  Governor  — tiny per-expert plasticity gates (HRM-style) over owned neurons
  Router    — live support-set g(x): pick expert by min loss on 32 shots
              (no task_id at test)

Hard isolation: during task t forward, only expert t's neurons are active
(non-owned FFN channels zeroed via hooks); only those receive gradients.
Everything else (attention, embeddings, LN, non-owned FFN) stays frozen.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from .registry import predict_required_weights


# ---------------------------------------------------------------------------
# Neuron <-> law conversion (fix 1)
# ---------------------------------------------------------------------------
# Law k is calibrated in scalar-weight units on the 17K 2D model.
# One GPT-Neo FFN neuron ≈ (hidden + 1 + hidden) = 2*hidden+1 scalar params.
# Using the raw law (k/1536) yields <1 neuron/task — unusable.
# neuron_div maps law-k -> neurons. Default 8 => ~700 neurons total (5.7% of
# 12288), enough for yes/no skills while staying a small expert budget.
DEFAULT_NEURON_DIV = 8


def k_to_neuron_counts(
    k_alloc: int,
    n_layers: int,
    neuron_div: int = DEFAULT_NEURON_DIV,
    min_total: int = 8,
) -> list[int]:
    """Distribute law-k allocation across layers as neuron counts."""
    n_total = max(min_total, int(math.ceil(k_alloc / max(1, neuron_div))))
    base = n_total // n_layers
    rem = n_total % n_layers
    return [base + (1 if i < rem else 0) for i in range(n_layers)]


def predict_neurons(task_name: str, n_layers: int, neuron_div: int = DEFAULT_NEURON_DIV,
                    target_acc: float = 98.0) -> tuple[int, list[int]]:
    """v2 law -> (k_alloc, per-layer neuron counts)."""
    k = predict_required_weights(task_name, target_acc=target_acc)
    counts = k_to_neuron_counts(k, n_layers, neuron_div=neuron_div)
    return k, counts


# ---------------------------------------------------------------------------
# Register: per-layer occupancy over the FFN neuron pool
# ---------------------------------------------------------------------------
class NeuronRegister:
    """Tracks which FFN neurons are owned by which expert/task.

    Extended with per-neuron protection scores captured after each task
    (|grad|*|weight| footprint on owned neurons). The governor reads these
    via get_protection_feats() so it can see what the network already
    learned and defend old-task territory.
    """

    def __init__(self, inters: list[int], device: torch.device):
        self.inters = list(inters)
        self.n_layers = len(inters)
        self.device = device
        self.occupied = [
            torch.zeros(inter, dtype=torch.bool, device=device) for inter in inters
        ]
        self.ownership = [
            torch.full((inter,), -1, dtype=torch.long, device=device)
            for inter in inters
        ]
        self.allocations: list[dict] = []
        # per-neuron protection from COMPLETED tasks (max footprint seen)
        self.protection = [
            torch.zeros(inter, device=device) for inter in inters
        ]
        self.n_captured: int = 0

    def allocate(self, task_name: str, counts: list[int], expert_id: int) -> list[torch.Tensor]:
        """Carve out free neurons per layer; returns per-layer Bool masks."""
        masks: list[torch.Tensor] = []
        for li, take in enumerate(counts):
            occ = self.occupied[li]
            free = torch.where(~occ)[0]
            t = min(int(take), free.numel())
            chosen = free[:t]
            mask = torch.zeros(self.inters[li], dtype=torch.bool, device=self.device)
            mask[chosen] = True
            occ[chosen] = True
            self.ownership[li][chosen] = expert_id
            masks.append(mask)
        n_owned = int(sum(m.sum().item() for m in masks))
        self.allocations.append(
            {"task": task_name, "expert_id": expert_id, "counts": [int(c) for c in counts],
             "owned": n_owned}
        )
        return masks

    def deallocate(self, masks: list[torch.Tensor], expert_id: int) -> None:
        """Free neurons back (adaptive-retry: failed allocation attempt)."""
        for li, m in enumerate(masks):
            self.occupied[li][m] = False
            self.ownership[li][m & (self.ownership[li] == expert_id)] = -1
        self.allocations = [
            a for a in self.allocations if a.get("expert_id") != expert_id
        ]

    def capture_protection(
        self,
        masks: list[torch.Tensor],
        weights: list[torch.Tensor],
        grads: list[torch.Tensor | None],
    ) -> None:
        """After learning a task, record |grad|*|weight| footprint on its neurons.

        weights/grads: per-layer c_fc.weight (inter, hidden). Protection is
        per-neuron (mean over hidden). Old-task max is preserved (never
        weakened by later captures on different neurons; same-neuron
        recapture takes max so protection is monotonic).
        """
        for li, m in enumerate(masks):
            if grads[li] is None:
                continue
            fp = (grads[li].abs() * weights[li].abs()).mean(dim=-1)  # (inter,)
            # only owned neurons carry this task's protection
            fp = fp * m.to(fp.dtype)
            self.protection[li] = torch.maximum(self.protection[li], fp.detach())
        self.n_captured += 1

    def get_protection_feats(self, mask: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """(N, 3) registry-derived features for owned neurons under mask.

        Features:
          0. protection_norm: log1p(prot) scaled to [0,1] vs layer max
          1. prior_owned: 1 if this neuron carries protection from an
             already-completed task (0 for freshly allocated neurons)
          2. layer_occupancy: fraction of this layer already occupied
             (context pressure — many old neurons => be conservative)
        """
        idx = torch.where(mask)[0]
        if idx.numel() == 0:
            return torch.zeros(0, 3, device=mask.device)
        prot = self.protection[layer_idx][idx]
        p_log = torch.log1p(prot)
        p_max = torch.log1p(self.protection[layer_idx]).max().detach()
        p_norm = p_log / p_max.clamp(min=1e-6)
        prior = (prot > 0).float()
        occ = self.occupied[layer_idx].float().mean().expand_as(p_norm)
        return torch.stack([p_norm, prior, occ], dim=-1)

    def total_occupied(self) -> int:
        return int(sum(o.sum().item() for o in self.occupied))

    def total_pool(self) -> int:
        return int(sum(self.inters))

    def summary(self) -> dict:
        return {
            "pool": self.total_pool(),
            "occupied": self.total_occupied(),
            "allocations": self.allocations,
            "captured_tasks": self.n_captured,
        }


# ---------------------------------------------------------------------------
# Hard-isolation forward hooks: zero non-owned FFN channels
# ---------------------------------------------------------------------------
class ExpertMaskContext:
    """Context manager zeroing non-active FFN channels (binary) and/or
    scaling active channels by differentiable governor gates.

    masks_per_expert[eid][li] -> Bool (inter,)
    gates: optional per-layer (inter,) float gates for the active expert only
           (non-owned entries must already be 0). When set, gates replace the
     binary mask so task loss can train TinyPerExpertGovernor.
    """

    def __init__(
        self,
        layers: nn.ModuleList,
        masks_per_expert: list[list[torch.Tensor]],
        expert_ids: list[int],
        gates: list[torch.Tensor] | None = None,
    ):
        self.layers = layers
        self.masks_per_expert = masks_per_expert
        self.expert_ids = expert_ids
        self.gates = gates
        self._handles: list[torch.utils.hooks.RemovableHandle] = []

    def _combined(self, layer_idx: int) -> torch.Tensor:
        inter = self.masks_per_expert[0][layer_idx].numel()
        acc = torch.zeros(
            inter,
            dtype=torch.bool,
            device=self.masks_per_expert[0][layer_idx].device,
        )
        for eid in self.expert_ids:
            acc |= self.masks_per_expert[eid][layer_idx]
        return acc

    def __enter__(self):
        for li, layer in enumerate(self.layers):
            binary = self._combined(li)
            gate = self.gates[li] if self.gates is not None else None

            def make_fc_hook(m_mask, g):
                def hook(module, inputs, output):
                    out = output * 1.0  # keep graph if g requires grad
                    if g is not None:
                        out = out * g.to(out.dtype)
                    out = out.masked_fill(~m_mask, 0)
                    return out
                return hook

            self._handles.append(
                layer.mlp.c_fc.register_forward_hook(make_fc_hook(binary, gate))
            )
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles.clear()
        return False


def hard_mask_grads(layers: nn.ModuleList, masks: list[torch.Tensor]) -> None:
    """Zero gradients outside the active expert's neurons (fix 3)."""
    for li, layer in enumerate(layers):
        m = masks[li]
        fc_w = layer.mlp.c_fc.weight
        if fc_w.grad is not None:
            # rows = neurons
            grad = fc_w.grad
            grad[~m, :] = 0
        if layer.mlp.c_fc.bias is not None and layer.mlp.c_fc.bias.grad is not None:
            layer.mlp.c_fc.bias.grad[~m] = 0
        cp_w = layer.mlp.c_proj.weight
        if cp_w.grad is not None:
            # cols = neurons
            cp_w.grad[:, ~m] = 0


def set_ffn_trainable(layers: nn.ModuleList, masks: list[torch.Tensor] | None) -> None:
    """Freeze entire model except FFN; if masks given, only owned rows trainable
    via requires_grad on full tensors + hard grad mask after backward."""
    for layer in layers:
        layer.mlp.c_fc.weight.requires_grad_(True)
        if layer.mlp.c_fc.bias is not None:
            layer.mlp.c_fc.bias.requires_grad_(True)
        layer.mlp.c_proj.weight.requires_grad_(True)


def freeze_all_but_ffn(model: nn.Module, layers: nn.ModuleList) -> None:
    for p in model.parameters():
        p.requires_grad_(False)
    for layer in layers:
        layer.mlp.c_fc.weight.requires_grad_(True)
        if layer.mlp.c_fc.bias is not None:
            layer.mlp.c_fc.bias.requires_grad_(True)
        layer.mlp.c_proj.weight.requires_grad_(True)


# ---------------------------------------------------------------------------
# Tiny per-expert HRM governor (plasticity gates over owned neurons)
# ---------------------------------------------------------------------------
class TinyPerExpertGovernor(nn.Module):
    """Tiny gate net: per-neuron features -> sigmoid plasticity in (0,1].

    Amortized like HRMIntentGovernor: shared (8+3)->hidden->1 MLP, so the
    governor stays ~100-400 params while controlling all of its expert's
    neurons. Init bias high so gates start ~open (1.0); light L1 encourages
    selectivity.

    Input features (11 total):
      0-7: neuron-local (weight/grad stats, position, layer, owned)
      8-10: registry-derived (protection_norm, prior_owned, layer_occupancy)
            — lets the governor SEE what the network already learned and
            keep gates closed on old-task territory when appropriate.
    """

    INPUT_DIM = 11
    REGISTRY_DIM = 3

    def __init__(self, hidden: int = 12):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(self.INPUT_DIM, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        # start near open
        nn.init.zeros_(self.net[-1].weight)
        nn.init.constant_(self.net[-1].bias, 4.0)  # sigmoid(4)≈0.982

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        """feats (N, 11) -> gates (N,) in (0,1)."""
        if feats.size(-1) != self.INPUT_DIM:
            # backwards compat: pad missing registry features with zeros
            pad = self.INPUT_DIM - feats.size(-1)
            if pad > 0:
                feats = torch.cat(
                    [feats, feats.new_zeros(feats.size(0), pad)], dim=-1
                )
            else:
                feats = feats[:, : self.INPUT_DIM]
        return torch.sigmoid(self.net(feats).squeeze(-1))


def neuron_features(
    weights: torch.Tensor,
    grads: torch.Tensor | None,
    mask: torch.Tensor,
    layer_idx: int,
    n_layers: int,
    registry_feats: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build (N, 11) features for governor over owned neurons.

    weights/grads: c_fc.weight (inter, hidden) — row i is neuron i.
    registry_feats: optional (N, 3) from NeuronRegister.get_protection_feats.
    Returns (feats, idx) where idx are owned neuron indices.
    """
    idx = torch.where(mask)[0]
    n_feat = TinyPerExpertGovernor.INPUT_DIM
    if idx.numel() == 0:
        return torch.zeros(0, n_feat, device=mask.device), idx
    w_rows = weights[idx]  # (N, hidden)
    if grads is not None:
        g_rows = grads[idx]
    else:
        g_rows = torch.zeros_like(w_rows)
    n = mask.numel()
    pos = idx.float() / max(n - 1, 1)
    layer_f = torch.full_like(pos, layer_idx / max(n_layers - 1, 1))
    owned = torch.ones_like(pos)
    local = torch.stack(
        [
            torch.log1p(w_rows.abs().mean(dim=-1)),
            torch.log1p(g_rows.abs().mean(dim=-1)),
            pos,
            layer_f,
            owned,
            torch.tanh(w_rows.mean(dim=-1)),
            torch.tanh(g_rows.mean(dim=-1)),
            w_rows.std(dim=-1),
        ],
        dim=-1,
    )
    if registry_feats is not None and registry_feats.size(0) == idx.numel():
        feats = torch.cat([local, registry_feats.to(local.dtype)], dim=-1)
    else:
        feats = torch.cat(
            [local, local.new_zeros(local.size(0), TinyPerExpertGovernor.REGISTRY_DIM)],
            dim=-1,
        )
    return feats, idx


# ---------------------------------------------------------------------------
# Live support-set router (fix 2): no task_id
# ---------------------------------------------------------------------------
@torch.no_grad()
def support_pick_expert(
    model: nn.Module,
    layers: nn.ModuleList,
    masks_per_expert: list[list[torch.Tensor]],
    tokenizer,
    pairs: list[tuple[str, int]],
    device,
    max_length: int,
) -> tuple[int, list[float]]:
    """Pick expert with min yes/no NLL on the support set under each mask."""
    from .llm_tasks import tokenize_pairs, yes_no_token_ids

    yes_id, no_id = yes_no_token_ids(tokenizer)
    enc = tokenize_pairs(tokenizer, pairs, max_length=max_length)
    input_ids = enc["input_ids"].to(device)
    attention = enc["attention_mask"].to(device)
    labels = enc["labels"]
    first_lab = (labels != -100).float().argmax(dim=1).to(device)
    b_idx = torch.arange(input_ids.size(0), device=device)
    pred_pos = (first_lab - 1).clamp(min=0)
    true = torch.tensor([lab for _, lab in pairs], device=device)

    losses: list[float] = []
    model.eval()
    for eid in range(len(masks_per_expert)):
        with ExpertMaskContext(layers, masks_per_expert, [eid]):
            out = model(input_ids=input_ids, attention_mask=attention)
        logits = out.logits[b_idx, pred_pos, :]  # (B, V)
        # NLL of the true yes/no token
        target = torch.where(true.bool(), yes_id, no_id)
        nll = torch.nn.functional.cross_entropy(logits, target)
        losses.append(float(nll.item()))
    best = int(min(range(len(losses)), key=lambda i: losses[i]))
    return best, losses


@torch.no_grad()
def support_pick_expert_qa(
    model: nn.Module,
    layers: nn.ModuleList,
    masks_per_expert: list[list[torch.Tensor]],
    tokenizer,
    qa_pairs: list[tuple[str, str]],
    device,
    max_length: int,
) -> tuple[int, list[float]]:
    """Pick expert with min free-text answer NLL (for mid-chat fact recall)."""
    from .llm_tasks import tokenize_qa

    enc = tokenize_qa(tokenizer, qa_pairs, max_length=max_length)
    input_ids = enc["input_ids"].to(device)
    attention = enc["attention_mask"].to(device)
    labels = enc["labels"].to(device)
    shift_logits_mask = labels[:, 1:].contiguous() != -100

    losses: list[float] = []
    model.eval()
    for eid in range(len(masks_per_expert)):
        with ExpertMaskContext(layers, masks_per_expert, [eid]):
            out = model(input_ids=input_ids, attention_mask=attention)
        shift_logits = out.logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        vocab = shift_logits.size(-1)
        flat_logits = shift_logits.view(-1, vocab)
        flat_lab = shift_labels.view(-1).clamp(min=0)
        nll = torch.nn.functional.cross_entropy(flat_logits, flat_lab, ignore_index=-100)
        losses.append(float(nll.item()))
    best = int(min(range(len(losses)), key=lambda i: losses[i]))
    return best, losses


def train_fact_expert(
    model,
    tokenizer,
    layers,
    qa_pairs: list[tuple[str, str]],
    masks: list[torch.Tensor],
    governor: TinyPerExpertGovernor,
    router: HRMRouter,
    router_expert_id: int,
    cfg: dict,
    device,
    max_length: int,
    epochs: int | None = None,
    lr: float | None = None,
    register: "NeuronRegister | None" = None,
) -> dict:
    """Mid-chat fact install: full EvAGI stack on free-text QA.

    Equation V4 chose `masks` via the caller (Register.allocate).
    Here: TinyPerExpertGovernor gates (registry-aware) + hard grad mask +
    HRM router step. If `register` is given, governor features include
    protection/prior_owned/layer_occupancy from completed tasks.
    """
    from torch.utils.data import DataLoader, TensorDataset

    from .llm_tasks import tokenize_qa

    enc = tokenize_qa(tokenizer, qa_pairs, max_length=max_length)
    ds = TensorDataset(enc["input_ids"], enc["attention_mask"], enc["labels"])
    bs = int(cfg.get("batch_size", 8))
    loader = DataLoader(ds, batch_size=min(bs, len(ds)), shuffle=True)

    ffn_params = [p for p in model.parameters() if p.requires_grad]
    gov_params = list(governor.parameters())
    router_params = list(router.parameters())
    _lr = float(lr if lr is not None else cfg.get("lr", 1e-4))
    opt = torch.optim.AdamW(ffn_params + gov_params, lr=_lr,
                            weight_decay=float(cfg.get("weight_decay", 0.01)))
    router_opt = torch.optim.AdamW(router_params, lr=_lr)
    _epochs = int(epochs if epochs is not None else cfg.get("epochs_per_task", 3))
    gov_l1 = float(cfg.get("gov_l1", 1e-4))

    model.train()
    governor.train()
    router.train()

    history = []
    for epoch in range(_epochs):
        running = 0.0
        r_running = 0.0
        n = 0
        for input_ids, attention_mask, labels in loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            gates: list[torch.Tensor] = []
            gate_means: list[torch.Tensor] = []
            for li, layer in enumerate(layers):
                m = masks[li]
                w = layer.mlp.c_fc.weight
                g = w.grad if w.grad is not None else torch.zeros_like(w)
                reg_feats = None
                if register is not None:
                    reg_feats = register.get_protection_feats(m, li)
                feats, idx = neuron_features(
                    w.detach(), g.detach(), m, li, len(layers),
                    registry_feats=reg_feats,
                )
                if idx.numel() == 0:
                    gates.append(torch.zeros(w.shape[0], device=device))
                    continue
                go = governor(feats)
                full = torch.zeros(w.shape[0], device=device)
                full = full.index_add(0, idx, go)
                gates.append(full)
                gate_means.append(go.mean())

            opt.zero_grad(set_to_none=True)
            router_opt.zero_grad(set_to_none=True)

            with ExpertMaskContext(layers, [masks], [0], gates=gates):
                out = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = out.logits
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = labels[:, 1:].contiguous()
                vocab = shift_logits.size(-1)
                loss = torch.nn.functional.cross_entropy(
                    shift_logits.view(-1, vocab),
                    shift_labels.view(-1).clamp(min=0),
                    ignore_index=-100,
                )

            loss.backward(retain_graph=True)
            hard_mask_grads(layers, masks)

            if gate_means:
                gov_l1_loss = torch.stack(gate_means).mean() * gov_l1
                gov_l1_loss.backward()
            else:
                gov_l1_loss = torch.tensor(0.0, device=device)

            # HRM router: this live batch belongs to the new fact expert
            with torch.no_grad():
                p = torch.sigmoid(out.logits[:, -1, 0])
                feats_r = torch.stack(
                    [
                        input_ids.float().mean() / 50256.0,
                        input_ids.float().std() / 50256.0,
                        labels.float().mean() / max(1.0, float(labels.max())),
                        torch.log1p(loss.detach()),
                        p.mean(),
                        p.std(),
                        (p > 0.5).float().mean(),
                        input_ids.new_tensor(float(input_ids.size(0))) / 128.0,
                        input_ids.new_tensor(float(len(layers))) / 8.0,
                    ]
                )
            logits_r = router(feats_r.unsqueeze(0))
            # expand-safe: if router is smaller, skip (caller should rebuild)
            if logits_r.size(-1) > router_expert_id:
                rloss = torch.nn.functional.cross_entropy(
                    logits_r, torch.tensor([router_expert_id], device=device)
                )
                rloss.backward()
                r_running += float(rloss.item())
            else:
                rloss = None

            torch.nn.utils.clip_grad_norm_(ffn_params + gov_params, 1.0)
            opt.step()
            router_opt.step()

            running += float(loss.item())
            n += 1
        rec = {
            "epoch": epoch + 1,
            "loss": running / max(n, 1),
            "router_loss": r_running / max(n, 1),
        }
        history.append(rec)
        print(f"    live epoch {epoch + 1}/{_epochs} loss={rec['loss']:.4f} "
              f"router_loss={rec['router_loss']:.3f}")

    # record footprint so FUTURE governors see this task's territory
    if register is not None:
        with torch.no_grad():
            w_list, g_list, m_list = [], [], []
            for li, layer in enumerate(layers):
                w = layer.mlp.c_fc.weight
                w_list.append(w.detach())
                g_list.append(w.grad.detach() if w.grad is not None else None)
                m_list.append(masks[li])
            register.capture_protection(m_list, w_list, g_list)

    for p in governor.parameters():
        p.requires_grad_(False)
    model.eval()
    governor.eval()
    router.eval()
    return {"history": history, "final_loss": history[-1]["loss"] if history else None}


def expand_router(router: HRMRouter, new_num_experts: int, device) -> HRMRouter:
    """Grow HRMRouter output head for a newly installed live expert."""
    old_sd = router.state_dict()
    # net: Linear-ReLU-Linear-ReLU-Linear
    feat_dim = router.net[0].in_features
    hidden = router.net[0].out_features
    new = HRMRouter(num_experts=new_num_experts, hidden=hidden, feat_dim=feat_dim).to(device)
    new_sd = new.state_dict()
    for k, v in old_sd.items():
        if k.endswith("net.4.weight") or k.endswith("net.4.bias"):
            old_rows = v.shape[0]
            new_sd[k][:old_rows] = v
        else:
            new_sd[k] = v
    new.load_state_dict(new_sd)
    return new


class HRMRouter(nn.Module):
    """Main HRM router: global batch features -> expert logits (9 -> h -> E)."""

    def __init__(self, num_experts: int, hidden: int = 16, feat_dim: int = 9):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_experts),
        )
        self.num_experts = num_experts

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        if feats.dim() == 1:
            feats = feats.unsqueeze(0)
        return self.net(feats)


def batch_global_feats(x: torch.Tensor, y: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
    """9 global features for the main router (input-conditional, no task_id)."""
    with torch.no_grad():
        p = torch.sigmoid(logits.view(-1))
        loss_ = torch.nn.functional.binary_cross_entropy(
            p.clamp(1e-6, 1 - 1e-6), y.view(-1)
        )
        feats = torch.tensor(
            [
                x.mean().item(),
                x.std().item(),
                x.abs().mean().item(),
                y.float().mean().item(),
                loss_.item(),
                p.mean().item(),
                p.std().item(),
                (p > 0.5).float().mean().item(),
                float(x.shape[0]),
            ],
            device=x.device,
        )
    return feats
