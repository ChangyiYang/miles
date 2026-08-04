"""Optimizer-inclusive per-slot checkpoints: the slot swap primitive.

One sidecar format serves slot swap-out/swap-in, periodic per-adapter saves,
and the future Tinker ``save_state``/load-with-optimizer. The payload must be
complete for numeric equivalence: bf16 slot weights, fp32
master params (NOT re-derivable from bf16 — reloading would drop the low
mantissa bits), Adam moments AND both step counters (per-param ``state["step"]``
and per-group ``group["step"]`` — FusedAdam tracks bias correction per group),
plus rank/alpha (non-persistent Bridge buffers, replayed via
``init_adapter_slot``). The LR scheduler is NOT serialized: its clock is the
optimizer step count, so ``install_slot_scheduler(..., resume_step)`` rebuilds
it exactly.

Stable parameter names strip the slot index (``...adapters.{slot}.`` ->
``...adapter.``), matching the exposed-slot export convention that
``load_adapter`` already consumes — state can move between slots and, later,
between training clients.

Distributed layout: every rank writes its own shard (tp/pp/ep-sharded weights,
rank-owned optimizer state under the LayerWise whole-param DP scatter); each
shard is written atomically (tmp + rename); rank 0 commits a manifest after a
barrier. A load first validates the manifest and topology, then mutates the
slot.
"""

import logging
import os
import re
from pathlib import Path

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)

FORMAT = "miles-multi-lora-slot-v2"
_SLOT_INDEX = re.compile(r"\.adapters\.(\d+)\.")


def stable_slot_param_name(name: str, slot: int) -> str:
    """``...adapters.{slot}.linear_in.weight`` -> ``...adapter.linear_in.weight``.

    Matches the exposed-slot naming that ``load_adapter`` consumes, so the
    weights section of a sidecar is directly loadable into ANY slot."""
    return _SLOT_INDEX.sub(lambda m: ".adapter." if int(m.group(1)) == slot else m.group(0), name)


def named_adapter_slot_parameters(model, slot: int):
    """Yield (stable_name, model_param) for one slot, in deterministic
    module-traversal order across chunks."""
    from megatron.bridge.peft.multi_lora_layers import MultiLoRALinear

    marker = f".adapters.{slot}."
    seen: set[int] = set()
    model_chunks = model if isinstance(model, (list, tuple)) else [model]
    for model_chunk in model_chunks:
        for module_name, module in model_chunk.named_modules():
            if not isinstance(module, MultiLoRALinear):
                continue
            for param_name, param in module.named_parameters(prefix=module_name):
                if marker in param_name and id(param) not in seen:
                    seen.add(id(param))
                    yield stable_slot_param_name(param_name, slot), param


def _slot_adam_index(optimizer, slot: int) -> dict[int, dict]:
    """id(main param or param) -> raw Adam state, across the slot's children."""
    from miles.backends.megatron_utils.multi_lora_optimizer import _slot_children

    index: dict[int, dict] = {}
    for child in _slot_children(optimizer, slot):
        raw = getattr(child, "optimizer", child)
        for param, state in raw.state.items():
            index[id(param)] = state
    return index


def _slot_group_steps(optimizer, slot: int) -> list:
    from miles.backends.megatron_utils.multi_lora_optimizer import _slot_children

    return [group.get("step", 0) for child in _slot_children(optimizer, slot) for group in child.param_groups]


def sidecar_dir(adapter) -> Path | None:
    save = adapter.config.save
    return Path(save) / "slot_state" if save is not None else None


def _shard_path(base: Path, rank: int) -> Path:
    return base / f"shard_rank{rank:05d}.pt"


def save_slot_state(args, model, optimizer, adapter, *, reason: str = "swap") -> Path | None:
    """Write-through sidecar: the durable record of the slot's full
    training state. Returns the manifest path (rank 0) or the shard path."""
    base = sidecar_dir(adapter)
    if base is None:
        # Adapters without a save dir are not swap-eligible; callers
        # must pin them instead. Reaching here is a caller bug for swaps.
        logger.warning(f"[multilora] ({adapter.name}) no save dir; slot state NOT persisted ({reason})")
        return None
    base.mkdir(parents=True, exist_ok=True)

    slot = adapter.slot
    adam_index = _slot_adam_index(optimizer, slot)
    weights: dict[str, torch.Tensor] = {}
    masters: dict[str, torch.Tensor] = {}
    adam_state: dict[str, dict] = {}
    for stable_name, param in named_adapter_slot_parameters(model, slot):
        weights[stable_name] = param.detach().cpu()
        main = getattr(param, "main_param", None)
        state = adam_index.get(id(main)) if main is not None else None
        if state is None:
            state = adam_index.get(id(param))
        if main is not None:
            masters[stable_name] = main.detach().cpu()
        if state:
            adam_state[stable_name] = {
                key: (value.detach().cpu() if torch.is_tensor(value) else value) for key, value in state.items()
            }

    rank = dist.get_rank() if dist.is_initialized() else 0
    payload = {
        "format": FORMAT,
        "name": adapter.name,
        "registration_id": adapter.registration_id,
        "rank_lora": adapter.config.rank,
        "alpha": adapter.config.alpha,
        "weights": weights,
        "master_params": masters,
        "adam_state": adam_state,
        "group_steps": _slot_group_steps(optimizer, slot),
        "clocks": {"optimizer_step": adapter.step, "serving_version": adapter.version},
        "topology": {
            "rank": rank,
            "world_size": dist.get_world_size() if dist.is_initialized() else 1,
        },
        "reason": reason,
    }
    shard = _shard_path(base, rank)
    tmp = shard.with_suffix(".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, shard)  # atomic per shard: a crash never leaves a torn file

    if dist.is_initialized():
        dist.barrier()
    manifest = base / "manifest.pt"
    if rank == 0:
        # Committed only after every rank's shard landed; the loader treats a
        # missing/older manifest as "no valid sidecar".
        tmp_manifest = manifest.with_suffix(".tmp")
        torch.save(
            {
                "format": FORMAT,
                "name": adapter.name,
                "registration_id": adapter.registration_id,
                "optimizer_step": adapter.step,
                "world_size": payload["topology"]["world_size"],
            },
            tmp_manifest,
        )
        os.replace(tmp_manifest, manifest)
    if dist.is_initialized():
        dist.barrier()
    logger.info(f"[multilora] ({adapter.name}) slot state saved at step {adapter.step} ({reason})")
    return manifest if rank == 0 else shard


def find_slot_state(adapter) -> Path | None:
    """The sidecar base dir, only if a committed manifest matches this
    registration's world topology."""
    base = sidecar_dir(adapter)
    if base is None or not (base / "manifest.pt").exists():
        return None
    manifest = torch.load(base / "manifest.pt", map_location="cpu", weights_only=True)
    if manifest.get("format") != FORMAT or manifest.get("name") != adapter.name:
        return None
    world = dist.get_world_size() if dist.is_initialized() else 1
    if manifest.get("world_size") != world:
        logger.warning(
            f"[multilora] ({adapter.name}) sidecar world_size {manifest.get('world_size')} != {world}; ignoring"
        )
        return None
    return base


def load_slot_state(args, model, optimizer, adapter) -> int:
    """Restore a slot from its sidecar. Ordering matters: weights ->
    rank/alpha replay -> slot-scoped master rebuild -> overwrite masters and
    Adam state (incl. both step counters) from the sidecar. Returns the
    restored optimizer step (0 = no sidecar; caller falls back to the
    weights-only checkpoint path with fresh Adam state)."""
    from megatron.bridge.peft.multi_lora_layers import init_adapter_slot, load_adapter

    from miles.backends.megatron_utils.multi_lora_optimizer import reload_adapter_slot_model_params

    base = find_slot_state(adapter)
    if base is None:
        return 0
    rank = dist.get_rank() if dist.is_initialized() else 0
    shard = _shard_path(base, rank)
    payload = torch.load(shard, map_location="cpu", weights_only=True)
    if payload.get("format") != FORMAT or payload.get("name") != adapter.name:
        raise ValueError(f"[multilora] ({adapter.name}) sidecar shard mismatch at {shard}")

    slot = adapter.slot
    loaded = load_adapter(model, slot, payload["weights"])
    assert loaded > 0, f"[multilora] ({adapter.name}) sidecar restored 0 weight tensors"
    init_adapter_slot(model, slot, rank=payload["rank_lora"], alpha=payload["alpha"])
    # Slot-scoped: a global reload would quantize every other resident slot's
    # fp32 master through bf16.
    reload_adapter_slot_model_params(optimizer, slot)

    adam_index = _slot_adam_index(optimizer, slot)
    masters = payload["master_params"]
    adam_state = payload["adam_state"]
    for stable_name, param in named_adapter_slot_parameters(model, slot):
        main = getattr(param, "main_param", None)
        if main is not None and stable_name in masters:
            main.data.copy_(masters[stable_name].to(device=main.device, dtype=main.dtype))
        state = adam_index.get(id(main)) if main is not None else adam_index.get(id(param))
        if state is not None and stable_name in adam_state:
            for key, value in adam_state[stable_name].items():
                if torch.is_tensor(value) and key in state and torch.is_tensor(state[key]):
                    state[key].copy_(value.to(device=state[key].device, dtype=state[key].dtype))
                else:
                    state[key] = value

    from miles.backends.megatron_utils.multi_lora_optimizer import _slot_children

    saved_group_steps = payload["group_steps"]
    groups = [g for child in _slot_children(optimizer, slot) for g in child.param_groups]
    for group, step in zip(groups, saved_group_steps, strict=False):
        if step:
            group["step"] = step

    restored_step = int(payload["clocks"]["optimizer_step"])
    logger.info(f"[multilora] ({adapter.name}) slot state restored at step {restored_step}")
    return restored_step


def swap_out(args, model, optimizer, adapter) -> None:
    """Persist the tenant's full state, then vacate the slot for the next
    tenant: optimizer state and retained grads must never leak across."""
    from megatron.bridge.peft.multi_lora_layers import clear_adapter_slot

    from miles.backends.megatron_utils.multi_lora_optimizer import zero_adapter_slot_grads
    from miles.backends.megatron_utils.multi_lora_scheduler import drop_slot_scheduler
    from miles.backends.megatron_utils.multi_lora_utils import zero_optimizer_state_for_adapter

    save_slot_state(args, model, optimizer, adapter, reason="swap")
    clear_adapter_slot(model, adapter.slot)
    zero_optimizer_state_for_adapter(optimizer, model, adapter.slot)
    zero_adapter_slot_grads(model, adapter.slot)
    drop_slot_scheduler(optimizer, adapter.slot)


def swap_in(args, model, optimizer, adapter) -> int:
    """Bind a tenant into a (vacated) slot: sidecar restore when one exists,
    otherwise the weights-only registration path. Installs the scheduler at
    the restored step (the scheduler clock IS the optimizer step count)."""
    from miles.backends.megatron_utils.multi_lora_scheduler import install_slot_scheduler

    restored_step = load_slot_state(args, model, optimizer, adapter)
    if restored_step == 0:
        from miles.backends.megatron_utils.multi_lora_utils import _register_adapter

        restored_step = _register_adapter(adapter, model)
        from miles.backends.megatron_utils.multi_lora_optimizer import reload_adapter_slot_model_params

        reload_adapter_slot_model_params(optimizer, adapter.slot)
    install_slot_scheduler(args, optimizer, adapter, restored_step)
    return restored_step
