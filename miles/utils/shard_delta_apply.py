"""Engine-side apply for the ``shard-delta`` transfer mode.

Registered with sglang via ``--custom-weight-loader``; sglang calls it with the
``(name, tensor)`` stream it just received over the broadcast group. Names
carrying the ``::idx`` / ``::val`` suffixes are a sparse pair for one parameter:
write ``val`` into the live storage at the flat positions in ``idx``. Anything
else is a whole tensor and falls through to sglang's own loader.
"""

from collections.abc import Iterable

import torch


def _params(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return dict(model.named_parameters())


def apply_shard_delta(model: torch.nn.Module, named_tensors: Iterable[tuple[str, torch.Tensor]]):
    """Apply sparse deltas in place; yield the dense leftovers for sglang.

    Returning the unconsumed entries keeps the seed sync (all dense) and any
    dense-fallback tensor on sglang's normal loading path, so fused/quantized
    parameters still go through their own weight_loader.
    """
    params = _params(model)
    pending: dict[str, torch.Tensor] = {}
    passthrough: list[tuple[str, torch.Tensor]] = []

    for name, tensor in named_tensors:
        if name.endswith("::idx") or name.endswith("::val"):
            base, kind = name.rsplit("::", 1)
            other = pending.pop(base, None)
            if other is None:
                pending[base] = tensor if kind == "idx" else tensor
                pending[f"{base}::kind"] = kind  # remember which half arrived
                continue
            first_kind = pending.pop(f"{base}::kind")
            idx, val = (other, tensor) if first_kind == "idx" else (tensor, other)
            target = params.get(base)
            if target is None:
                raise KeyError(f"shard-delta: no live parameter named {base!r}")
            flat = target.data.reshape(-1)
            flat.index_copy_(0, idx.to(device=flat.device, dtype=torch.long), val.to(flat.dtype).to(flat.device))
        else:
            passthrough.append((name, tensor))

    if pending:
        raise RuntimeError(f"shard-delta: unpaired halves for {sorted(k for k in pending if '::' not in k)}")
    return passthrough
