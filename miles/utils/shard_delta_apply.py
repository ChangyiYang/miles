"""Engine-side apply for the ``shard-delta`` transfer mode.

Registered with sglang via ``--custom-weight-loader``; sglang calls it with the
``(name, tensor)`` stream it just received over the broadcast group.

A bucket's sparse payload arrives as exactly three entries rather than a pair
per parameter:

* ``__delta_meta__|<name1>\\x1f<name2>...`` -- an int64 tensor of cumulative
  element counts, one per sparse parameter, with the ordered parameter names
  carried in the entry name itself (names travel over Ray as plain strings, so
  this costs no extra entry).
* ``__delta_idx__`` -- int32 positions into the concatenation of those
  parameters, flattened in the same order.
* ``__delta_val__`` -- the matching values.

Everything else is a whole tensor and falls through to sglang's own loader, so
the seed sync and any dense-fallback parameter still go through their normal
weight_loader. Packing the bucket this way is the point: ``dist.broadcast`` is
issued per entry, and a per-parameter pair made entry count -- not bytes -- the
dominant cost.
"""

from collections.abc import Iterable

import torch

_META_PREFIX = "__delta_meta__|"
_NAME_SEP = "\x1f"
_IDX_NAME = "__delta_idx__"
_VAL_NAME = "__delta_val__"


def _params(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return dict(model.named_parameters())


def apply_shard_delta(model: torch.nn.Module, named_tensors: Iterable[tuple[str, torch.Tensor]]):
    """Apply the packed bucket delta in place; yield dense leftovers for sglang."""
    meta_names: list[str] | None = None
    bounds: torch.Tensor | None = None
    idx: torch.Tensor | None = None
    val: torch.Tensor | None = None
    passthrough: list[tuple[str, torch.Tensor]] = []

    for name, tensor in named_tensors:
        if name.startswith(_META_PREFIX):
            meta_names = name[len(_META_PREFIX) :].split(_NAME_SEP)
            bounds = tensor
        elif name == _IDX_NAME:
            idx = tensor
        elif name == _VAL_NAME:
            val = tensor
        else:
            passthrough.append((name, tensor))

    if meta_names is None:
        if idx is not None or val is not None:
            raise RuntimeError("shard-delta: got idx/val without the meta entry")
        return passthrough
    if idx is None or val is None:
        raise RuntimeError("shard-delta: meta entry arrived without idx/val")
    if bounds.numel() != len(meta_names):
        raise RuntimeError(
            f"shard-delta: {len(meta_names)} names but {bounds.numel()} bounds"
        )

    params = _params(model)
    dev = idx.device
    idx = idx.to(torch.int64)
    # split the packed positions back per parameter: the bounds are the
    # cumulative element counts, so one searchsorted replaces a per-parameter
    # scan
    cut = torch.searchsorted(idx, bounds.to(dev), right=False).tolist()
    start_idx = 0
    start_off = 0
    for i, pname in enumerate(meta_names):
        end_idx = cut[i]
        target = params.get(pname)
        if target is None:
            raise KeyError(f"shard-delta: no live parameter named {pname!r}")
        local = idx[start_idx:end_idx] - start_off
        if local.numel():
            flat = target.data.reshape(-1)
            flat.index_copy_(
                0,
                local.to(flat.device),
                val[start_idx:end_idx].to(flat.dtype).to(flat.device),
            )
        start_idx = end_idx
        start_off = int(bounds[i])

    return passthrough
