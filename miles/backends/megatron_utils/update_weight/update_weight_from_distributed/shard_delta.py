"""Delta weight sync over the existing NCCL broadcast group.

Same wire as ``broadcast`` (one NCCL group per PP stage, metadata over Ray) but
each sync ships only the elements that changed since the previous one. The
engine writes those positions straight into the live parameter storage, so a
sync costs O(changed) instead of O(model).

Where the diff happens is the whole design problem here, and the first version
got it wrong in a way worth recording. ``_update_weight_implementation`` is
handed tensors that are *already TP/EP all-gathered and converted to HF layout*,
and ``mixin.py`` runs it on the source rank only. Diffing there means one rank
holding a full-model host snapshot and pushing it across PCIe twice per sync --
~120 GB of round trip for a 30B model. Measured: 55s per sync against plain
broadcast's 8.14s, i.e. shipping 1% of the bytes cost 7x more time than shipping
all of them.

So the diff happens one layer down, as verl's ``delta_sync/sharded.py`` does it:
each rank compares its *local shard* against a GPU snapshot of that shard (1/16
of the model here, no host traffic at all). What crosses the gather is then only
a uint8 changed-mask -- half the bytes of a value gather, and a fifteenth of
what the naive version moved.

verl can stop there because FSDP parameters are already in HF layout, so a local
index maps to a global one by adding an offset. Megatron is not: between the
shard and HF index space sit ``all_gather_param`` (concat along ``partition_dim``
with ``partition_stride``) and ``convert_to_hf`` (rename, ``chunk(2, dim=0)`` for
the GLU pair, QKV split). Rather than reimplement that permutation and risk
silently writing weights to the wrong offsets, the mask rides the identical path
as the values -- so it arrives already in HF index space, correct by
construction. ``quantize_params`` no-ops without a quantization config and
``remove_padding`` is a slice, so the uint8 mask survives the trip unchanged.
"""

import logging
import time
from typing import Optional

import ray
import torch
from tqdm import tqdm

from miles.backends.training_utils.parallel import get_parallel_state

from ...megatron_to_hf import convert_to_hf
from ..common import all_gather_param
from .broadcast import (
    UpdateWeightFromDistributed,
    update_weights_from_distributed,
)

logger = logging.getLogger(__name__)

# A tensor whose changed fraction exceeds this is cheaper to send whole: the
# sparse encoding costs 4 bytes of index on top of each 2-byte value, so a
# sparse element is 3x a dense one and the break-even sits at 1/3.
DENSE_FALLBACK_RATIO = 0.25

# Wire contract for the packed bucket payload, mirrored in
# miles/utils/shard_delta_apply.py.
_META_PREFIX = "__delta_meta__|"
_NAME_SEP = "\x1f"
_IDX_NAME = "__delta_idx__"
_VAL_NAME = "__delta_val__"


# all_gather_param() reads Megatron's sharding metadata straight off the
# parameter object, so a bare .clone() of the data trips
# "does not have tensor_model_parallel attribute". The mask has to carry the
# same marks to travel the same path.
_SHARD_ATTRS = (
    "tensor_model_parallel",
    "partition_dim",
    "partition_stride",
    "parallel_mode",
    "allreduce",
    "sequence_parallel",
)


def _mark_like(param: torch.Tensor, other: torch.Tensor) -> torch.Tensor:
    for attr in _SHARD_ATTRS:
        if hasattr(param, attr):
            setattr(other, attr, getattr(param, attr))
    return other


def _clone_as_shard(param: torch.Tensor) -> torch.Tensor:
    return _mark_like(param, param.data.detach().clone())


def _split_changed(
    tensors: list[torch.Tensor], masks: list[Optional[torch.Tensor]]
) -> Optional[list[Optional[torch.Tensor]]]:
    """Return the changed-position vector per tensor, or None if unseeded.

    ``nonzero`` runs once over one concatenated mask rather than once per
    tensor: a bucket holds ~336 tensors, and the kernel launches plus the
    implicit device sync inside ``nonzero`` dominate the arithmetic.
    """
    # a partially-masked bucket would mean the shard layout changed under us;
    # fall back to dense rather than guess
    if any(m is None for m in masks):
        return None

    flat_mask = torch.cat([m.reshape(-1) for m in masks])
    changed = flat_mask.nonzero(as_tuple=True)[0]

    bounds: list[int] = []
    running = 0
    for t in tensors:
        running += t.numel()
        bounds.append(running)
    # keep the offsets as python ints: reading them back off a cuda tensor would
    # cost one device sync per tensor, which is what this batching avoids
    cut = torch.searchsorted(
        changed, torch.tensor(bounds, device=changed.device), right=False
    ).tolist()
    out: list[Optional[torch.Tensor]] = []
    start_idx = 0
    start_off = 0
    for i, cut_i in enumerate(cut):
        out.append(changed[start_idx:cut_i] - start_off)
        start_idx = cut_i
        start_off = bounds[i]
    return out


class UpdateWeightFromShardDelta(UpdateWeightFromDistributed):
    """Broadcast-based transport that ships per-tensor deltas."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # local-shard snapshot, on GPU: name -> what this rank last published
        self._shard_snaps: dict[str, torch.Tensor] = {}
        # HF-space changed-mask for the bucket being built; cleared every
        # flush so at most one bucket is resident
        self._mask_hf: dict[str, torch.Tensor] = {}
        # TP-gathered masks accumulated across update units, drained by
        # whichever path consumes it (the expert path buckets several units
        # before converting, so this cannot be per-unit state)
        self._pending_mask: list[tuple[str, torch.Tensor]] = []

    # -- mask plumbing ------------------------------------------------------

    def _all_gather_update_unit(self, update_unit):
        """Gather the live unit, and the changed-mask through the identical path."""
        mask_unit = []
        for name, param in update_unit:
            snap = self._shard_snaps.get(name)
            if snap is not None:
                # diff in place on the local shard, then ship only the *mask*
                # through the gather. A uint8 mask is half a bf16 value gather,
                # and because it rides the identical all_gather_param +
                # convert_to_hf path it lands already in HF index space -- no
                # analytic remapping through chunk()/QKV splits to get wrong.
                mask = _mark_like(param, (param.data != snap).to(torch.uint8))
                mask_unit.append((name, all_gather_param(self.args, name, mask)))

        gathered_params, unit_size = super()._all_gather_update_unit(update_unit)

        # refresh the snapshot only after the mask was computed, and in place so
        # steady-state syncs allocate nothing
        for name, param in update_unit:
            snap = self._shard_snaps.get(name)
            if snap is None:
                self._shard_snaps[name] = _clone_as_shard(param)
            else:
                snap.copy_(param.data)
        self._pending_mask.extend(mask_unit)
        return gathered_params, unit_size

    def _gather_and_update_non_expert_weights(self, update_bucket_weight_func, pbar=None):
        """Mirror of the mixin's non-expert path, converting the mask too."""
        buffer_size = 0
        converted_named_tensors: list[tuple[str, torch.Tensor]] = []

        for update_unit in self._get_weight_transfer_update_units(is_expert=False):
            gathered_params, unit_size = self._all_gather_update_unit(update_unit)

            if not self._is_source:
                # non-source ranks join the collectives but publish nothing, so
                # drop the masks they gathered or they grow without bound
                self._pending_mask = []
                continue

            if buffer_size + unit_size > self.args.update_weight_buffer_size and converted_named_tensors:
                update_bucket_weight_func(converted_named_tensors, pbar)
                converted_named_tensors = []
                buffer_size = 0

            for name, param in self._pending_mask:
                for hf_name, hf_param in convert_to_hf(
                    self.args, self.model_name, name, param, self.quantization_config
                ):
                    self._mask_hf[hf_name] = hf_param
            self._pending_mask = []
            for name, param in gathered_params:
                converted_named_tensors += convert_to_hf(
                    self.args, self.model_name, name, param, self.quantization_config
                )
            buffer_size += unit_size

        if converted_named_tensors:
            update_bucket_weight_func(converted_named_tensors, pbar)

    def _gather_and_update_expert_weights(self, update_bucket_weight_func, pbar=None):
        """Mirror of the mixin's expert path, carrying masks in lockstep.

        The parent flushes a bucket *before* appending the current unit, so a
        mask list drained at flush time would contain a unit whose parameters
        are not in that bucket. The mismatch is fail-safe (the bucket falls back
        to dense) but it silently costs every expert bucket its sparsity, which
        on a MoE model is nearly the whole model.
        """
        buffer_size = 0
        named_tensors: list[tuple[str, torch.Tensor]] = []
        masks: list[tuple[str, torch.Tensor]] = []

        for update_unit in self._get_weight_transfer_update_units(is_expert=True):
            gathered_params, unit_size = self._all_gather_update_unit(update_unit)
            unit_masks = self._pending_mask
            self._pending_mask = []

            if (
                buffer_size + unit_size
            ) * get_parallel_state().ep.size > self.args.update_weight_buffer_size and named_tensors:
                self._flush_expert_bucket(named_tensors, masks, update_bucket_weight_func, pbar)
                named_tensors = []
                masks = []
                buffer_size = 0

            named_tensors.extend(gathered_params)
            masks.extend(unit_masks)
            buffer_size += unit_size

        if named_tensors:
            self._flush_expert_bucket(named_tensors, masks, update_bucket_weight_func, pbar)

    def _flush_expert_bucket(self, named_tensors, masks, update_bucket_weight_func, pbar=None):
        """Run the parent's EP-gather+convert twice: once to capture the mask,
        once for real. The mask pass sinks into a dict instead of the wire."""
        if masks:
            captured: list[tuple[str, torch.Tensor]] = []
            self._update_expert_bucket_weights(
                masks, lambda tensors, _pbar=None: captured.extend(tensors), None
            )
            self._mask_hf.update(dict(captured))
        self._update_expert_bucket_weights(named_tensors, update_bucket_weight_func, pbar)

    # -- transport -----------------------------------------------------------

    def _update_weight_implementation(
        self, converted_named_tensors: list[tuple[str, torch.Tensor]], pbar: tqdm | None = None
    ) -> None:
        while not ray.get(self.rollout_engine_lock.acquire.remote()):
            time.sleep(0.1)

        t0 = time.perf_counter()
        names = [n for n, _ in converted_named_tensors]
        datas = [p.data for _, p in converted_named_tensors]
        changed_per_tensor = _split_changed(datas, [self._mask_hf.get(n) for n in names])
        t_diff = time.perf_counter() - t0

        t0 = time.perf_counter()
        payload: list[tuple[str, torch.Tensor]] = []
        n_sparse = 0
        wire_bytes = 0
        dense_bytes = 0
        changed_elems = 0
        sparse_elems = 0
        # One (idx, val) pair for the whole bucket, not per tensor. Measured on
        # the per-tensor form: broadcasting ~13 MB of sparse payload as 672
        # entries took 0.044s while broadcasting the full 1.06 GB as 336 entries
        # took 0.026s -- 77x fewer bytes, 70% more time. dist.broadcast is
        # issued per entry, so entry count is the cost, and doubling it is what
        # ate the entire saving.
        sparse_names: list[str] = []
        sparse_idx: list[torch.Tensor] = []
        sparse_val: list[torch.Tensor] = []
        bounds: list[int] = []
        running = 0
        for i, (name, data) in enumerate(zip(names, datas)):
            dense_bytes += data.numel() * data.element_size()
            changed = None if changed_per_tensor is None else changed_per_tensor[i]
            if changed is None or changed.numel() > data.numel() * DENSE_FALLBACK_RATIO:
                payload.append((name, data.contiguous()))
                wire_bytes += data.numel() * data.element_size()
                continue
            sparse_names.append(name)
            # positions are relative to the concatenation of the sparse tensors
            # only, so dense fallbacks do not leave holes in the index space
            sparse_idx.append(changed.to(torch.int32) + running)
            sparse_val.append(data.reshape(-1)[changed])
            running += data.numel()
            bounds.append(running)
            n_sparse += 1
            changed_elems += changed.numel()
            sparse_elems += data.numel()

        if sparse_names:
            assert running < 2**31, f"bucket too large for int32 positions: {running}"
            idx_cat = torch.cat(sparse_idx).contiguous()
            val_cat = torch.cat(sparse_val).contiguous()
            meta = torch.tensor(bounds, dtype=torch.int64, device=idx_cat.device)
            # the ordered names ride in the entry name: names travel over Ray as
            # plain strings, so this needs no extra tensor and no extra entry
            payload.append((_META_PREFIX + _NAME_SEP.join(sparse_names), meta))
            payload.append((_IDX_NAME, idx_cat))
            payload.append((_VAL_NAME, val_cat))
            wire_bytes += (
                val_cat.numel() * val_cat.element_size()
                + idx_cat.numel() * idx_cat.element_size()
                + meta.numel() * meta.element_size()
            )
        t_encode = time.perf_counter() - t0

        t0 = time.perf_counter()
        refs = update_weights_from_distributed(
            self._group_name,
            self._model_update_groups,
            self.weight_version,
            self.rollout_engines,
            payload,
            selector=self._weight_update_selector,
        )
        t_bcast = time.perf_counter() - t0

        t0 = time.perf_counter()
        ray.get(refs)
        t_apply = time.perf_counter() - t0

        converted_named_tensors.clear()
        # the mask for this bucket has been consumed; drop it so only one
        # bucket's worth of HF-space mask is ever resident
        self._mask_hf.clear()
        ray.get(self.rollout_engine_lock.release.remote())
        if self._is_source and dense_bytes:
            logger.info(
                "SHARD-DELTA stage: diff %.3f encode %.3f bcast %.3f apply %.3f | "
                "%d tensors (%d sparse), wire %.3f%% of dense (%.2f GB dense), "
                "changed %.3f%% of sparse-path elems",
                t_diff,
                t_encode,
                t_bcast,
                t_apply,
                len(payload),
                n_sparse,
                100.0 * wire_bytes / dense_bytes,
                dense_bytes / 1e9,
                (100.0 * changed_elems / sparse_elems) if sparse_elems else 0.0,
            )
        if pbar:
            pbar.update(1)
