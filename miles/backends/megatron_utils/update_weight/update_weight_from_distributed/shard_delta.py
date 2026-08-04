"""Delta weight sync over the existing NCCL broadcast group.

Same wire as ``broadcast`` (one NCCL group per PP stage, metadata over Ray) but
each sync ships only the elements that changed since the previous one: the
source rank keeps a snapshot of what it last published, diffs each converted HF
tensor against it, and broadcasts ``(int32 positions, values)`` pairs. The
engine writes those positions straight into the live parameter storage, so a
sync costs O(changed) instead of O(model).

The first sync has no baseline, so it publishes full tensors exactly like
``broadcast`` and then primes the snapshot; every later sync is sparse.
"""

import logging
import time
from typing import Optional

import ray
import torch
from tqdm import tqdm

from .broadcast import (
    UpdateWeightFromDistributed,
    update_weights_from_distributed,
)

logger = logging.getLogger(__name__)

# A tensor whose changed fraction exceeds this is cheaper to send whole: the
# sparse encoding costs 4 bytes of index per element on top of the value.
DENSE_FALLBACK_RATIO = 0.25


def _encode_delta(
    cur: torch.Tensor, snap: Optional[torch.Tensor]
) -> tuple[Optional[torch.Tensor], torch.Tensor]:
    """Return ``(positions, values)`` for one tensor.

    ``positions is None`` means "dense": ``values`` is the whole tensor, used
    for the seed sync and for tensors that changed too much to be worth
    encoding sparsely.
    """
    if snap is None:
        return None, cur
    flat_cur = cur.reshape(-1)
    # snapshots live on the host to keep GPU memory free, so bring the baseline
    # to the tensor's device for the comparison
    flat_snap = snap.reshape(-1).to(flat_cur.device, non_blocking=True)
    changed = (flat_cur != flat_snap).nonzero(as_tuple=True)[0]
    if changed.numel() > flat_cur.numel() * DENSE_FALLBACK_RATIO:
        return None, cur
    return changed.to(torch.int32), flat_cur[changed]


class UpdateWeightFromShardDelta(UpdateWeightFromDistributed):
    """Broadcast-based transport that ships per-tensor deltas."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # name -> CPU copy of what the engines currently hold
        self._snapshots: dict[str, torch.Tensor] = {}
        self._seeded = False

    def _update_weight_implementation(
        self, converted_named_tensors: list[tuple[str, torch.Tensor]], pbar: tqdm | None = None
    ) -> None:
        while not ray.get(self.rollout_engine_lock.acquire.remote()):
            time.sleep(0.1)

        payload: list[tuple[str, torch.Tensor]] = []
        n_sparse = 0
        wire_elems = 0
        total_elems = 0
        for name, param in converted_named_tensors:
            data = param.data
            total_elems += data.numel()
            positions, values = _encode_delta(data, self._snapshots.get(name))
            if positions is None:
                payload.append((name, values.contiguous()))
                wire_elems += values.numel()
            else:
                # two entries per changed tensor: the engine pairs them by name
                payload.append((f"{name}::idx", positions.contiguous()))
                payload.append((f"{name}::val", values.contiguous()))
                n_sparse += 1
                wire_elems += values.numel() * 2  # values + int32 indices
            # snapshot what the engines will hold after this sync
            self._snapshots[name] = data.detach().to("cpu", copy=True)

        refs = update_weights_from_distributed(
            self._group_name,
            self._model_update_groups,
            self.weight_version,
            self.rollout_engines,
            payload,
            selector=self._weight_update_selector,
        )
        ray.get(refs)
        converted_named_tensors.clear()
        ray.get(self.rollout_engine_lock.release.remote())
        if self._is_source and total_elems:
            logger.info(
                "SHARD-DELTA bucket: %d tensors (%d sparse), wire %.1f%% of dense",
                len(payload),
                n_sparse,
                100.0 * wire_elems / total_elems,
            )
        if pbar:
            pbar.update(1)
