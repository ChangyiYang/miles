from argparse import Namespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from miles.ray.rollout.inference_controller import InferenceController
from miles.utils.context_lock import ContextLock


class _OrderRecordingInferenceController:
    def __init__(self, order: list[str]):
        self._order = order

    def __getattr__(self, name: str):
        recorder = self

        async def _method(*args, **kwargs):
            recorder._order.append(name)
            return MagicMock()

        return _method


@pytest.mark.asyncio
async def test_group_pauses_health_checks_before_snapshotting_the_engines():
    """``start_update_weights`` pauses the health monitor before it reads the engine set."""
    order: list[str] = []
    controller = InferenceController.__new__(InferenceController)

    async def _record_pause() -> None:
        order.append("health_monitoring_pause")

    def _record_snapshot():
        order.append("get_updatable_server")
        return None

    controller.context_lock = ContextLock("InferenceController")
    controller.args = Namespace(colocate=False)
    controller.servers = {}
    controller._health_monitoring_pause = _record_pause
    controller._get_updatable_server = _record_snapshot

    await controller.start_update_weights()

    assert order == ["health_monitoring_pause", "get_updatable_server"]


@pytest.mark.asyncio
async def test_the_trainer_brackets_the_broadcast_with_start_and_end_update_weights():
    """The engine snapshot is taken before, and released after, the trainer broadcast."""
    from miles.ray.train.group import TrainerController

    order: list[str] = []
    group = TrainerController.__new__(TrainerController)
    group.args = Namespace(debug_train_only=False, debug_rollout_only=False)
    group._inference_controller = _OrderRecordingInferenceController(order)
    group._execute_first_alive = AsyncMock()
    group._maybe_log_inference_engine_weight_checksums = AsyncMock()

    await group.update_weights()

    assert order[:2] == ["start_update_weights", "end_update_weights"]


def test_fsdp_updater_flushes_only_after_every_engine_is_paused():
    """Every engine is paused before any engine is flushed."""
    from unittest.mock import patch

    from miles.backends.experimental.fsdp_utils.update_weight_utils import UpdateWeightFromTensor

    order: list[str] = []
    pause_modes: list[str] = []

    class _Client:
        def __init__(self, index: int):
            self._index = index

        async def pause_generation(self, *, mode: str = "retract"):
            order.append(f"pause-{self._index}")
            pause_modes.append(mode)

        async def flush_cache(self):
            order.append(f"flush-{self._index}")

        async def begin_weight_update(self):
            order.append(f"begin-{self._index}")

        async def end_weight_update(self):
            order.append(f"end-{self._index}")

        async def continue_generation(self):
            order.append(f"continue-{self._index}")

    updater = UpdateWeightFromTensor.__new__(UpdateWeightFromTensor)
    updater.args = Namespace(update_weight_buffer_size=1 << 30)
    updater.weight_version = 0
    updater.model = MagicMock()
    updater.model.state_dict.return_value = {}
    updater.rollout_engines = [_Client(0), _Client(1)]

    module = "miles.backends.experimental.fsdp_utils.update_weight_utils"
    with patch(f"{module}.dist") as dist_mock, patch(f"{module}.get_gloo_group", return_value=MagicMock()):
        dist_mock.get_rank.return_value = 0
        updater.update_weights()

    assert set(order[:2]) == {"pause-0", "pause-1"}
    assert set(order[2:4]) == {"flush-0", "flush-1"}
    assert set(order[4:6]) == {"begin-0", "begin-1"}
    assert set(order[6:8]) == {"end-0", "end-1"}
    assert set(order[8:]) == {"continue-0", "continue-1"}
    assert pause_modes == ["retract", "retract"]
