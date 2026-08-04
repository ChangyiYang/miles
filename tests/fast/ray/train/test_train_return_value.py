from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import ray
from tests.fast.ray.train.conftest import get_raw_actor_handles, make_alive_cell

from miles.backends.megatron_utils.ft.types import TrainStepOutcome, TrainStepOutput
from miles.ray.train.group import TrainerController
from miles.utils.retry_utils import NonRetryableError

pytestmark = pytest.mark.asyncio

_DUMMY_DATA_PACK = {"data_ref": "data", "sample_indices": [0]}


def _make_group(cells: list) -> TrainerController:
    group = object.__new__(TrainerController)
    group._cells_by_id = {cell.cell_id: cell for cell in cells}
    group.args = SimpleNamespace(enable_event_analyzer=False, save_debug_event_data=None)
    group._witness_allocator = None
    group._indep_dp_quorum_id = 0
    group._health_checker_activeness = True
    group._test_action_executor = SimpleNamespace(run_after_step=AsyncMock())
    return group


def _normal_output(values=None) -> TrainStepOutput:
    return TrainStepOutput(outcome=TrainStepOutcome.NORMAL, values=values)


class TestTrainReturnValue:
    async def test_one_result_per_worker_reaches_the_caller(self):
        """The critic values leave the group per worker so the driver can feed them to the actor."""
        cell = make_alive_cell(0, alive_cell_indices=[0])
        for handle in get_raw_actor_handles(cell):
            ray.get(handle.set_train_return_value.remote(_normal_output()))
        group = _make_group([cell])

        results = await group.train(3, _DUMMY_DATA_PACK)

        assert results == [_normal_output()] * 2

    async def test_results_of_several_cells_are_concatenated_in_cell_order(self):
        """Independent DP ranks are positional, so a reordered result list misroutes values."""
        cells = [make_alive_cell(index, alive_cell_indices=[0, 1]) for index in range(2)]
        for index, cell in enumerate(cells):
            for handle in get_raw_actor_handles(cell):
                ray.get(handle.set_train_return_value.remote(_normal_output(values=index)))
        group = _make_group(cells)

        results = await group.train(3, _DUMMY_DATA_PACK)

        assert [result.values for result in results] == [0, 0, 1, 1]

    async def test_a_failed_cell_contributes_no_result(self):
        """A raw exception object in the returned list would be fed straight into the next train call."""
        cells = [make_alive_cell(index, alive_cell_indices=[0, 1]) for index in range(2)]
        ray.get(get_raw_actor_handles(cells[0])[0].set_fail_methods.remote(["train"]))
        for handle in get_raw_actor_handles(cells[1]):
            ray.get(handle.set_train_return_value.remote(_normal_output(values="ok")))
        group = _make_group(cells)

        results = await group.train(3, _DUMMY_DATA_PACK)

        assert [result.values for result in results] == ["ok", "ok"]


class TestRetryReturnsTheValue:
    async def test_the_value_of_the_successful_attempt_is_returned(self):
        """train() reads its result through retry, so retry must stop swallowing it."""
        from miles.utils.retry_utils import retry

        attempts = []

        async def _fn(attempt: int) -> str:
            attempts.append(attempt)
            if attempt == 0:
                raise RuntimeError("boom")
            return "second"

        async def _no_sleep(_seconds: float) -> None:
            return None

        assert await retry(_fn, sleep_fn=_no_sleep) == "second"
        assert attempts == [0, 1]


class TestWorkerResultShape:
    async def test_a_normal_outcome_does_not_trip_the_discarded_check(self):
        """A normal step must not be mistaken for a retry request, or every step would be retried."""
        cell = make_alive_cell(0, alive_cell_indices=[0])
        for handle in get_raw_actor_handles(cell):
            ray.get(handle.set_train_return_value.remote(_normal_output()))
        group = _make_group([cell])

        await group.train(3, _DUMMY_DATA_PACK)

    async def test_a_discarded_outcome_is_seen(self):
        """Missing the discarded outcome would let the group commit a step it asked to redo."""
        cell = make_alive_cell(0, alive_cell_indices=[0])
        results = [[TrainStepOutput(outcome=TrainStepOutcome.DISCARDED_SHOULD_RETRY)]]

        outcomes = TrainerController._compute_attempt_outcomes([cell], results)

        assert outcomes["discarded"] == [0]

    async def test_a_worker_result_of_the_wrong_type_is_not_retried(self):
        """Retrying a broken worker contract 30 times turns an instant bug into a 20-minute stall."""
        cell = make_alive_cell(0, alive_cell_indices=[0])
        results = [[{"train_step_outcome": TrainStepOutcome.NORMAL}]]

        with pytest.raises(NonRetryableError):
            TrainerController._compute_attempt_outcomes([cell], results)
