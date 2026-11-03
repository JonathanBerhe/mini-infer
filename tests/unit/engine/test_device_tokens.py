"""`DeviceTokens`: ids that stay on the device between two decode steps."""

from __future__ import annotations

import pytest
import torch

from mini_infer.cache.paged_kv_cache import PagedKVCache
from mini_infer.engine.device_tokens import DeviceTokens
from mini_infer.engine.model_runner import ModelRunner
from tests.unit._dspark_doubles import micro_setup


def test_gather_takes_the_rows_in_slot_order() -> None:
    source = torch.tensor([10, 11, 12, 13])
    assert DeviceTokens(source, [3, 0, 2]).gather().tolist() == [13, 10, 12]
    assert len(DeviceTokens(source, [1])) == 1


def test_leading_rows_in_order_are_a_view_of_the_source() -> None:
    """No gather kernel when the slots take rows 0, 1, ... of the sampled ids."""
    source = torch.tensor([10, 11, 12, 13])
    leading = DeviceTokens(source, [0, 1, 2]).gather()
    assert leading.tolist() == [10, 11, 12]
    assert leading.data_ptr() == source.data_ptr()
    assert DeviceTokens(source, range(4)).gather().data_ptr() == source.data_ptr()
    # A row out of order, or a skipped row, needs the gather.
    assert DeviceTokens(source, [0, 2]).gather().data_ptr() != source.data_ptr()
    assert DeviceTokens(source, [1, 0]).gather().tolist() == [11, 10]


def test_rejects_a_bad_row_or_a_matrix() -> None:
    with pytest.raises(ValueError, match="outside"):
        DeviceTokens(torch.tensor([1, 2]), [0, 2])
    with pytest.raises(ValueError, match="1-D"):
        DeviceTokens(torch.zeros(2, 2, dtype=torch.long), [0])


def _runner_with_two_slots() -> tuple[ModelRunner, PagedKVCache]:
    micro, _ = micro_setup(seed=4)
    runner = ModelRunner(micro._model, None, "cpu", micro.block_pool)  # type: ignore[arg-type]
    cache = PagedKVCache(runner.block_pool)
    cache.add_request_slot()
    cache.add_request_slot()
    runner.forward_step_packed(cache, list(range(10, 19)), [0, 6, 9], [0, 0])
    return runner, cache


def test_the_eager_step_with_device_ids_matches_the_step_with_host_ids() -> None:
    host_runner, host_cache = _runner_with_two_slots()
    device_runner, device_cache = _runner_with_two_slots()
    sampled = torch.tensor([7, 5, 9])  # slot 0 takes row 2, slot 1 takes row 0
    expected = host_runner.forward_step_packed(host_cache, [9, 7], [0, 1, 2], [6, 3])
    got = device_runner.forward_step_packed(
        device_cache, DeviceTokens(sampled, [2, 0]), [0, 1, 2], [6, 3]
    )
    assert torch.equal(got, expected)
    assert device_cache.seq_lens_list() == host_cache.seq_lens_list()


def test_the_runner_refuses_device_ids_that_do_not_cover_the_batch() -> None:
    runner, cache = _runner_with_two_slots()
    short = DeviceTokens(torch.tensor([1, 2]), [0])
    with pytest.raises(ValueError, match="one-token-per-slot"):
        runner.forward_step_packed(cache, short, [0, 1, 2], [6, 3])
