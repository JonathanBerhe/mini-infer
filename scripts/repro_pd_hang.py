"""Standalone (no pytest, no Modal, CPU/gloo) reproduction of the two-process
PD hang that the 2x H100 smoke hit.

`tests/unit/test_workers_mp.py` is skipped on the theory that the stall is a
macOS/CPU/pytest-thread-contention artifact that "runs cleanly on multi-GPU
CUDA." The H100 smoke disproved that (it hung with no pytest involved). This
script removes pytest from the picture entirely and instruments each rank with
`faulthandler`, so if a rank wedges we get its full stack instead of guessing.

Each rank:
  - registers a repeating faulthandler stack dump to /tmp/pd_repro_rank{r}.stacks
  - prints flushed checkpoints around every blocking step (PG init, model load,
    prefill, send / recv, decode)

The parent enforces a hard deadline and force-terminates children, so this can
never hang the caller. After it returns, the per-rank stack files show where
(if anywhere) each rank was stuck.

Run:
    uv run python scripts/repro_pd_hang.py
"""

from __future__ import annotations

import faulthandler
import socket
import sys
import time

import torch
import torch.multiprocessing as mp

_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
_DUMP_EVERY = 20.0  # seconds: if a rank is stuck this long, dump its stacks
_DEADLINE = 150.0  # hard parent deadline; model load + prefill + decode << this


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _child(rank: int, world_size: int, master_port: int) -> None:
    stack_path = f"/tmp/pd_repro_rank{rank}.stacks"
    # Must stay open for the child's whole lifetime: the repeating
    # faulthandler timer writes to it throughout (a context manager would
    # close it immediately). Closed in the finally block.
    stack_file = open(stack_path, "w")  # noqa: SIM115
    # Repeating dump of ALL thread stacks: a hang shows up as a stack here.
    faulthandler.dump_traceback_later(_DUMP_EVERY, repeat=True, file=stack_file)

    def ck(msg: str) -> None:
        print(f"[rank {rank}] {msg}", flush=True)

    from mini_infer.distributed.group import (
        destroy_distributed,
        init_distributed,
        replica_scope,
    )
    from mini_infer.engine.model_runner import ModelRunner
    from mini_infer.engine.sampler import SamplingParams
    from mini_infer.scheduler.request_state import Request
    from mini_infer.workers import DECODE_RANK, PREFILL_RANK, DecodeWorker, PrefillWorker
    from mini_infer.workers.kv_transfer import recv_handoff, send_handoff
    from mini_infer.workers.multi_process import _limit_child_threads

    ck("start; init_distributed (gloo)...")
    init_distributed(
        world_size=world_size,
        rank=rank,
        backend="gloo",
        master_addr="127.0.0.1",
        master_port=master_port,
    )
    try:
        ck("PG init done; capping threads + loading model (replica_scope)...")
        _limit_child_threads()
        with replica_scope():
            runner = ModelRunner.from_pretrained(_MODEL, device="cpu", dtype=torch.float32)
            ck("model loaded")
            params = SamplingParams(temperature=0.0, top_k=0, top_p=1.0)

            if rank == PREFILL_RANK:
                worker = PrefillWorker(runner)
                req = Request(
                    prompt="The capital of France is", sampling_params=params, max_tokens=8
                )
                ck("prefilling...")
                handoff = worker.prefill(req)
                ck(f"prefilled (len={handoff.prefill_len}); send_handoff -> rank {DECODE_RANK}...")
                send_handoff(handoff, dst_rank=DECODE_RANK)
                ck("send_handoff returned; DONE")
            elif rank == DECODE_RANK:
                worker = DecodeWorker(runner)
                ck(f"recv_handoff <- rank {PREFILL_RANK}...")
                handoff = recv_handoff(src_rank=PREFILL_RANK, pool=runner.block_pool)
                ck(f"recv_handoff returned (len={handoff.prefill_len}); decoding...")
                tokens = list(worker.decode(handoff))
                ck(f"decoded {len(tokens)} tokens; DONE tokens={tokens}")
    finally:
        faulthandler.cancel_dump_traceback_later()
        stack_file.flush()
        stack_file.close()
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            destroy_distributed()


def main() -> None:
    ctx = mp.get_context("spawn")
    port = _free_port()
    procs = [ctx.Process(target=_child, args=(r, 2, port)) for r in range(2)]
    print(f"launching 2-process PD on CPU/gloo (port={port}, deadline={_DEADLINE:.0f}s)")
    for p in procs:
        p.start()

    deadline = time.monotonic() + _DEADLINE
    while time.monotonic() < deadline:
        if all(not p.is_alive() for p in procs):
            break
        time.sleep(1.0)

    hung = [i for i, p in enumerate(procs) if p.is_alive()]
    for p in procs:
        if p.is_alive():
            p.terminate()
        p.join(timeout=5)

    print()
    if hung:
        print(f"HANG: ranks {hung} still alive at deadline; stacks below")
    else:
        print(f"both ranks exited (codes={[p.exitcode for p in procs]})")
    for r in range(2):
        try:
            with open(f"/tmp/pd_repro_rank{r}.stacks") as f:
                content = f.read().strip()
        except FileNotFoundError:
            content = "(no stack file)"
        shown = content or "(none; rank did not hang past dump window)"
        print(f"\n===== rank {r} stack dumps =====\n{shown}")


if __name__ == "__main__":
    main()
    sys.exit(0)
