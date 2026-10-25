# ADR-048: One folder per model family

Date: 2026-10-25
Status: Accepted

## Context

mini-infer ports a new model family every few weeks. Before this change, the
code of one family was in many places. DeepSeek-V4 had its model in
`models/deepseek_v4.py`, 15 files in `models/blocks/` and `models/blocks/v4/`,
an attention kernel in `cache/`, about 25 tests with three name patterns in
the flat `tests/unit/` folder, and about 12 scripts in the flat `scripts/`
folder. The caches or kernels of Kimi Linear, MiniMax-M3, Nemotron-H and
Qwen3.5 were in the shared `cache/` folder. `models/blocks/` had 47 files that
mixed shared parts, parts of one model, and GPU kernels. A reader had to search
the whole tree to find the code of one model.

## Decision

1. Each model family has one package in `src/mini_infer/models/`: `llama/`,
   `mistral/`, `qwen/`, `gemma/`, `deepseek_v2/`, `deepseek_v4/`, `glm/`,
   `minimax_m3/`, `inkling/`, `kimi_linear/` and `nemotron_h/`. A family
   holds the models of one maker that share code, for example Qwen2, Qwen3
   and Qwen3.5.
2. A file that only one family uses is in the package of that family. This
   includes its state cache and its kernels. A file that two or more
   families use is in `models/blocks/`. The shared MoE parts and their FP8
   and FP4 kernels are in `models/blocks/moe/`.
3. The main file of a family keeps its name, for example
   `deepseek_v4/deepseek_v4.py`. A file drops its model prefix when the
   package name gives it, for example `blocks/inkling_moe.py` is now
   `inkling/moe.py`.
4. `models/blocks/__init__.py` exports only the shared blocks. Code imports
   a family block from its module, for example
   `from mini_infer.models.deepseek_v4.csa import CSAAttention`.
5. `tests/unit/` has the same folders as `src/mini_infer/`. A test named
   after one model is in the folder of that model. Each other test is in the
   folder of the code that it tests. Test file names do not change.
6. `scripts/` has one folder for each family, plus `bench/`, `smoke/`,
   `dspark/` and `tpu/`. Script names do not change, so the `modal_` prefix
   still marks a script that starts a paid GPU run.

## Alternatives Considered

- One file per model and one flat folder of layers, as in vLLM and SGLang.
  A port here has up to 16 files of its own (DeepSeek-V4), so a flat folder
  stays crowded.
- One package per architecture, for example `qwen2/`, `qwen3/` and
  `qwen3_5/`. Qwen2 and Qwen3 are one file each and share code, and a
  reader looks for "Qwen" first.
- Keep the family names in `models/blocks/__init__.py`, so that the old
  imports still work. Then `blocks` depends on every family. No code outside
  this repository imports these paths.
- Move the ADRs, plans and benchmark reports into model folders. They
  record decisions in time order, and many of them cover more than one
  model.

## Consequences

- The change moves 245 files and changes no code behavior. The CPU suite
  (1095 tests) and the real-model suite (93 tests) give the same outcome for
  each test before and after the move.
- Speed does not change. On the CPU, a small random-weight model of each of
  the 15 architectures runs within 4% of the old layout in both directions.
  The geometric mean of after/before is 0.999 for prefill and 0.987 for
  decode. On an M1 Pro GPU (MPS), four real checkpoints run within 2%. The
  greedy tokens and the logits are bit-identical in every run.
- The 45 Triton and Gluon kernels keep their code. Only the Python module
  path in the Triton cache key changes, so the first GPU run compiles each
  kernel again. One kernel docstring names a moved script. That kernel
  compiles to the same PTX and SASS for sm90 and sm100.
- Every open branch that touches a moved file needs a rebase. Git detects
  the renames, so most conflicts resolve automatically.
- A new port adds one package, one test folder and one script folder. It
  also imports its model module in `_register_builtin_models()`.
