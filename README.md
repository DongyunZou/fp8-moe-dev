# FP8 MoE Big-Shape Kernel Snapshot

This directory is a minimal kernel snapshot from:

- source repo: `/home/dongyun/workspace/projects/ablation/fp8_moe`
- source commit: `9400819`
- starter-kit commit used for rule/context checks: `75ccd05`
- benchmark source: `profile/current_all19_verify_py/summary.json`

Only two things are included here:

- `kernel/`: CUDA/C++ source files needed by the current kernel
- `README.md`: this note, dependencies, run commands, and per-shape results

## What This Version Does

The entry point is `moe_fp8.cu::run`.

This is a hybrid implementation:

- Small and medium shapes use the existing TRTLLM/FlashInfer BMM path for selected GEMM phases.
- Large shapes use our CUTLASS grouped FP8 blockwise GEMM1 path from `cutlass_gemm.cu`.
- Routing, packing, pruning, SwiGLU/quantization, scatter/reduce, descriptor construction, and graph/cache logic live in `moe_fp8.cu`.

Important routing details:

- GEMM1 uses TRTLLM BMM only when `T <= 80`, or `901 <= T < 10000`, or very small `M_total_unpadded <= 128`, with `T=7` explicitly excluded.
- Therefore the official large shapes `T=11948` and `T=14107` take the custom CUTLASS grouped FP8 GEMM1 path.
- GEMM2 still uses the existing TRTLLM BMM path when `M_total_unpadded <= 20000`; otherwise it falls back to the CUTLASS grouped FP8 path.

So the large-shape speedups below are not from a pure all-BMM solution: the large-shape GEMM1 path is our own CUTLASS-based kernel path.

## Files

```text
kernel/
  moe_fp8.cu
  cutlass_gemm.cu
  trtllm_fp8_bmm.cu
  trtllm_batched_gemm_runner.cu
  envUtils.cpp
  logger.cpp
  stringUtils.cpp
  tllmException.cpp
```

## Dependencies

Runtime/test environment used locally:

- NVIDIA B200 / SM100-capable GPU
- CUDA 13.2-era toolchain with `nvcc` support for `sm_100a`
- Python environment with `flashinfer-bench`, `flashinfer`, `tvm_ffi`, PyTorch CUDA build
- FlashInfer packaged CUTLASS headers:
  - `flashinfer/data/cutlass/include`
  - `flashinfer/data/cutlass/tools/util/include`
- FlashInfer/TensorRT-LLM BMM generated headers and cubins in the local cache:
  - `~/.cache/flashinfer/cubins/*/batched_gemm-*/include/trtllmGen_bmm_export`
  - `Bmm_Bfloat16_E4m3E4m3*noShflA*dsFp8*dynB*.cubin`
- Link flags include `-lcuda -ldl`.

The local benchmark wrapper in the source repo patches the build environment to add these include paths, `TVM_FFI_CUDA_ARCH_LIST=10.0a`, `--expt-relaxed-constexpr`, and `TLLM_GEN_GEMM_CUBIN_PATH`.

## How To Run And Benchmark

The minimal snapshot intentionally does not include the full starter-kit or local helper scripts. To benchmark this exact version, place the files in a FlashInfer-bench CUDA solution directory with:

```text
[build]
language = "cuda"
entry_point = "moe_fp8.cu::run"
destination_passing_style = true
```

The exact local command used from the source workspace is:

```bash
cd /home/dongyun/workspace/projects/ablation/fp8_moe

PY=/home/dongyun/workspace/projects/fp8-moe/.venv/bin/python
export PYTHONPATH=/home/dongyun/.cache/uv/archive-v0/PN6qqGHi_dhz5nCMR4zKI

python3 /home/dongyun/workspace/skills/gpu-lock-skill/scripts/gpu_lock.py run \
  --gpu 0 \
  --timeout 45m \
  --owner "$USER-fp8-moe-bench" \
  -- \
  env PYTHONPATH="$PYTHONPATH" \
  "$PY" verify.py --all --baseline \
    --dump-json profile/current_all19_verify_py/summary.json
```

To smoke-test only the two large shapes on the agent kernel:

```bash
cd /home/dongyun/workspace/projects/ablation/fp8_moe

PY=/home/dongyun/workspace/projects/fp8-moe/.venv/bin/python
export PYTHONPATH=/home/dongyun/.cache/uv/archive-v0/PN6qqGHi_dhz5nCMR4zKI

python3 /home/dongyun/workspace/skills/gpu-lock-skill/scripts/gpu_lock.py run \
  --gpu 0 \
  --timeout 30m \
  --owner "$USER-fp8-moe-bigshape" \
  -- \
  env PYTHONPATH="$PYTHONPATH" \
  "$PY" verify.py \
    --uuid 58a34f27 \
    --uuid 5e8dc11c
```

For large-shape speedup numbers, prefer the full `--all --baseline` command above. In a fresh large-only run on 2026-06-11, the agent passed both large workloads, but the FlashInfer baseline hit the benchmark runner's 300s timeout on both large workloads. The per-shape baseline numbers below therefore come from the successful full 19-workload benchmark summary.

MoE correctness tolerance used by `verify.py`:

```text
atol = 1.0
rtol = 0.3
required_matched_ratio = 0.9
```

## Full Benchmark Summary

Measured against FlashInfer baseline `flashinfer_wrapper_9sdjf3` on all 19 workloads:

```text
baseline mean latency: 493.728 us
agent mean latency:    335.394 us
ratio of means:        1.472x
mean per-shape speedup: 1.444x
min per-shape speedup:  1.264x
correctness:           19/19 PASSED
```

Large-shape aggregate:

```text
T=11948 and T=14107 ratio of summed baseline/agent latency: 1.585x
large-shape mean per-shape speedup:                         1.587x
```

Large-shape agent-only smoke rerun on 2026-06-11:

```text
T=11948: 1123.027 us, PASSED
T=14107: 1532.356 us, PASSED
FlashInfer baseline in that large-only rerun: TIMEOUT at 300s for both shapes
```

## Per-Shape Speedup

| seq_len | uuid | baseline us | agent us | speedup | GEMM1 path |
|---:|---|---:|---:|---:|---|
| 1 | `e05c6c03` | 203.425 | 108.255 | 1.879x | TRTLLM BMM |
| 7 | `b8f4f012` | 212.457 | 143.990 | 1.475x | CUTLASS grouped FP8 |
| 14 | `8cba5890` | 265.967 | 174.308 | 1.526x | TRTLLM BMM |
| 15 | `2e69caee` | 202.835 | 114.267 | 1.775x | TRTLLM BMM |
| 16 | `a7c2bcfd` | 254.074 | 179.555 | 1.415x | TRTLLM BMM |
| 32 | `6230e838` | 307.987 | 232.557 | 1.324x | TRTLLM BMM |
| 52 | `f7d6ac7c` | 283.012 | 206.819 | 1.368x | TRTLLM BMM |
| 53 | `fc378037` | 364.681 | 245.837 | 1.483x | TRTLLM BMM |
| 54 | `76010cb4` | 320.950 | 227.433 | 1.411x | TRTLLM BMM |
| 55 | `81955b1e` | 334.480 | 260.429 | 1.284x | TRTLLM BMM |
| 56 | `4822167c` | 341.984 | 245.670 | 1.392x | TRTLLM BMM |
| 57 | `74d7ff04` | 333.920 | 259.711 | 1.286x | TRTLLM BMM |
| 58 | `e626d3e6` | 337.645 | 267.078 | 1.264x | TRTLLM BMM |
| 59 | `eedc63b2` | 300.046 | 226.919 | 1.322x | TRTLLM BMM |
| 62 | `5eadab1e` | 294.599 | 207.190 | 1.422x | TRTLLM BMM |
| 80 | `8f1ff9f1` | 366.890 | 287.478 | 1.276x | TRTLLM BMM |
| 901 | `1a4c6ba1` | 459.392 | 337.650 | 1.361x | TRTLLM BMM |
| 11948 | `58a34f27` | 1786.778 | 1119.045 | 1.597x | CUTLASS grouped FP8 |
| 14107 | `5e8dc11c` | 2409.706 | 1528.301 | 1.577x | CUTLASS grouped FP8 |
