# FP8 MoE Kernel Snapshot

CUDA/C++ snapshot for:

```text
moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048
entry point: moe_fp8.cu::run
source commit: 9400819
```

This repo intentionally contains only `kernel/` and this README. It is a development snapshot, not a standalone FlashInfer starter-kit checkout: use it by copying `kernel/` into a runner workspace that has `config.toml`, `verify.py`, and the build-env patch for CUTLASS/TRTLLM include paths.

## GEMM Routing

The MoE path has two matrix multiplies:

- **GEMM1**: `hidden -> 2 * intermediate`, before SwiGLU.
- **GEMM2**: `intermediate -> hidden`, after SwiGLU and FP8 requant.

Current defaults in `kernel/moe_fp8.cu`:

```text
MOE_USE_TRTLLM_BMM_GEMM1 = 1
MOE_USE_TRTLLM_BMM_GEMM2 = 1
MOE_MMA_GEMM2_M_THRESHOLD = 0
MOE_TRTLLM_BMM_GEMM2_M_THRESHOLD = 20000
```

GEMM1 branch:

```text
TRTLLM BMM if:
  trtllm_bmm_available
  M_total_unpadded > 0
  T != 7
  and (M_total_unpadded <= 128 or T <= 80 or 901 <= T < 10000)

otherwise:
  our CUTLASS grouped FP8 blockwise GEMM in cutlass_gemm.cu
```

So the two large official shapes use our GEMM1:

```text
T=11948 -> CUTLASS grouped FP8 GEMM1
T=14107 -> CUTLASS grouped FP8 GEMM1
```

GEMM2 branch:

```text
TRTLLM BMM if:
  trtllm_bmm_available
  M_total_unpadded > 0
  M_total_unpadded <= 20000

our CUTLASS grouped FP8 GEMM2 only if:
  TRTLLM BMM is unavailable/disabled, or M_total_unpadded > 20000
```

For the recorded v34 run on the official 19 workloads, **GEMM2 did not use our CUTLASS path**. The routed `M_total_unpadded` stays below `20000` for all 19 shapes, including the two large shapes:

```text
T=11948 -> M_total_unpadded ~= 8830
T=14107 -> M_total_unpadded ~= 13428
```

So GEMM2 is TRTLLM BMM in the recorded benchmark. Our CUTLASS GEMM2 code exists and is wired as a fallback, but it is not responsible for the recorded 1.4x result.

## Files

```text
kernel/
  moe_fp8.cu                         # routing, packing, pruning, graph/cache, non-GEMM CUDA kernels
  cutlass_gemm.cu                    # our CUTLASS grouped FP8 blockwise GEMM implementation
  trtllm_fp8_bmm.cu                  # adapter into TRTLLM/FlashInfer generated BMM cubins
  trtllm_batched_gemm_runner.cu      # TRTLLM BMM runner glue
  envUtils.cpp logger.cpp stringUtils.cpp tllmException.cpp
```

## Environment

The uv setup follows the MIT Kernel Mafia release README/reproduction notes.

From a runner workspace with `pyproject.toml`/`uv.lock` equivalent to the original ablation workspace:

```bash
git clone https://github.com/flashinfer-ai/flashinfer-bench.git /tmp/flashinfer-bench-main
uv sync

# Required by the contest stack and by submissions using CUTLASS/CuTe headers.
git clone https://github.com/deepseek-ai/DeepGEMM.git /tmp/DeepGEMM
uv pip install -e /tmp/DeepGEMM --no-build-isolation

# Dataset path; alternatively use the runner's download script.
export FIB_DATASET_PATH=/path/to/flashinfer-trace
```

Expected pinned stack:

```text
Python >= 3.12
flashinfer-python == 0.6.8.post1
torch >= 2.12.0 from the CUDA 13.2 PyTorch index
triton == 3.6.0
ninja >= 1.13.0
flashinfer-bench from /tmp/flashinfer-bench-main
```

The runner must also add these build settings before `tvm_ffi` compiles the CUDA extension:

```text
TVM_FFI_CUDA_ARCH_LIST=10.0a
CUDA flags:
  -I $flashinfer/data/cutlass/include
  -I $flashinfer/data/cutlass/tools/util/include
  -I FlashInfer/TRTLLM BMM generated headers
  --expt-relaxed-constexpr
  -DTLLM_ENABLE_CUDA
  -DTLLM_GEN_EXPORT_INTERFACE
  -DTLLM_GEN_EXPORT_FLASHINFER
  -DTLLM_GEN_GEMM_CUBIN_PATH="..."
link flags:
  -lcuda -ldl
```

In the local ablation workspace this is handled by `verify.py`.

## Run

Use a runner workspace whose `config.toml` has:

```toml
[build]
language = "cuda"
entry_point = "moe_fp8.cu::run"
destination_passing_style = true
```

Copy this repo's kernel files into the runner:

```bash
rm -rf /path/to/runner/solution/cuda
mkdir -p /path/to/runner/solution
cp -r /path/to/fp8-moe-dev/kernel /path/to/runner/solution/cuda
```

Then benchmark:

```bash
cd /path/to/runner
uv run python verify.py --all --baseline
```

The local command used for the recorded run was equivalent to:

```bash
cd /home/dongyun/workspace/projects/ablation/fp8_moe
uv run python verify.py --all --baseline \
  --dump-json profile/current_all19_verify_py/summary.json
```

MoE tolerance:

```text
atol = 1.0
rtol = 0.3
required_matched_ratio = 0.9
```

## Results

Measured against FlashInfer baseline `flashinfer_wrapper_9sdjf3`:

```text
correctness:             19/19 PASSED
baseline mean latency:   493.728 us
agent mean latency:      335.394 us
ratio of means:          1.472x
mean per-shape speedup:  1.444x
min per-shape speedup:   1.264x
```

Large shapes:

```text
T=11948: 1786.778 us -> 1119.045 us, 1.597x, GEMM1 = our CUTLASS path
T=14107: 2409.706 us -> 1528.301 us, 1.577x, GEMM1 = our CUTLASS path
GEMM2 for both large shapes = TRTLLM BMM
large-shape summed-latency speedup: 1.585x
```

Per-shape table:

| seq_len | uuid | baseline us | agent us | speedup | GEMM1 path |
|---:|---|---:|---:|---:|---|
| 1 | `e05c6c03` | 203.425 | 108.255 | 1.879x | TRTLLM BMM |
| 7 | `b8f4f012` | 212.457 | 143.990 | 1.475x | our CUTLASS |
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
| 11948 | `58a34f27` | 1786.778 | 1119.045 | 1.597x | our CUTLASS |
| 14107 | `5e8dc11c` | 2409.706 | 1528.301 | 1.577x | our CUTLASS |
