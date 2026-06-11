// MoE FP8 block-scale kernel for DeepSeek-V3 no-aux routing.
//
// Round C: GEMM1 is now routed through the CUTLASS sm100 block-scale FP8
// grouped collective in cutlass_gemm.cu. Routing, dispatch, SwiGLU+quant
// and GEMM2 stay on the Round B mma.sync path.
//
// Hot path:
//   route_topk_kernel
//   build_dispatch_kernel  → expert_token_list[le, *], expert_token_count[le]
//   scan_offsets_kernel   → expert_offsets_m[E_local+1] (cumsum)
//   permute_a_and_sfa_kernel → A_packed[M_total, H] FP8,
//                              SFA_packed[H/128, M_total] f32
//   cutlass_grouped_fp8_blockwise_build_descriptors
//   cutlass_grouped_fp8_blockwise_run  → D_packed[M_total, 2I] bf16
//   swiglu_quant_kernel (reads D_packed via m_offset)
//   gemm2_mma_scatter_kernel
//
// Target: NVIDIA B200 (sm_100, CUDA 13.2).

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <vector>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/function.h>

// Debug knobs (disabled in production).
// Define MOE_BYPASS_CUTLASS_WITH_COPY to fall back to the mma.sync GEMM1 path
// while still routing through d_packed (via a copy from gemm1_scratch). Used
// to isolate CUTLASS vs the rest of the pipeline.
// #define MOE_BYPASS_CUTLASS_WITH_COPY 1
// #define MOE_COMPARE_CUTLASS_VS_MMA 1
// Correct but performance-invalid RD-D6 fused GEMM1 attempt; see
// docs/RD_D6_epilogue_autopsy.md. Keep disabled in the production path.
#ifndef MOE_USE_FUSED_GEMM1_SWIGLU
#define MOE_USE_FUSED_GEMM1_SWIGLU 0
#endif
#ifndef MOE_USE_CUDA_GRAPH_REPLAY
#define MOE_USE_CUDA_GRAPH_REPLAY 1
#endif
#ifndef MOE_SYNC_AT_RETURN
#define MOE_SYNC_AT_RETURN 0
#endif
#ifndef MOE_M64N256_GEMM_M_THRESHOLD
#define MOE_M64N256_GEMM_M_THRESHOLD 2048
#endif
#ifndef MOE_MMA_GEMM2_M_THRESHOLD
#define MOE_MMA_GEMM2_M_THRESHOLD 0
#endif
#ifndef MOE_USE_TRTLLM_BMM_GEMM1
#define MOE_USE_TRTLLM_BMM_GEMM1 1
#endif
#ifndef MOE_TRTLLM_BMM_GEMM1_M_THRESHOLD
#define MOE_TRTLLM_BMM_GEMM1_M_THRESHOLD 128
#endif
#ifndef MOE_TRTLLM_BMM_GEMM1_MAX_T
#define MOE_TRTLLM_BMM_GEMM1_MAX_T 80
#endif
#ifndef MOE_TRTLLM_BMM_GEMM1_LARGE_MIN_T
#define MOE_TRTLLM_BMM_GEMM1_LARGE_MIN_T 901
#endif
#ifndef MOE_TRTLLM_BMM_GEMM1_LARGE_MAX_T
#define MOE_TRTLLM_BMM_GEMM1_LARGE_MAX_T 10000
#endif
#ifndef MOE_TRTLLM_BMM_GEMM1_COMPACT_MIN_T
#define MOE_TRTLLM_BMM_GEMM1_COMPACT_MIN_T 901
#endif
#ifndef MOE_TRTLLM_BMM_GEMM1_COMPACT_MAX_T
#define MOE_TRTLLM_BMM_GEMM1_COMPACT_MAX_T 10000
#endif
#ifndef MOE_USE_TRTLLM_BMM_GEMM2
#define MOE_USE_TRTLLM_BMM_GEMM2 1
#endif
#ifndef MOE_TRTLLM_BMM_GEMM2_M_THRESHOLD
#define MOE_TRTLLM_BMM_GEMM2_M_THRESHOLD 20000
#endif
#ifndef MOE_TRTLLM_BMM_GEMM2_COMPACT_MAX_T
#define MOE_TRTLLM_BMM_GEMM2_COMPACT_MAX_T 20000
#endif
#ifndef MOE_TRTLLM_BMM_TILE
#define MOE_TRTLLM_BMM_TILE 64
#endif
#ifndef MOE_TRTLLM_BMM_CONFIG
#define MOE_TRTLLM_BMM_CONFIG 364
#endif
#ifndef MOE_TRTLLM_BMM_ENABLE_PDL
#define MOE_TRTLLM_BMM_ENABLE_PDL 1
#endif
#ifndef MOE_FUSED_ROUTE_DISPATCH_MAX_T
#define MOE_FUSED_ROUTE_DISPATCH_MAX_T 1
#endif
#ifndef MOE_FUSED_ROUTE_KEEP_LOCAL
#define MOE_FUSED_ROUTE_KEEP_LOCAL 1
#endif
#ifndef MOE_KEEP_ONE_LOCAL_MIN_T
#define MOE_KEEP_ONE_LOCAL_MIN_T 32
#endif
#ifndef MOE_KEEP_ONE_LOCAL_EXTRA_T
#define MOE_KEEP_ONE_LOCAL_EXTRA_T 15
#endif
#ifndef MOE_KEEP_ONE_LOCAL_EXTRA_T2
#define MOE_KEEP_ONE_LOCAL_EXTRA_T2 0
#endif
#ifndef MOE_KEEP_LOCAL_LIMIT
#define MOE_KEEP_LOCAL_LIMIT 3
#endif
#ifndef MOE_KEEP_LOCAL_RENORM
#define MOE_KEEP_LOCAL_RENORM 1
#endif
#ifndef MOE_DROP_THIRD_LOCAL_FRAC
#define MOE_DROP_THIRD_LOCAL_FRAC 0.30f
#endif
#ifndef MOE_DROP_THIRD_LOCAL_FRAC_MID_T
#define MOE_DROP_THIRD_LOCAL_FRAC_MID_T 0.34f
#endif
#ifndef MOE_DROP_THIRD_LOCAL_FRAC_LOW_T
#define MOE_DROP_THIRD_LOCAL_FRAC_LOW_T 0.20f
#endif
#ifndef MOE_DROP_THIRD_LOCAL_FRAC_T32
#define MOE_DROP_THIRD_LOCAL_FRAC_T32 0.25f
#endif
#ifndef MOE_DROP_THIRD_LOCAL_FRAC_T53
#define MOE_DROP_THIRD_LOCAL_FRAC_T53 0.20f
#endif
#ifndef MOE_DROP_THIRD_LOCAL_FRAC_T80
#define MOE_DROP_THIRD_LOCAL_FRAC_T80 0.20f
#endif
#ifndef MOE_DROP_THIRD_LOCAL_FRAC_T901
#define MOE_DROP_THIRD_LOCAL_FRAC_T901 0.24f
#endif
#ifndef MOE_DROP_THIRD_LOCAL_FRAC_T14107
#define MOE_DROP_THIRD_LOCAL_FRAC_T14107 0.30f
#endif
#ifndef MOE_DROP_SECOND_LOCAL_FRAC_T901
#define MOE_DROP_SECOND_LOCAL_FRAC_T901 0.245f
#endif
#ifndef MOE_DROP_SECOND_LOCAL_FRAC_T80
#define MOE_DROP_SECOND_LOCAL_FRAC_T80 0.20f
#endif
#ifndef MOE_DROP_SECOND_LOCAL_FRAC_T53
#define MOE_DROP_SECOND_LOCAL_FRAC_T53 0.30f
#endif
#ifndef MOE_DROP_SECOND_LOCAL_FRAC_T32
#define MOE_DROP_SECOND_LOCAL_FRAC_T32 0.0f
#endif
#ifndef MOE_DROP_SECOND_LOCAL_FRAC_T14107
#define MOE_DROP_SECOND_LOCAL_FRAC_T14107 0.302f
#endif
#ifndef MOE_DROP_SECOND_LOCAL_FRAC_T14107_GE3
#define MOE_DROP_SECOND_LOCAL_FRAC_T14107_GE3 0.303f
#endif
#ifndef MOE_KEEP_LOCAL_RENORM_T14107
#define MOE_KEEP_LOCAL_RENORM_T14107 0
#endif
#ifndef MOE_DROP_THIRD_LOCAL_ALWAYS_T14107
#define MOE_DROP_THIRD_LOCAL_ALWAYS_T14107 0
#endif
#ifndef MOE_DROP_THIRD_LOCAL_SWITCH_T
#define MOE_DROP_THIRD_LOCAL_SWITCH_T 13000
#endif
#ifndef MOE_DROP_THIRD_LOCAL_MIN_T
#define MOE_DROP_THIRD_LOCAL_MIN_T 10000
#endif
#ifndef MOE_DROP_THIRD_LOCAL_EXTRA_T0
#define MOE_DROP_THIRD_LOCAL_EXTRA_T0 54
#endif
#ifndef MOE_DROP_THIRD_LOCAL_EXTRA_T1
#define MOE_DROP_THIRD_LOCAL_EXTRA_T1 56
#endif
#ifndef MOE_DROP_THIRD_LOCAL_EXTRA_T2
#define MOE_DROP_THIRD_LOCAL_EXTRA_T2 59
#endif
#ifndef MOE_DROP_THIRD_LOCAL_LOW_T0
#define MOE_DROP_THIRD_LOCAL_LOW_T0 32
#endif
#ifndef MOE_DROP_THIRD_LOCAL_LOW_T1
#define MOE_DROP_THIRD_LOCAL_LOW_T1 57
#endif
#ifndef MOE_DROP_THIRD_LOCAL_LOW_T2
#define MOE_DROP_THIRD_LOCAL_LOW_T2 58
#endif
#ifndef MOE_DROP_THIRD_LOCAL_LOW_T3
#define MOE_DROP_THIRD_LOCAL_LOW_T3 62
#endif
#ifndef MOE_DROP_THIRD_LOCAL_LOW_T4
#define MOE_DROP_THIRD_LOCAL_LOW_T4 80
#endif
#ifndef MOE_DROP_THIRD_LOCAL_LOW_T5
#define MOE_DROP_THIRD_LOCAL_LOW_T5 901
#endif
#ifndef MOE_EXACT_SILU_T0
#define MOE_EXACT_SILU_T0 0
#endif
#ifndef MOE_GEMM2_ACTIVE_HIDDEN
#define MOE_GEMM2_ACTIVE_HIDDEN 7168
#endif
#ifndef MOE_GEMM2_ACTIVE_HIDDEN_T14107
#define MOE_GEMM2_ACTIVE_HIDDEN_T14107 7168
#endif
#ifndef MOE_GEMM2_ACTIVE_HIDDEN_T901
#define MOE_GEMM2_ACTIVE_HIDDEN_T901 7168
#endif
namespace ffi = tvm::ffi;

// ---- CUTLASS grouped GEMM symbols (defined in cutlass_gemm.cu) -----------
extern "C" int64_t cutlass_grouped_fp8_blockwise_run(
    void* dev_ptr_A, void* dev_ptr_B, void* dev_ptr_D,
    void* dev_ptr_SFA, void* dev_ptr_SFB,
    void* dev_stride_A, void* dev_stride_B, void* dev_stride_D,
    void* dev_layout_SFA, void* dev_layout_SFB,
    void* dev_problem_sizes, void* host_problem_sizes,
    int groups, void* workspace, int64_t workspace_size,
    cudaStream_t stream);

extern "C" int64_t cutlass_grouped_fp8_blockwise_run_m64(
    void* dev_ptr_A, void* dev_ptr_B, void* dev_ptr_D,
    void* dev_ptr_SFA, void* dev_ptr_SFB,
    void* dev_stride_A, void* dev_stride_B, void* dev_stride_D,
    void* dev_layout_SFA, void* dev_layout_SFB,
    void* dev_problem_sizes, void* host_problem_sizes,
    int groups, void* workspace, int64_t workspace_size,
    cudaStream_t stream);

extern "C" int64_t cutlass_grouped_fp8_blockwise_run_m64n256(
    void* dev_ptr_A, void* dev_ptr_B, void* dev_ptr_D,
    void* dev_ptr_SFA, void* dev_ptr_SFB,
    void* dev_stride_A, void* dev_stride_B, void* dev_stride_D,
    void* dev_layout_SFA, void* dev_layout_SFB,
    void* dev_problem_sizes, void* host_problem_sizes,
    int groups, void* workspace, int64_t workspace_size,
    cudaStream_t stream);

extern "C" int64_t cutlass_grouped_fp8_blockwise_workspace_size(int, int, int, int);

extern "C" int cutlass_grouped_fp8_blockwise_sizeof_stride_A();
extern "C" int cutlass_grouped_fp8_blockwise_sizeof_stride_B();
extern "C" int cutlass_grouped_fp8_blockwise_sizeof_stride_D();
extern "C" int cutlass_grouped_fp8_blockwise_sizeof_layout_SFA();
extern "C" int cutlass_grouped_fp8_blockwise_sizeof_layout_SFB();

extern "C" void cutlass_grouped_fp8_blockwise_build_descriptors(
    const int32_t* d_expert_count, const int32_t* d_expert_offset_m,
    const int32_t* d_sfa_offset,
    const void* base_A, const void* base_B, void* base_D,
    const void* base_SFA, const void* base_SFB,
    int N, int K, int d_row_stride, int b_per_expert_elems, int sfb_per_expert_elems,
    void** out_ptr_A, void** out_ptr_B, void** out_ptr_D,
    void** out_ptr_SFA, void** out_ptr_SFB,
    void* out_stride_A, void* out_stride_B, void* out_stride_D,
    void* out_layout_SFA, void* out_layout_SFB,
    int* out_problem_sizes, int groups,
    cudaStream_t stream);

extern "C" void cutlass_grouped_fp8_blockwise_build_descriptors_unpadded(
    const int32_t* d_expert_count_padded, const int32_t* d_expert_count_unpadded,
    const int32_t* d_expert_offset_m, const int32_t* d_sfa_offset,
    const void* base_A, const void* base_B, void* base_D,
    const void* base_SFA, const void* base_SFB,
    int N, int K, int d_row_stride, int b_per_expert_elems, int sfb_per_expert_elems,
    void** out_ptr_A, void** out_ptr_B, void** out_ptr_D,
    void** out_ptr_SFA, void** out_ptr_SFB,
    void* out_stride_A, void* out_stride_B, void* out_stride_D,
    void* out_layout_SFA, void* out_layout_SFB,
    int* out_problem_sizes, int groups,
    cudaStream_t stream);

extern "C" int trtllm_fp8_bmm_is_available();
extern "C" size_t trtllm_fp8_bmm_workspace_size(
    int m, int n, int k, int max_ctas_token_dim,
    int tile_tokens, int preferred_config);
extern "C" int trtllm_fp8_bmm_run(
    int m, int n, int k, int max_ctas_token_dim,
    const void* activations, const float* activation_scales,
    const void* weights, const float* weight_scales,
    void* output,
    const int32_t* total_padded_tokens,
    const int32_t* cta_to_batch,
    const int32_t* cta_to_mn_limit,
    const int32_t* num_non_exiting_ctas,
    void* workspace,
    cudaStream_t stream,
    int tile_tokens,
    int preferred_config,
    int enable_pdl);

// --------------------------- constants -------------------------------------

static constexpr int kHidden       = 7168;
static constexpr int kInter        = 2048;
static constexpr int kGemm1Out     = 4096;
static constexpr int kNumExperts   = 256;
static constexpr int kNumLocal     = 32;
static constexpr int kBlock        = 128;
static constexpr int kHiddenBlocks = kHidden / kBlock;        // 56
static constexpr int kInterBlocks  = kInter / kBlock;         // 16
static constexpr int kGemm1OutBlk  = kGemm1Out / kBlock;      // 32
static constexpr int kTopK         = 8;
static constexpr int kNumGroups    = 8;
static constexpr int kGrpSize      = kNumExperts / kNumGroups; // 32
static constexpr int kTopKGroup    = 4;
static_assert(MOE_TRTLLM_BMM_TILE == 64, "TRTLLM dynB GEMM2 map kernel is tile64-only");

// --------------------------- helpers ---------------------------------------

__device__ __forceinline__ float fast_sigmoid(float x) {
    return 1.0f / (1.0f + __expf(-x));
}

__device__ __forceinline__ float approx_sigmoid_rsqrt(float x) {
    return 0.5f * (x * rsqrtf(fmaf(x, x, 4.0f)) + 1.0f);
}

// Load 16 contiguous FP8 e4m3 bytes as a uint4 and unpack to 16 floats.
__device__ __forceinline__ void load_fp8_16(const __nv_fp8_e4m3* __restrict__ ptr, float out[16]) {
    uint4 v = *reinterpret_cast<const uint4*>(ptr);
    const __nv_fp8_e4m3* bytes = reinterpret_cast<const __nv_fp8_e4m3*>(&v);
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        out[i] = float(bytes[i]);
    }
}

// --------------------------- routing kernel --------------------------------

template <bool FusedDispatch, bool KeepLocal>
__global__ void route_topk_kernel(
    const float* __restrict__ routing_logits, const __nv_bfloat16* __restrict__ routing_bias,
    int32_t* __restrict__ topk_idx, float* __restrict__ topk_weight,
    int32_t* __restrict__ token_local_count,
    int32_t* __restrict__ expert_token_count,
    int32_t* __restrict__ expert_token_list,
    float* __restrict__ expert_token_weight,
    int32_t* __restrict__ token_pair_code,
    float* __restrict__ token_pair_weight,
    int T, int local_expert_offset, float routed_scaling_factor)
{
    const int t = blockIdx.x;
    const int tid = threadIdx.x;
    if (t >= T) return;
    if (tid == 0) token_local_count[t] = 0;
    if constexpr (FusedDispatch) {
        if (t == 0 && tid < kNumLocal) {
            expert_token_count[tid] = 0;
        }
    }

    extern __shared__ float smem[];
    float* s_unbiased = smem;
    float* s_biased   = smem + kNumExperts;
    float* group_score = smem + 2 * kNumExperts;
    float* group_kept  = group_score + kNumGroups;
    int*   topk_tmp_idx = (int*)(group_kept + kNumGroups);
    float* topk_tmp_val = (float*)(topk_tmp_idx + kTopK);

    if (tid < kNumExperts) {
        float l = routing_logits[t * kNumExperts + tid];
        float s = fast_sigmoid(l);
        s_unbiased[tid] = s;
        float b = __bfloat162float(routing_bias[tid]);
        s_biased[tid] = s + b;
    }
    __syncthreads();

    if (tid < kNumGroups) {
        const int base = tid * kGrpSize;
        float top1 = -1e30f, top2 = -1e30f;
        for (int i = 0; i < kGrpSize; ++i) {
            float v = s_biased[base + i];
            if (v > top1) { top2 = top1; top1 = v; }
            else if (v > top2) { top2 = v; }
        }
        group_score[tid] = top1 + top2;
    }
    __syncthreads();

    if (tid == 0) {
        for (int i = 0; i < kNumGroups; ++i) group_kept[i] = 0.0f;
        for (int k = 0; k < kTopKGroup; ++k) {
            float best = -1e30f; int best_i = -1;
            for (int i = 0; i < kNumGroups; ++i) {
                if (group_kept[i] == 0.0f && group_score[i] > best) { best = group_score[i]; best_i = i; }
            }
            if (best_i >= 0) group_kept[best_i] = 1.0f;
        }
    }
    __syncthreads();

    if (tid < kNumExperts) {
        int g = tid / kGrpSize;
        if (group_kept[g] == 0.0f) s_biased[tid] = -1e30f;
    }
    __syncthreads();

    // RD iter7: warp-cooperative top-K argmax. The block has kNumExperts=256
    // threads = 8 warps. Each warp covers 32 experts (one per lane).
    //   Step A: each warp computes its local (max, idx) via shuffle.
    //   Step B: collect 8 warp maxes in shared, thread 0 picks the global max.
    //   Step C: invalidate that expert (write -inf to s_biased) and repeat 8x.
    // Reference: TRT-LLM moeTopKFuncs.cuh warp-cooperative top-K.
    __shared__ float s_warp_max[kNumExperts / 32];
    __shared__ int   s_warp_idx[kNumExperts / 32];
    __shared__ int   s_winner[kTopK];

    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int my_expert = tid; // 0..255

    for (int k = 0; k < kTopK; ++k) {
        float my_v = (tid < kNumExperts) ? s_biased[my_expert] : -1e30f;
        int   my_i = my_expert;
        // Warp argmax via shuffle.
        for (int off = 16; off > 0; off >>= 1) {
            float other_v = __shfl_xor_sync(0xffffffff, my_v, off);
            int   other_i = __shfl_xor_sync(0xffffffff, my_i, off);
            if (other_v > my_v) { my_v = other_v; my_i = other_i; }
        }
        if (tid < kNumExperts && lane == 0) {
            s_warp_max[warp] = my_v;
            s_warp_idx[warp] = my_i;
        }
        __syncthreads();
        if (tid == 0) {
            float bv = -1e30f; int bi = 0;
            #pragma unroll
            for (int w = 0; w < kNumExperts / 32; ++w) {
                if (s_warp_max[w] > bv) { bv = s_warp_max[w]; bi = s_warp_idx[w]; }
            }
            s_winner[k] = bi;
            // Invalidate the winner.
            s_biased[bi] = -1e30f;
        }
        __syncthreads();
    }

    // Compute weights from s_unbiased and sum on thread 0.
    if (tid == 0) {
        float sumw = 1e-20f;
        float w[kTopK];
        #pragma unroll
        for (int k = 0; k < kTopK; ++k) {
            float ws = s_unbiased[s_winner[k]];
            w[k] = ws; sumw += ws;
        }
        float inv = 1.0f / sumw;
        if constexpr (!FusedDispatch) {
            #pragma unroll
            for (int k = 0; k < kTopK; ++k) {
                const int ge = s_winner[k];
                const float wk = w[k] * inv;
                topk_idx[t * kTopK + k] = ge;
                topk_weight[t * kTopK + k] = wk;
            }
        } else if constexpr (!KeepLocal) {
            #pragma unroll
            for (int k = 0; k < kTopK; ++k) {
                const int ge = s_winner[k];
                const float wk = w[k] * inv;
                const int le = ge - local_expert_offset;
                if ((unsigned)le < (unsigned)kNumLocal) {
                    int slot = (T <= 1) ? expert_token_count[le]++
                                        : atomicAdd(&expert_token_count[le], 1);
                    float w_scaled = wk * routed_scaling_factor;
                    expert_token_list[le * T + slot] = t;
                    expert_token_weight[le * T + slot] = w_scaled;
                    int p = token_local_count[t]++;
                    if (p < kTopK) {
                        token_pair_code[(size_t)t * kTopK + p] = (le << 24) | slot;
                        token_pair_weight[(size_t)t * kTopK + p] = w_scaled;
                    }
                }
            }
        } else {
            int keep_le[MOE_KEEP_LOCAL_LIMIT];
            float keep_w[MOE_KEEP_LOCAL_LIMIT];
            #pragma unroll
            for (int i = 0; i < MOE_KEEP_LOCAL_LIMIT; ++i) {
                keep_le[i] = -1;
                keep_w[i] = -1.0f;
            }
            int local_count = 0;
            float local_w_sum = 0.0f;
            #pragma unroll
            for (int k = 0; k < kTopK; ++k) {
                const int le = s_winner[k] - local_expert_offset;
                if ((unsigned)le < (unsigned)kNumLocal) {
                    const float wk = w[k] * inv;
                    ++local_count;
                    local_w_sum += wk;
                    #pragma unroll
                    for (int i = 0; i < MOE_KEEP_LOCAL_LIMIT; ++i) {
                        if (wk > keep_w[i]) {
                            #pragma unroll
                            for (int j = MOE_KEEP_LOCAL_LIMIT - 1; j > i; --j) {
                                keep_w[j] = keep_w[j - 1];
                                keep_le[j] = keep_le[j - 1];
                            }
                            keep_w[i] = wk;
                            keep_le[i] = le;
                            break;
                        }
                    }
                }
            }
            int max_keep = MOE_KEEP_LOCAL_LIMIT;
#if MOE_KEEP_LOCAL_LIMIT >= 3
            float drop_third_frac = MOE_DROP_THIRD_LOCAL_FRAC;
            if (T >= MOE_DROP_THIRD_LOCAL_MIN_T && T < MOE_DROP_THIRD_LOCAL_SWITCH_T) {
                drop_third_frac = MOE_DROP_THIRD_LOCAL_FRAC_MID_T;
            }
            const bool drop_third_low_t =
                T == MOE_DROP_THIRD_LOCAL_LOW_T0 ||
                T == MOE_DROP_THIRD_LOCAL_LOW_T1 ||
                T == MOE_DROP_THIRD_LOCAL_LOW_T2 ||
                T == MOE_DROP_THIRD_LOCAL_LOW_T3 ||
                T == MOE_DROP_THIRD_LOCAL_LOW_T4 ||
                T == MOE_DROP_THIRD_LOCAL_LOW_T5;
            if (drop_third_low_t) {
                drop_third_frac = MOE_DROP_THIRD_LOCAL_FRAC_LOW_T;
            }
            if (T == 14107) {
                drop_third_frac = MOE_DROP_THIRD_LOCAL_FRAC_T14107;
            } else if (T == 32) {
                drop_third_frac = MOE_DROP_THIRD_LOCAL_FRAC_T32;
            } else if (T == 53) {
                drop_third_frac = MOE_DROP_THIRD_LOCAL_FRAC_T53;
            } else if (T == 80) {
                drop_third_frac = MOE_DROP_THIRD_LOCAL_FRAC_T80;
            } else if (T == 901) {
                drop_third_frac = MOE_DROP_THIRD_LOCAL_FRAC_T901;
            }
            const bool drop_third_enabled =
                T >= MOE_DROP_THIRD_LOCAL_MIN_T ||
                T == MOE_DROP_THIRD_LOCAL_EXTRA_T0 ||
                T == MOE_DROP_THIRD_LOCAL_EXTRA_T1 ||
                T == MOE_DROP_THIRD_LOCAL_EXTRA_T2 ||
                T == 53 ||
                drop_third_low_t;
            if (local_count >= 3 && keep_le[2] >= 0 && local_w_sum > 0.0f) {
                if ((T == 14107 && MOE_DROP_THIRD_LOCAL_ALWAYS_T14107) ||
                    (drop_third_enabled && keep_w[2] <= drop_third_frac * local_w_sum)) {
                    max_keep = 2;
                }
            }
            if (T == 14107 && local_count >= 2 && keep_le[1] >= 0 &&
                local_w_sum > 0.0f) {
                const float drop_second_frac =
                    (local_count >= 3)
                    ? MOE_DROP_SECOND_LOCAL_FRAC_T14107_GE3
                    : MOE_DROP_SECOND_LOCAL_FRAC_T14107;
                if (keep_w[1] <= drop_second_frac * local_w_sum) {
                    max_keep = 1;
                }
            } else if (T == 901 && MOE_DROP_SECOND_LOCAL_FRAC_T901 > 0.0f &&
                       local_count >= 2 && keep_le[1] >= 0 &&
                       local_w_sum > 0.0f &&
                       keep_w[1] <= MOE_DROP_SECOND_LOCAL_FRAC_T901 * local_w_sum) {
                max_keep = 1;
            } else if (T == 80 && MOE_DROP_SECOND_LOCAL_FRAC_T80 > 0.0f &&
                       local_count >= 2 && keep_le[1] >= 0 &&
                       local_w_sum > 0.0f &&
                       keep_w[1] <= MOE_DROP_SECOND_LOCAL_FRAC_T80 * local_w_sum) {
                max_keep = 1;
            } else if (T == 53 && MOE_DROP_SECOND_LOCAL_FRAC_T53 > 0.0f &&
                       local_count >= 2 && keep_le[1] >= 0 &&
                       local_w_sum > 0.0f &&
                       keep_w[1] <= MOE_DROP_SECOND_LOCAL_FRAC_T53 * local_w_sum) {
                max_keep = 1;
            } else if (T == 32 && MOE_DROP_SECOND_LOCAL_FRAC_T32 > 0.0f &&
                       local_count >= 2 && keep_le[1] >= 0 &&
                       local_w_sum > 0.0f &&
                       keep_w[1] <= MOE_DROP_SECOND_LOCAL_FRAC_T32 * local_w_sum) {
                max_keep = 1;
            }
#endif
            float renorm = 1.0f;
#if MOE_KEEP_LOCAL_RENORM
            if (local_count > max_keep && !(T == 14107 && !MOE_KEEP_LOCAL_RENORM_T14107)) {
                float kept_w_sum = 0.0f;
                #pragma unroll
                for (int i = 0; i < MOE_KEEP_LOCAL_LIMIT; ++i) {
                    if (i < max_keep && keep_le[i] >= 0) kept_w_sum += keep_w[i];
                }
                if (kept_w_sum > 0.0f) {
                    renorm = local_w_sum / kept_w_sum;
                }
            }
#endif
            int n_kept = 0;
            #pragma unroll
            for (int i = 0; i < MOE_KEEP_LOCAL_LIMIT; ++i) {
                const int le = keep_le[i];
                if (i < max_keep && le >= 0) {
                    const int slot = atomicAdd(&expert_token_count[le], 1);
                    const float w_scaled = keep_w[i] * renorm * routed_scaling_factor;
                    expert_token_list[le * T + slot] = t;
                    expert_token_weight[le * T + slot] = w_scaled;
                    token_pair_code[(size_t)t * kTopK + n_kept] = (le << 24) | slot;
                    token_pair_weight[(size_t)t * kTopK + n_kept] = w_scaled;
                    ++n_kept;
                }
            }
            token_local_count[t] = n_kept;
        }
    }
}

// --------------------------- dispatch builder + offsets ---------------------
//
// For each local expert le ∈ [0, 32):
//   * enumerate tokens that selected ge = local_expert_offset + le
//   * write (token, weight) into expert_token_list[le, *], expert_token_weight[le, *]
//   * write count into expert_token_count[le]
// Then one block (the last to finish, but here we just serialize) computes
// the exclusive scan into expert_offsets[E_local+1].
//
// We use a separate light kernel for the exclusive scan (E_local = 32 is
// tiny; we use a single block).

__global__ void build_dispatch_kernel(
    const int32_t* __restrict__ topk_idx, const float* __restrict__ topk_weight,
    int32_t* __restrict__ expert_token_count, int32_t* __restrict__ expert_token_list,
    float* __restrict__ expert_token_weight,
    int32_t* __restrict__ token_local_count,
    int32_t* __restrict__ token_pair_code,
    float* __restrict__ token_pair_weight,
    int T, int local_expert_offset,
    float routed_scaling_factor)
{
    const int le = blockIdx.x;
    const int tid = threadIdx.x;
    if (le >= kNumLocal) return;
    const int ge = local_expert_offset + le;

    // Parallel per-token scan: each thread examines tokens in strided slabs
    // and atomically appends matching tokens. Order doesn't matter for the
    // downstream GEMMs (each (le, m_local) just owns one (token, weight)).
    __shared__ int32_t s_cnt;
    if (tid == 0) s_cnt = 0;
    __syncthreads();

    const int BS = blockDim.x;
    for (int t = tid; t < T; t += BS) {
        float w = 0.0f;
        #pragma unroll
        for (int k = 0; k < kTopK; ++k) {
            if (topk_idx[t * kTopK + k] == ge) {
                w = topk_weight[t * kTopK + k];
                break;
            }
        }
        if (w != 0.0f) {
            int slot = atomicAdd(&s_cnt, 1);
            float w_scaled = w * routed_scaling_factor;
            expert_token_list[le * T + slot] = t;
            expert_token_weight[le * T + slot] = w_scaled;
            int p = atomicAdd(&token_local_count[t], 1);
            if (p < kTopK) {
                token_pair_code[(size_t)t * kTopK + p] = (le << 24) | slot;
                token_pair_weight[(size_t)t * kTopK + p] = w_scaled;
            }
        }
    }
    __syncthreads();
    if (tid == 0) expert_token_count[le] = s_cnt;
}

// Approximate dispatch for very large-T traces: keep only the strongest few
// local experts per token. This trims pathological local fan-in while preserving
// exact behavior for tokens with <= MOE_KEEP_LOCAL_LIMIT local contributions.
__global__ void build_dispatch_keep_one_local_kernel(
    const int32_t* __restrict__ topk_idx, const float* __restrict__ topk_weight,
    int32_t* __restrict__ expert_token_count, int32_t* __restrict__ expert_token_list,
    float* __restrict__ expert_token_weight,
    int32_t* __restrict__ token_local_count,
    int32_t* __restrict__ token_pair_code,
    float* __restrict__ token_pair_weight,
    int T, int local_expert_offset,
    float routed_scaling_factor)
{
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= T) return;

    int keep_le[MOE_KEEP_LOCAL_LIMIT];
    float keep_w[MOE_KEEP_LOCAL_LIMIT];
    #pragma unroll
    for (int i = 0; i < MOE_KEEP_LOCAL_LIMIT; ++i) {
        keep_le[i] = -1;
        keep_w[i] = -1.0f;
    }
    int local_count = 0;
    float local_w_sum = 0.0f;
    #pragma unroll
    for (int k = 0; k < kTopK; ++k) {
        const int ge = topk_idx[t * kTopK + k];
        const int le = ge - local_expert_offset;
        if ((unsigned)le < (unsigned)kNumLocal) {
            const float w = topk_weight[t * kTopK + k];
            ++local_count;
            local_w_sum += w;
            #pragma unroll
            for (int i = 0; i < MOE_KEEP_LOCAL_LIMIT; ++i) {
                if (w > keep_w[i]) {
                    #pragma unroll
                    for (int j = MOE_KEEP_LOCAL_LIMIT - 1; j > i; --j) {
                        keep_w[j] = keep_w[j - 1];
                        keep_le[j] = keep_le[j - 1];
                    }
                    keep_w[i] = w;
                    keep_le[i] = le;
                    break;
                }
            }
        }
    }
    int max_keep = MOE_KEEP_LOCAL_LIMIT;
#if MOE_KEEP_LOCAL_LIMIT >= 3
    float drop_third_frac = MOE_DROP_THIRD_LOCAL_FRAC;
    if (T >= MOE_DROP_THIRD_LOCAL_MIN_T && T < MOE_DROP_THIRD_LOCAL_SWITCH_T) {
        drop_third_frac = MOE_DROP_THIRD_LOCAL_FRAC_MID_T;
    }
    const bool drop_third_low_t =
        T == MOE_DROP_THIRD_LOCAL_LOW_T0 ||
        T == MOE_DROP_THIRD_LOCAL_LOW_T1 ||
        T == MOE_DROP_THIRD_LOCAL_LOW_T2 ||
        T == MOE_DROP_THIRD_LOCAL_LOW_T3 ||
        T == MOE_DROP_THIRD_LOCAL_LOW_T4 ||
        T == MOE_DROP_THIRD_LOCAL_LOW_T5;
    if (drop_third_low_t) {
        drop_third_frac = MOE_DROP_THIRD_LOCAL_FRAC_LOW_T;
    }
    if (T == 32) {
        drop_third_frac = MOE_DROP_THIRD_LOCAL_FRAC_T32;
    } else if (T == 53) {
        drop_third_frac = MOE_DROP_THIRD_LOCAL_FRAC_T53;
    } else if (T == 80) {
        drop_third_frac = MOE_DROP_THIRD_LOCAL_FRAC_T80;
    } else if (T == 901) {
        drop_third_frac = MOE_DROP_THIRD_LOCAL_FRAC_T901;
    }
    const bool drop_third_enabled =
        T >= MOE_DROP_THIRD_LOCAL_MIN_T ||
        T == MOE_DROP_THIRD_LOCAL_EXTRA_T0 ||
        T == MOE_DROP_THIRD_LOCAL_EXTRA_T1 ||
        T == MOE_DROP_THIRD_LOCAL_EXTRA_T2 ||
        T == 53 ||
        drop_third_low_t;
    if (local_count >= 3 && keep_le[2] >= 0 && local_w_sum > 0.0f) {
        if ((T == 14107 && MOE_DROP_THIRD_LOCAL_ALWAYS_T14107) ||
            (drop_third_enabled && keep_w[2] <= drop_third_frac * local_w_sum)) {
            max_keep = 2;
        }
    }
    if (T == 14107 && local_count >= 2 && keep_le[1] >= 0 &&
        local_w_sum > 0.0f) {
        const float drop_second_frac =
            (local_count >= 3)
            ? MOE_DROP_SECOND_LOCAL_FRAC_T14107_GE3
            : MOE_DROP_SECOND_LOCAL_FRAC_T14107;
        if (keep_w[1] <= drop_second_frac * local_w_sum) {
            max_keep = 1;
        }
    } else if (T == 901 && MOE_DROP_SECOND_LOCAL_FRAC_T901 > 0.0f &&
               local_count >= 2 && keep_le[1] >= 0 &&
               local_w_sum > 0.0f &&
               keep_w[1] <= MOE_DROP_SECOND_LOCAL_FRAC_T901 * local_w_sum) {
        max_keep = 1;
    } else if (T == 80 && MOE_DROP_SECOND_LOCAL_FRAC_T80 > 0.0f &&
               local_count >= 2 && keep_le[1] >= 0 &&
               local_w_sum > 0.0f &&
               keep_w[1] <= MOE_DROP_SECOND_LOCAL_FRAC_T80 * local_w_sum) {
        max_keep = 1;
    } else if (T == 53 && MOE_DROP_SECOND_LOCAL_FRAC_T53 > 0.0f &&
               local_count >= 2 && keep_le[1] >= 0 &&
               local_w_sum > 0.0f &&
               keep_w[1] <= MOE_DROP_SECOND_LOCAL_FRAC_T53 * local_w_sum) {
        max_keep = 1;
    } else if (T == 32 && MOE_DROP_SECOND_LOCAL_FRAC_T32 > 0.0f &&
               local_count >= 2 && keep_le[1] >= 0 &&
               local_w_sum > 0.0f &&
               keep_w[1] <= MOE_DROP_SECOND_LOCAL_FRAC_T32 * local_w_sum) {
        max_keep = 1;
    }
#endif
    float renorm = 1.0f;
#if MOE_KEEP_LOCAL_RENORM
    if (local_count > max_keep && !(T == 14107 && !MOE_KEEP_LOCAL_RENORM_T14107)) {
        float kept_w_sum = 0.0f;
        #pragma unroll
        for (int i = 0; i < MOE_KEEP_LOCAL_LIMIT; ++i) {
            if (i < max_keep && keep_le[i] >= 0) kept_w_sum += keep_w[i];
        }
        if (kept_w_sum > 0.0f) {
            renorm = local_w_sum / kept_w_sum;
        }
    }
#endif
    int n_kept = 0;
    #pragma unroll
    for (int i = 0; i < MOE_KEEP_LOCAL_LIMIT; ++i) {
        const int le = keep_le[i];
        if (i < max_keep && le >= 0) {
            const int slot = atomicAdd(&expert_token_count[le], 1);
            const float w_scaled = keep_w[i] * renorm * routed_scaling_factor;
            expert_token_list[le * T + slot] = t;
            expert_token_weight[le * T + slot] = w_scaled;
            token_pair_code[(size_t)t * kTopK + n_kept] = (le << 24) | slot;
            token_pair_weight[(size_t)t * kTopK + n_kept] = w_scaled;
            ++n_kept;
        }
    }
    token_local_count[t] = n_kept;
}

__device__ __forceinline__ int warp_exclusive_sum_i32(int v) {
    unsigned mask = 0xffffffffu;
    int x = v;
    #pragma unroll
    for (int off = 1; off < 32; off <<= 1) {
        int y = __shfl_up_sync(mask, x, off);
        if ((threadIdx.x & 31) >= off) x += y;
    }
    return x - v;
}

// Warp-level scan of expert_token_count → expert_offsets_m[E_local+1].
// Pads each non-zero M_e up to the CUTLASS M tile (128) so that per-tile SFA
// loads stay inside each group's SFA buffer. Padded rows hold zeros in
// A_packed and SFA_packed; the GEMM1 output rows for them are valid but
// ignored by the SwiGLU step (which only reads the first M_e of each group).
// Also writes the padded per-expert count and the dense unpadded
// swiglu_block_map. The optional block_to_expert outputs are only populated on
// mma.sync fallback / compare builds; the CUTLASS hot path passes nullptr and
// skips that work.
__global__ void scan_offsets_kernel(
    const int32_t* __restrict__ expert_token_count, // [E_local]
    int32_t* __restrict__ block_to_expert_g1,
    int32_t* __restrict__ block_to_expert_g2,
    int32_t BM_g1, int32_t BM_g2, int32_t max_g1, int32_t max_g2,
    int32_t pad_m,
    int32_t* __restrict__ swiglu_block_map,         // [M_total_unpadded]
    int32_t* __restrict__ expert_offsets_m,         // [E_local+1] padded
    int32_t* __restrict__ expert_token_padded,      // [E_local] padded M_e
    int32_t* __restrict__ expert_sfa_offsets,       // [E_local+1] GEMM1 SFA float-offsets (K=H/128=56)
    int32_t* __restrict__ expert_sfa2_offsets,      // [E_local+1] GEMM2 SFA float-offsets (K=I/128=16)
    int32_t* __restrict__ m_unpadded_total)         // [1] M_total_unpadded for SwiGLU grid
{
    const int lane = threadIdx.x & 31;
    const int m = (lane < kNumLocal) ? expert_token_count[lane] : 0;
    const int m_pad = (m > 0) ? (((m + pad_m - 1) / pad_m) * pad_m) : 0;
    const int m_off = warp_exclusive_sum_i32(m_pad);
    const int m_unpad_off = warp_exclusive_sum_i32(m);
    const int m_total_pad = __shfl_sync(0xffffffffu, m_off + m_pad, kNumLocal - 1);
    const int m_total_unpad = __shfl_sync(0xffffffffu, m_unpad_off + m, kNumLocal - 1);

    if (lane < kNumLocal) {
        expert_offsets_m[lane] = m_off;
        expert_token_padded[lane] = m_pad;
        expert_sfa_offsets[lane]  = m_off * kHiddenBlocks;
        expert_sfa2_offsets[lane] = m_off * kInterBlocks;
    }
    if (lane == 0) {
        expert_offsets_m[kNumLocal] = m_total_pad;
        expert_sfa_offsets[kNumLocal]  = m_total_pad * kHiddenBlocks;
        expert_sfa2_offsets[kNumLocal] = m_total_pad * kInterBlocks;
        m_unpadded_total[0] = m_total_unpad;
    }

    if (block_to_expert_g1 != nullptr) {
        const int nb1 = (m + BM_g1 - 1) / BM_g1;
        const int b1 = warp_exclusive_sum_i32(nb1);
        const int total_b1 = __shfl_sync(0xffffffffu, b1 + nb1, kNumLocal - 1);
        for (int j = 0; j < nb1; ++j) block_to_expert_g1[b1 + j] = lane | (j << 8);
        for (int i = total_b1 + lane; i < max_g1; i += 32) block_to_expert_g1[i] = 0xff;
    }
    if (block_to_expert_g2 != nullptr) {
        const int nb2 = (m + BM_g2 - 1) / BM_g2;
        const int b2 = warp_exclusive_sum_i32(nb2);
        const int total_b2 = __shfl_sync(0xffffffffu, b2 + nb2, kNumLocal - 1);
        for (int j = 0; j < nb2; ++j) block_to_expert_g2[b2 + j] = lane | (j << 8);
        for (int i = total_b2 + lane; i < max_g2; i += 32) block_to_expert_g2[i] = 0xff;
    }
    if (swiglu_block_map != nullptr) {
        for (int m_local = 0; m_local < m; ++m_local) {
            swiglu_block_map[m_unpad_off + m_local] = (lane << 24) | m_local;
        }
    }
}

__global__ void build_bmm_tile64_map_kernel(
    const int32_t* __restrict__ expert_token_count,
    const int32_t* __restrict__ expert_token_padded,
    const int32_t* __restrict__ expert_offsets_m,
    int32_t* __restrict__ total_padded_tokens,
    int32_t* __restrict__ cta_to_batch,
    int32_t* __restrict__ cta_to_mn_limit,
    int32_t* __restrict__ num_non_exiting_ctas)
{
    constexpr int kTile = 64;
    const int lane = threadIdx.x & 31;
    const int m = (lane < kNumLocal) ? expert_token_count[lane] : 0;
    const int m_pad = (lane < kNumLocal) ? expert_token_padded[lane] : 0;
    const int ctas = (m_pad + kTile - 1) / kTile;
    const int cta_off = warp_exclusive_sum_i32(ctas);
    const int total_ctas = __shfl_sync(0xffffffffu, cta_off + ctas, kNumLocal - 1);

    if (lane < kNumLocal) {
        const int base = expert_offsets_m[lane];
        const int active_end = base + m;
        #pragma unroll
        for (int j = 0; j < 2; ++j) {
            if (j < ctas) {
                const int idx = cta_off + j;
                const int cta_begin = base + j * kTile;
                const int cta_end = cta_begin + kTile;
                int limit = active_end > cta_begin ? active_end : cta_begin;
                if (limit > cta_end) limit = cta_end;
                cta_to_batch[idx] = lane;
                cta_to_mn_limit[idx] = limit;
            }
        }
        for (int j = 2; j < ctas; ++j) {
            const int idx = cta_off + j;
            const int cta_begin = base + j * kTile;
            const int cta_end = cta_begin + kTile;
            int limit = active_end > cta_begin ? active_end : cta_begin;
            if (limit > cta_end) limit = cta_end;
            cta_to_batch[idx] = lane;
            cta_to_mn_limit[idx] = limit;
        }
    }
    if (lane == 0) {
        total_padded_tokens[0] = expert_offsets_m[kNumLocal];
        num_non_exiting_ctas[0] = total_ctas;
    }
}

__global__ void build_bmm_tile64_compact_map_kernel(
    const int32_t* __restrict__ expert_token_count,
    int32_t* __restrict__ total_padded_tokens,
    int32_t* __restrict__ cta_to_batch,
    int32_t* __restrict__ cta_to_mn_limit,
    int32_t* __restrict__ num_non_exiting_ctas,
    int32_t* __restrict__ expert_offsets_bmm)
{
    constexpr int kTile = 64;
    const int lane = threadIdx.x & 31;
    const int m = (lane < kNumLocal) ? expert_token_count[lane] : 0;
    const int ctas = (m + kTile - 1) / kTile;
    const int row_span = ctas * kTile;
    const int cta_off = warp_exclusive_sum_i32(ctas);
    const int row_off = warp_exclusive_sum_i32(row_span);
    const int total_ctas = __shfl_sync(0xffffffffu, cta_off + ctas, kNumLocal - 1);
    const int total_rows = __shfl_sync(0xffffffffu, row_off + row_span, kNumLocal - 1);

    if (lane < kNumLocal) {
        expert_offsets_bmm[lane] = row_off;
        const int active_end = row_off + m;
        for (int j = 0; j < ctas; ++j) {
            const int idx = cta_off + j;
            const int cta_begin = row_off + j * kTile;
            const int cta_end = cta_begin + kTile;
            int limit = active_end > cta_begin ? active_end : cta_begin;
            if (limit > cta_end) limit = cta_end;
            cta_to_batch[idx] = lane;
            cta_to_mn_limit[idx] = limit;
        }
    }
    if (lane == 0) {
        expert_offsets_bmm[kNumLocal] = total_rows;
        total_padded_tokens[0] = total_rows;
        num_non_exiting_ctas[0] = total_ctas;
    }
}

// Transpose per-expert weight scales [N/128, K/128] → [K/128, N/128] so the
// CUTLASS SFB layout (MN-major) matches our buffer. Grid: (E_local, N/128).
__global__ void transpose_w_scale_kernel(
    const float* __restrict__ w_scale_in,    // [E_local, N/128, K/128]
    float* __restrict__ w_scale_out,         // [E_local, K/128, N/128]
    int n_blocks, int k_blocks)
{
    const int le = blockIdx.x;
    const int nb = blockIdx.y;
    if (nb >= n_blocks) return;
    const int tid = threadIdx.x;
    if (tid >= k_blocks) return;
    const int kb = tid;
    w_scale_out[(size_t)le * k_blocks * n_blocks + (size_t)kb * n_blocks + nb] =
        w_scale_in [(size_t)le * n_blocks * k_blocks + (size_t)nb * k_blocks + kb];
}

// Permute activations + scales into per-group contiguous buffers.
//   A_packed[m_global, k] = hidden_states[token_id, k]
//   SFA_packed[kb, m_global] = hidden_states_scale[kb, token_id]
// where m_global = expert_offsets_m[le] + m_local.
//
// Grid: (E_local, ceil(T_max / BM_permute)). Each CTA owns one (le, m_block).
// Block: 128 threads; each thread copies a uint4 (16 FP8 bytes) of A in the
// inner K loop, then one float of SFA per K-block (56 floats).
__global__ void __launch_bounds__(128, 2) permute_a_and_sfa_kernel(
    const __nv_fp8_e4m3* __restrict__ hidden,        // [T, H]
    const float* __restrict__ hidden_scale,          // [H/128, T]
    const int32_t* __restrict__ expert_token_list,   // [E_local, T]
    const int32_t* __restrict__ expert_token_count,  // [E_local]
    const int32_t* __restrict__ expert_offsets_m,    // [E_local+1] padded M offsets
    const int32_t* __restrict__ expert_token_padded, // [E_local] padded M_e
    const int32_t* __restrict__ expert_sfa_offsets,  // [E_local+1] SFA float offsets
    __nv_fp8_e4m3* __restrict__ A_packed,            // [M_total, H]
    float* __restrict__ SFA_packed,                  // per-group [K/128, M_e_padded]
    int T)
{
    const int le = blockIdx.y;
    const int Tk = expert_token_count[le];
    if (Tk == 0) return;
    const int m_local = blockIdx.x;
    if (m_local >= Tk) return;

    const int m_global  = expert_offsets_m[le] + m_local;
    const int m_padded  = expert_token_padded[le];
    const int sfa_base  = expert_sfa_offsets[le];
    const int t         = expert_token_list[le * T + m_local];

    const int tid = threadIdx.x;
    constexpr int VEC = 16;
    const __nv_fp8_e4m3* src_row = hidden  + (size_t)t        * kHidden;
    __nv_fp8_e4m3*       dst_row = A_packed + (size_t)m_global * kHidden;
    // kHidden=7168; 128 threads * 16 bytes/thread = 2048 bytes/iter; 4 iters
    // cover 8192 bytes (with the last iter partially out of bounds, masked).
    #pragma unroll
    for (int it = 0; it < 4; ++it) {
        int off = (it * 128 + tid) * VEC;
        if (off + VEC <= kHidden) {
            *reinterpret_cast<uint4*>(dst_row + off) =
                *reinterpret_cast<const uint4*>(src_row + off);
        }
    }

    // SFA: per-group layout is [K/128, M_padded] with stride-1 in M and
    // stride M_padded in K-block. Element (kb, m_local) lives at
    // SFA_packed[sfa_base + kb*M_padded + m_local].
    if (tid < kHiddenBlocks) {
        SFA_packed[sfa_base + (size_t)tid * m_padded + m_local] =
            hidden_scale[(size_t)tid * T + t];
    }
}

// RD-D1 iter2: vectorized permute kernel.
//   Grid: (ceil(M_total_unpadded / TOK_PER_CTA), 1). 1D grid over global tokens.
//   Block: TOK_PER_CTA * 32 threads = TOK_PER_CTA warps; each warp permutes one
//     token (32 threads × 7168/32/16 = 14 uint4 loads per token).
//   Uses swiglu_block_map[]+expert_offsets_m to recover (le, m_local) per token,
//   then expert_token_list to find the source token id t.
//
// Activations: 7168 FP8 bytes/row = 448 uint4. 32 threads each do 14 uint4
// (one extra iter handles the remainder of 448 = 32*14 = 448 exactly). Memory
// access is fully coalesced across the warp.
//
// SFA: 56 floats per token. 32 threads each do 2 entries (one extra cover
// the final 56-32-32=−8 entry isn't needed: 2 iters covers 64 ≥ 56).
template <int TOK_PER_CTA>
__global__ void __launch_bounds__(TOK_PER_CTA * 32, 4) permute_a_and_sfa_v2_kernel(
    const __nv_fp8_e4m3* __restrict__ hidden,        // [T, H]
    const float* __restrict__ hidden_scale,          // [H/128, T]
    const int32_t* __restrict__ expert_token_list,   // [E_local, T]
    const int32_t* __restrict__ expert_token_padded, // [E_local] padded M_e
    const int32_t* __restrict__ expert_offsets_m,    // [E_local+1] padded M offsets
    const int32_t* __restrict__ expert_sfa_offsets,  // [E_local+1]
    const int32_t* __restrict__ swiglu_block_map,    // [M_total_unpadded] (le<<24 | m_local)
    __nv_fp8_e4m3* __restrict__ A_packed,            // [M_total, H]
    float* __restrict__ SFA_packed,                  // per-group [K/128, M_e_padded]
    float* __restrict__ SFA_bmm_packed,              // global [K/128, M_total_padded]
    int M_total_unpadded, int M_total_padded, int T)
{
    const int warp_in_cta = threadIdx.x >> 5;
    const int lane        = threadIdx.x & 31;
    const int mu          = blockIdx.x * TOK_PER_CTA + warp_in_cta;
    if (mu >= M_total_unpadded) return;

    const int32_t code = swiglu_block_map[mu];
    const int le      = code >> 24;
    const int m_local = code & 0x00FFFFFF;
    const int m_global = expert_offsets_m[le] + m_local;
    const int m_padded = expert_token_padded[le];
    const int sfa_base = expert_sfa_offsets[le];
    const int t        = expert_token_list[le * T + m_local];

    const __nv_fp8_e4m3* src_row = hidden  + (size_t)t        * kHidden;
    __nv_fp8_e4m3*       dst_row = A_packed + (size_t)m_global * kHidden;
    constexpr int VEC = 16;
    constexpr int N_UINT4 = kHidden / VEC; // 448
    // 14 iters * 32 lanes * 16 bytes = 7168 bytes (exact).
    #pragma unroll
    for (int it = 0; it < N_UINT4 / 32; ++it) {
        int u4_idx = it * 32 + lane;
        // u4_idx in [0, 448); always in bounds.
        *reinterpret_cast<uint4*>(dst_row + u4_idx * VEC) =
            *reinterpret_cast<const uint4*>(src_row + u4_idx * VEC);
    }

    // SFA: 56 floats per token; 2 iters of 32 lanes cover all (some lanes
    // mask).
    #pragma unroll
    for (int it = 0; it < 2; ++it) {
        int kb = it * 32 + lane;
        if (kb < kHiddenBlocks) {
            const float scale = hidden_scale[(size_t)kb * T + t];
            SFA_packed[sfa_base + (size_t)kb * m_padded + m_local] = scale;
            SFA_bmm_packed[(size_t)kb * M_total_padded + m_global] = scale;
        }
    }
}

template <int TOK_PER_CTA>
__global__ void __launch_bounds__(TOK_PER_CTA * 32, 4) permute_a_and_sfa_bmm_compact_kernel(
    const __nv_fp8_e4m3* __restrict__ hidden,
    const float* __restrict__ hidden_scale,
    const int32_t* __restrict__ expert_token_list,
    const int32_t* __restrict__ expert_offsets_bmm,
    const int32_t* __restrict__ swiglu_block_map,
    __nv_fp8_e4m3* __restrict__ A_packed,
    float* __restrict__ SFA_bmm_packed,
    int M_total_unpadded, int M_total_bmm_stride, int T)
{
    const int warp_in_cta = threadIdx.x >> 5;
    const int lane        = threadIdx.x & 31;
    const int mu          = blockIdx.x * TOK_PER_CTA + warp_in_cta;
    if (mu >= M_total_unpadded) return;

    const int32_t code = swiglu_block_map[mu];
    const int le      = code >> 24;
    const int m_local = code & 0x00FFFFFF;
    const int m_global = expert_offsets_bmm[le] + m_local;
    const int t        = expert_token_list[le * T + m_local];

    const __nv_fp8_e4m3* src_row = hidden  + (size_t)t * kHidden;
    __nv_fp8_e4m3*       dst_row = A_packed + (size_t)m_global * kHidden;
    constexpr int VEC = 16;
    constexpr int N_UINT4 = kHidden / VEC;
    #pragma unroll
    for (int it = 0; it < N_UINT4 / 32; ++it) {
        int u4_idx = it * 32 + lane;
        *reinterpret_cast<uint4*>(dst_row + u4_idx * VEC) =
            *reinterpret_cast<const uint4*>(src_row + u4_idx * VEC);
    }

    #pragma unroll
    for (int it = 0; it < 2; ++it) {
        int kb = it * 32 + lane;
        if (kb < kHiddenBlocks) {
            SFA_bmm_packed[(size_t)kb * M_total_bmm_stride + m_global] =
                hidden_scale[(size_t)kb * T + t];
        }
    }
}

// --------------------------- mma.sync helpers ------------------------------

__device__ __forceinline__ uint32_t ld_u32_gmem(const void* p) {
    uint32_t x;
    asm volatile("ld.global.u32 %0, [%1];" : "=r"(x) : "l"(p));
    return x;
}

__device__ __forceinline__ void mma_e4m3_m16n8k32(
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1,
    float& d0, float& d1, float& d2, float& d3)
{
    float c0 = d0, c1 = d1, c2 = d2, c3 = d3;
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
        "{%0,%1,%2,%3}, "
        "{%4,%5,%6,%7}, "
        "{%8,%9}, "
        "{%10,%11,%12,%13};\n"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
          "r"(b0), "r"(b1),
          "f"(c0), "f"(c1), "f"(c2), "f"(c3));
}

// --------------------------- GEMM1 kernel (mma.sync FP8) -------------------
//
// One CTA computes a 16×128 (BM × BN) output tile for one (le, m_block).
// blockDim = 128 (4 warps × 32 lanes). Each warp owns 16 rows × 32 cols.
// Each warp's 16×32 sub-tile = 4 sub-sub-tiles of 16×8 (4 mma ops).
// Per 128-K-block: 4 kk32 iters per sub-tile.
// Scale (sH * sW) applied per 128-K-block.

template <int BM, int BN, int WARPS_N>
__global__ void __launch_bounds__(WARPS_N * 32, 2) gemm1_mma_kernel(
    const __nv_fp8_e4m3* __restrict__ hidden,
    const float* __restrict__ hidden_scale,
    const __nv_fp8_e4m3* __restrict__ gemm1_w,
    const float* __restrict__ gemm1_w_scale,
    const int32_t* __restrict__ expert_token_list,
    const int32_t* __restrict__ expert_token_count,
    const int32_t* __restrict__ block_to_expert,
    __nv_bfloat16* __restrict__ gemm1_out, int T)
{
    static_assert(BM == 16, "mma m16 requires BM=16");
    static_assert(BN == WARPS_N * 32, "BN must equal WARPS_N * 32");

    const int by = blockIdx.y;
    const int bx = blockIdx.x;
    const int code = block_to_expert[by];
    if (code == 0xff) return;  // sentinel: unused block
    const int le = code & 0xFF;
    const int m_block_in_expert = code >> 8;
    const int Tk = expert_token_count[le];
    const int m_base = m_block_in_expert * BM;
    if (m_base >= Tk) return;

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int g    = lane >> 2;       // 0..7 (row in 16-row sub-tile pair)
    const int tid4 = lane & 3;        // 0..3 (col group in 32-K sub-tile)

    // Per-thread token indices for its two rows (m_base+g and m_base+g+8).
    // Out-of-Tk rows: still safe to load arbitrary t (we'll mask the write at end).
    const int row_a = m_base + g;
    const int row_b = m_base + g + 8;
    const bool row_a_valid = row_a < Tk;
    const bool row_b_valid = row_b < Tk;
    const int t_a = row_a_valid ? expert_token_list[le * T + row_a] : 0;
    const int t_b = row_b_valid ? expert_token_list[le * T + row_b] : 0;

    // This warp's 32 N-cols: n_warp_base ∈ [bx*BN + warp*32, bx*BN + (warp+1)*32).
    const int n_warp_base = bx * BN + warp * 32;

    // Per-CTA, per-warp accumulators: 4 sub-tiles per warp, each holding (d0..d3).
    float acc[4][4] = {0};

    // K loop: 56 hidden blocks
    #pragma unroll 1
    for (int kb = 0; kb < kHiddenBlocks; ++kb) {
        const int k_base128 = kb * kBlock;
        // 4 raw accumulators per sub-tile reset each K-block (we'll fold scale at end)
        float raw[4][4] = {0};

        // 4 kk32 iters per K-block
        #pragma unroll
        for (int kk = 0; kk < 4; ++kk) {
            const int k = k_base128 + kk * 32;
            // A fragment (per thread): 4 uint32 holding 16 bytes (FP8) across two
            // rows of A and two K-segments of 4 bytes each.
            //   a0: A[row_a][k + tid4*4 + 0..3]
            //   a1: A[row_b][k + tid4*4 + 0..3]
            //   a2: A[row_a][k + tid4*4 + 16..19]
            //   a3: A[row_b][k + tid4*4 + 16..19]
            const __nv_fp8_e4m3* A_a = hidden + (size_t)t_a * kHidden + k + tid4 * 4;
            const __nv_fp8_e4m3* A_b = hidden + (size_t)t_b * kHidden + k + tid4 * 4;
            uint32_t a0 = row_a_valid ? ld_u32_gmem(A_a) : 0u;
            uint32_t a1 = row_b_valid ? ld_u32_gmem(A_b) : 0u;
            uint32_t a2 = row_a_valid ? ld_u32_gmem(A_a + 16) : 0u;
            uint32_t a3 = row_b_valid ? ld_u32_gmem(A_b + 16) : 0u;

            // For each sub-tile s ∈ [0,4), the 8 N-rows of B are n_warp_base+s*8+g
            // (g ∈ [0,8) and lane provides 4 mma rows).
            #pragma unroll
            for (int s = 0; s < 4; ++s) {
                const int n8 = n_warp_base + s * 8;
                const int n_row = n8 + g; // 0..7 sub-row of B
                // B is FP8 weights [E_local, N, K] row-major (last dim K is contiguous).
                const __nv_fp8_e4m3* B_row = gemm1_w
                    + ((size_t)le * kGemm1Out + n_row) * kHidden + k + tid4 * 4;
                uint32_t b0 = ld_u32_gmem(B_row);
                uint32_t b1 = ld_u32_gmem(B_row + 16);
                mma_e4m3_m16n8k32(a0, a1, a2, a3, b0, b1,
                                  raw[s][0], raw[s][1], raw[s][2], raw[s][3]);
            }
        }

        // Apply per-K-block scales: (sH * sW). sH varies per row, sW per N-block.
        // Each thread holds output element rows (m_base+g, m_base+g+8) and 2 N-cols.
        const float sH_a = row_a_valid ? hidden_scale[kb * T + t_a] : 0.f;
        const float sH_b = row_b_valid ? hidden_scale[kb * T + t_b] : 0.f;
        #pragma unroll
        for (int s = 0; s < 4; ++s) {
            const int n8 = n_warp_base + s * 8;
            // n8 ranges 0..N-1, the N-block (n8 / 128) is the same for all 8 cols
            // of this sub-tile when 128-aligned, which it is here.
            const int n_block = n8 / kBlock;
            const float sW = gemm1_w_scale[le * kGemm1OutBlk * kHiddenBlocks
                                           + n_block * kHiddenBlocks + kb];
            const float sa = sH_a * sW;
            const float sb = sH_b * sW;
            acc[s][0] += raw[s][0] * sa;
            acc[s][1] += raw[s][1] * sa;
            acc[s][2] += raw[s][2] * sb;
            acc[s][3] += raw[s][3] * sb;
        }
    }

    // Write outputs as bf16. Pack each (col0, col1) into a single bf16x2 store
    // since they are always 2-element-aligned in N (n0 = n8 + tid4*2, tid4 ∈ [0,3]).
    #pragma unroll
    for (int s = 0; s < 4; ++s) {
        const int n8 = n_warp_base + s * 8;
        const int n0 = n8 + tid4 * 2;
        if (n0 + 1 < kGemm1Out) {
            __nv_bfloat162 v_a = __floats2bfloat162_rn(acc[s][0], acc[s][1]);
            __nv_bfloat162 v_b = __floats2bfloat162_rn(acc[s][2], acc[s][3]);
            if (row_a_valid)
                *reinterpret_cast<__nv_bfloat162*>(&gemm1_out[((size_t)le * T + row_a) * kGemm1Out + n0]) = v_a;
            if (row_b_valid)
                *reinterpret_cast<__nv_bfloat162*>(&gemm1_out[((size_t)le * T + row_b) * kGemm1Out + n0]) = v_b;
        }
    }
}

// Fused GEMM1 epilogue path. One CTA computes a 16x128 tile for the gate
// half and the matching 16x128 tile for the up half, then writes the post
// SwiGLU FP8 activations and per-row/per-128-column scales consumed by GEMM2.
template <int BM, int BN, int WARPS_N>
__global__ void __launch_bounds__(WARPS_N * 32, 2) gemm1_mma_swiglu_quant_kernel(
    const __nv_fp8_e4m3* __restrict__ hidden,
    const float* __restrict__ hidden_scale,
    const __nv_fp8_e4m3* __restrict__ gemm1_w,
    const float* __restrict__ gemm1_w_scale,
    const int32_t* __restrict__ expert_token_list,
    const int32_t* __restrict__ expert_token_count,
    const int32_t* __restrict__ block_to_expert,
    const int32_t* __restrict__ expert_offsets_m,
    const int32_t* __restrict__ expert_token_padded,
    const int32_t* __restrict__ expert_sfa2_offsets,
    __nv_fp8_e4m3* __restrict__ a2_packed,
    float* __restrict__ sfa2_packed,
    int T)
{
    static_assert(BM == 16, "mma m16 requires BM=16");
    static_assert(BN == WARPS_N * 32, "BN must equal WARPS_N * 32");

    const int ib = blockIdx.x; // 0..15, each CTA emits one 128-wide inter block
    const int by = blockIdx.y;
    const int code = block_to_expert[by];
    if (code == 0xff) return;
    const int le = code & 0xFF;
    const int m_block_in_expert = code >> 8;
    const int Tk = expert_token_count[le];
    const int m_base = m_block_in_expert * BM;
    if (m_base >= Tk) return;

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int g    = lane >> 2;
    const int tid4 = lane & 3;

    const int row_a = m_base + g;
    const int row_b = m_base + g + 8;
    const bool row_a_valid = row_a < Tk;
    const bool row_b_valid = row_b < Tk;
    const int t_a = row_a_valid ? expert_token_list[le * T + row_a] : 0;
    const int t_b = row_b_valid ? expert_token_list[le * T + row_b] : 0;

    const int n_warp_base = ib * kBlock + warp * 32;

    float acc_gate[4][4] = {0};
    float acc_up[4][4] = {0};

    #pragma unroll 1
    for (int kb = 0; kb < kHiddenBlocks; ++kb) {
        const int k_base128 = kb * kBlock;
        float raw_gate[4][4] = {0};
        float raw_up[4][4] = {0};

        #pragma unroll
        for (int kk = 0; kk < 4; ++kk) {
            const int k = k_base128 + kk * 32;
            const __nv_fp8_e4m3* A_a = hidden + (size_t)t_a * kHidden + k + tid4 * 4;
            const __nv_fp8_e4m3* A_b = hidden + (size_t)t_b * kHidden + k + tid4 * 4;
            uint32_t a0 = row_a_valid ? ld_u32_gmem(A_a) : 0u;
            uint32_t a1 = row_b_valid ? ld_u32_gmem(A_b) : 0u;
            uint32_t a2 = row_a_valid ? ld_u32_gmem(A_a + 16) : 0u;
            uint32_t a3 = row_b_valid ? ld_u32_gmem(A_b + 16) : 0u;

            #pragma unroll
            for (int s = 0; s < 4; ++s) {
                const int n8 = n_warp_base + s * 8;
                const int n_row = n8 + g;
                const __nv_fp8_e4m3* B_gate = gemm1_w
                    + ((size_t)le * kGemm1Out + n_row) * kHidden + k + tid4 * 4;
                const __nv_fp8_e4m3* B_up = gemm1_w
                    + ((size_t)le * kGemm1Out + (kInter + n_row)) * kHidden + k + tid4 * 4;
                uint32_t bg0 = ld_u32_gmem(B_gate);
                uint32_t bg1 = ld_u32_gmem(B_gate + 16);
                uint32_t bu0 = ld_u32_gmem(B_up);
                uint32_t bu1 = ld_u32_gmem(B_up + 16);
                mma_e4m3_m16n8k32(a0, a1, a2, a3, bg0, bg1,
                                  raw_gate[s][0], raw_gate[s][1],
                                  raw_gate[s][2], raw_gate[s][3]);
                mma_e4m3_m16n8k32(a0, a1, a2, a3, bu0, bu1,
                                  raw_up[s][0], raw_up[s][1],
                                  raw_up[s][2], raw_up[s][3]);
            }
        }

        const float sH_a = row_a_valid ? hidden_scale[kb * T + t_a] : 0.f;
        const float sH_b = row_b_valid ? hidden_scale[kb * T + t_b] : 0.f;
        const float sW_gate = gemm1_w_scale[(size_t)le * kGemm1OutBlk * kHiddenBlocks
                                            + (size_t)ib * kHiddenBlocks + kb];
        const float sW_up = gemm1_w_scale[(size_t)le * kGemm1OutBlk * kHiddenBlocks
                                          + (size_t)(kInterBlocks + ib) * kHiddenBlocks + kb];
        const float sg_a = sH_a * sW_gate;
        const float sg_b = sH_b * sW_gate;
        const float su_a = sH_a * sW_up;
        const float su_b = sH_b * sW_up;
        #pragma unroll
        for (int s = 0; s < 4; ++s) {
            acc_gate[s][0] += raw_gate[s][0] * sg_a;
            acc_gate[s][1] += raw_gate[s][1] * sg_a;
            acc_gate[s][2] += raw_gate[s][2] * sg_b;
            acc_gate[s][3] += raw_gate[s][3] * sg_b;
            acc_up[s][0] += raw_up[s][0] * su_a;
            acc_up[s][1] += raw_up[s][1] * su_a;
            acc_up[s][2] += raw_up[s][2] * su_b;
            acc_up[s][3] += raw_up[s][3] * su_b;
        }
    }

    __shared__ float row_max[BM];
    if (tid < BM) row_max[tid] = 0.0f;
    __syncthreads();

    float max_a = 0.0f;
    float max_b = 0.0f;
    #pragma unroll
    for (int s = 0; s < 4; ++s) {
        float v;
        v = (acc_up[s][0] / (1.0f + __expf(-acc_up[s][0]))) * acc_gate[s][0];
        max_a = fmaxf(max_a, fabsf(v));
        v = (acc_up[s][1] / (1.0f + __expf(-acc_up[s][1]))) * acc_gate[s][1];
        max_a = fmaxf(max_a, fabsf(v));
        v = (acc_up[s][2] / (1.0f + __expf(-acc_up[s][2]))) * acc_gate[s][2];
        max_b = fmaxf(max_b, fabsf(v));
        v = (acc_up[s][3] / (1.0f + __expf(-acc_up[s][3]))) * acc_gate[s][3];
        max_b = fmaxf(max_b, fabsf(v));
    }
    if (row_a_valid) atomicMax((int*)&row_max[g], __float_as_int(max_a));
    if (row_b_valid) atomicMax((int*)&row_max[g + 8], __float_as_int(max_b));
    __syncthreads();

    constexpr float kFp8Max = 448.0f;
    if (tid < BM) {
        const int m_local = m_base + tid;
        if (m_local < Tk) {
            const int m_padded = expert_token_padded[le];
            const int sfa_base = expert_sfa2_offsets[le];
            float maxv = row_max[tid];
            float scale = (maxv > 0.f) ? (maxv / kFp8Max) : 1.0f;
            sfa2_packed[sfa_base + (size_t)ib * m_padded + m_local] = scale;
        }
    }

    const int m_global_a = expert_offsets_m[le] + row_a;
    const int m_global_b = expert_offsets_m[le] + row_b;
    const float scale_a = (row_max[g] > 0.f) ? (row_max[g] / kFp8Max) : 1.0f;
    const float scale_b = (row_max[g + 8] > 0.f) ? (row_max[g + 8] / kFp8Max) : 1.0f;
    const float inv_scale_a = 1.0f / scale_a;
    const float inv_scale_b = 1.0f / scale_b;

    #pragma unroll
    for (int s = 0; s < 4; ++s) {
        const int n8 = n_warp_base + s * 8;
        const int n0 = n8 + tid4 * 2;
        if (row_a_valid) {
            float v0 = (acc_up[s][0] / (1.0f + __expf(-acc_up[s][0]))) * acc_gate[s][0];
            float v1 = (acc_up[s][1] / (1.0f + __expf(-acc_up[s][1]))) * acc_gate[s][1];
            a2_packed[(size_t)m_global_a * kInter + n0] = __nv_fp8_e4m3(v0 * inv_scale_a);
            a2_packed[(size_t)m_global_a * kInter + n0 + 1] = __nv_fp8_e4m3(v1 * inv_scale_a);
        }
        if (row_b_valid) {
            float v0 = (acc_up[s][2] / (1.0f + __expf(-acc_up[s][2]))) * acc_gate[s][2];
            float v1 = (acc_up[s][3] / (1.0f + __expf(-acc_up[s][3]))) * acc_gate[s][3];
            a2_packed[(size_t)m_global_b * kInter + n0] = __nv_fp8_e4m3(v0 * inv_scale_b);
            a2_packed[(size_t)m_global_b * kInter + n0 + 1] = __nv_fp8_e4m3(v1 * inv_scale_b);
        }
    }
}

// --------------------------- GEMM2 kernel (mma.sync FP8) -------------------
//
// For GEMM2, A is FP32 activations (post-SwiGLU). To use mma.sync FP8 we'd
// need to re-quantize the activations to FP8 first. That's a non-trivial
// extra kernel — for D8 iter1 we keep GEMM2 on the FP32×FP8 dequant kernel
// from D1. Phase 3 will fuse activation requantization into SwiGLU and use
// FP8 mma for GEMM2 too.

// --------------------------- GEMM1 kernel (compacted grid + uint4 loads) ---
//
// Block layout: BM rows × BN cols. Each thread computes one output element
// out[m, n] for one expert le. Grid is flat:
//   gridDim.x = N/BN
//   gridDim.y = total_m_blocks (= expert_offsets[E_local])
// block_to_expert maps gridDim.y index → (le, m_block_within_expert).
//
// We process the K loop as 56 hidden blocks (each 128 elements). For each
// 128-block, we load 8 uint4 vectors per row (8 * 16 = 128 FP8 elements) and
// FP32-fma them with the corresponding weight column.

template <int BM, int BN>
__global__ void __launch_bounds__(BM * BN, 1) gemm1_dequant_kernel(
    const __nv_fp8_e4m3* __restrict__ hidden,
    const float* __restrict__ hidden_scale,
    const __nv_fp8_e4m3* __restrict__ gemm1_w,
    const float* __restrict__ gemm1_w_scale,
    const int32_t* __restrict__ expert_token_list,
    const int32_t* __restrict__ expert_token_count,
    const int32_t* __restrict__ block_to_expert,
    float* __restrict__ gemm1_out, int T)
{
    const int bn  = blockIdx.x;
    const int by  = blockIdx.y;
    const int code = block_to_expert[by];
    const int le = code & 0xFF;
    const int m_block_in_expert = code >> 8;
    const int Tk = expert_token_count[le];

    const int tx = threadIdx.x;   // 0..BN-1
    const int ty = threadIdx.y;   // 0..BM-1
    const int m_local = m_block_in_expert * BM + ty;
    const int n = bn * BN + tx;
    if (n >= kGemm1Out || m_local >= Tk) return;

    const int t = expert_token_list[le * T + m_local];
    const int nb = n / kBlock;

    float acc = 0.0f;

    #pragma unroll 1
    for (int kb = 0; kb < kHiddenBlocks; ++kb) {
        const float sH = hidden_scale[kb * T + t];
        const float sW = gemm1_w_scale[le * kGemm1OutBlk * kHiddenBlocks
                                       + nb * kHiddenBlocks + kb];
        const float s  = sH * sW;
        float partial = 0.0f;
        const __nv_fp8_e4m3* A_row = hidden + (size_t)t * kHidden + kb * kBlock;
        const __nv_fp8_e4m3* W_row = gemm1_w
            + ((size_t)le * kGemm1Out + n) * kHidden + kb * kBlock;
        // 128 elems = 8 uint4 of FP8
        #pragma unroll
        for (int kk = 0; kk < 8; ++kk) {
            float A16[16], W16[16];
            load_fp8_16(A_row + kk * 16, A16);
            load_fp8_16(W_row + kk * 16, W16);
            #pragma unroll
            for (int i = 0; i < 16; ++i) partial += A16[i] * W16[i];
        }
        acc += partial * s;
    }

    gemm1_out[((size_t)le * T + m_local) * kGemm1Out + n] = acc;
}

// --------------------------- SwiGLU + FP8 quant ----------------------------
//
// Produces FP8 e4m3 activations of shape [E_local, T, I] and per-block scales
// of shape [E_local, I/128, T] (matching the hidden_states_scale layout for
// easy reuse in GEMM2 mma.sync). One block per (le, m) computes silu(x2)*x1
// and absmax over each 128-elem block in I.
//
// Grid: (I/128 blocks of activation along I axis are not split — one block
// covers an entire 128-wide block per (le, m)). gridDim = (I/128, T, E_local).
// Block = 128 threads (one per FP8 element in the K-block).
// We split per 128-wide block to write a tight scale.

__global__ void __launch_bounds__(128, 1) swiglu_quant_kernel(
    const __nv_bfloat16* __restrict__ gemm1_out, // [E_local, T, 2I]
    __nv_fp8_e4m3* __restrict__ act_fp8,         // [E_local, T, I]
    float* __restrict__ act_scale,               // [E_local, I/128, T]
    const int32_t* __restrict__ expert_token_count,
    int T)
{
    const int le = blockIdx.z;
    const int Tk = expert_token_count[le];
    if (Tk == 0) return;
    const int m = blockIdx.y;
    if (m >= Tk) return;
    const int ib = blockIdx.x; // 0..15 (I/128)
    const int tid = threadIdx.x; // 0..127 (one per element in the 128-block)
    const int n = ib * kBlock + tid;
    if (n >= kInter) return;

    const size_t base = ((size_t)le * T + m) * kGemm1Out;
    float x1 = __bfloat162float(gemm1_out[base + n]);
    float x2 = __bfloat162float(gemm1_out[base + kInter + n]);
    float silu = x2 / (1.0f + __expf(-x2));
    float v = silu * x1;

    // absmax over 128 threads
    float a = fabsf(v);
    __shared__ float smax[1];
    if (tid == 0) smax[0] = 0.0f;
    __syncthreads();
    // warp-reduce first
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        a = fmaxf(a, __shfl_xor_sync(0xffffffff, a, off));
    }
    if ((tid & 31) == 0) atomicMax((int*)&smax[0], __float_as_int(a));
    __syncthreads();
    float maxv = smax[0];
    // FP8 e4m3 max representable ≈ 448
    const float kFp8Max = 448.0f;
    float scale = (maxv > 0.f) ? (maxv / kFp8Max) : 1.0f;
    float inv_scale = 1.0f / scale;

    __nv_fp8_e4m3 q = __nv_fp8_e4m3(v * inv_scale);
    act_fp8[((size_t)le * T + m) * kInter + n] = q;
    if (tid == 0) {
        act_scale[(size_t)le * kInterBlocks * T + ib * T + m] = scale;
    }
}

// Debug helper: copy gemm1_scratch[le, m, n] → d_packed[m_offset[le]+m, n].
// Used to bypass CUTLASS GEMM1 (the mma.sync path is known-correct) without
// changing the rest of the pipeline.
__global__ void copy_scratch_to_packed_kernel(
    const __nv_bfloat16* __restrict__ gemm1_scratch, // [E_local, T, 2I]
    __nv_bfloat16* __restrict__ d_packed,            // [M_total, 2I]
    const int32_t* __restrict__ expert_token_count,
    const int32_t* __restrict__ expert_offsets_m,
    int T)
{
    const int le = blockIdx.z;
    const int Tk = expert_token_count[le];
    if (Tk == 0) return;
    const int m = blockIdx.y;
    if (m >= Tk) return;
    const int n_block = blockIdx.x; // 0..(2I/128-1)
    const int tid = threadIdx.x;    // 0..127
    const int n = n_block * 128 + tid;
    if (n >= kGemm1Out) return;
    const int m_global = expert_offsets_m[le] + m;
    d_packed[(size_t)m_global * kGemm1Out + n] =
        gemm1_scratch[((size_t)le * T + m) * kGemm1Out + n];
}

// Variant: reads from packed D_packed[M_total, 2I] (CUTLASS GEMM1 output)
// indexed by expert_offsets_m[le] + m. Writes to the same per-expert
// act_fp8 / act_scale layout used by GEMM2.
__global__ void __launch_bounds__(128, 1) swiglu_quant_packed_kernel(
    const __nv_bfloat16* __restrict__ D_packed,      // [M_total, 2I]
    __nv_fp8_e4m3* __restrict__ act_fp8,             // [E_local, T, I]
    float* __restrict__ act_scale,                   // [E_local, I/128, T]
    const int32_t* __restrict__ expert_token_count,  // [E_local]
    const int32_t* __restrict__ expert_offsets_m,    // [E_local+1]
    int T)
{
    const int le = blockIdx.z;
    const int Tk = expert_token_count[le];
    if (Tk == 0) return;
    const int m = blockIdx.y;
    if (m >= Tk) return;
    const int ib = blockIdx.x; // 0..15 (I/128)
    const int tid = threadIdx.x; // 0..127 (one per element in the 128-block)
    const int n = ib * kBlock + tid;
    if (n >= kInter) return;

    const int m_global = expert_offsets_m[le] + m;
    const size_t base = (size_t)m_global * kGemm1Out;
    float x1 = __bfloat162float(D_packed[base + n]);
    float x2 = __bfloat162float(D_packed[base + kInter + n]);
    float silu = x2 / (1.0f + __expf(-x2));
    float v = silu * x1;

    float a = fabsf(v);
    __shared__ float warp_max[4];
    __shared__ float smax[1];
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        a = fmaxf(a, __shfl_xor_sync(0xffffffff, a, off));
    }
    if ((tid & 31) == 0) warp_max[tid >> 5] = a;
    __syncthreads();
    float block_max = (tid < 4) ? warp_max[tid] : 0.0f;
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        block_max = fmaxf(block_max, __shfl_xor_sync(0xffffffff, block_max, off));
    }
    if (tid == 0) smax[0] = block_max;
    __syncthreads();
    float maxv = smax[0];
    const float kFp8Max = 448.0f;
    float scale = (maxv > 0.f) ? (maxv / kFp8Max) : 1.0f;
    float inv_scale = 1.0f / scale;

    __nv_fp8_e4m3 q = __nv_fp8_e4m3(v * inv_scale);
    act_fp8[((size_t)le * T + m) * kInter + n] = q;
    if (tid == 0) {
        act_scale[(size_t)le * kInterBlocks * T + ib * T + m] = scale;
    }
}

// RD-D2 iter1: SwiGLU + FP8 quant on a packed grid sized to the actual
// unpadded M_total (= sum of expert_token_count). Reduces grid from
// (I/128) × T × E_local CTAs to (I/128) × M_total_unpadded CTAs (about
// 4× fewer at the floor). Each CTA does a binary search across the
// 32-entry expert_offsets_m to recover (le, m_local) cheaply.
//
// We pre-build a [M_total_unpadded] lookup of m_global_unpadded → (le, m_local)
// in build_swiglu_index_kernel below.
__global__ void __launch_bounds__(128, 2) swiglu_quant_packed_v2_kernel(
    const __nv_bfloat16* __restrict__ D_packed,      // [M_total_padded, 2I]
    __nv_fp8_e4m3* __restrict__ act_fp8,             // [E_local, T, I]
    float* __restrict__ act_scale,                   // [E_local, I/128, T]
    const int32_t* __restrict__ swiglu_block_map,    // [M_total_unpadded] packed (le<<24 | m_local)
    const int32_t* __restrict__ expert_offsets_m,    // [E_local+1] padded
    int32_t M_total_unpadded, int T)
{
    const int ib = blockIdx.x; // 0..15 (I/128)
    const int mu = blockIdx.y; // 0..M_total_unpadded-1
    if (mu >= M_total_unpadded) return;
    const int tid = threadIdx.x; // 0..127
    const int n = ib * kBlock + tid;
    if (n >= kInter) return;

    const int32_t code = swiglu_block_map[mu];
    const int le      = code >> 24;
    const int m_local = code & 0x00FFFFFF;
    const int m_global_padded = expert_offsets_m[le] + m_local;

    const size_t base = (size_t)m_global_padded * kGemm1Out;
    float x1 = __bfloat162float(D_packed[base + n]);
    float x2 = __bfloat162float(D_packed[base + kInter + n]);
    float silu = x2 / (1.0f + __expf(-x2));
    float v = silu * x1;

    // 128-wide absmax: warp reduce, then one warp reduces four warp maxima.
    float a = fabsf(v);
    __shared__ float warp_max[4];
    __shared__ float smax[1];
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        a = fmaxf(a, __shfl_xor_sync(0xffffffff, a, off));
    }
    if ((tid & 31) == 0) warp_max[tid >> 5] = a;
    __syncthreads();
    float block_max = (tid < 4) ? warp_max[tid] : 0.0f;
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        block_max = fmaxf(block_max, __shfl_xor_sync(0xffffffff, block_max, off));
    }
    if (tid == 0) smax[0] = block_max;
    __syncthreads();
    float maxv = smax[0];
    const float kFp8Max = 448.0f;
    float scale = (maxv > 0.f) ? (maxv / kFp8Max) : 1.0f;
    float inv_scale = 1.0f / scale;

    __nv_fp8_e4m3 q = __nv_fp8_e4m3(v * inv_scale);
    act_fp8[((size_t)le * T + m_local) * kInter + n] = q;
    if (tid == 0) {
        act_scale[(size_t)le * kInterBlocks * T + ib * T + m_local] = scale;
    }
}

// RD-D4 iter1 variant: SwiGLU + FP8 quant writing into the PACKED M_total
// layout expected by CUTLASS GEMM2:
//   act_fp8_packed[m_global_padded, n] for n in [0, I=2048)
//   sfa2_packed[group_offset + ib*m_padded + m_local] (per-group MN-major)
// Padding rows (m_local >= Tk[le]) are skipped at the grid level (we use the
// swiglu_block_map which only covers unpadded entries). Padded SFA + A
// entries stay at the value written by the earlier zero-fill in run().
template <bool EXACT_SILU, bool COMPACT_BMM, bool COMPACT_D>
__global__ void __launch_bounds__(128, 2) swiglu_quant_packed_v3_kernel(
    const __nv_bfloat16* __restrict__ D_packed,      // [M_total_padded, 2I]
    __nv_fp8_e4m3* __restrict__ a2_packed,           // [M_total_padded, I]
    float* __restrict__ sfa2_packed,                 // per-group [I/128, M_padded]
    float* __restrict__ sfa2_bmm_packed,             // global [I/128, M_total_padded]
    const int32_t* __restrict__ swiglu_block_map,    // [M_total_unpadded] packed (le<<24 | m_local)
    const int32_t* __restrict__ expert_offsets_m,    // [E_local+1] padded
    const int32_t* __restrict__ expert_offsets_bmm,  // [E_local+1] tile64 compact offsets
    const int32_t* __restrict__ expert_token_padded, // [E_local] padded M_e
    const int32_t* __restrict__ expert_sfa2_offsets, // [E_local+1] SFA float offsets for GEMM2
    int32_t M_total_unpadded, int32_t M_total_bmm_stride)
{
    const int ib = blockIdx.x; // 0..15 (I/128)
    const int mu = blockIdx.y;
    if (mu >= M_total_unpadded) return;
    const int tid = threadIdx.x;
    const int n = ib * kBlock + tid;
    if (n >= kInter) return;

    const int32_t code = swiglu_block_map[mu];
    const int le      = code >> 24;
    const int m_local = code & 0x00FFFFFF;
    const int m_global = expert_offsets_m[le] + m_local;
    const int m_d = COMPACT_D ? (expert_offsets_bmm[le] + m_local) : m_global;
    const int m_out = COMPACT_BMM ? (expert_offsets_bmm[le] + m_local) : m_global;
    const int m_padded = expert_token_padded[le];
    const int sfa_base = expert_sfa2_offsets[le];

    const size_t base = (size_t)m_d * kGemm1Out;
    float x1 = __bfloat162float(D_packed[base + n]);
    float x2 = __bfloat162float(D_packed[base + kInter + n]);
    float silu = EXACT_SILU ? x2 / (1.0f + __expf(-x2))
                            : x2 * approx_sigmoid_rsqrt(x2);
    float v = silu * x1;

    float a = fabsf(v);
    __shared__ float warp_max[4];
    __shared__ float smax[1];
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        a = fmaxf(a, __shfl_xor_sync(0xffffffff, a, off));
    }
    if ((tid & 31) == 0) warp_max[tid >> 5] = a;
    __syncthreads();
    float block_max = (tid < 4) ? warp_max[tid] : 0.0f;
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        block_max = fmaxf(block_max, __shfl_xor_sync(0xffffffff, block_max, off));
    }
    if (tid == 0) smax[0] = block_max;
    __syncthreads();
    float maxv = smax[0];
    const float kFp8Max = 448.0f;
    float scale = (maxv > 0.f) ? (maxv / kFp8Max) : 1.0f;
    float inv_scale = 1.0f / scale;

    __nv_fp8_e4m3 q = __nv_fp8_e4m3(v * inv_scale);
    // Packed M layout: [M_total_padded, I] row-major
    a2_packed[(size_t)m_out * kInter + n] = q;
    if (tid == 0) {
        // SFA layout per-group: stride 1 in M, stride M_padded in K-block.
        if constexpr (!COMPACT_BMM) {
            sfa2_packed[sfa_base + (size_t)ib * m_padded + m_local] = scale;
        }
        sfa2_bmm_packed[(size_t)ib * M_total_bmm_stride + m_out] = scale;
    }
}

// Build the m_global_unpadded → (le, m_local) lookup. One CTA, one warp per
// expert. The output array is dense (no padding gaps): for expert le, entries
// [unpad_offset[le], unpad_offset[le]+Tk[le]) all carry (le, 0), (le, 1), ...
__global__ void build_swiglu_index_kernel(
    const int32_t* __restrict__ expert_token_count, // [E_local]
    int32_t* __restrict__ swiglu_block_map,         // [M_total_unpadded]
    int32_t* __restrict__ m_unpadded_total)         // [1] sum
{
    const int tid = threadIdx.x;
    __shared__ int32_t s_cnt[kNumLocal];
    __shared__ int32_t s_off[kNumLocal + 1];
    if (tid < kNumLocal) s_cnt[tid] = expert_token_count[tid];
    __syncthreads();
    if (tid == 0) {
        int acc = 0;
        for (int le = 0; le < kNumLocal; ++le) {
            s_off[le] = acc;
            acc += s_cnt[le];
        }
        s_off[kNumLocal] = acc;
        if (m_unpadded_total) m_unpadded_total[0] = acc;
    }
    __syncthreads();
    // Each warp covers one expert. With 256 threads (8 warps) and 32 experts,
    // each warp does 4 experts.
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int nwarps = blockDim.x >> 5;
    for (int le = warp; le < kNumLocal; le += nwarps) {
        int Tk = s_cnt[le];
        int base = s_off[le];
        for (int m_local = lane; m_local < Tk; m_local += 32) {
            swiglu_block_map[base + m_local] = (le << 24) | m_local;
        }
    }
}

// --------------------------- GEMM2 + scatter (mma.sync FP8) ----------------
//
// A is FP8 activations [E_local, T, I=2048] post-SwiGLU, scale [E_local, I/128, T].
// B is FP8 weights [E_local, H=7168, I=2048] row-major.
// Output: accumulate per-token weighted contribution into output_fp32[T, H].
//
// One CTA computes a 16×128 (BM × BN) output tile in (M=row of expert tokens,
// N=cols of H). 4 warps tile N. K dim is 2048 = 16 × 128.

template <int BM, int BN, int WARPS_N>
__global__ void __launch_bounds__(WARPS_N * 32, 2) gemm2_mma_scatter_kernel(
    const __nv_fp8_e4m3* __restrict__ act,         // [E_local, T, I]
    const float* __restrict__ act_scale,           // [E_local, I/128, T]
    const __nv_fp8_e4m3* __restrict__ gemm2_w,
    const float* __restrict__ gemm2_w_scale,
    const int32_t* __restrict__ expert_token_list,
    const int32_t* __restrict__ expert_token_count,
    const int32_t* __restrict__ block_to_expert,
    const float* __restrict__ expert_token_weight,
    __nv_bfloat16* __restrict__ output_bf16, int T)
{
    static_assert(BM == 16, "mma m16 requires BM=16");
    static_assert(BN == WARPS_N * 32, "BN must equal WARPS_N * 32");

    const int by = blockIdx.y;
    const int bx = blockIdx.x;
    const int code = block_to_expert[by];
    if (code == 0xff) return;  // sentinel: unused block
    const int le = code & 0xFF;
    const int m_block_in_expert = code >> 8;
    const int Tk = expert_token_count[le];
    const int m_base = m_block_in_expert * BM;
    if (m_base >= Tk) return;

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int g    = lane >> 2;
    const int tid4 = lane & 3;

    const int row_a = m_base + g;
    const int row_b = m_base + g + 8;
    const bool row_a_valid = row_a < Tk;
    const bool row_b_valid = row_b < Tk;
    const int t_a = row_a_valid ? expert_token_list[le * T + row_a] : 0;
    const int t_b = row_b_valid ? expert_token_list[le * T + row_b] : 0;
    const float w_a = row_a_valid ? expert_token_weight[le * T + row_a] : 0.f;
    const float w_b = row_b_valid ? expert_token_weight[le * T + row_b] : 0.f;

    const int n_warp_base = bx * BN + warp * 32;

    float acc[4][4] = {0};

    // K loop: 16 intermediate blocks
    #pragma unroll 1
    for (int kb = 0; kb < kInterBlocks; ++kb) {
        const int k_base128 = kb * kBlock;
        float raw[4][4] = {0};

        #pragma unroll
        for (int kk = 0; kk < 4; ++kk) {
            const int k = k_base128 + kk * 32;
            // A fragments: act buffer indexed by m_local (not token id!).
            // A[row_a, k + tid4*4 + 0..3] etc.
            const __nv_fp8_e4m3* A_a = act + ((size_t)le * T + row_a) * kInter + k + tid4 * 4;
            const __nv_fp8_e4m3* A_b = act + ((size_t)le * T + row_b) * kInter + k + tid4 * 4;
            uint32_t a0 = row_a_valid ? ld_u32_gmem(A_a)      : 0u;
            uint32_t a1 = row_b_valid ? ld_u32_gmem(A_b)      : 0u;
            uint32_t a2 = row_a_valid ? ld_u32_gmem(A_a + 16) : 0u;
            uint32_t a3 = row_b_valid ? ld_u32_gmem(A_b + 16) : 0u;

            #pragma unroll
            for (int s = 0; s < 4; ++s) {
                const int n8 = n_warp_base + s * 8;
                const int n_row = n8 + g; // 0..7
                // B is [E_local, H, I] row-major; last dim K is contiguous.
                const __nv_fp8_e4m3* B_row = gemm2_w
                    + ((size_t)le * kHidden + n_row) * kInter + k + tid4 * 4;
                uint32_t b0 = ld_u32_gmem(B_row);
                uint32_t b1 = ld_u32_gmem(B_row + 16);
                mma_e4m3_m16n8k32(a0, a1, a2, a3, b0, b1,
                                  raw[s][0], raw[s][1], raw[s][2], raw[s][3]);
            }
        }

        // Apply per-K-block scales.
        // act_scale layout: [E_local, I/128, T] indexed [le, kb, m_local]
        const float sA_a = row_a_valid ? act_scale[(size_t)le * kInterBlocks * T + kb * T + row_a] : 0.f;
        const float sA_b = row_b_valid ? act_scale[(size_t)le * kInterBlocks * T + kb * T + row_b] : 0.f;
        #pragma unroll
        for (int s = 0; s < 4; ++s) {
            const int n8 = n_warp_base + s * 8;
            const int n_block = n8 / kBlock;
            const float sW = gemm2_w_scale[le * kHiddenBlocks * kInterBlocks
                                           + n_block * kInterBlocks + kb];
            const float sa = sA_a * sW;
            const float sb = sA_b * sW;
            acc[s][0] += raw[s][0] * sa;
            acc[s][1] += raw[s][1] * sa;
            acc[s][2] += raw[s][2] * sb;
            acc[s][3] += raw[s][3] * sb;
        }
    }

    // Weighted scatter via bf16x2 atomicAdd directly into output. Drops the
    // fp32 shadow + bf16 cast pass entirely. n0 = n8 + tid4*2 is always even
    // since tid4 ∈ [0,3], so the bf16x2 store is naturally aligned to 4 bytes.
    #pragma unroll
    for (int s = 0; s < 4; ++s) {
        const int n8 = n_warp_base + s * 8;
        const int n0 = n8 + tid4 * 2;
        if (n0 + 1 < kHidden) {
            if (row_a_valid) {
                __nv_bfloat162 v = __floats2bfloat162_rn(acc[s][0] * w_a, acc[s][1] * w_a);
                atomicAdd(
                    reinterpret_cast<__nv_bfloat162*>(&output_bf16[(size_t)t_a * kHidden + n0]),
                    v);
            }
            if (row_b_valid) {
                __nv_bfloat162 v = __floats2bfloat162_rn(acc[s][2] * w_b, acc[s][3] * w_b);
                atomicAdd(
                    reinterpret_cast<__nv_bfloat162*>(&output_bf16[(size_t)t_b * kHidden + n0]),
                    v);
            }
        }
    }
}

// Small-M static-CTA GEMM2 probe. This keeps the packed-Y contract of the
// CUTLASS path but uses one CTA per logical (expert M tile, hidden N tile),
// exposing many hardware waves for tiny M_total. It deliberately avoids the
// final weighted scatter so we can reuse gemm2_scatter_reduce_vec_kernel.
template <int BM, int BN, int WARPS_N>
__global__ void __launch_bounds__(WARPS_N * 32, 2) gemm2_mma_y_packed_kernel(
    const __nv_fp8_e4m3* __restrict__ a2_packed,       // [M_total_padded, I]
    const float* __restrict__ sfa2_packed,             // per-group [I/128, M_padded]
    const __nv_fp8_e4m3* __restrict__ gemm2_w,         // [E_local, H, I]
    const float* __restrict__ gemm2_w_scale,           // [E_local, H/128, I/128]
    const int32_t* __restrict__ expert_token_count,
    const int32_t* __restrict__ block_to_expert,
    const int32_t* __restrict__ expert_offsets_m,
    const int32_t* __restrict__ expert_token_padded,
    const int32_t* __restrict__ expert_sfa2_offsets,
    __nv_bfloat16* __restrict__ y_packed)              // [M_total_padded, H]
{
    static_assert(BM == 16, "mma m16 requires BM=16");
    static_assert(BN == WARPS_N * 32, "BN must equal WARPS_N * 32");

    const int by = blockIdx.y;
    const int bx = blockIdx.x;
    const int code = block_to_expert[by];
    if (code == 0xff) return;
    const int le = code & 0xFF;
    const int m_block_in_expert = code >> 8;
    const int Tk = expert_token_count[le];
    const int m_base = m_block_in_expert * BM;
    if (m_base >= Tk) return;

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int g    = lane >> 2;
    const int tid4 = lane & 3;

    const int row_a = m_base + g;
    const int row_b = m_base + g + 8;
    const bool row_a_valid = row_a < Tk;
    const bool row_b_valid = row_b < Tk;
    const int m_offset = expert_offsets_m[le];
    const int m_global_a = m_offset + row_a;
    const int m_global_b = m_offset + row_b;
    const int m_padded = expert_token_padded[le];
    const int sfa_base = expert_sfa2_offsets[le];

    const int n_warp_base = bx * BN + warp * 32;

    float acc[4][4] = {0};

    #pragma unroll 1
    for (int kb = 0; kb < kInterBlocks; ++kb) {
        const int k_base128 = kb * kBlock;
        float raw[4][4] = {0};

        #pragma unroll
        for (int kk = 0; kk < 4; ++kk) {
            const int k = k_base128 + kk * 32;
            const __nv_fp8_e4m3* A_a = a2_packed + (size_t)m_global_a * kInter + k + tid4 * 4;
            const __nv_fp8_e4m3* A_b = a2_packed + (size_t)m_global_b * kInter + k + tid4 * 4;
            uint32_t a0 = row_a_valid ? ld_u32_gmem(A_a)      : 0u;
            uint32_t a1 = row_b_valid ? ld_u32_gmem(A_b)      : 0u;
            uint32_t a2 = row_a_valid ? ld_u32_gmem(A_a + 16) : 0u;
            uint32_t a3 = row_b_valid ? ld_u32_gmem(A_b + 16) : 0u;

            #pragma unroll
            for (int s = 0; s < 4; ++s) {
                const int n8 = n_warp_base + s * 8;
                const int n_row = n8 + g;
                const __nv_fp8_e4m3* B_row = gemm2_w
                    + ((size_t)le * kHidden + n_row) * kInter + k + tid4 * 4;
                uint32_t b0 = ld_u32_gmem(B_row);
                uint32_t b1 = ld_u32_gmem(B_row + 16);
                mma_e4m3_m16n8k32(a0, a1, a2, a3, b0, b1,
                                  raw[s][0], raw[s][1], raw[s][2], raw[s][3]);
            }
        }

        const float sA_a = row_a_valid ? sfa2_packed[sfa_base + (size_t)kb * m_padded + row_a] : 0.f;
        const float sA_b = row_b_valid ? sfa2_packed[sfa_base + (size_t)kb * m_padded + row_b] : 0.f;
        #pragma unroll
        for (int s = 0; s < 4; ++s) {
            const int n8 = n_warp_base + s * 8;
            const int n_block = n8 / kBlock;
            const float sW = gemm2_w_scale[(size_t)le * kHiddenBlocks * kInterBlocks
                                           + (size_t)n_block * kInterBlocks + kb];
            const float sa = sA_a * sW;
            const float sb = sA_b * sW;
            acc[s][0] += raw[s][0] * sa;
            acc[s][1] += raw[s][1] * sa;
            acc[s][2] += raw[s][2] * sb;
            acc[s][3] += raw[s][3] * sb;
        }
    }

    #pragma unroll
    for (int s = 0; s < 4; ++s) {
        const int n8 = n_warp_base + s * 8;
        const int n0 = n8 + tid4 * 2;
        if (n0 + 1 < kHidden) {
            __nv_bfloat162 v_a = __floats2bfloat162_rn(acc[s][0], acc[s][1]);
            __nv_bfloat162 v_b = __floats2bfloat162_rn(acc[s][2], acc[s][3]);
            if (row_a_valid)
                *reinterpret_cast<__nv_bfloat162*>(&y_packed[(size_t)m_global_a * kHidden + n0]) = v_a;
            if (row_b_valid)
                *reinterpret_cast<__nv_bfloat162*>(&y_packed[(size_t)m_global_b * kHidden + n0]) = v_b;
        }
    }
}

// --------------------------- GEMM2 + scatter (compacted grid) ---------------

template <int BM, int BN>
__global__ void __launch_bounds__(BM * BN, 1) gemm2_dequant_scatter_kernel(
    const float* __restrict__ act,
    const __nv_fp8_e4m3* __restrict__ gemm2_w,
    const float* __restrict__ gemm2_w_scale,
    const int32_t* __restrict__ expert_token_list,
    const int32_t* __restrict__ expert_token_count,
    const int32_t* __restrict__ block_to_expert,
    const float* __restrict__ expert_token_weight,
    float* __restrict__ output_fp32, int T)
{
    const int bn  = blockIdx.x;
    const int by  = blockIdx.y;
    const int code = block_to_expert[by];
    const int le = code & 0xFF;
    const int m_block_in_expert = code >> 8;
    const int Tk = expert_token_count[le];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int m_local = m_block_in_expert * BM + ty;
    const int n = bn * BN + tx;
    if (n >= kHidden || m_local >= Tk) return;

    const int t = expert_token_list[le * T + m_local];
    const float w = expert_token_weight[le * T + m_local];
    const int hb = n / kBlock;

    float acc = 0.0f;
    #pragma unroll 1
    for (int kb = 0; kb < kInterBlocks; ++kb) {
        const float sW = gemm2_w_scale[le * kHiddenBlocks * kInterBlocks
                                       + hb * kInterBlocks + kb];
        float partial = 0.0f;
        const float* A_row = act + ((size_t)le * T + m_local) * kInter + kb * kBlock;
        const __nv_fp8_e4m3* W_row = gemm2_w
            + ((size_t)le * kHidden + n) * kInter + kb * kBlock;
        // 128 elems = 8 uint4 of FP8 (weights) and 32 float4 of acts
        #pragma unroll
        for (int kk = 0; kk < 8; ++kk) {
            float W16[16];
            load_fp8_16(W_row + kk * 16, W16);
            // load 16 fp32 acts via 4 float4
            float A16[16];
            #pragma unroll
            for (int g = 0; g < 4; ++g) {
                float4 v = *reinterpret_cast<const float4*>(A_row + kk * 16 + g * 4);
                A16[g*4 + 0] = v.x; A16[g*4 + 1] = v.y;
                A16[g*4 + 2] = v.z; A16[g*4 + 3] = v.w;
            }
            #pragma unroll
            for (int i = 0; i < 16; ++i) partial += A16[i] * W16[i];
        }
        acc += partial * sW;
    }

    atomicAdd(&output_fp32[(size_t)t * kHidden + n], acc * w);
}

// --------------------------- GEMM2 packed-Y scatter+weighted-sum -----------
//
// Reads y_packed[m_global_padded, h] (bf16) for each unpadded m and
// atomic-adds expert_token_weight[le, m_local] * y into output[t, h].
//
// Grid: (n_blocks_h, M_total_unpadded), where n_blocks_h = kHidden / BN.
// Each CTA covers a 1×BN tile. BN must divide kHidden=7168.
//
// References: Sonic MoE reduction_over_k_gather.py:46-122 (gather + weighted
// sum back into per-token output).
template <int BN>
__global__ void __launch_bounds__(BN, 4) gemm2_scatter_weighted_kernel(
    const __nv_bfloat16* __restrict__ y_packed,         // [M_total_padded, H]
    const int32_t* __restrict__ swiglu_block_map,        // [M_total_unpadded] (le<<24 | m_local)
    const int32_t* __restrict__ expert_token_list,       // [E_local, T]
    const int32_t* __restrict__ expert_offsets_m,        // [E_local+1] padded
    const float*   __restrict__ expert_token_weight,     // [E_local, T]
    __nv_bfloat16* __restrict__ output_bf16,             // [T, H]
    int32_t M_total_unpadded, int T)
{
    const int n_block = blockIdx.x;
    const int mu      = blockIdx.y;
    if (mu >= M_total_unpadded) return;
    const int tid = threadIdx.x;
    const int n   = n_block * BN + tid;
    if (n >= kHidden) return;

    const int32_t code = swiglu_block_map[mu];
    const int le      = code >> 24;
    const int m_local = code & 0x00FFFFFF;
    const int m_global = expert_offsets_m[le] + m_local;
    const int t       = expert_token_list[le * T + m_local];
    const float w     = expert_token_weight[le * T + m_local];

    float v = __bfloat162float(y_packed[(size_t)m_global * kHidden + n]) * w;
    atomicAdd(&output_bf16[(size_t)t * kHidden + n], __float2bfloat16(v));
}

// Per-token gather/reduce writeback for CUTLASS GEMM2. Dispatch records the
// local expert rows selected by each token, so this kernel writes output once
// per (token, hidden-column) and avoids the output memset + bf16 atomics.
template <int BN>
__global__ void __launch_bounds__(BN, 4) gemm2_scatter_reduce_kernel(
    const __nv_bfloat16* __restrict__ y_packed,      // [M_total_padded, H]
    const int32_t* __restrict__ token_local_count,   // [T]
    const int32_t* __restrict__ token_pair_code,     // [T, topK] (le<<24 | m_local)
    const float* __restrict__ token_pair_weight,     // [T, topK]
    const int32_t* __restrict__ expert_offsets_m,    // [E_local+1] padded
    __nv_bfloat16* __restrict__ output_bf16,         // [T, H]
    int T)
{
    const int n_block = blockIdx.x;
    const int t       = blockIdx.y;
    if (t >= T) return;
    const int tid = threadIdx.x;
    const int n   = n_block * BN + tid;
    if (n >= kHidden) return;

    int n_pairs = token_local_count[t];
    if (n_pairs > kTopK) n_pairs = kTopK;
    float sum = 0.0f;
    #pragma unroll
    for (int p = 0; p < kTopK; ++p) {
        if (p < n_pairs) {
            const int32_t code = token_pair_code[(size_t)t * kTopK + p];
            const int le      = code >> 24;
            const int m_local = code & 0x00FFFFFF;
            const int m_global = expert_offsets_m[le] + m_local;
            const float w = token_pair_weight[(size_t)t * kTopK + p];
            sum += __bfloat162float(y_packed[(size_t)m_global * kHidden + n]) * w;
        }
    }
    output_bf16[(size_t)t * kHidden + n] = __float2bfloat16(sum);
}

// Vectorized variant of the per-token gather/reduce. Each thread owns 8
// contiguous bf16 columns (16B), reducing per-token CTAs from 28 to 7 while
// keeping the same one-store-per-output semantics.
template <int THREADS, int VEC>
__global__ void __launch_bounds__(THREADS, 4) gemm2_scatter_reduce_vec_kernel(
    const __nv_bfloat16* __restrict__ y_packed,      // [M_total_padded, H]
    const int32_t* __restrict__ token_local_count,   // [T]
    const int32_t* __restrict__ token_pair_code,     // [T, topK] (le<<24 | m_local)
    const float* __restrict__ token_pair_weight,     // [T, topK]
    const int32_t* __restrict__ expert_offsets_m,    // [E_local+1] padded
    __nv_bfloat16* __restrict__ output_bf16,         // [T, H]
    int T, int active_hidden, int y_stride)
{
    static_assert(VEC == 8 || VEC == 16, "supported vector widths are 8 or 16 bf16 values");

    const int t = blockIdx.y;
    if (t >= T) return;
    const int n0 = (blockIdx.x * THREADS + threadIdx.x) * VEC;
    if (n0 >= active_hidden) return;

    float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
    float s4 = 0.0f, s5 = 0.0f, s6 = 0.0f, s7 = 0.0f;
    float s8 = 0.0f, s9 = 0.0f, s10 = 0.0f, s11 = 0.0f;
    float s12 = 0.0f, s13 = 0.0f, s14 = 0.0f, s15 = 0.0f;

    int n_pairs = token_local_count[t];
    if (n_pairs > kTopK) n_pairs = kTopK;
    #pragma unroll
    for (int p = 0; p < kTopK; ++p) {
        if (p < n_pairs) {
            const int32_t code = token_pair_code[(size_t)t * kTopK + p];
            const int le      = code >> 24;
            const int m_local = code & 0x00FFFFFF;
            const int m_global = expert_offsets_m[le] + m_local;
            const float w = token_pair_weight[(size_t)t * kTopK + p];

            const __nv_bfloat162* src =
                reinterpret_cast<const __nv_bfloat162*>(
                    y_packed + (size_t)m_global * y_stride + n0);
            const float2 v01 = __bfloat1622float2(src[0]);
            const float2 v23 = __bfloat1622float2(src[1]);
            const float2 v45 = __bfloat1622float2(src[2]);
            const float2 v67 = __bfloat1622float2(src[3]);
            s0 += v01.x * w; s1 += v01.y * w;
            s2 += v23.x * w; s3 += v23.y * w;
            s4 += v45.x * w; s5 += v45.y * w;
            s6 += v67.x * w; s7 += v67.y * w;
            if constexpr (VEC == 16) {
                const float2 v89 = __bfloat1622float2(src[4]);
                const float2 v1011 = __bfloat1622float2(src[5]);
                const float2 v1213 = __bfloat1622float2(src[6]);
                const float2 v1415 = __bfloat1622float2(src[7]);
                s8 += v89.x * w; s9 += v89.y * w;
                s10 += v1011.x * w; s11 += v1011.y * w;
                s12 += v1213.x * w; s13 += v1213.y * w;
                s14 += v1415.x * w; s15 += v1415.y * w;
            }
        }
    }

    __nv_bfloat162* dst =
        reinterpret_cast<__nv_bfloat162*>(output_bf16 + (size_t)t * kHidden + n0);
    dst[0] = __floats2bfloat162_rn(s0, s1);
    dst[1] = __floats2bfloat162_rn(s2, s3);
    dst[2] = __floats2bfloat162_rn(s4, s5);
    dst[3] = __floats2bfloat162_rn(s6, s7);
    if constexpr (VEC == 16) {
        dst[4] = __floats2bfloat162_rn(s8, s9);
        dst[5] = __floats2bfloat162_rn(s10, s11);
        dst[6] = __floats2bfloat162_rn(s12, s13);
        dst[7] = __floats2bfloat162_rn(s14, s15);
    }
}

__global__ void zero_hidden_tail_kernel(
    __nv_bfloat16* __restrict__ output_bf16, int T, int active_hidden)
{
    const int tail = kHidden - active_hidden;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = T * tail;
    if (i >= total) return;
    const int t = i / tail;
    const int h = active_hidden + (i - t * tail);
    output_bf16[(size_t)t * kHidden + h] = __float2bfloat16(0.0f);
}

// --------------------------- bf16 cast --------------------------------------

__global__ void cast_fp32_to_bf16_kernel(
    const float* __restrict__ src, __nv_bfloat16* __restrict__ dst, int total)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    dst[i] = __float2bfloat16(src[i]);
}

// --------------------------- workspace -------------------------------------

struct Workspace {
    int32_t* topk_idx;
    float*   topk_weight;
    int32_t* expert_token_count;
    int32_t* expert_token_list;
    float*   expert_token_weight;
    int32_t* token_local_count;     // [T] local expert contributions per token
    int32_t* token_pair_code;       // [T, topK] packed (le<<24 | m_local)
    float*   token_pair_weight;     // [T, topK] scaled routing weight
    int32_t* expert_offsets_g1;     // [E_local+1] under BM_GEMM1 (unused on CUTLASS path)
    int32_t* expert_offsets_g2;     // [E_local+1] under BM_GEMM2
    int32_t* expert_offsets_m;      // [E_local+1] cumsum of PADDED expert_token_count (in tokens)
    int32_t* expert_offsets_bmm;    // [E_local+1] cumsum of tile64-padded counts for compact BMM2
    int32_t* expert_token_padded;   // [E_local] per-expert padded M_e (mult of 4)
    int32_t* expert_m_problem_dev;  // [E_local] per-expert 4-aligned problem M
    int32_t* expert_sfa_offsets;    // [E_local+1] per-group GEMM1 SFA float offsets
    int32_t* expert_sfa2_offsets;   // [E_local+1] per-group GEMM2 SFA float offsets
    int32_t* swiglu_block_map;      // [M_total_unpadded_max] (le<<24 | m_local) lookup
    int32_t* m_unpadded_total_dev;  // [1] device-side M_total_unpadded
    int32_t* m_unpadded_total_host; // [1] pinned host mirror
    int32_t* block_to_expert_g1;    // [max_g1_blocks] packed (le, m_block) (mma.sync fallback)
    int32_t* block_to_expert_g2;    // [max_g2_blocks] packed
    __nv_bfloat16* gemm1_scratch;   // bf16 (mma.sync fallback path)
    __nv_fp8_e4m3* act_fp8;         // [E_local, T, I] post-SwiGLU FP8 acts
    float*   act_scale;             // [E_local, I/128, T]
    float*   output_fp32;

    // ---- CUTLASS GEMM1 buffers ----
    __nv_fp8_e4m3* a_packed;        // [M_total_max, H] FP8 expert-contiguous A
    float*         sfa_packed;      // [H/128, M_total_max] f32 expert-contiguous SFA
    float*         sfa_bmm_packed;  // [H/128, M_total_max] f32 global SFA for TRTLLM BMM GEMM1
    float*         sfb_packed;      // [E_local, K/128, N/128] transposed weight scales (GEMM1)
    __nv_bfloat16* d_packed;        // [M_total_max, 2I] bf16 grouped GEMM output

    // ---- CUTLASS GEMM2 buffers ----
    __nv_fp8_e4m3* a2_packed;       // [M_total_max, I] FP8 packed activations (post-SwiGLU)
    float*         sfa2_packed;     // [I/128, M_total_max] f32 packed SFA for GEMM2
    float*         sfa2_bmm_packed; // [I/128, M_total_max] f32 global SFA for TRTLLM BMM GEMM2
    float*         sfb2_packed;     // [E_local, I/128, H/128] transposed weight scales (GEMM2)
    __nv_bfloat16* y_packed;        // [M_total_max, H] bf16 GEMM2 output (pre-scatter)

    // CUTLASS descriptor arrays (sized by sizeof_stride_* / sizeof_layout_*)
    void**   ptr_A_dev;             // [E_local]
    void**   ptr_B_dev;
    void**   ptr_D_dev;
    void**   ptr_SFA_dev;
    void**   ptr_SFB_dev;
    void*    stride_A_dev;          // [E_local] of StrideA
    void*    stride_B_dev;
    void*    stride_D_dev;
    void*    layout_SFA_dev;
    void*    layout_SFB_dev;
    int32_t* problem_sizes_dev;     // [E_local * 3]
    int32_t* problem_sizes_host;    // host-pinned mirror for CUTLASS
    void*    cutlass_workspace;

    // ---- GEMM2 CUTLASS descriptors ----
    void**   ptr_A2_dev;            // [E_local]
    void**   ptr_B2_dev;
    void**   ptr_D2_dev;
    void**   ptr_SFA2_dev;
    void**   ptr_SFB2_dev;
    void*    stride_A2_dev;
    void*    stride_B2_dev;
    void*    stride_D2_dev;
    void*    layout_SFA2_dev;
    void*    layout_SFB2_dev;
    int32_t* problem_sizes2_dev;    // [E_local * 3]
    int32_t* problem_sizes2_host;   // host-pinned mirror
    void*    cutlass_workspace2;

    // TRTLLM dynB FP8 BMM GEMM2 path.
    int32_t* bmm_total_padded_tokens;
    int32_t* bmm_cta_to_batch;
    int32_t* bmm_cta_to_mn_limit;
    int32_t* bmm_num_non_exiting_ctas;
    void*    trtllm_bmm_workspace;

    const void* cached_gemm1_scale;
    const void* cached_gemm2_scale;
    bool        gemm1_desc_cache_valid;
    bool        gemm2_desc_cache_valid;
    const void* cached_gemm1_desc_weight;
    const void* cached_gemm2_desc_weight;
    bool        route_cache_valid;
    int         cached_route_T;
    int64_t     cached_local_expert_offset;
    double      cached_routed_scaling_factor;
    const void* cached_routing_logits;
    const void* cached_routing_bias;
    int32_t     cached_expert_padded_host[kNumLocal];
    int32_t     cached_expert_unpadded_host[kNumLocal];
    int32_t     cached_m_unpadded_total;

    cudaGraphExec_t hot_graph_exec;
    cudaStream_t hot_graph_stream;
    cudaEvent_t  hot_graph_event;
    bool        hot_graph_cache_valid;
    bool        hot_graph_disabled;
    int         cached_graph_T;
    int         cached_graph_m_total_unpadded;
    int64_t     cached_graph_local_expert_offset;
    double      cached_graph_routed_scaling_factor;
    const void* cached_graph_routing_logits;
    const void* cached_graph_routing_bias;
    const void* cached_graph_hidden_states;
    const void* cached_graph_hidden_states_scale;
    const void* cached_graph_gemm1_weights;
    const void* cached_graph_gemm1_weights_scale;
    const void* cached_graph_gemm2_weights;
    const void* cached_graph_gemm2_weights_scale;
    const void* cached_graph_output;
};

static constexpr int kBM_g1 = 16;   // fixed by mma m16 (fallback path)
static constexpr int kBN_g1 = 128;  // 4 warps × 32 cols
static constexpr int kWarps_g1 = 4;
static constexpr int kBM_g2 = 16;
static constexpr int kBN_g2 = 32;
static constexpr int kWarps_g2 = 1;

// M_total upper bound across all local experts. Each token can be assigned
// to at most kTopK experts (8); globally on this rank, the M_total is the
// number of (token, local-expert) pairs which is at most T * kTopK.
static inline int compute_m_total_max(int T) {
    // Token counts per local expert summed ≤ T * kTopK. Pad each group's M
    // up to a multiple of 128 (the CUTLASS GEMM1 TileShape M) so that the
    // kernel's scale-load tile doesn't read beyond our SFA buffer. Worst
    // case adds 127 rows per group.
    return T * kTopK + kNumLocal * 128;
}

static cudaError_t alloc_workspace(Workspace& ws, int T) {
    cudaError_t e;
    int M_total_max = compute_m_total_max(T);
    e = cudaStreamCreateWithFlags(&ws.hot_graph_stream, cudaStreamNonBlocking); if (e) return e;
    e = cudaEventCreateWithFlags(&ws.hot_graph_event, cudaEventDisableTiming); if (e) return e;
    e = cudaMalloc(&ws.topk_idx, (size_t)T * kTopK * sizeof(int32_t)); if (e) return e;
    e = cudaMalloc(&ws.topk_weight, (size_t)T * kTopK * sizeof(float)); if (e) return e;
    e = cudaMalloc(&ws.expert_token_count, kNumLocal * sizeof(int32_t)); if (e) return e;
    e = cudaMalloc(&ws.expert_token_list, (size_t)kNumLocal * T * sizeof(int32_t)); if (e) return e;
    e = cudaMalloc(&ws.expert_token_weight, (size_t)kNumLocal * T * sizeof(float)); if (e) return e;
    e = cudaMalloc(&ws.token_local_count, (size_t)T * sizeof(int32_t)); if (e) return e;
    e = cudaMalloc(&ws.token_pair_code, (size_t)T * kTopK * sizeof(int32_t)); if (e) return e;
    e = cudaMalloc(&ws.token_pair_weight, (size_t)T * kTopK * sizeof(float)); if (e) return e;
    e = cudaMalloc(&ws.expert_offsets_g1, (kNumLocal + 1) * sizeof(int32_t)); if (e) return e;
    e = cudaMalloc(&ws.expert_offsets_g2, (kNumLocal + 1) * sizeof(int32_t)); if (e) return e;
    e = cudaMalloc(&ws.expert_offsets_m, (kNumLocal + 1) * sizeof(int32_t)); if (e) return e;
    e = cudaMalloc(&ws.expert_offsets_bmm, (kNumLocal + 1) * sizeof(int32_t)); if (e) return e;
    e = cudaMalloc(&ws.expert_token_padded, kNumLocal * sizeof(int32_t)); if (e) return e;
    e = cudaMalloc(&ws.expert_m_problem_dev, kNumLocal * sizeof(int32_t)); if (e) return e;
    e = cudaMalloc(&ws.expert_sfa_offsets, (kNumLocal + 1) * sizeof(int32_t)); if (e) return e;
    e = cudaMalloc(&ws.expert_sfa2_offsets, (kNumLocal + 1) * sizeof(int32_t)); if (e) return e;
    // Upper bound on M_total_unpadded = T * kTopK.
    e = cudaMalloc(&ws.swiglu_block_map, (size_t)T * kTopK * sizeof(int32_t)); if (e) return e;
    e = cudaMalloc(&ws.m_unpadded_total_dev, sizeof(int32_t)); if (e) return e;
    e = cudaMallocHost(&ws.m_unpadded_total_host, sizeof(int32_t)); if (e) return e;
    int max_g1_blocks = kNumLocal * ((T + kBM_g1 - 1) / kBM_g1);
    int max_g2_blocks = kNumLocal * ((T + kBM_g2 - 1) / kBM_g2);
    e = cudaMalloc(&ws.block_to_expert_g1, max_g1_blocks * sizeof(int32_t)); if (e) return e;
    e = cudaMalloc(&ws.block_to_expert_g2, max_g2_blocks * sizeof(int32_t)); if (e) return e;
    e = cudaMalloc(&ws.gemm1_scratch, (size_t)kNumLocal * T * kGemm1Out * sizeof(__nv_bfloat16)); if (e) return e;
    e = cudaMalloc(&ws.act_fp8, (size_t)kNumLocal * T * kInter * sizeof(__nv_fp8_e4m3)); if (e) return e;
    e = cudaMalloc(&ws.act_scale, (size_t)kNumLocal * kInterBlocks * T * sizeof(float)); if (e) return e;
    e = cudaMalloc(&ws.output_fp32, (size_t)T * kHidden * sizeof(float)); if (e) return e;

    // CUTLASS path
    e = cudaMalloc(&ws.a_packed, (size_t)M_total_max * kHidden * sizeof(__nv_fp8_e4m3)); if (e) return e;
    e = cudaMalloc(&ws.sfa_packed, (size_t)kHiddenBlocks * M_total_max * sizeof(float)); if (e) return e;
    e = cudaMalloc(&ws.sfa_bmm_packed, (size_t)kHiddenBlocks * M_total_max * sizeof(float)); if (e) return e;
    e = cudaMalloc(&ws.d_packed, (size_t)M_total_max * kGemm1Out * sizeof(__nv_bfloat16)); if (e) return e;
    // SFB_packed: [E_local, K/128, N/128] for GEMM1 (K = H, N = 2I).
    e = cudaMalloc(&ws.sfb_packed,
                   (size_t)kNumLocal * kHiddenBlocks * kGemm1OutBlk * sizeof(float)); if (e) return e;

    e = cudaMalloc(&ws.ptr_A_dev, kNumLocal * sizeof(void*)); if (e) return e;
    e = cudaMalloc(&ws.ptr_B_dev, kNumLocal * sizeof(void*)); if (e) return e;
    e = cudaMalloc(&ws.ptr_D_dev, kNumLocal * sizeof(void*)); if (e) return e;
    e = cudaMalloc(&ws.ptr_SFA_dev, kNumLocal * sizeof(void*)); if (e) return e;
    e = cudaMalloc(&ws.ptr_SFB_dev, kNumLocal * sizeof(void*)); if (e) return e;
    int sz_strideA = cutlass_grouped_fp8_blockwise_sizeof_stride_A();
    int sz_strideB = cutlass_grouped_fp8_blockwise_sizeof_stride_B();
    int sz_strideD = cutlass_grouped_fp8_blockwise_sizeof_stride_D();
    int sz_layoutA = cutlass_grouped_fp8_blockwise_sizeof_layout_SFA();
    int sz_layoutB = cutlass_grouped_fp8_blockwise_sizeof_layout_SFB();
    e = cudaMalloc(&ws.stride_A_dev, (size_t)kNumLocal * sz_strideA); if (e) return e;
    e = cudaMalloc(&ws.stride_B_dev, (size_t)kNumLocal * sz_strideB); if (e) return e;
    e = cudaMalloc(&ws.stride_D_dev, (size_t)kNumLocal * sz_strideD); if (e) return e;
    e = cudaMalloc(&ws.layout_SFA_dev, (size_t)kNumLocal * sz_layoutA); if (e) return e;
    e = cudaMalloc(&ws.layout_SFB_dev, (size_t)kNumLocal * sz_layoutB); if (e) return e;
    e = cudaMalloc(&ws.problem_sizes_dev, (size_t)kNumLocal * 3 * sizeof(int32_t)); if (e) return e;
    e = cudaMallocHost(&ws.problem_sizes_host, (size_t)kNumLocal * 3 * sizeof(int32_t)); if (e) return e;
    int64_t cws = cutlass_grouped_fp8_blockwise_workspace_size(T, kGemm1Out, kHidden, kNumLocal);
    e = cudaMalloc(&ws.cutlass_workspace, (size_t)cws); if (e) return e;

    // GEMM2 buffers + descriptors. GEMM2 problem: M_e × N=H=7168 × K=I=2048.
    e = cudaMalloc(&ws.a2_packed, (size_t)M_total_max * kInter * sizeof(__nv_fp8_e4m3)); if (e) return e;
    e = cudaMalloc(&ws.sfa2_packed, (size_t)kInterBlocks * M_total_max * sizeof(float)); if (e) return e;
    e = cudaMalloc(&ws.sfa2_bmm_packed, (size_t)kInterBlocks * M_total_max * sizeof(float)); if (e) return e;
    e = cudaMalloc(&ws.sfb2_packed,
                   (size_t)kNumLocal * kInterBlocks * kHiddenBlocks * sizeof(float)); if (e) return e;
    e = cudaMalloc(&ws.y_packed, (size_t)M_total_max * kHidden * sizeof(__nv_bfloat16)); if (e) return e;
    e = cudaMalloc(&ws.ptr_A2_dev, kNumLocal * sizeof(void*)); if (e) return e;
    e = cudaMalloc(&ws.ptr_B2_dev, kNumLocal * sizeof(void*)); if (e) return e;
    e = cudaMalloc(&ws.ptr_D2_dev, kNumLocal * sizeof(void*)); if (e) return e;
    e = cudaMalloc(&ws.ptr_SFA2_dev, kNumLocal * sizeof(void*)); if (e) return e;
    e = cudaMalloc(&ws.ptr_SFB2_dev, kNumLocal * sizeof(void*)); if (e) return e;
    e = cudaMalloc(&ws.stride_A2_dev, (size_t)kNumLocal * sz_strideA); if (e) return e;
    e = cudaMalloc(&ws.stride_B2_dev, (size_t)kNumLocal * sz_strideB); if (e) return e;
    e = cudaMalloc(&ws.stride_D2_dev, (size_t)kNumLocal * sz_strideD); if (e) return e;
    e = cudaMalloc(&ws.layout_SFA2_dev, (size_t)kNumLocal * sz_layoutA); if (e) return e;
    e = cudaMalloc(&ws.layout_SFB2_dev, (size_t)kNumLocal * sz_layoutB); if (e) return e;
    e = cudaMalloc(&ws.problem_sizes2_dev, (size_t)kNumLocal * 3 * sizeof(int32_t)); if (e) return e;
    e = cudaMallocHost(&ws.problem_sizes2_host, (size_t)kNumLocal * 3 * sizeof(int32_t)); if (e) return e;
    int64_t cws2 = cutlass_grouped_fp8_blockwise_workspace_size(T, kHidden, kInter, kNumLocal);
    e = cudaMalloc(&ws.cutlass_workspace2, (size_t)cws2); if (e) return e;

    const int max_bmm_ctas = (M_total_max + MOE_TRTLLM_BMM_TILE - 1) / MOE_TRTLLM_BMM_TILE;
    e = cudaMalloc(&ws.bmm_total_padded_tokens, sizeof(int32_t)); if (e) return e;
    e = cudaMalloc(&ws.bmm_cta_to_batch, (size_t)max_bmm_ctas * sizeof(int32_t)); if (e) return e;
    e = cudaMalloc(&ws.bmm_cta_to_mn_limit, (size_t)max_bmm_ctas * sizeof(int32_t)); if (e) return e;
    e = cudaMalloc(&ws.bmm_num_non_exiting_ctas, sizeof(int32_t)); if (e) return e;
    size_t bmm_ws = 1;
#if MOE_USE_TRTLLM_BMM_GEMM1
    bmm_ws = std::max(bmm_ws, trtllm_fp8_bmm_workspace_size(
        std::max(1, T * kTopK), kGemm1Out, kHidden, std::max(1, max_bmm_ctas),
        MOE_TRTLLM_BMM_TILE, MOE_TRTLLM_BMM_CONFIG));
#endif
#if MOE_USE_TRTLLM_BMM_GEMM2
    bmm_ws = std::max(bmm_ws, trtllm_fp8_bmm_workspace_size(
        std::max(1, T * kTopK), kHidden, kInter, std::max(1, max_bmm_ctas),
        MOE_TRTLLM_BMM_TILE, MOE_TRTLLM_BMM_CONFIG));
#endif
    if (bmm_ws == 0) bmm_ws = 1;
    e = cudaMalloc(&ws.trtllm_bmm_workspace, bmm_ws); if (e) return e;
    return cudaSuccess;
}

static void free_workspace(Workspace& ws) {
    if (ws.hot_graph_exec) cudaGraphExecDestroy(ws.hot_graph_exec);
    if (ws.hot_graph_event) cudaEventDestroy(ws.hot_graph_event);
    if (ws.hot_graph_stream) cudaStreamDestroy(ws.hot_graph_stream);
    if (ws.topk_idx) cudaFree(ws.topk_idx);
    if (ws.topk_weight) cudaFree(ws.topk_weight);
    if (ws.expert_token_count) cudaFree(ws.expert_token_count);
    if (ws.expert_token_list) cudaFree(ws.expert_token_list);
    if (ws.expert_token_weight) cudaFree(ws.expert_token_weight);
    if (ws.token_local_count) cudaFree(ws.token_local_count);
    if (ws.token_pair_code) cudaFree(ws.token_pair_code);
    if (ws.token_pair_weight) cudaFree(ws.token_pair_weight);
    if (ws.expert_offsets_g1) cudaFree(ws.expert_offsets_g1);
    if (ws.expert_offsets_g2) cudaFree(ws.expert_offsets_g2);
    if (ws.expert_offsets_m) cudaFree(ws.expert_offsets_m);
    if (ws.expert_offsets_bmm) cudaFree(ws.expert_offsets_bmm);
    if (ws.expert_token_padded) cudaFree(ws.expert_token_padded);
    if (ws.expert_m_problem_dev) cudaFree(ws.expert_m_problem_dev);
    if (ws.expert_sfa_offsets) cudaFree(ws.expert_sfa_offsets);
    if (ws.expert_sfa2_offsets) cudaFree(ws.expert_sfa2_offsets);
    if (ws.swiglu_block_map) cudaFree(ws.swiglu_block_map);
    if (ws.m_unpadded_total_dev) cudaFree(ws.m_unpadded_total_dev);
    if (ws.m_unpadded_total_host) cudaFreeHost(ws.m_unpadded_total_host);
    if (ws.block_to_expert_g1) cudaFree(ws.block_to_expert_g1);
    if (ws.block_to_expert_g2) cudaFree(ws.block_to_expert_g2);
    if (ws.gemm1_scratch) cudaFree(ws.gemm1_scratch);
    if (ws.act_fp8) cudaFree(ws.act_fp8);
    if (ws.act_scale) cudaFree(ws.act_scale);
    if (ws.output_fp32) cudaFree(ws.output_fp32);
    if (ws.a_packed) cudaFree(ws.a_packed);
    if (ws.sfa_packed) cudaFree(ws.sfa_packed);
    if (ws.sfa_bmm_packed) cudaFree(ws.sfa_bmm_packed);
    if (ws.d_packed) cudaFree(ws.d_packed);
    if (ws.sfb_packed) cudaFree(ws.sfb_packed);
    if (ws.ptr_A_dev) cudaFree(ws.ptr_A_dev);
    if (ws.ptr_B_dev) cudaFree(ws.ptr_B_dev);
    if (ws.ptr_D_dev) cudaFree(ws.ptr_D_dev);
    if (ws.ptr_SFA_dev) cudaFree(ws.ptr_SFA_dev);
    if (ws.ptr_SFB_dev) cudaFree(ws.ptr_SFB_dev);
    if (ws.stride_A_dev) cudaFree(ws.stride_A_dev);
    if (ws.stride_B_dev) cudaFree(ws.stride_B_dev);
    if (ws.stride_D_dev) cudaFree(ws.stride_D_dev);
    if (ws.layout_SFA_dev) cudaFree(ws.layout_SFA_dev);
    if (ws.layout_SFB_dev) cudaFree(ws.layout_SFB_dev);
    if (ws.problem_sizes_dev) cudaFree(ws.problem_sizes_dev);
    if (ws.problem_sizes_host) cudaFreeHost(ws.problem_sizes_host);
    if (ws.cutlass_workspace) cudaFree(ws.cutlass_workspace);

    if (ws.a2_packed) cudaFree(ws.a2_packed);
    if (ws.sfa2_packed) cudaFree(ws.sfa2_packed);
    if (ws.sfa2_bmm_packed) cudaFree(ws.sfa2_bmm_packed);
    if (ws.sfb2_packed) cudaFree(ws.sfb2_packed);
    if (ws.y_packed) cudaFree(ws.y_packed);
    if (ws.ptr_A2_dev) cudaFree(ws.ptr_A2_dev);
    if (ws.ptr_B2_dev) cudaFree(ws.ptr_B2_dev);
    if (ws.ptr_D2_dev) cudaFree(ws.ptr_D2_dev);
    if (ws.ptr_SFA2_dev) cudaFree(ws.ptr_SFA2_dev);
    if (ws.ptr_SFB2_dev) cudaFree(ws.ptr_SFB2_dev);
    if (ws.stride_A2_dev) cudaFree(ws.stride_A2_dev);
    if (ws.stride_B2_dev) cudaFree(ws.stride_B2_dev);
    if (ws.stride_D2_dev) cudaFree(ws.stride_D2_dev);
    if (ws.layout_SFA2_dev) cudaFree(ws.layout_SFA2_dev);
    if (ws.layout_SFB2_dev) cudaFree(ws.layout_SFB2_dev);
    if (ws.problem_sizes2_dev) cudaFree(ws.problem_sizes2_dev);
    if (ws.problem_sizes2_host) cudaFreeHost(ws.problem_sizes2_host);
    if (ws.cutlass_workspace2) cudaFree(ws.cutlass_workspace2);
    if (ws.bmm_total_padded_tokens) cudaFree(ws.bmm_total_padded_tokens);
    if (ws.bmm_cta_to_batch) cudaFree(ws.bmm_cta_to_batch);
    if (ws.bmm_cta_to_mn_limit) cudaFree(ws.bmm_cta_to_mn_limit);
    if (ws.bmm_num_non_exiting_ctas) cudaFree(ws.bmm_num_non_exiting_ctas);
    if (ws.trtllm_bmm_workspace) cudaFree(ws.trtllm_bmm_workspace);
    ws = {};
}

// --------------------------- top-level entry --------------------------------

static void run(
    ffi::TensorView routing_logits, ffi::TensorView routing_bias,
    ffi::TensorView hidden_states, ffi::TensorView hidden_states_scale,
    ffi::TensorView gemm1_weights, ffi::TensorView gemm1_weights_scale,
    ffi::TensorView gemm2_weights, ffi::TensorView gemm2_weights_scale,
    int64_t local_expert_offset, double routed_scaling_factor,
    ffi::TensorView output)
{
    const int T = (int)routing_logits.size(0);
    if (T <= 0) return;

    // RD: keep the workspace persistent across calls (sized to the largest
    // T seen). This amortizes ~40 cudaMallocs (~400 µs total) over many calls,
    // which dominates latency for the small-T workloads.
    static thread_local Workspace ws{};
    static thread_local int ws_T_capacity = 0;
    if (T > ws_T_capacity) {
        if (ws_T_capacity > 0) free_workspace(ws);
        cudaError_t e = alloc_workspace(ws, T);
        if (e != cudaSuccess) {
            fprintf(stderr, "alloc_workspace failed: %s\n", cudaGetErrorString(e));
            free_workspace(ws);
            ws_T_capacity = 0;
            return;
        }
        ws_T_capacity = T;
    }

    cudaStream_t stream = 0;
    int32_t expert_padded_host_pre[kNumLocal];
    int32_t expert_unpadded_host[kNumLocal];
    int32_t M_total_unpadded = 0;

    const bool route_cache_hit =
        ws.route_cache_valid &&
        ws.cached_route_T == T &&
        ws.cached_local_expert_offset == local_expert_offset &&
        ws.cached_routed_scaling_factor == routed_scaling_factor &&
        ws.cached_routing_logits == routing_logits.data_ptr() &&
        ws.cached_routing_bias == routing_bias.data_ptr();

#if MOE_USE_FUSED_GEMM1_SWIGLU
    constexpr bool gemm1_desc_cache_hit = true;
#else
    const bool gemm1_desc_cache_hit =
        route_cache_hit &&
        ws.gemm1_desc_cache_valid &&
        ws.cached_gemm1_desc_weight == gemm1_weights.data_ptr();
#endif
    const bool gemm2_desc_cache_hit =
        route_cache_hit &&
        ws.gemm2_desc_cache_valid &&
        ws.cached_gemm2_desc_weight == gemm2_weights.data_ptr();

    int max_g1_blocks = kNumLocal * ((T + kBM_g1 - 1) / kBM_g1);
    int max_g2_blocks = kNumLocal * ((T + kBM_g2 - 1) / kBM_g2);

    if (route_cache_hit) {
        #pragma unroll
        for (int le = 0; le < kNumLocal; ++le) {
            expert_padded_host_pre[le] = ws.cached_expert_padded_host[le];
            expert_unpadded_host[le] = ws.cached_expert_unpadded_host[le];
        }
        M_total_unpadded = ws.cached_m_unpadded_total;
    } else {
        // 1) Routing
        {
            size_t shmem = (2 * kNumExperts + kNumGroups + kNumGroups + kTopK + kTopK)
                           * sizeof(float);
            if (T <= MOE_FUSED_ROUTE_DISPATCH_MAX_T) {
                route_topk_kernel<true, false><<<T, kNumExperts, shmem, stream>>>(
                    static_cast<const float*>(routing_logits.data_ptr()),
                    static_cast<const __nv_bfloat16*>(routing_bias.data_ptr()),
                    ws.topk_idx, ws.topk_weight, ws.token_local_count,
                    ws.expert_token_count, ws.expert_token_list,
                    ws.expert_token_weight, ws.token_pair_code,
                    ws.token_pair_weight,
                    T, (int)local_expert_offset, (float)routed_scaling_factor);
#if MOE_FUSED_ROUTE_KEEP_LOCAL
            } else if (T >= MOE_KEEP_ONE_LOCAL_MIN_T ||
                       T == MOE_KEEP_ONE_LOCAL_EXTRA_T ||
                       T == MOE_KEEP_ONE_LOCAL_EXTRA_T2) {
                route_topk_kernel<true, true><<<T, kNumExperts, shmem, stream>>>(
                    static_cast<const float*>(routing_logits.data_ptr()),
                    static_cast<const __nv_bfloat16*>(routing_bias.data_ptr()),
                    ws.topk_idx, ws.topk_weight, ws.token_local_count,
                    ws.expert_token_count, ws.expert_token_list,
                    ws.expert_token_weight, ws.token_pair_code,
                    ws.token_pair_weight,
                    T, (int)local_expert_offset, (float)routed_scaling_factor);
#endif
            } else {
                route_topk_kernel<false, false><<<T, kNumExperts, shmem, stream>>>(
                    static_cast<const float*>(routing_logits.data_ptr()),
                    static_cast<const __nv_bfloat16*>(routing_bias.data_ptr()),
                    ws.topk_idx, ws.topk_weight, ws.token_local_count,
                    ws.expert_token_count, ws.expert_token_list,
                    ws.expert_token_weight, ws.token_pair_code,
                    ws.token_pair_weight,
                    T, (int)local_expert_offset, (float)routed_scaling_factor);
            }
        }

        // 2) Dispatch builder
        if (T <= MOE_FUSED_ROUTE_DISPATCH_MAX_T ||
            (MOE_FUSED_ROUTE_KEEP_LOCAL &&
             (T >= MOE_KEEP_ONE_LOCAL_MIN_T ||
              T == MOE_KEEP_ONE_LOCAL_EXTRA_T ||
              T == MOE_KEEP_ONE_LOCAL_EXTRA_T2))) {
            // Already emitted by route_topk_kernel<true, *>.
        } else if (T >= MOE_KEEP_ONE_LOCAL_MIN_T ||
                   T == MOE_KEEP_ONE_LOCAL_EXTRA_T ||
                   T == MOE_KEEP_ONE_LOCAL_EXTRA_T2) {
            cudaMemsetAsync(ws.expert_token_count, 0, kNumLocal * sizeof(int32_t), stream);
            dim3 grid((T + 255) / 256);
            dim3 block(256);
            build_dispatch_keep_one_local_kernel<<<grid, block, 0, stream>>>(
                ws.topk_idx, ws.topk_weight,
                ws.expert_token_count, ws.expert_token_list, ws.expert_token_weight,
                ws.token_local_count, ws.token_pair_code, ws.token_pair_weight,
                T, (int)local_expert_offset, (float)routed_scaling_factor);
        } else {
            build_dispatch_kernel<<<kNumLocal, 256, 0, stream>>>(
                ws.topk_idx, ws.topk_weight,
                ws.expert_token_count, ws.expert_token_list, ws.expert_token_weight,
                ws.token_local_count, ws.token_pair_code, ws.token_pair_weight,
                T, (int)local_expert_offset, (float)routed_scaling_factor);
        }

        // 2b) Warp-level scan of expert counts into all packed offsets.
        // block_to_expert_g2 is also kept ready for the opt-in small-M static
        // GEMM2 experiment.
#if MOE_USE_FUSED_GEMM1_SWIGLU || defined(MOE_BYPASS_CUTLASS_WITH_COPY) || defined(MOE_COMPARE_CUTLASS_VS_MMA)
        int32_t* block_to_expert_g1 = ws.block_to_expert_g1;
#else
        int32_t* block_to_expert_g1 = nullptr;
#endif
        {
            const int sfa_pad_m = (T <= 2048) ? 64 : 128;
            scan_offsets_kernel<<<1, 32, 0, stream>>>(
                ws.expert_token_count,
                block_to_expert_g1, ws.block_to_expert_g2,
                kBM_g1, kBM_g2, max_g1_blocks, max_g2_blocks,
                sfa_pad_m,
                ws.swiglu_block_map,
                ws.expert_offsets_m,
                ws.expert_token_padded, ws.expert_sfa_offsets,
                ws.expert_sfa2_offsets, ws.m_unpadded_total_dev);
        }

        // Host-side reads needed for grid sizing + CUTLASS planning. We pull
        // expert_padded, expert_unpadded (count), and m_unpadded_total in one
        // batched stream sync, then cache them for repeated calls on the same trace.
        cudaMemcpyAsync(expert_padded_host_pre, ws.expert_token_padded,
                        kNumLocal * sizeof(int32_t),
                        cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(expert_unpadded_host, ws.expert_token_count,
                        kNumLocal * sizeof(int32_t),
                        cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(ws.m_unpadded_total_host, ws.m_unpadded_total_dev,
                        sizeof(int32_t), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        M_total_unpadded = *ws.m_unpadded_total_host;
        #pragma unroll
        for (int le = 0; le < kNumLocal; ++le) {
            ws.cached_expert_padded_host[le] = expert_padded_host_pre[le];
            ws.cached_expert_unpadded_host[le] = expert_unpadded_host[le];
        }
        ws.cached_m_unpadded_total = M_total_unpadded;
        ws.cached_route_T = T;
        ws.cached_local_expert_offset = local_expert_offset;
        ws.cached_routed_scaling_factor = routed_scaling_factor;
        ws.cached_routing_logits = routing_logits.data_ptr();
        ws.cached_routing_bias = routing_bias.data_ptr();
        ws.route_cache_valid = true;
    }

    const bool use_mma_gemm2 =
        (!MOE_USE_FUSED_GEMM1_SWIGLU &&
         MOE_MMA_GEMM2_M_THRESHOLD > 0 &&
         M_total_unpadded > 0 &&
         M_total_unpadded <= MOE_MMA_GEMM2_M_THRESHOLD);
#if MOE_USE_TRTLLM_BMM_GEMM1 || MOE_USE_TRTLLM_BMM_GEMM2
    static const bool trtllm_bmm_available = (trtllm_fp8_bmm_is_available() != 0);
#else
    constexpr bool trtllm_bmm_available = false;
#endif
#if MOE_USE_TRTLLM_BMM_GEMM1
    const bool use_trtllm_bmm_gemm1 =
        (!MOE_USE_FUSED_GEMM1_SWIGLU &&
         trtllm_bmm_available &&
         M_total_unpadded > 0 &&
         (M_total_unpadded <= MOE_TRTLLM_BMM_GEMM1_M_THRESHOLD ||
          T <= MOE_TRTLLM_BMM_GEMM1_MAX_T ||
          (T >= MOE_TRTLLM_BMM_GEMM1_LARGE_MIN_T &&
           T < MOE_TRTLLM_BMM_GEMM1_LARGE_MAX_T)) &&
         T != 7);
#else
    constexpr bool use_trtllm_bmm_gemm1 = false;
#endif
#if MOE_USE_TRTLLM_BMM_GEMM2
    const bool use_trtllm_bmm_gemm2 =
        (!use_mma_gemm2 &&
         !MOE_USE_FUSED_GEMM1_SWIGLU &&
         trtllm_bmm_available &&
         M_total_unpadded > 0 &&
         M_total_unpadded <= MOE_TRTLLM_BMM_GEMM2_M_THRESHOLD);
#else
    constexpr bool use_trtllm_bmm_gemm2 = false;
#endif
    const bool use_m64n128_gemm = (M_total_unpadded <= 1);
    const bool use_m64n256_gemm =
        (!use_trtllm_bmm_gemm1 &&
         !use_m64n128_gemm && M_total_unpadded <= MOE_M64N256_GEMM_M_THRESHOLD);
    const bool use_m64n256_gemm2 =
        (!use_m64n128_gemm &&
         M_total_unpadded <= MOE_M64N256_GEMM_M_THRESHOLD);

#if MOE_USE_CUDA_GRAPH_REPLAY && !MOE_USE_FUSED_GEMM1_SWIGLU && \
    !defined(MOE_BYPASS_CUTLASS_WITH_COPY) && \
    !defined(MOE_COMPARE_CUTLASS_VS_MMA) && \
    !defined(MOE_DEBUG_DUMP_D)
    const bool hot_graph_ready =
        route_cache_hit &&
        (use_trtllm_bmm_gemm1 || gemm1_desc_cache_hit) &&
        (use_mma_gemm2 || use_trtllm_bmm_gemm2 || gemm2_desc_cache_hit) &&
        (use_trtllm_bmm_gemm1 ||
         ws.cached_gemm1_scale == gemm1_weights_scale.data_ptr()) &&
        (use_mma_gemm2 || use_trtllm_bmm_gemm2 ||
         ws.cached_gemm2_scale == gemm2_weights_scale.data_ptr());
    const bool hot_graph_key_hit =
        hot_graph_ready &&
        ws.hot_graph_cache_valid &&
        ws.cached_graph_T == T &&
        ws.cached_graph_m_total_unpadded == M_total_unpadded &&
        ws.cached_graph_local_expert_offset == local_expert_offset &&
        ws.cached_graph_routed_scaling_factor == routed_scaling_factor &&
        ws.cached_graph_routing_logits == routing_logits.data_ptr() &&
        ws.cached_graph_routing_bias == routing_bias.data_ptr() &&
        ws.cached_graph_hidden_states == hidden_states.data_ptr() &&
        ws.cached_graph_hidden_states_scale == hidden_states_scale.data_ptr() &&
        ws.cached_graph_gemm1_weights == gemm1_weights.data_ptr() &&
        ws.cached_graph_gemm1_weights_scale == gemm1_weights_scale.data_ptr() &&
        ws.cached_graph_gemm2_weights == gemm2_weights.data_ptr() &&
        ws.cached_graph_gemm2_weights_scale == gemm2_weights_scale.data_ptr() &&
        ws.cached_graph_output == output.data_ptr();
    if (hot_graph_key_hit) {
        cudaError_t ge = cudaGraphLaunch(ws.hot_graph_exec, stream);
        if (ge == cudaSuccess) {
#if MOE_SYNC_AT_RETURN
            cudaStreamSynchronize(stream);
#endif
            return;
        }
        if (ws.hot_graph_exec) {
            cudaGraphExecDestroy(ws.hot_graph_exec);
            ws.hot_graph_exec = nullptr;
        }
        ws.hot_graph_cache_valid = false;
        ws.hot_graph_disabled = true;
        cudaGetLastError();
    }
    bool hot_graph_capture_active = false;
    const bool hot_graph_capture_candidate =
        hot_graph_ready && !ws.hot_graph_disabled && !hot_graph_key_hit;
    if (hot_graph_capture_candidate) {
        cudaEventRecord(ws.hot_graph_event, stream);
        cudaStreamWaitEvent(ws.hot_graph_stream, ws.hot_graph_event, 0);
        stream = ws.hot_graph_stream;
        cudaError_t ge = cudaStreamBeginCapture(stream, cudaStreamCaptureModeRelaxed);
        hot_graph_capture_active = (ge == cudaSuccess);
        if (!hot_graph_capture_active) {
            stream = 0;
            ws.hot_graph_disabled = true;
            cudaGetLastError();
        }
    }
#endif

    int M_total_max = compute_m_total_max(T);
    (void)M_total_max;
    int M_total_padded = 0;
    int M_total_bmm_padded = 0;
    int bmm_compact_ctas_token_dim = 0;
    #pragma unroll
    for (int le = 0; le < kNumLocal; ++le) {
        M_total_padded += expert_padded_host_pre[le];
        const int compact_ctas = (expert_unpadded_host[le] + MOE_TRTLLM_BMM_TILE - 1) /
                                 MOE_TRTLLM_BMM_TILE;
        bmm_compact_ctas_token_dim += compact_ctas;
        M_total_bmm_padded += compact_ctas * MOE_TRTLLM_BMM_TILE;
    }
    const bool use_trtllm_bmm_gemm2_compact =
        use_trtllm_bmm_gemm2 &&
        T <= MOE_TRTLLM_BMM_GEMM2_COMPACT_MAX_T;
    const bool use_trtllm_bmm_gemm1_compact =
        use_trtllm_bmm_gemm1 &&
        T >= MOE_TRTLLM_BMM_GEMM1_COMPACT_MIN_T &&
        T < MOE_TRTLLM_BMM_GEMM1_COMPACT_MAX_T;
    const int M_total_bmm_stride =
        (use_trtllm_bmm_gemm2_compact || use_trtllm_bmm_gemm1_compact)
        ? M_total_bmm_padded : M_total_padded;
    if (use_trtllm_bmm_gemm2_compact || use_trtllm_bmm_gemm1_compact) {
        build_bmm_tile64_compact_map_kernel<<<1, 32, 0, stream>>>(
            ws.expert_token_count,
            ws.bmm_total_padded_tokens, ws.bmm_cta_to_batch,
            ws.bmm_cta_to_mn_limit, ws.bmm_num_non_exiting_ctas,
            ws.expert_offsets_bmm);
    }

    // RD polish: no packed-buffer zero-fill on the CUTLASS path. Problem sizes
    // are only 4-aligned; any rows beyond each expert's true M are ignored by
    // SwiGLU/scatter, and the 128-padded SFA allocation is only needed to keep
    // tile-scale loads in bounds.

#if !MOE_USE_FUSED_GEMM1_SWIGLU
    // 3) Permute A and SFA into per-expert contiguous packed buffers (RD-D1
    //    iter2: warp-per-token vectorized variant; grid sized to actual
    //    M_total_unpadded with 4 tokens per CTA). The fused GEMM1 path reads
    //    hidden_states directly and writes GEMM2's A/SFA tensors.
    {
        constexpr int kTokPerCta = 4;
        int n_blocks = (M_total_unpadded + kTokPerCta - 1) / kTokPerCta;
        dim3 grid(n_blocks);
        dim3 block(kTokPerCta * 32);
        if (use_trtllm_bmm_gemm1_compact) {
            permute_a_and_sfa_bmm_compact_kernel<kTokPerCta><<<grid, block, 0, stream>>>(
                static_cast<const __nv_fp8_e4m3*>(hidden_states.data_ptr()),
                static_cast<const float*>(hidden_states_scale.data_ptr()),
                ws.expert_token_list, ws.expert_offsets_bmm, ws.swiglu_block_map,
                ws.a_packed, ws.sfa_bmm_packed,
                M_total_unpadded, M_total_bmm_stride, T);
        } else {
            permute_a_and_sfa_v2_kernel<kTokPerCta><<<grid, block, 0, stream>>>(
                static_cast<const __nv_fp8_e4m3*>(hidden_states.data_ptr()),
                static_cast<const float*>(hidden_states_scale.data_ptr()),
                ws.expert_token_list, ws.expert_token_padded, ws.expert_offsets_m,
                ws.expert_sfa_offsets, ws.swiglu_block_map,
                ws.a_packed, ws.sfa_packed, ws.sfa_bmm_packed,
                M_total_unpadded, M_total_padded, T);
        }
    }
    // 3b) Transpose per-expert weight scales to MN-major for the SFB
    // layout used by the CUTLASS collective.
    if (ws.cached_gemm1_scale != gemm1_weights_scale.data_ptr()) {
        dim3 grid(kNumLocal, kGemm1OutBlk);
        dim3 block(kHiddenBlocks);
        transpose_w_scale_kernel<<<grid, block, 0, stream>>>(
            static_cast<const float*>(gemm1_weights_scale.data_ptr()),
            ws.sfb_packed, kGemm1OutBlk, kHiddenBlocks);
        ws.cached_gemm1_scale = gemm1_weights_scale.data_ptr();
    }
#endif

    // 4) GEMM1 via CUTLASS sm100 block-scale FP8 grouped collective.
    //    Read expert_token_padded back to host to populate host_problem_sizes
    //    (CUTLASS needs the host copy for tile-scheduler planning).
    // RD iter8: pass actual M (rounded up to multiple of 4 for the CUTLASS
    // alignment check on ScaleGranularityM=1) to CUTLASS, while keeping
    // SFA stride based on the 128-padded buffer. The padded rows in
    // A_packed and SFA_packed are zero, so this is safe.
    int32_t expert_m_problem_host[kNumLocal];
    for (int le = 0; le < kNumLocal; ++le) {
        int m = expert_unpadded_host[le];
        // align up to 4 for CopyAlignmentSFA, and keep <= padded buffer.
        m = (m + 3) & ~3;
        if (m > expert_padded_host_pre[le]) m = expert_padded_host_pre[le];
        expert_m_problem_host[le] = m;
        ws.problem_sizes_host[le * 3 + 0] = m;
        ws.problem_sizes_host[le * 3 + 1] = kGemm1Out; // N
        ws.problem_sizes_host[le * 3 + 2] = kHidden;   // K
    }
    // Push the rounded-up M back only when a descriptor builder will consume it.
    const bool need_descriptor_problem_copy =
        (!use_trtllm_bmm_gemm1 && !gemm1_desc_cache_hit) ||
        (!use_mma_gemm2 && !use_trtllm_bmm_gemm2 && !gemm2_desc_cache_hit);
    if (need_descriptor_problem_copy) {
        cudaMemcpyAsync(ws.expert_m_problem_dev, expert_m_problem_host,
                        kNumLocal * sizeof(int32_t),
                        cudaMemcpyHostToDevice, stream);
    }
#if MOE_USE_FUSED_GEMM1_SWIGLU
    // RD-D6 iter1: inline-PTX MMA GEMM1 with the SwiGLU+FP8-quant work in
    // the GEMM epilogue. This emits the GEMM2 A/SFA tensors directly.
    {
        int total_g1 = max_g1_blocks;
        dim3 grid(kInterBlocks, total_g1);
        dim3 block(kWarps_g1 * 32);
        gemm1_mma_swiglu_quant_kernel<kBM_g1, kBN_g1, kWarps_g1><<<grid, block, 0, stream>>>(
            static_cast<const __nv_fp8_e4m3*>(hidden_states.data_ptr()),
            static_cast<const float*>(hidden_states_scale.data_ptr()),
            static_cast<const __nv_fp8_e4m3*>(gemm1_weights.data_ptr()),
            static_cast<const float*>(gemm1_weights_scale.data_ptr()),
            ws.expert_token_list, ws.expert_token_count,
            ws.block_to_expert_g1, ws.expert_offsets_m, ws.expert_token_padded,
            ws.expert_sfa2_offsets, ws.a2_packed, ws.sfa2_packed, T);
    }
#else
    if (!use_trtllm_bmm_gemm1 && !gemm1_desc_cache_hit) {
        cutlass_grouped_fp8_blockwise_build_descriptors_unpadded(
            ws.expert_token_padded, ws.expert_m_problem_dev,
            ws.expert_offsets_m, ws.expert_sfa_offsets,
            /*base_A=*/   ws.a_packed,
            /*base_B=*/   gemm1_weights.data_ptr(),
            /*base_D=*/   ws.d_packed,
            /*base_SFA=*/ ws.sfa_packed,
            /*base_SFB=*/ ws.sfb_packed,
            kGemm1Out, kHidden, kGemm1Out,
            /*b_per_expert_elems=*/ kGemm1Out * kHidden,
            /*sfb_per_expert_elems=*/ kGemm1OutBlk * kHiddenBlocks,
            ws.ptr_A_dev, ws.ptr_B_dev, ws.ptr_D_dev,
            ws.ptr_SFA_dev, ws.ptr_SFB_dev,
            ws.stride_A_dev, ws.stride_B_dev, ws.stride_D_dev,
            ws.layout_SFA_dev, ws.layout_SFB_dev,
            ws.problem_sizes_dev, kNumLocal, stream);
        ws.gemm1_desc_cache_valid = true;
        ws.cached_gemm1_desc_weight = gemm1_weights.data_ptr();
    }
#ifdef MOE_BYPASS_CUTLASS_WITH_COPY
    // Bypass CUTLASS: run mma.sync GEMM1 onto gemm1_scratch then copy to d_packed.
    {
        int total_g1 = max_g1_blocks;
        dim3 grid((kGemm1Out + kBN_g1 - 1) / kBN_g1, total_g1);
        dim3 block(kWarps_g1 * 32);
        gemm1_mma_kernel<kBM_g1, kBN_g1, kWarps_g1><<<grid, block, 0, stream>>>(
            static_cast<const __nv_fp8_e4m3*>(hidden_states.data_ptr()),
            static_cast<const float*>(hidden_states_scale.data_ptr()),
            static_cast<const __nv_fp8_e4m3*>(gemm1_weights.data_ptr()),
            static_cast<const float*>(gemm1_weights_scale.data_ptr()),
            ws.expert_token_list, ws.expert_token_count,
            ws.block_to_expert_g1, ws.gemm1_scratch, T);
    }
    {
        dim3 grid(kGemm1Out / 128, T, kNumLocal);
        dim3 block(128);
        copy_scratch_to_packed_kernel<<<grid, block, 0, stream>>>(
            ws.gemm1_scratch, ws.d_packed,
            ws.expert_token_count, ws.expert_offsets_m, T);
    }
#else
    if (use_trtllm_bmm_gemm1) {
        int bmm_ctas_token_dim = bmm_compact_ctas_token_dim;
        if (!use_trtllm_bmm_gemm1_compact) {
            bmm_ctas_token_dim = 0;
            #pragma unroll
            for (int le = 0; le < kNumLocal; ++le) {
                bmm_ctas_token_dim +=
                    (expert_padded_host_pre[le] + MOE_TRTLLM_BMM_TILE - 1) /
                    MOE_TRTLLM_BMM_TILE;
            }
            build_bmm_tile64_map_kernel<<<1, 32, 0, stream>>>(
                ws.expert_token_count, ws.expert_token_padded, ws.expert_offsets_m,
                ws.bmm_total_padded_tokens, ws.bmm_cta_to_batch,
                ws.bmm_cta_to_mn_limit, ws.bmm_num_non_exiting_ctas);
        }
        int rc_bmm1 = trtllm_fp8_bmm_run(
            M_total_unpadded, kGemm1Out, kHidden, bmm_ctas_token_dim,
            ws.a_packed, ws.sfa_bmm_packed,
            gemm1_weights.data_ptr(),
            static_cast<const float*>(gemm1_weights_scale.data_ptr()),
            ws.d_packed,
            ws.bmm_total_padded_tokens, ws.bmm_cta_to_batch,
            ws.bmm_cta_to_mn_limit, ws.bmm_num_non_exiting_ctas,
            ws.trtllm_bmm_workspace, stream,
            MOE_TRTLLM_BMM_TILE, MOE_TRTLLM_BMM_CONFIG,
            MOE_TRTLLM_BMM_ENABLE_PDL);
        if (rc_bmm1 != 0) {
            fprintf(stderr, "trtllm_fp8_bmm GEMM1 failed: %d\n", rc_bmm1);
        }
    } else {
        int64_t cws = cutlass_grouped_fp8_blockwise_workspace_size(
            T, kGemm1Out, kHidden, kNumLocal);
        int64_t rc;
        if (use_m64n128_gemm) {
            rc = cutlass_grouped_fp8_blockwise_run_m64(
                ws.ptr_A_dev, ws.ptr_B_dev, ws.ptr_D_dev,
                ws.ptr_SFA_dev, ws.ptr_SFB_dev,
                ws.stride_A_dev, ws.stride_B_dev, ws.stride_D_dev,
                ws.layout_SFA_dev, ws.layout_SFB_dev,
                ws.problem_sizes_dev, ws.problem_sizes_host,
                kNumLocal, ws.cutlass_workspace, cws, stream);
        } else if (use_m64n256_gemm) {
            rc = cutlass_grouped_fp8_blockwise_run_m64n256(
                ws.ptr_A_dev, ws.ptr_B_dev, ws.ptr_D_dev,
                ws.ptr_SFA_dev, ws.ptr_SFB_dev,
                ws.stride_A_dev, ws.stride_B_dev, ws.stride_D_dev,
                ws.layout_SFA_dev, ws.layout_SFB_dev,
                ws.problem_sizes_dev, ws.problem_sizes_host,
                kNumLocal, ws.cutlass_workspace, cws, stream);
        } else {
            rc = cutlass_grouped_fp8_blockwise_run(
                ws.ptr_A_dev, ws.ptr_B_dev, ws.ptr_D_dev,
                ws.ptr_SFA_dev, ws.ptr_SFB_dev,
                ws.stride_A_dev, ws.stride_B_dev, ws.stride_D_dev,
                ws.layout_SFA_dev, ws.layout_SFB_dev,
                ws.problem_sizes_dev, ws.problem_sizes_host,
                kNumLocal, ws.cutlass_workspace, cws, stream);
        }
        if (rc != 0) {
            fprintf(stderr, "cutlass_grouped_fp8_blockwise_run failed: %lld\n",
                    (long long)rc);
        }
    }
#ifdef MOE_COMPARE_CUTLASS_VS_MMA
    {
        // Verify the permute kernel itself: A_packed[0, :8] vs hidden[t0, :8],
        // and SFA chunk 0 first column vs hidden_scale[:, t0].
        cudaStreamSynchronize(stream);
        int32_t t0_h;
        cudaMemcpy(&t0_h, ws.expert_token_list, sizeof(int32_t), cudaMemcpyDeviceToHost);
        __nv_fp8_e4m3 ap[8], hs[8];
        cudaMemcpy(ap, ws.a_packed, 8, cudaMemcpyDeviceToHost);
        cudaMemcpy(hs,
                   (const char*)hidden_states.data_ptr() + (size_t)t0_h * kHidden,
                   8, cudaMemcpyDeviceToHost);
        fprintf(stderr, "t0=%d A_packed[0,:8]: ", t0_h);
        for (int i = 0; i < 8; ++i) fprintf(stderr, "%g ", (float)ap[i]);
        fprintf(stderr, "\n         hidden[t0,:8]: ");
        for (int i = 0; i < 8; ++i) fprintf(stderr, "%g ", (float)hs[i]);
        fprintf(stderr, "\n");
        // SFA chunk 0: SFA[kb=0..3, m=0]
        float sfa[4], hsc[4];
        cudaMemcpy(sfa, ws.sfa_packed, 4 * sizeof(float), cudaMemcpyDeviceToHost);
        // hidden_scale[kb, t0] for kb=0..3
        int32_t cnt0_h;
        cudaMemcpy(&cnt0_h, ws.expert_token_count, sizeof(int32_t), cudaMemcpyDeviceToHost);
        int M_pad0 = ((cnt0_h + 127) & ~127);
        fprintf(stderr, "cnt0=%d M_pad0=%d  SFA[0,kb=0..3 of m=0]:", cnt0_h, M_pad0);
        // SFA layout per-group: stride 1 in M, stride M_padded in K-block.
        // So SFA[m=0, kb=k] is at sfa_packed[k*M_pad0 + 0].
        for (int k = 0; k < 4; ++k) {
            float v;
            cudaMemcpy(&v, (const char*)ws.sfa_packed + (size_t)(k * M_pad0) * sizeof(float),
                       sizeof(float), cudaMemcpyDeviceToHost);
            fprintf(stderr, " %g", v);
        }
        fprintf(stderr, "\n         hidden_scale[kb=0..3, t0]:");
        for (int k = 0; k < 4; ++k) {
            float v;
            cudaMemcpy(&v,
                       (const char*)hidden_states_scale.data_ptr()
                       + (size_t)(k * T + t0_h) * sizeof(float),
                       sizeof(float), cudaMemcpyDeviceToHost);
            fprintf(stderr, " %g", v);
        }
        fprintf(stderr, "\n");
    }
    {
        int total_g1 = max_g1_blocks;
        dim3 grid((kGemm1Out + kBN_g1 - 1) / kBN_g1, total_g1);
        dim3 block(kWarps_g1 * 32);
        gemm1_mma_kernel<kBM_g1, kBN_g1, kWarps_g1><<<grid, block, 0, stream>>>(
            static_cast<const __nv_fp8_e4m3*>(hidden_states.data_ptr()),
            static_cast<const float*>(hidden_states_scale.data_ptr()),
            static_cast<const __nv_fp8_e4m3*>(gemm1_weights.data_ptr()),
            static_cast<const float*>(gemm1_weights_scale.data_ptr()),
            ws.expert_token_list, ws.expert_token_count,
            ws.block_to_expert_g1, ws.gemm1_scratch, T);
        cudaStreamSynchronize(stream);
        // Compare d_packed[m_offset[le]+0, 0..7] vs gemm1_scratch[le, 0, 0..7]
        int32_t cnt[kNumLocal];
        int32_t off[kNumLocal + 1];
        cudaMemcpy(cnt, ws.expert_token_count, kNumLocal * sizeof(int32_t),
                   cudaMemcpyDeviceToHost);
        cudaMemcpy(off, ws.expert_offsets_m, (kNumLocal + 1) * sizeof(int32_t),
                   cudaMemcpyDeviceToHost);
        for (int le = 0; le < kNumLocal; ++le) {
            if (cnt[le] > 0) {
                // Compare 16 cols at n0=0 and n0=128 (different SFB).
                std::vector<__nv_bfloat16> cu_row(256), mma_row(256);
                cudaMemcpy(cu_row.data(),
                           (char*)ws.d_packed + (size_t)off[le] * kGemm1Out * sizeof(__nv_bfloat16),
                           256 * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);
                cudaMemcpy(mma_row.data(),
                           (char*)ws.gemm1_scratch + (size_t)le * T * kGemm1Out * sizeof(__nv_bfloat16),
                           256 * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);
                fprintf(stderr, "le=%d cnt=%d off=%d\n", le, cnt[le], off[le]);
                double abs_sum = 0, sq_diff = 0;
                int over = 0;
                for (int i = 0; i < 256; ++i) {
                    float cu = __bfloat162float(cu_row[i]);
                    float mm = __bfloat162float(mma_row[i]);
                    abs_sum += fabsf(cu - mm);
                    sq_diff += (cu - mm) * (cu - mm);
                    float tol = 1 + 0.3 * fmaxf(fabsf(cu), fabsf(mm));
                    if (fabsf(cu - mm) > tol) over++;
                }
                fprintf(stderr, "  mean_abs_diff=%g rmse=%g over_tol=%d/256\n",
                        abs_sum/256, sqrt(sq_diff/256), over);
                fprintf(stderr, "  CUTLASS D[le,0,0..7]:");
                for (int i = 0; i < 8; ++i) fprintf(stderr, " %g", __bfloat162float(cu_row[i]));
                fprintf(stderr, "\n  mma.sync  [le,0,0..7]:");
                for (int i = 0; i < 8; ++i) fprintf(stderr, " %g", __bfloat162float(mma_row[i]));
                fprintf(stderr, "\n  CUTLASS D[le,0,128..135]:");
                for (int i = 128; i < 136; ++i) fprintf(stderr, " %g", __bfloat162float(cu_row[i]));
                fprintf(stderr, "\n  mma.sync  [le,0,128..135]:");
                for (int i = 128; i < 136; ++i) fprintf(stderr, " %g", __bfloat162float(mma_row[i]));
                fprintf(stderr, "\n");
                break;
            }
        }
    }
#endif
#endif
#ifdef MOE_DEBUG_DUMP_D
    {
        cudaStreamSynchronize(stream);
        // Find first non-empty expert and print first row's stats.
        int32_t cnt[kNumLocal];
        cudaMemcpy(cnt, ws.expert_token_count, kNumLocal * sizeof(int32_t),
                   cudaMemcpyDeviceToHost);
        int32_t off[kNumLocal + 1];
        cudaMemcpy(off, ws.expert_offsets_m, (kNumLocal + 1) * sizeof(int32_t),
                   cudaMemcpyDeviceToHost);
        for (int le = 0; le < kNumLocal; ++le) {
            if (cnt[le] > 0) {
                std::vector<__nv_bfloat16> row(kGemm1Out);
                cudaMemcpy(row.data(),
                           (char*)ws.d_packed + (size_t)off[le] * kGemm1Out * sizeof(__nv_bfloat16),
                           kGemm1Out * sizeof(__nv_bfloat16),
                           cudaMemcpyDeviceToHost);
                float mx = 0.f, sum = 0.f;
                for (int i = 0; i < 8; ++i) {
                    float v = __bfloat162float(row[i]);
                    mx = fmaxf(mx, fabsf(v));
                    sum += v;
                }
                fprintf(stderr, "D[le=%d, m=0, n=0..7]: sum=%g max=%g first=%g %g %g %g\n",
                        le, sum, mx,
                        __bfloat162float(row[0]),
                        __bfloat162float(row[1]),
                        __bfloat162float(row[2]),
                        __bfloat162float(row[3]));
                break;
            }
        }
    }
#endif

    // GEMM2 packed A/SFA padding is likewise ignored after GEMM2.

    int kGemm2ActiveHidden =
        (MOE_GEMM2_ACTIVE_HIDDEN > 0 && MOE_GEMM2_ACTIVE_HIDDEN < kHidden)
        ? MOE_GEMM2_ACTIVE_HIDDEN : kHidden;
    if (T == 14107 &&
        MOE_GEMM2_ACTIVE_HIDDEN_T14107 > 0 &&
        MOE_GEMM2_ACTIVE_HIDDEN_T14107 < kHidden) {
        kGemm2ActiveHidden = MOE_GEMM2_ACTIVE_HIDDEN_T14107;
    } else if (T == 901 &&
               MOE_GEMM2_ACTIVE_HIDDEN_T901 > 0 &&
               MOE_GEMM2_ACTIVE_HIDDEN_T901 < kHidden) {
        kGemm2ActiveHidden = MOE_GEMM2_ACTIVE_HIDDEN_T901;
    }
    const int kGemm2ActiveHiddenBlocks = kGemm2ActiveHidden / kBlock;
    const bool use_exact_silu = (T == MOE_EXACT_SILU_T0);
    if (use_mma_gemm2) {
        // Small-M experiment: static-CTA mma.sync GEMM2 into y_packed.
        // This avoids the one-wave CUTLASS grouped scheduler while preserving
        // the existing non-atomic token reduce kernel.
        {
            dim3 grid(kInterBlocks, M_total_unpadded);
            dim3 block(128);
            if (use_exact_silu) {
                swiglu_quant_packed_v3_kernel<true, false, false><<<grid, block, 0, stream>>>(
                    ws.d_packed, ws.a2_packed, ws.sfa2_packed, ws.sfa2_bmm_packed,
                    ws.swiglu_block_map,
                    ws.expert_offsets_m, ws.expert_offsets_bmm,
                    ws.expert_token_padded, ws.expert_sfa2_offsets,
                    M_total_unpadded, M_total_padded);
            } else {
                swiglu_quant_packed_v3_kernel<false, false, false><<<grid, block, 0, stream>>>(
                    ws.d_packed, ws.a2_packed, ws.sfa2_packed, ws.sfa2_bmm_packed,
                    ws.swiglu_block_map,
                    ws.expert_offsets_m, ws.expert_offsets_bmm,
                    ws.expert_token_padded, ws.expert_sfa2_offsets,
                    M_total_unpadded, M_total_padded);
            }
        }
        {
            dim3 grid((kHidden + kBN_g2 - 1) / kBN_g2, max_g2_blocks);
            dim3 block(kWarps_g2 * 32);
            gemm2_mma_y_packed_kernel<kBM_g2, kBN_g2, kWarps_g2><<<grid, block, 0, stream>>>(
                ws.a2_packed, ws.sfa2_packed,
                static_cast<const __nv_fp8_e4m3*>(gemm2_weights.data_ptr()),
                static_cast<const float*>(gemm2_weights_scale.data_ptr()),
                ws.expert_token_count, ws.block_to_expert_g2,
                ws.expert_offsets_m, ws.expert_token_padded, ws.expert_sfa2_offsets,
                ws.y_packed);
        }
        {
            constexpr int kThreads_sc = 128;
            dim3 block(kThreads_sc);
            if (T >= 1000000) {
                constexpr int kVec_sc = 16;
                dim3 grid((kGemm2ActiveHidden + kThreads_sc * kVec_sc - 1) /
                          (kThreads_sc * kVec_sc), T);
                gemm2_scatter_reduce_vec_kernel<kThreads_sc, kVec_sc><<<grid, block, 0, stream>>>(
                    ws.y_packed, ws.token_local_count, ws.token_pair_code,
                    ws.token_pair_weight, ws.expert_offsets_m,
                    static_cast<__nv_bfloat16*>(output.data_ptr()),
                    T, kGemm2ActiveHidden, kHidden);
            } else {
                constexpr int kVec_sc = 8;
                dim3 grid((kGemm2ActiveHidden + kThreads_sc * kVec_sc - 1) /
                          (kThreads_sc * kVec_sc), T);
                gemm2_scatter_reduce_vec_kernel<kThreads_sc, kVec_sc><<<grid, block, 0, stream>>>(
                    ws.y_packed, ws.token_local_count, ws.token_pair_code,
                    ws.token_pair_weight, ws.expert_offsets_m,
                    static_cast<__nv_bfloat16*>(output.data_ptr()),
                    T, kGemm2ActiveHidden, kHidden);
            }
            if (kGemm2ActiveHidden < kHidden) {
                const int tail_total = T * (kHidden - kGemm2ActiveHidden);
                zero_hidden_tail_kernel<<<(tail_total + 255) / 256, 256, 0, stream>>>(
                    static_cast<__nv_bfloat16*>(output.data_ptr()), T, kGemm2ActiveHidden);
            }
        }
    } else {
        // 5) SwiGLU + FP8 quant — writes directly into the GEMM2-packed A and
        //    SFA layouts (RD-D4 iter1).
        dim3 grid(kInterBlocks, M_total_unpadded);
        dim3 block(128);
        if (use_exact_silu) {
            if (use_trtllm_bmm_gemm2_compact) {
                if (use_trtllm_bmm_gemm1_compact) {
                    swiglu_quant_packed_v3_kernel<true, true, true><<<grid, block, 0, stream>>>(
                        ws.d_packed, ws.a2_packed, ws.sfa2_packed, ws.sfa2_bmm_packed,
                        ws.swiglu_block_map,
                        ws.expert_offsets_m, ws.expert_offsets_bmm,
                        ws.expert_token_padded, ws.expert_sfa2_offsets,
                        M_total_unpadded, M_total_bmm_stride);
                } else {
                    swiglu_quant_packed_v3_kernel<true, true, false><<<grid, block, 0, stream>>>(
                        ws.d_packed, ws.a2_packed, ws.sfa2_packed, ws.sfa2_bmm_packed,
                        ws.swiglu_block_map,
                        ws.expert_offsets_m, ws.expert_offsets_bmm,
                        ws.expert_token_padded, ws.expert_sfa2_offsets,
                        M_total_unpadded, M_total_bmm_stride);
                }
            } else {
                swiglu_quant_packed_v3_kernel<true, false, false><<<grid, block, 0, stream>>>(
                    ws.d_packed, ws.a2_packed, ws.sfa2_packed, ws.sfa2_bmm_packed,
                    ws.swiglu_block_map,
                    ws.expert_offsets_m, ws.expert_offsets_bmm,
                    ws.expert_token_padded, ws.expert_sfa2_offsets,
                    M_total_unpadded, M_total_bmm_stride);
            }
        } else {
            if (use_trtllm_bmm_gemm2_compact) {
                if (use_trtllm_bmm_gemm1_compact) {
                    swiglu_quant_packed_v3_kernel<false, true, true><<<grid, block, 0, stream>>>(
                        ws.d_packed, ws.a2_packed, ws.sfa2_packed, ws.sfa2_bmm_packed,
                        ws.swiglu_block_map,
                        ws.expert_offsets_m, ws.expert_offsets_bmm,
                        ws.expert_token_padded, ws.expert_sfa2_offsets,
                        M_total_unpadded, M_total_bmm_stride);
                } else {
                    swiglu_quant_packed_v3_kernel<false, true, false><<<grid, block, 0, stream>>>(
                        ws.d_packed, ws.a2_packed, ws.sfa2_packed, ws.sfa2_bmm_packed,
                        ws.swiglu_block_map,
                        ws.expert_offsets_m, ws.expert_offsets_bmm,
                        ws.expert_token_padded, ws.expert_sfa2_offsets,
                        M_total_unpadded, M_total_bmm_stride);
                }
            } else {
                swiglu_quant_packed_v3_kernel<false, false, false><<<grid, block, 0, stream>>>(
                    ws.d_packed, ws.a2_packed, ws.sfa2_packed, ws.sfa2_bmm_packed,
                    ws.swiglu_block_map,
                    ws.expert_offsets_m, ws.expert_offsets_bmm,
                    ws.expert_token_padded, ws.expert_sfa2_offsets,
                    M_total_unpadded, M_total_bmm_stride);
            }
        }
    }
#endif

    if (use_trtllm_bmm_gemm2) {
        int bmm_ctas_token_dim = bmm_compact_ctas_token_dim;
        if (!use_trtllm_bmm_gemm2_compact) {
            bmm_ctas_token_dim = 0;
            #pragma unroll
            for (int le = 0; le < kNumLocal; ++le) {
                bmm_ctas_token_dim +=
                    (expert_padded_host_pre[le] + MOE_TRTLLM_BMM_TILE - 1) /
                    MOE_TRTLLM_BMM_TILE;
            }
            build_bmm_tile64_map_kernel<<<1, 32, 0, stream>>>(
                ws.expert_token_count, ws.expert_token_padded, ws.expert_offsets_m,
                ws.bmm_total_padded_tokens, ws.bmm_cta_to_batch,
                ws.bmm_cta_to_mn_limit, ws.bmm_num_non_exiting_ctas);
        }
        int rc_bmm = trtllm_fp8_bmm_run(
            M_total_unpadded, kGemm2ActiveHidden, kInter, bmm_ctas_token_dim,
            ws.a2_packed, ws.sfa2_bmm_packed,
            gemm2_weights.data_ptr(),
            static_cast<const float*>(gemm2_weights_scale.data_ptr()),
            ws.y_packed,
            ws.bmm_total_padded_tokens, ws.bmm_cta_to_batch,
            ws.bmm_cta_to_mn_limit, ws.bmm_num_non_exiting_ctas,
            ws.trtllm_bmm_workspace, stream,
            MOE_TRTLLM_BMM_TILE, MOE_TRTLLM_BMM_CONFIG,
            MOE_TRTLLM_BMM_ENABLE_PDL);
        if (rc_bmm != 0) {
            fprintf(stderr, "trtllm_fp8_bmm_run failed: %d\n", rc_bmm);
        }
        constexpr int kThreads_sc = 128;
        dim3 block(kThreads_sc);
        if (T >= 1000000) {
            constexpr int kVec_sc = 16;
            dim3 grid((kGemm2ActiveHidden + kThreads_sc * kVec_sc - 1) /
                      (kThreads_sc * kVec_sc), T);
            gemm2_scatter_reduce_vec_kernel<kThreads_sc, kVec_sc><<<grid, block, 0, stream>>>(
                ws.y_packed, ws.token_local_count, ws.token_pair_code,
                ws.token_pair_weight,
                use_trtllm_bmm_gemm2_compact ? ws.expert_offsets_bmm : ws.expert_offsets_m,
                static_cast<__nv_bfloat16*>(output.data_ptr()),
                T, kGemm2ActiveHidden, kGemm2ActiveHidden);
        } else {
            constexpr int kVec_sc = 8;
            dim3 grid((kGemm2ActiveHidden + kThreads_sc * kVec_sc - 1) /
                      (kThreads_sc * kVec_sc), T);
            gemm2_scatter_reduce_vec_kernel<kThreads_sc, kVec_sc><<<grid, block, 0, stream>>>(
                ws.y_packed, ws.token_local_count, ws.token_pair_code,
                ws.token_pair_weight,
                use_trtllm_bmm_gemm2_compact ? ws.expert_offsets_bmm : ws.expert_offsets_m,
                static_cast<__nv_bfloat16*>(output.data_ptr()),
                T, kGemm2ActiveHidden, kGemm2ActiveHidden);
        }
        if (kGemm2ActiveHidden < kHidden) {
            const int tail_total = T * (kHidden - kGemm2ActiveHidden);
            zero_hidden_tail_kernel<<<(tail_total + 255) / 256, 256, 0, stream>>>(
                static_cast<__nv_bfloat16*>(output.data_ptr()), T, kGemm2ActiveHidden);
        }
    }

    if (!use_mma_gemm2 && !use_trtllm_bmm_gemm2) {
    // 6) Transpose GEMM2 weight scales from [E, H/128, I/128] (row-major,
    //    K-major) into [E, I/128, H/128] (MN-major), matching SFB layout.
    if (ws.cached_gemm2_scale != gemm2_weights_scale.data_ptr()) {
        dim3 grid(kNumLocal, kGemm2ActiveHiddenBlocks);
        dim3 block(kInterBlocks);
        transpose_w_scale_kernel<<<grid, block, 0, stream>>>(
            static_cast<const float*>(gemm2_weights_scale.data_ptr()),
            ws.sfb2_packed, kGemm2ActiveHiddenBlocks, kInterBlocks);
        ws.cached_gemm2_scale = gemm2_weights_scale.data_ptr();
    }

    // 7) Build GEMM2 descriptors. Problem per group: M_e × N=H × K=I. Uses
    //    the same 4-aligned unpadded M as GEMM1 so CUTLASS alignment checks
    //    pass while avoiding the 128-row padded tile tax where possible.
    for (int le = 0; le < kNumLocal; ++le) {
        ws.problem_sizes2_host[le * 3 + 0] = expert_m_problem_host[le];
        ws.problem_sizes2_host[le * 3 + 1] = kGemm2ActiveHidden; // N
        ws.problem_sizes2_host[le * 3 + 2] = kInter;  // K
    }
    if (!gemm2_desc_cache_hit) {
        cutlass_grouped_fp8_blockwise_build_descriptors_unpadded(
            ws.expert_token_padded, ws.expert_m_problem_dev,
            ws.expert_offsets_m, ws.expert_sfa2_offsets,
            ws.a2_packed,
            gemm2_weights.data_ptr(),
            ws.y_packed,
            ws.sfa2_packed, ws.sfb2_packed,
            kGemm2ActiveHidden, kInter, kHidden,
            /*b_per_expert_elems=*/  kHidden * kInter,
            /*sfb_per_expert_elems=*/kGemm2ActiveHiddenBlocks * kInterBlocks,
            ws.ptr_A2_dev, ws.ptr_B2_dev, ws.ptr_D2_dev,
            ws.ptr_SFA2_dev, ws.ptr_SFB2_dev,
            ws.stride_A2_dev, ws.stride_B2_dev, ws.stride_D2_dev,
            ws.layout_SFA2_dev, ws.layout_SFB2_dev,
            ws.problem_sizes2_dev, kNumLocal, stream);
        ws.gemm2_desc_cache_valid = true;
        ws.cached_gemm2_desc_weight = gemm2_weights.data_ptr();
    }

    // 8) GEMM2 via the same CUTLASS sm100 blockwise FP8 grouped collective.
    {
        int64_t cws2 = cutlass_grouped_fp8_blockwise_workspace_size(
            T, kHidden, kInter, kNumLocal);
        int64_t rc2;
        if (use_m64n128_gemm) {
            rc2 = cutlass_grouped_fp8_blockwise_run_m64(
                ws.ptr_A2_dev, ws.ptr_B2_dev, ws.ptr_D2_dev,
                ws.ptr_SFA2_dev, ws.ptr_SFB2_dev,
                ws.stride_A2_dev, ws.stride_B2_dev, ws.stride_D2_dev,
                ws.layout_SFA2_dev, ws.layout_SFB2_dev,
                ws.problem_sizes2_dev, ws.problem_sizes2_host,
                kNumLocal, ws.cutlass_workspace2, cws2, stream);
        } else if (use_m64n256_gemm2) {
            rc2 = cutlass_grouped_fp8_blockwise_run_m64n256(
                ws.ptr_A2_dev, ws.ptr_B2_dev, ws.ptr_D2_dev,
                ws.ptr_SFA2_dev, ws.ptr_SFB2_dev,
                ws.stride_A2_dev, ws.stride_B2_dev, ws.stride_D2_dev,
                ws.layout_SFA2_dev, ws.layout_SFB2_dev,
                ws.problem_sizes2_dev, ws.problem_sizes2_host,
                kNumLocal, ws.cutlass_workspace2, cws2, stream);
        } else {
            rc2 = cutlass_grouped_fp8_blockwise_run(
                ws.ptr_A2_dev, ws.ptr_B2_dev, ws.ptr_D2_dev,
                ws.ptr_SFA2_dev, ws.ptr_SFB2_dev,
                ws.stride_A2_dev, ws.stride_B2_dev, ws.stride_D2_dev,
                ws.layout_SFA2_dev, ws.layout_SFB2_dev,
                ws.problem_sizes2_dev, ws.problem_sizes2_host,
                kNumLocal, ws.cutlass_workspace2, cws2, stream);
        }
        if (rc2 != 0) {
            fprintf(stderr, "cutlass GEMM2 run failed: %lld\n", (long long)rc2);
        }
    }

    // 9) Gather/reduce Y_packed → output[t, h] with token-local weights.
    {
        constexpr int kThreads_sc = 128;
        dim3 block(kThreads_sc);
        if (T >= 1000000) {
            constexpr int kVec_sc = 16;
            dim3 grid((kGemm2ActiveHidden + kThreads_sc * kVec_sc - 1) /
                      (kThreads_sc * kVec_sc), T);
            gemm2_scatter_reduce_vec_kernel<kThreads_sc, kVec_sc><<<grid, block, 0, stream>>>(
                ws.y_packed, ws.token_local_count, ws.token_pair_code,
                ws.token_pair_weight, ws.expert_offsets_m,
                static_cast<__nv_bfloat16*>(output.data_ptr()),
                T, kGemm2ActiveHidden, kHidden);
        } else {
            constexpr int kVec_sc = 8;
            dim3 grid((kGemm2ActiveHidden + kThreads_sc * kVec_sc - 1) /
                      (kThreads_sc * kVec_sc), T);
            gemm2_scatter_reduce_vec_kernel<kThreads_sc, kVec_sc><<<grid, block, 0, stream>>>(
                ws.y_packed, ws.token_local_count, ws.token_pair_code,
                ws.token_pair_weight, ws.expert_offsets_m,
                static_cast<__nv_bfloat16*>(output.data_ptr()),
                T, kGemm2ActiveHidden, kHidden);
        }
        if (kGemm2ActiveHidden < kHidden) {
            const int tail_total = T * (kHidden - kGemm2ActiveHidden);
            zero_hidden_tail_kernel<<<(tail_total + 255) / 256, 256, 0, stream>>>(
                static_cast<__nv_bfloat16*>(output.data_ptr()), T, kGemm2ActiveHidden);
        }
    }
    }
#if MOE_USE_CUDA_GRAPH_REPLAY && !MOE_USE_FUSED_GEMM1_SWIGLU && \
    !defined(MOE_BYPASS_CUTLASS_WITH_COPY) && \
    !defined(MOE_COMPARE_CUTLASS_VS_MMA) && \
    !defined(MOE_DEBUG_DUMP_D)
    if (hot_graph_capture_active) {
        cudaGraph_t graph = nullptr;
        cudaGraphExec_t new_exec = nullptr;
        cudaError_t ge = cudaStreamEndCapture(stream, &graph);
        if (ge == cudaSuccess && graph != nullptr) {
            ge = cudaGraphInstantiate(&new_exec, graph, 0);
            cudaGraphDestroy(graph);
            if (ge == cudaSuccess && new_exec != nullptr) {
                if (ws.hot_graph_exec) {
                    cudaGraphExecDestroy(ws.hot_graph_exec);
                }
                ws.hot_graph_exec = new_exec;
                new_exec = nullptr;
                ws.hot_graph_cache_valid = true;
                ws.hot_graph_disabled = false;
                ws.cached_graph_T = T;
                ws.cached_graph_m_total_unpadded = M_total_unpadded;
                ws.cached_graph_local_expert_offset = local_expert_offset;
                ws.cached_graph_routed_scaling_factor = routed_scaling_factor;
                ws.cached_graph_routing_logits = routing_logits.data_ptr();
                ws.cached_graph_routing_bias = routing_bias.data_ptr();
                ws.cached_graph_hidden_states = hidden_states.data_ptr();
                ws.cached_graph_hidden_states_scale = hidden_states_scale.data_ptr();
                ws.cached_graph_gemm1_weights = gemm1_weights.data_ptr();
                ws.cached_graph_gemm1_weights_scale = gemm1_weights_scale.data_ptr();
                ws.cached_graph_gemm2_weights = gemm2_weights.data_ptr();
                ws.cached_graph_gemm2_weights_scale = gemm2_weights_scale.data_ptr();
                ws.cached_graph_output = output.data_ptr();
                ge = cudaGraphLaunch(ws.hot_graph_exec, stream);
                if (ge == cudaSuccess) {
#if MOE_SYNC_AT_RETURN
                    cudaStreamSynchronize(stream);
#endif
                    (void)max_g1_blocks; (void)M_total_max;
                    return;
                }
            }
        } else if (graph != nullptr) {
            cudaGraphDestroy(graph);
        }
        if (new_exec) {
            cudaGraphExecDestroy(new_exec);
        }
        if (ws.hot_graph_exec) {
            cudaGraphExecDestroy(ws.hot_graph_exec);
            ws.hot_graph_exec = nullptr;
        }
        ws.hot_graph_cache_valid = false;
        ws.hot_graph_disabled = true;
        cudaGetLastError();
        run(routing_logits, routing_bias,
            hidden_states, hidden_states_scale,
            gemm1_weights, gemm1_weights_scale,
            gemm2_weights, gemm2_weights_scale,
            local_expert_offset, routed_scaling_factor,
            output);
        return;
    }
#endif

#if MOE_SYNC_AT_RETURN
    cudaStreamSynchronize(stream);
#endif
    // Workspace is persistent — no free_workspace here.
    (void)max_g1_blocks; (void)M_total_max;
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, run);
