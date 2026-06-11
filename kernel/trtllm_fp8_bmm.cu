// Thin C ABI wrapper around FlashInfer's TensorRT-LLM Gen FP8 dynB BMM runner.
//
// The MoE hot path is plain CUDA C++ and should not include the C++ runner
// headers directly. Keep the dependency isolated here and expose only stable
// C-callable helpers.

#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

#include "flashinfer/trtllm/batched_gemm/KernelRunner.h"

#ifndef TLLM_GEN_GEMM_CUBIN_PATH
#define TLLM_GEN_GEMM_CUBIN_PATH ""
#endif

namespace flashinfer {
namespace trtllm_cubin_loader {

std::string getCubin(const std::string& name, const std::string&) {
    std::string path = name;
    std::ifstream file(path, std::ios::binary);
    if (!file && !path.empty() && path[0] != '/') {
        const char* home = std::getenv("HOME");
        if (home) {
            path = std::string(home) + "/.cache/flashinfer/cubins/" + name;
            file.open(path, std::ios::binary);
        }
    }
    if (!file) {
        throw std::runtime_error("Missing local cubin: " + path);
    }
    return std::string(std::istreambuf_iterator<char>(file),
                       std::istreambuf_iterator<char>());
}

}  // namespace trtllm_cubin_loader
}  // namespace flashinfer

namespace {

namespace btg = batchedGemm::trtllm::gen;
namespace bgg = batchedGemm::gemm;
using tensorrt_llm::kernels::EltwiseActType;
using tensorrt_llm::kernels::TrtllmGenBatchedGemmRunner;
using tensorrt_llm::kernels::TrtllmGenBatchedGemmRunnerOptions;

constexpr int kNumLocalExperts = 32;

TrtllmGenBatchedGemmRunnerOptions make_options(int tile_tokens) {
    TrtllmGenBatchedGemmRunnerOptions opts;
    opts.dtypeA = btg::Dtype::E4m3;
    opts.dtypeB = btg::Dtype::E4m3;
    opts.dtypeC = btg::Dtype::Bfloat16;
    opts.eltwiseActType = EltwiseActType::None;
    opts.deepSeekFp8 = true;
    opts.fusedAct = false;
    opts.routeAct = false;
    opts.staticBatch = false;
    opts.transposeMmaOutput = true;
    opts.tileSize = tile_tokens;
    opts.epilogueTileM = 64;
    opts.useShuffledMatrix = false;
    opts.weightLayout = bgg::MatrixLayout::MajorK;
    return opts;
}

TrtllmGenBatchedGemmRunner& runner_for_tile(int tile_tokens) {
    static TrtllmGenBatchedGemmRunner runner8(make_options(8));
    static TrtllmGenBatchedGemmRunner runner16(make_options(16));
    static TrtllmGenBatchedGemmRunner runner64(make_options(64));
    if (tile_tokens == 8) return runner8;
    if (tile_tokens == 16) return runner16;
    if (tile_tokens == 64) return runner64;
    throw std::runtime_error("unsupported TRTLLM BMM tile size");
}

int resolve_config(TrtllmGenBatchedGemmRunner& runner, int m, int n, int k,
                   int max_ctas_token_dim, int preferred_config) {
    std::vector<int32_t> empty_batched_tokens;
    if (preferred_config >= 0 &&
        runner.isValidConfigIndex(preferred_config, m, n, k, empty_batched_tokens,
                                  m, kNumLocalExperts, max_ctas_token_dim)) {
        return preferred_config;
    }
    return static_cast<int>(runner.getDefaultValidConfigIndex(
        m, n, k, empty_batched_tokens, m, kNumLocalExperts, max_ctas_token_dim));
}

}  // namespace

extern "C" __attribute__((visibility("default")))
int trtllm_fp8_bmm_is_available() {
    try {
        (void)runner_for_tile(64);
        return 1;
    } catch (...) {
        return 0;
    }
}

extern "C" __attribute__((visibility("default")))
size_t trtllm_fp8_bmm_workspace_size(
    int m, int n, int k, int max_ctas_token_dim,
    int tile_tokens, int preferred_config) {
    try {
        m = std::max(m, 1);
        max_ctas_token_dim = std::max(max_ctas_token_dim, 1);
        TrtllmGenBatchedGemmRunner& runner = runner_for_tile(tile_tokens);
        int config = resolve_config(runner, m, n, k, max_ctas_token_dim, preferred_config);
        std::vector<int32_t> empty_batched_tokens;
        return runner.getWorkspaceSizeInBytes(
            m, n, k, empty_batched_tokens, m, kNumLocalExperts,
            max_ctas_token_dim, config);
    } catch (...) {
        return 0;
    }
}

extern "C" __attribute__((visibility("default")))
int trtllm_fp8_bmm_run(
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
    int enable_pdl) {
    try {
        if (m <= 0 || max_ctas_token_dim <= 0) return 0;
        TrtllmGenBatchedGemmRunner& runner = runner_for_tile(tile_tokens);
        int config = resolve_config(runner, m, n, k, max_ctas_token_dim, preferred_config);
        std::vector<int32_t> empty_batched_tokens;
        int device = 0;
        cudaError_t e = cudaGetDevice(&device);
        if (e != cudaSuccess) return -2;
        runner.run(
            m, n, k, empty_batched_tokens, m, kNumLocalExperts, max_ctas_token_dim,
            activations, activation_scales, weights, weight_scales,
            /*perTokensSfA=*/nullptr, /*perTokensSfB=*/nullptr,
            /*scaleC=*/nullptr, /*scaleGateC=*/nullptr,
            /*bias=*/nullptr, /*gatedActAlpha=*/nullptr, /*gatedActBeta=*/nullptr,
            /*clampLimit=*/nullptr, output,
            /*outSfC=*/nullptr,
            /*routeMap=*/nullptr, total_padded_tokens,
            cta_to_batch, cta_to_mn_limit, num_non_exiting_ctas,
            workspace, reinterpret_cast<CUstream>(stream), device,
            config, enable_pdl != 0);
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "trtllm_fp8_bmm_run exception: %s\n", e.what());
        return -1;
    } catch (...) {
        std::fprintf(stderr, "trtllm_fp8_bmm_run unknown exception\n");
        return -1;
    }
}
