// CUTLASS sm100 block-scale FP8 grouped GEMM, wrapped for use by moe_fp8.cu.
//
// This is the Round C wiring: DeepSeek-V3 quantization uses a per-token
// scale along M (granularity 1) and a per-128-block scale along K. The
// previous Round B wrapper used 128x128 vec on A which is the wrong layout
// for token-major activations. We fix that here.
//
// Scale layout:
//   A: ElementA = e4m3, LayoutA = RowMajor. Per-group A buffer is packed as
//      [M_e, K] row-major. M_e is the per-expert token count.
//   B: ElementB = e4m3, LayoutB = ColumnMajor (= weights stored as
//      [N, K] row-major, last-dim K contiguous = effectively column-major
//      under the CUTLASS convention).
//   SFA: ScaleGranularity = (1, 128, 128). MN-major in M.
//        The SFA buffer is shared across groups: SFA_packed[kb, m_global]
//        with stride 1 in M and stride M_total in K-block. Each group's
//        pointer is base + m_offset (m_offset in tokens).
//        Layout shape per group: ((1, M_e), (128, K/128), 1), stride
//        ((_0, _1), (_0, M_total), M_total*K/128).
//   SFB: K-major. Per-expert dense weight scale [N/128, K/128] row-major,
//        size N/128*K/128 per group; ptr = base + le*N/128*K/128.
//
// References:
//   /tmp/cutlass-src/examples/81_blackwell_gemm_blockwise/81_blackwell_grouped_gemm_groupwise.cu

#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <type_traits>

// CUTLASS host-side trace, controlled by an env-time macro. Default off.
#ifndef CUTLASS_DEBUG_TRACE_LEVEL
#define CUTLASS_DEBUG_TRACE_LEVEL 0
#endif

#include <cutlass/cutlass.h>
#include <cute/tensor.hpp>
#include <cutlass/tensor_ref.h>
#include <cutlass/epilogue/thread/activation.h>
#include <cutlass/gemm/dispatch_policy.hpp>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/epilogue/dispatch_policy.hpp>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/gemm/kernel/gemm_universal.hpp>
#include <cutlass/gemm/kernel/tile_scheduler_params.h>
#include <cutlass/util/packed_stride.hpp>

using namespace cute;

#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)

namespace cutlass_grouped_fp8_blockwise {

using ProblemShape  = cutlass::gemm::GroupProblemShape<Shape<int, int, int>>;
using ElementA      = cutlass::float_e4m3_t;
using LayoutA       = cutlass::layout::RowMajor;
constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;
using ElementB      = cutlass::float_e4m3_t;
// Our gemm1_weights is [N, K] row-major (K = contracting dim is contiguous
// in memory). CUTLASS B operand is shape (K, N) so K-major contiguous data
// means LayoutB = ColumnMajor (i.e. stride 1 in K, stride K in N).
using LayoutB       = cutlass::layout::ColumnMajor;
constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;
using ElementC      = cutlass::bfloat16_t;
using LayoutC       = cutlass::layout::RowMajor;
constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;
using ElementD      = ElementC;
using LayoutD       = LayoutC;
constexpr int AlignmentD = AlignmentC;
using ElementAccumulator = float;
using ElementCompute     = float;

// Default tile shape: 128x128x128 with 1-CTA cluster (RD iter5 baseline winner).
// Tile sweep results (mean µs at dev-8):
//   <128,128,128> × <1,1,1>:  644  (winner)
//   <128,256,128> × <1,1,1>: 1600  (wider N hurts wave count)
//   <256,128,128> × <2,1,1>:  896  (2-CTA + 256-pad-tax hurts small/mid T)
//   <256,128,128> × <1,1,1>:  N/A   (1-CTA builder rejects M=256)
using MmaTileShape_MNK = Shape<_128, _128, _128>;
using ClusterShape_MNK = Shape<_1, _1, _1>;

// SFA: ScaleGranularity=(1,128,128). Per-token scale along M.
//   majorSFA = MN → inner-M stride 1, K-block stride = M.
//   majorSFB = MN → inner-N stride 1, K-block stride = N/128.
// (matches the upstream groupwise example; we transpose our K-major weight
// scales into MN-major in moe_fp8.cu before the GEMM.)
using ScaleConfig = cutlass::detail::Sm100BlockwiseScaleConfig<
    1, 128, 128, UMMA::Major::MN, UMMA::Major::MN>;
using LayoutSFA = decltype(ScaleConfig::deduce_layoutSFA());
using LayoutSFB = decltype(ScaleConfig::deduce_layoutSFB());

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
    MmaTileShape_MNK, ClusterShape_MNK,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementCompute,
    ElementC, LayoutC*, AlignmentC,
    ElementD, LayoutC*, AlignmentD,
    cutlass::epilogue::PtrArrayTmaWarpSpecialized1Sm
  >::CollectiveOp;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
    ElementA, cute::tuple<LayoutA*, LayoutSFA*>, AlignmentA,
    ElementB, cute::tuple<LayoutB*, LayoutSFB*>, AlignmentB,
    ElementAccumulator,
    MmaTileShape_MNK, ClusterShape_MNK,
    cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    cutlass::gemm::KernelPtrArrayTmaWarpSpecializedBlockwise1SmSm100
  >::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    ProblemShape,
    CollectiveMainloop,
    CollectiveEpilogue,
    void>;

using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

using StrideA = typename Gemm::GemmKernel::InternalStrideA;
using StrideB = typename Gemm::GemmKernel::InternalStrideB;
using StrideC = typename Gemm::GemmKernel::InternalStrideC;
using StrideD = typename Gemm::GemmKernel::InternalStrideD;

}  // namespace cutlass_grouped_fp8_blockwise

namespace cutlass_grouped_fp8_blockwise_m64 {

using ProblemShape  = cutlass_grouped_fp8_blockwise::ProblemShape;
using ElementA      = cutlass_grouped_fp8_blockwise::ElementA;
using LayoutA       = cutlass_grouped_fp8_blockwise::LayoutA;
constexpr int AlignmentA = cutlass_grouped_fp8_blockwise::AlignmentA;
using ElementB      = cutlass_grouped_fp8_blockwise::ElementB;
using LayoutB       = cutlass_grouped_fp8_blockwise::LayoutB;
constexpr int AlignmentB = cutlass_grouped_fp8_blockwise::AlignmentB;
using ElementC      = cutlass_grouped_fp8_blockwise::ElementC;
using LayoutC       = cutlass_grouped_fp8_blockwise::LayoutC;
constexpr int AlignmentC = cutlass_grouped_fp8_blockwise::AlignmentC;
using ElementD      = cutlass_grouped_fp8_blockwise::ElementD;
using LayoutD       = cutlass_grouped_fp8_blockwise::LayoutD;
constexpr int AlignmentD = cutlass_grouped_fp8_blockwise::AlignmentD;
using ElementAccumulator = cutlass_grouped_fp8_blockwise::ElementAccumulator;
using ElementCompute     = cutlass_grouped_fp8_blockwise::ElementCompute;

// Small-M / narrow-N variant. This wins for the tiny T=1 workload.
using MmaTileShape_MNK = Shape<_64, _128, _128>;
using ClusterShape_MNK = cutlass_grouped_fp8_blockwise::ClusterShape_MNK;
using ScaleConfig = cutlass_grouped_fp8_blockwise::ScaleConfig;
using LayoutSFA = cutlass_grouped_fp8_blockwise::LayoutSFA;
using LayoutSFB = cutlass_grouped_fp8_blockwise::LayoutSFB;

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
    MmaTileShape_MNK, ClusterShape_MNK,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementCompute,
    ElementC, LayoutC*, AlignmentC,
    ElementD, LayoutC*, AlignmentD,
    cutlass::epilogue::PtrArrayTmaWarpSpecialized1Sm
  >::CollectiveOp;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
    ElementA, cute::tuple<LayoutA*, LayoutSFA*>, AlignmentA,
    ElementB, cute::tuple<LayoutB*, LayoutSFB*>, AlignmentB,
    ElementAccumulator,
    MmaTileShape_MNK, ClusterShape_MNK,
    cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    cutlass::gemm::KernelPtrArrayTmaWarpSpecializedBlockwise1SmSm100
  >::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    ProblemShape,
    CollectiveMainloop,
    CollectiveEpilogue,
    void>;

using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

using StrideA = typename Gemm::GemmKernel::InternalStrideA;
using StrideB = typename Gemm::GemmKernel::InternalStrideB;
using StrideC = typename Gemm::GemmKernel::InternalStrideC;
using StrideD = typename Gemm::GemmKernel::InternalStrideD;

static_assert(std::is_same_v<StrideA, cutlass_grouped_fp8_blockwise::StrideA>);
static_assert(std::is_same_v<StrideB, cutlass_grouped_fp8_blockwise::StrideB>);
static_assert(std::is_same_v<StrideD, cutlass_grouped_fp8_blockwise::StrideD>);
static_assert(std::is_same_v<LayoutSFA, cutlass_grouped_fp8_blockwise::LayoutSFA>);
static_assert(std::is_same_v<LayoutSFB, cutlass_grouped_fp8_blockwise::LayoutSFB>);

}  // namespace cutlass_grouped_fp8_blockwise_m64

namespace cutlass_grouped_fp8_blockwise_m64n256 {

using ProblemShape  = cutlass_grouped_fp8_blockwise::ProblemShape;
using ElementA      = cutlass_grouped_fp8_blockwise::ElementA;
using LayoutA       = cutlass_grouped_fp8_blockwise::LayoutA;
constexpr int AlignmentA = cutlass_grouped_fp8_blockwise::AlignmentA;
using ElementB      = cutlass_grouped_fp8_blockwise::ElementB;
using LayoutB       = cutlass_grouped_fp8_blockwise::LayoutB;
constexpr int AlignmentB = cutlass_grouped_fp8_blockwise::AlignmentB;
using ElementC      = cutlass_grouped_fp8_blockwise::ElementC;
using LayoutC       = cutlass_grouped_fp8_blockwise::LayoutC;
constexpr int AlignmentC = cutlass_grouped_fp8_blockwise::AlignmentC;
using ElementD      = cutlass_grouped_fp8_blockwise::ElementD;
using LayoutD       = cutlass_grouped_fp8_blockwise::LayoutD;
constexpr int AlignmentD = cutlass_grouped_fp8_blockwise::AlignmentD;
using ElementAccumulator = cutlass_grouped_fp8_blockwise::ElementAccumulator;
using ElementCompute     = cutlass_grouped_fp8_blockwise::ElementCompute;

// Small/medium-M variant from the standalone GEMM sweep. Wider N improves
// GEMM1/GEMM2 once there is more than a single routed local token.
using MmaTileShape_MNK = Shape<_64, _256, _128>;
using ClusterShape_MNK = cutlass_grouped_fp8_blockwise::ClusterShape_MNK;
using ScaleConfig = cutlass_grouped_fp8_blockwise::ScaleConfig;
using LayoutSFA = cutlass_grouped_fp8_blockwise::LayoutSFA;
using LayoutSFB = cutlass_grouped_fp8_blockwise::LayoutSFB;

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
    MmaTileShape_MNK, ClusterShape_MNK,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementCompute,
    ElementC, LayoutC*, AlignmentC,
    ElementD, LayoutC*, AlignmentD,
    cutlass::epilogue::PtrArrayTmaWarpSpecialized1Sm
  >::CollectiveOp;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
    ElementA, cute::tuple<LayoutA*, LayoutSFA*>, AlignmentA,
    ElementB, cute::tuple<LayoutB*, LayoutSFB*>, AlignmentB,
    ElementAccumulator,
    MmaTileShape_MNK, ClusterShape_MNK,
    cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    cutlass::gemm::KernelPtrArrayTmaWarpSpecializedBlockwise1SmSm100
  >::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    ProblemShape,
    CollectiveMainloop,
    CollectiveEpilogue,
    void>;

using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

using StrideA = typename Gemm::GemmKernel::InternalStrideA;
using StrideB = typename Gemm::GemmKernel::InternalStrideB;
using StrideC = typename Gemm::GemmKernel::InternalStrideC;
using StrideD = typename Gemm::GemmKernel::InternalStrideD;

static_assert(std::is_same_v<StrideA, cutlass_grouped_fp8_blockwise::StrideA>);
static_assert(std::is_same_v<StrideB, cutlass_grouped_fp8_blockwise::StrideB>);
static_assert(std::is_same_v<StrideD, cutlass_grouped_fp8_blockwise::StrideD>);
static_assert(std::is_same_v<LayoutSFA, cutlass_grouped_fp8_blockwise::LayoutSFA>);
static_assert(std::is_same_v<LayoutSFB, cutlass_grouped_fp8_blockwise::LayoutSFB>);

}  // namespace cutlass_grouped_fp8_blockwise_m64n256

struct CutlassGroupedFp8BlockwiseM128Traits {
    using ProblemShape = cutlass_grouped_fp8_blockwise::ProblemShape;
    using ElementA = cutlass_grouped_fp8_blockwise::ElementA;
    using ElementB = cutlass_grouped_fp8_blockwise::ElementB;
    using ElementD = cutlass_grouped_fp8_blockwise::ElementD;
    using ElementAccumulator = cutlass_grouped_fp8_blockwise::ElementAccumulator;
    using Gemm = cutlass_grouped_fp8_blockwise::Gemm;
    using StrideA = cutlass_grouped_fp8_blockwise::StrideA;
    using StrideB = cutlass_grouped_fp8_blockwise::StrideB;
    using StrideC = cutlass_grouped_fp8_blockwise::StrideC;
    using StrideD = cutlass_grouped_fp8_blockwise::StrideD;
    using LayoutSFA = cutlass_grouped_fp8_blockwise::LayoutSFA;
    using LayoutSFB = cutlass_grouped_fp8_blockwise::LayoutSFB;
};

struct CutlassGroupedFp8BlockwiseM64Traits {
    using ProblemShape = cutlass_grouped_fp8_blockwise_m64::ProblemShape;
    using ElementA = cutlass_grouped_fp8_blockwise_m64::ElementA;
    using ElementB = cutlass_grouped_fp8_blockwise_m64::ElementB;
    using ElementD = cutlass_grouped_fp8_blockwise_m64::ElementD;
    using ElementAccumulator = cutlass_grouped_fp8_blockwise_m64::ElementAccumulator;
    using Gemm = cutlass_grouped_fp8_blockwise_m64::Gemm;
    using StrideA = cutlass_grouped_fp8_blockwise_m64::StrideA;
    using StrideB = cutlass_grouped_fp8_blockwise_m64::StrideB;
    using StrideC = cutlass_grouped_fp8_blockwise_m64::StrideC;
    using StrideD = cutlass_grouped_fp8_blockwise_m64::StrideD;
    using LayoutSFA = cutlass_grouped_fp8_blockwise_m64::LayoutSFA;
    using LayoutSFB = cutlass_grouped_fp8_blockwise_m64::LayoutSFB;
};

struct CutlassGroupedFp8BlockwiseM64N256Traits {
    using ProblemShape = cutlass_grouped_fp8_blockwise_m64n256::ProblemShape;
    using ElementA = cutlass_grouped_fp8_blockwise_m64n256::ElementA;
    using ElementB = cutlass_grouped_fp8_blockwise_m64n256::ElementB;
    using ElementD = cutlass_grouped_fp8_blockwise_m64n256::ElementD;
    using ElementAccumulator = cutlass_grouped_fp8_blockwise_m64n256::ElementAccumulator;
    using Gemm = cutlass_grouped_fp8_blockwise_m64n256::Gemm;
    using StrideA = cutlass_grouped_fp8_blockwise_m64n256::StrideA;
    using StrideB = cutlass_grouped_fp8_blockwise_m64n256::StrideB;
    using StrideC = cutlass_grouped_fp8_blockwise_m64n256::StrideC;
    using StrideD = cutlass_grouped_fp8_blockwise_m64n256::StrideD;
    using LayoutSFA = cutlass_grouped_fp8_blockwise_m64n256::LayoutSFA;
    using LayoutSFB = cutlass_grouped_fp8_blockwise_m64n256::LayoutSFB;
};

template <class Traits>
static int64_t cutlass_grouped_fp8_blockwise_run_impl(
    // Per-group device arrays (already populated):
    void* dev_ptr_A,         // ElementA**  [groups]
    void* dev_ptr_B,         // ElementB**  [groups]
    void* dev_ptr_D,         // ElementD**  [groups]
    void* dev_ptr_SFA,       // float**     [groups]
    void* dev_ptr_SFB,       // float**     [groups]
    void* dev_stride_A,      // StrideA[groups]
    void* dev_stride_B,      // StrideB[groups]
    void* dev_stride_D,      // StrideD[groups]
    void* dev_layout_SFA,    // LayoutSFA[groups]
    void* dev_layout_SFB,    // LayoutSFB[groups]
    void* dev_problem_sizes, // {M,N,K}[groups]
    void* host_problem_sizes,// {M,N,K}[groups] host copy
    int   groups,
    void* workspace,
    int64_t workspace_size,
    cudaStream_t stream)
{
    using ProblemShape = typename Traits::ProblemShape;
    using ElementA = typename Traits::ElementA;
    using ElementB = typename Traits::ElementB;
    using ElementD = typename Traits::ElementD;
    using ElementAccumulator = typename Traits::ElementAccumulator;
    using Gemm = typename Traits::Gemm;
    using StrideA = typename Traits::StrideA;
    using StrideB = typename Traits::StrideB;
    using StrideC = typename Traits::StrideC;
    using StrideD = typename Traits::StrideD;
    using LayoutSFA = typename Traits::LayoutSFA;
    using LayoutSFB = typename Traits::LayoutSFB;

    cutlass::KernelHardwareInfo hw_info;
    hw_info.device_id = 0;
    hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);

    using PSH = typename ProblemShape::UnderlyingProblemShape;
    typename Gemm::Arguments arguments{
        cutlass::gemm::GemmUniversalMode::kGrouped,
        {groups,
         static_cast<PSH*>(dev_problem_sizes),
         static_cast<PSH*>(host_problem_sizes)},
        {static_cast<const ElementA**>(dev_ptr_A), static_cast<StrideA*>(dev_stride_A),
         static_cast<const ElementB**>(dev_ptr_B), static_cast<StrideB*>(dev_stride_B),
         static_cast<const ElementAccumulator**>(dev_ptr_SFA), static_cast<LayoutSFA*>(dev_layout_SFA),
         static_cast<const ElementAccumulator**>(dev_ptr_SFB), static_cast<LayoutSFB*>(dev_layout_SFB)},
        {{}, /*C=*/nullptr, /*stride_C=*/static_cast<StrideC*>(dev_stride_D),
              static_cast<ElementD**>(dev_ptr_D), static_cast<StrideD*>(dev_stride_D)},
        hw_info
    };

    auto& fusion_args = arguments.epilogue.thread;
    fusion_args.alpha = 1.f;
    fusion_args.beta  = 0.f;

    size_t needed_ws = Gemm::get_workspace_size(arguments);
    if ((int64_t)needed_ws > workspace_size) {
        fprintf(stderr, "cutlass workspace too small: need %zu, have %lld\n",
                needed_ws, (long long)workspace_size);
        return -13;
    }
    Gemm gemm;
    cutlass::Status s = gemm.can_implement(arguments);
    if (s != cutlass::Status::kSuccess) {
        fprintf(stderr, "cutlass can_implement failed: %d (%s) [groups=%d]\n",
                (int)s, cutlass::cutlassGetStatusString(s), groups);
        return -10;
    }
    s = gemm.initialize(arguments, workspace);
    if (s != cutlass::Status::kSuccess) {
        fprintf(stderr, "cutlass initialize failed: %d (%s)\n",
                (int)s, cutlass::cutlassGetStatusString(s));
        return -11;
    }
    s = gemm.run(stream);
    if (s != cutlass::Status::kSuccess) {
        fprintf(stderr, "cutlass run failed: %d (%s)\n",
                (int)s, cutlass::cutlassGetStatusString(s));
        return -12;
    }
    return 0;
}

extern "C" __attribute__((visibility("default")))
int64_t cutlass_grouped_fp8_blockwise_run(
    // Per-group device arrays (already populated):
    void* dev_ptr_A,         // ElementA**  [groups]
    void* dev_ptr_B,         // ElementB**  [groups]
    void* dev_ptr_D,         // ElementD**  [groups]
    void* dev_ptr_SFA,       // float**     [groups]
    void* dev_ptr_SFB,       // float**     [groups]
    void* dev_stride_A,      // StrideA[groups]
    void* dev_stride_B,      // StrideB[groups]
    void* dev_stride_D,      // StrideD[groups]
    void* dev_layout_SFA,    // LayoutSFA[groups]
    void* dev_layout_SFB,    // LayoutSFB[groups]
    void* dev_problem_sizes, // {M,N,K}[groups]
    void* host_problem_sizes,// {M,N,K}[groups] host copy
    int   groups,
    void* workspace,
    int64_t workspace_size,
    cudaStream_t stream)
{
    return cutlass_grouped_fp8_blockwise_run_impl<CutlassGroupedFp8BlockwiseM128Traits>(
        dev_ptr_A, dev_ptr_B, dev_ptr_D, dev_ptr_SFA, dev_ptr_SFB,
        dev_stride_A, dev_stride_B, dev_stride_D,
        dev_layout_SFA, dev_layout_SFB,
        dev_problem_sizes, host_problem_sizes,
        groups, workspace, workspace_size, stream);
}

extern "C" __attribute__((visibility("default")))
int64_t cutlass_grouped_fp8_blockwise_run_m64(
    // Per-group device arrays (already populated):
    void* dev_ptr_A,         // ElementA**  [groups]
    void* dev_ptr_B,         // ElementB**  [groups]
    void* dev_ptr_D,         // ElementD**  [groups]
    void* dev_ptr_SFA,       // float**     [groups]
    void* dev_ptr_SFB,       // float**     [groups]
    void* dev_stride_A,      // StrideA[groups]
    void* dev_stride_B,      // StrideB[groups]
    void* dev_stride_D,      // StrideD[groups]
    void* dev_layout_SFA,    // LayoutSFA[groups]
    void* dev_layout_SFB,    // LayoutSFB[groups]
    void* dev_problem_sizes, // {M,N,K}[groups]
    void* host_problem_sizes,// {M,N,K}[groups] host copy
    int   groups,
    void* workspace,
    int64_t workspace_size,
    cudaStream_t stream)
{
    return cutlass_grouped_fp8_blockwise_run_impl<CutlassGroupedFp8BlockwiseM64Traits>(
        dev_ptr_A, dev_ptr_B, dev_ptr_D, dev_ptr_SFA, dev_ptr_SFB,
        dev_stride_A, dev_stride_B, dev_stride_D,
        dev_layout_SFA, dev_layout_SFB,
        dev_problem_sizes, host_problem_sizes,
        groups, workspace, workspace_size, stream);
}

extern "C" __attribute__((visibility("default")))
int64_t cutlass_grouped_fp8_blockwise_run_m64n256(
    // Per-group device arrays (already populated):
    void* dev_ptr_A,         // ElementA**  [groups]
    void* dev_ptr_B,         // ElementB**  [groups]
    void* dev_ptr_D,         // ElementD**  [groups]
    void* dev_ptr_SFA,       // float**     [groups]
    void* dev_ptr_SFB,       // float**     [groups]
    void* dev_stride_A,      // StrideA[groups]
    void* dev_stride_B,      // StrideB[groups]
    void* dev_stride_D,      // StrideD[groups]
    void* dev_layout_SFA,    // LayoutSFA[groups]
    void* dev_layout_SFB,    // LayoutSFB[groups]
    void* dev_problem_sizes, // {M,N,K}[groups]
    void* host_problem_sizes,// {M,N,K}[groups] host copy
    int   groups,
    void* workspace,
    int64_t workspace_size,
    cudaStream_t stream)
{
    return cutlass_grouped_fp8_blockwise_run_impl<CutlassGroupedFp8BlockwiseM64N256Traits>(
        dev_ptr_A, dev_ptr_B, dev_ptr_D, dev_ptr_SFA, dev_ptr_SFB,
        dev_stride_A, dev_stride_B, dev_stride_D,
        dev_layout_SFA, dev_layout_SFB,
        dev_problem_sizes, host_problem_sizes,
        groups, workspace, workspace_size, stream);
}

extern "C" __attribute__((visibility("default")))
int64_t cutlass_grouped_fp8_blockwise_workspace_size(int M_max, int N_max, int K_max, int groups)
{
    using namespace cutlass_grouped_fp8_blockwise;
    (void)M_max; (void)N_max; (void)K_max; (void)groups;
    // Heuristic: ~16 MB headroom (group strides + scheduler params + epilogue scratch).
    return 16 * 1024 * 1024;
}

// Type sizes for moe_fp8.cu to allocate correctly.
extern "C" __attribute__((visibility("default")))
int cutlass_grouped_fp8_blockwise_sizeof_stride_A() { return (int)sizeof(cutlass_grouped_fp8_blockwise::StrideA); }
extern "C" __attribute__((visibility("default")))
int cutlass_grouped_fp8_blockwise_sizeof_stride_B() { return (int)sizeof(cutlass_grouped_fp8_blockwise::StrideB); }
extern "C" __attribute__((visibility("default")))
int cutlass_grouped_fp8_blockwise_sizeof_stride_D() { return (int)sizeof(cutlass_grouped_fp8_blockwise::StrideD); }
extern "C" __attribute__((visibility("default")))
int cutlass_grouped_fp8_blockwise_sizeof_layout_SFA() { return (int)sizeof(cutlass_grouped_fp8_blockwise::LayoutSFA); }
extern "C" __attribute__((visibility("default")))
int cutlass_grouped_fp8_blockwise_sizeof_layout_SFB() { return (int)sizeof(cutlass_grouped_fp8_blockwise::LayoutSFB); }

// Build per-group descriptors on device.
//
// Inputs:
//   d_expert_count[groups], d_expert_offset_m[groups+1] (token offsets, in tokens)
//   base_A: packed activations [M_total, K] row-major, FP8.
//   base_B: weights [groups, N, K] row-major, FP8 (stride per group = N*K).
//   base_D: output [M_total, N] row-major, bf16.
//   base_SFA: packed activation scale [K/128, M_total] (stride-1 in M).
//             Shared across groups: ptr_SFA[le] = base + m_offset.
//             K-block stride is M_total (the shared buffer's M extent).
//   base_SFB: weight scale [groups, N/128, K/128] row-major (stride per group
//             = N/128 * K/128 = N*K/128/128).
//
//   N, K: per-group N, K (constant across groups).
//   M_total: total token count (= sum of M_e).
__global__ void build_descriptors_kernel(
    const int32_t* __restrict__ d_expert_count,    // [groups] padded M_e (for SFA layout)
    const int32_t* __restrict__ d_expert_count_unpadded, // [groups] actual M (for problem_size)
    const int32_t* __restrict__ d_expert_offset_m, // [groups+1] cumsum padded M_e (in tokens)
    const int32_t* __restrict__ d_sfa_offset,      // [groups+1] cumsum (K/128 * padded M_e)
    const void* base_A,
    const void* base_B,
    void*       base_D,
    const void* base_SFA,
    const void* base_SFB,
    int N, int K, int d_row_stride,
    int b_per_expert_elems,         // N*K
    int sfb_per_expert_elems,       // (N/128)*(K/128)
    void** out_ptr_A,
    void** out_ptr_B,
    void** out_ptr_D,
    void** out_ptr_SFA,
    void** out_ptr_SFB,
    cutlass_grouped_fp8_blockwise::StrideA*    out_stride_A,
    cutlass_grouped_fp8_blockwise::StrideB*    out_stride_B,
    cutlass_grouped_fp8_blockwise::StrideD*    out_stride_D,
    cutlass_grouped_fp8_blockwise::LayoutSFA*  out_layout_SFA,
    cutlass_grouped_fp8_blockwise::LayoutSFB*  out_layout_SFB,
    int* out_problem_sizes,
    int groups)
{
    using namespace cutlass_grouped_fp8_blockwise;
    int tid = threadIdx.x;
    if (tid >= groups) return;

    int M_padded   = d_expert_count[tid];
    int M_unpadded = d_expert_count_unpadded ? d_expert_count_unpadded[tid] : M_padded;
    int m_offset = d_expert_offset_m[tid];
    int sfa_offset = d_sfa_offset[tid];

    out_ptr_A[tid] = (void*)((const cutlass::float_e4m3_t*)base_A + (size_t)m_offset * K);
    out_ptr_B[tid] = (void*)((const cutlass::float_e4m3_t*)base_B + (size_t)tid * b_per_expert_elems);
    out_ptr_D[tid] = (void*) ((cutlass::bfloat16_t*)base_D       + (size_t)m_offset * d_row_stride);
    out_ptr_SFA[tid] = (void*)((const float*)base_SFA + (size_t)sfa_offset);
    out_ptr_SFB[tid] = (void*)((const float*)base_SFB + (size_t)tid * sfb_per_expert_elems);

    // Strides for A/B/D depend only on N and K (row major), not M.
    out_stride_A[tid] = cutlass::make_cute_packed_stride(StrideA{}, {M_unpadded, K, 1});
    out_stride_B[tid] = cutlass::make_cute_packed_stride(StrideB{}, {N, K, 1});
    out_stride_D[tid] = cutlass::make_cute_packed_stride(StrideD{}, {M_unpadded, d_row_stride, 1});

    // SFA layout uses PADDED M so the K-block stride matches our SFA buffer.
    out_layout_SFA[tid] = ScaleConfig::tile_atom_to_shape_SFA(make_shape(M_padded, N, K, 1));
    out_layout_SFB[tid] = ScaleConfig::tile_atom_to_shape_SFB(make_shape(M_padded, N, K, 1));

    // Problem size uses the UNPADDED M so the tile scheduler doesn't waste
    // FLOPs on the zero-padded rows.
    out_problem_sizes[tid * 3 + 0] = M_unpadded;
    out_problem_sizes[tid * 3 + 1] = N;
    out_problem_sizes[tid * 3 + 2] = K;
}

extern "C" __attribute__((visibility("default")))
void cutlass_grouped_fp8_blockwise_build_descriptors(
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
    cudaStream_t stream)
{
    using namespace cutlass_grouped_fp8_blockwise;
    int block = ((groups + 31) / 32) * 32;
    if (block < 32) block = 32;
    build_descriptors_kernel<<<1, block, 0, stream>>>(
        d_expert_count, nullptr, d_expert_offset_m, d_sfa_offset,
        base_A, base_B, base_D, base_SFA, base_SFB,
        N, K, d_row_stride, b_per_expert_elems, sfb_per_expert_elems,
        out_ptr_A, out_ptr_B, out_ptr_D, out_ptr_SFA, out_ptr_SFB,
        (StrideA*)out_stride_A, (StrideB*)out_stride_B, (StrideD*)out_stride_D,
        (LayoutSFA*)out_layout_SFA, (LayoutSFB*)out_layout_SFB,
        out_problem_sizes, groups);
}

extern "C" __attribute__((visibility("default")))
void cutlass_grouped_fp8_blockwise_build_descriptors_unpadded(
    const int32_t* d_expert_count_padded,
    const int32_t* d_expert_count_unpadded,
    const int32_t* d_expert_offset_m,
    const int32_t* d_sfa_offset,
    const void* base_A, const void* base_B, void* base_D,
    const void* base_SFA, const void* base_SFB,
    int N, int K, int d_row_stride, int b_per_expert_elems, int sfb_per_expert_elems,
    void** out_ptr_A, void** out_ptr_B, void** out_ptr_D,
    void** out_ptr_SFA, void** out_ptr_SFB,
    void* out_stride_A, void* out_stride_B, void* out_stride_D,
    void* out_layout_SFA, void* out_layout_SFB,
    int* out_problem_sizes, int groups,
    cudaStream_t stream)
{
    using namespace cutlass_grouped_fp8_blockwise;
    int block = ((groups + 31) / 32) * 32;
    if (block < 32) block = 32;
    build_descriptors_kernel<<<1, block, 0, stream>>>(
        d_expert_count_padded, d_expert_count_unpadded,
        d_expert_offset_m, d_sfa_offset,
        base_A, base_B, base_D, base_SFA, base_SFB,
        N, K, d_row_stride, b_per_expert_elems, sfb_per_expert_elems,
        out_ptr_A, out_ptr_B, out_ptr_D, out_ptr_SFA, out_ptr_SFB,
        (StrideA*)out_stride_A, (StrideB*)out_stride_B, (StrideD*)out_stride_D,
        (LayoutSFA*)out_layout_SFA, (LayoutSFB*)out_layout_SFB,
        out_problem_sizes, groups);
}

#endif  // CUTLASS_ARCH_MMA_SM100_SUPPORTED
