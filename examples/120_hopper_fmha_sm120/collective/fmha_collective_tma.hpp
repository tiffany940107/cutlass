/***************************************************************************************************
 * Copyright (c) 2024 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/gemm/collective/collective_builder.hpp"

#include "../collective/fmha_common.hpp"
#include "../collective/fmha_collective_load.hpp"
#include "../collective/fmha_collective_softmax.hpp"
#include "../kernel/fmha_options.hpp"
#include "../cute_extension.h"

namespace cutlass::fmha::collective {

using namespace cute;
using cutlass::fmha::kernel::Tag;
using cutlass::fmha::kernel::find_option_t;

template<
  typename Element_,
  typename ElementAccumulator_,
  typename TileShape_, // BlockQO, BlockKV, BlockHead
  class Fusion,
  class... Options
>
struct FmhaMainloopTma {

  using Element = Element_;
  using ElementAccumulator = ElementAccumulator_;
  using TileShape = TileShape_;

  // Options
  using kClusterM = find_option_t<Tag::kClusterM, Int<1>, Options...>;
  static constexpr int StageCount = find_option_t<Tag::kStagesKV, Int<4>, Options...>::value;
  static constexpr int StageCountQ = find_option_t<Tag::kStagesQ, Int<1>, Options...>::value;

  using StagesQ = cutlass::gemm::collective::StageCount<StageCountQ>;
  using Stages = cutlass::gemm::collective::StageCount<StageCount>;
  using ClusterShape = Shape<kClusterM, _1, _1>;

  // 16B alignment lets us use TMA
  // For NVFP4, we need 32 elements alignment (32 * 4 bytes = 128 bytes)
  static constexpr int Alignment = 32;

  using TileShapeQK = TileShape;
  using TileShapePV = decltype(select<0,2,1>(TileShapeQK{}));

  using LayoutQKV = cute::tuple<int, _1, cute::tuple<int, int>>;
  using LayoutQ = LayoutQKV;
  using LayoutK = LayoutQKV;
  using LayoutV = LayoutQKV;

  // Define NVFP4 types for internal GEMM operations
  using ElementNVFP4 = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
  using ElementScaleFactor = typename ElementNVFP4::ScaleFactorType;
  
  // BlockScaledConfig for NVFP4 scale factors (based on Sage3 design)
  template<int SFVecSize_>
  struct BlockScaledConfig {
    static constexpr int SFVecSize = SFVecSize_;
    static constexpr int MMA_NSF = 4; // SFVecSize, MMA_NSF
    using Blk_MN = _64;
    using Blk_SF = _4; 
    using mnBasicBlockShape = Shape<_16,_4>;
    using mnBasicBlockStride = Stride<_16,_4>;
    using kBasicBlockShape = Shape<Int<SFVecSize>, Int<MMA_NSF>>;
    using kBasicBlockStride = Stride<_0, _1>;
    using SfAtom = Layout<Shape<mnBasicBlockShape, kBasicBlockShape>, 
                          Stride<mnBasicBlockStride, kBasicBlockStride>>;

    using LayoutSF = decltype(blocked_product(SfAtom{}, 
                                  make_layout(
                                      make_shape(int32_t(0), int32_t(0), int32_t(0), int32_t(0)),
                                      make_stride(int32_t(0), _1{}, int32_t(0), int32_t(0)))));
    
    using Blk_Elems = decltype(Blk_MN{} * Blk_SF{});
    using sSF_strideMN = decltype(prepend(Blk_Elems{}, mnBasicBlockStride{}));
    
    // Function to create scale factor layout for QKV
    template <class ProblemShape>
    CUTE_HOST_DEVICE
    static constexpr auto
    tile_atom_to_shape_SFQKV(ProblemShape problem_shape) {
      auto [Seqlen, Dim, HeadNum, Batch] = problem_shape;
      return tile_to_shape(SfAtom{}, make_shape(Seqlen, Dim, HeadNum, Batch), Step<_2,_1,_3,_4>{});
    }
    
    // Function to create scale factor layout for Vt
    template <class ProblemShape>
    CUTE_HOST_DEVICE
    static constexpr auto
    tile_atom_to_shape_SFVt(ProblemShape problem_shape) {
      auto [Dim, Seqlen, HeadNum, Batch] = problem_shape;
      return tile_to_shape(SfAtom{}, make_shape(Dim, Seqlen, HeadNum, Batch), Step<_2,_1,_3,_4>{});
    }
    
    // Deduce shared memory layout for scale factors (based on Sage3)
    template<class TiledMma, class TileShape_MNK>
    CUTE_HOST_DEVICE
    static constexpr auto
    deduce_smem_layoutSFQ(TiledMma tiled_mma, TileShape_MNK tileshape_mnk) {
      using sSFQ_shapeK = decltype(prepend(make_shape(Blk_SF{}/Int<MMA_NSF>{}, size<2>(TileShape_MNK{}) / Int<SFVecSize>{} / Blk_SF{}), kBasicBlockShape{}));
      using sSFQ_shapeM = decltype(prepend(size<0>(TileShape_MNK{}) / Blk_MN{}, mnBasicBlockShape{}));
      using sSFQ_strideM = sSF_strideMN;
      using sSFQ_strideK = decltype(prepend(make_stride(Int<MMA_NSF>{}, size<0>(TileShape_MNK{}) / Blk_MN{} * Blk_Elems{}), kBasicBlockStride{}));
      using sSFQ_shape = decltype(make_shape(sSFQ_shapeM{}, sSFQ_shapeK{}));
      using sSFQ_stride = decltype(make_stride(sSFQ_strideM{}, sSFQ_strideK{}));
      using SmemLayoutAtomSFQ = decltype(make_layout(sSFQ_shape{}, sSFQ_stride{}));
      return SmemLayoutAtomSFQ{};
    }
    
    template<class TiledMma, class TileShape_MNK>
    CUTE_HOST_DEVICE
    static constexpr auto
    deduce_smem_layoutSFKV(TiledMma tiled_mma, TileShape_MNK tileshape_mnk) {
      using sSFK_shapeK = decltype(prepend(make_shape(Blk_SF{}/Int<MMA_NSF>{}, size<2>(TileShape_MNK{}) / Int<SFVecSize>{} / Blk_SF{}), kBasicBlockShape{}));
      using sSFK_shapeN = decltype(prepend(size<1>(TileShape_MNK{}) / Blk_MN{}, mnBasicBlockShape{}));
      using sSFK_strideN = sSF_strideMN;
      using sSFK_strideK = decltype(prepend(make_stride(Int<MMA_NSF>{}, size<1>(TileShape_MNK{}) / Blk_MN{} * Blk_Elems{}), kBasicBlockStride{}));
      using sSFK_shape = decltype(make_shape(sSFK_shapeN{}, sSFK_shapeK{}));
      using sSFK_stride = decltype(make_stride(sSFK_strideN{}, sSFK_strideK{}));
      using SmemLayoutAtomSFK = decltype(make_layout(sSFK_shape{}, sSFK_stride{}));
      return SmemLayoutAtomSFK{};
    }
  };
  
  // Direct TiledMma configuration (based on Sage3 approach)
  static constexpr int SFVectorSize = 16;
  using BlkScaledConfig = BlockScaledConfig<SFVectorSize>;
  
  // Define atom layouts for TiledMma
  using AtomLayoutMNK = Layout<Shape<_4, _1, _1>>;
  
  // Create TiledMma directly (bypassing CollectiveBuilder)
  using TiledMmaQK = decltype(cute::make_tiled_mma(
      cute::SM120::BLOCKSCALED::SM120_16x32x64_TN_VS_NVFP4{},
      AtomLayoutMNK{},
      TileShapeQK{}
  ));
  
  using TiledMmaPV = decltype(cute::make_tiled_mma(
      cute::SM120::BLOCKSCALED::SM120_16x32x64_TN_VS_NVFP4{},
      AtomLayoutMNK{},
      TileShapePV{}
  ));

  // Scale factor layouts for NVFP4 GEMM operations (using our BlockScaledConfig)
  using LayoutSF = typename BlkScaledConfig::LayoutSF;
  using LayoutSFQ = LayoutSF;
  using LayoutSFK = LayoutSF;
  using LayoutSFV = LayoutSF;

  // Shared memory layouts (based on Sage3 approach)
  using SmemLayoutAtomQ = decltype(cutlass::gemm::collective::detail::sm120_rr_smem_selector<Element, decltype(size<2>(TileShapeQK{}))>());
  using SmemLayoutAtomK = decltype(cutlass::gemm::collective::detail::sm120_rr_smem_selector<Element, decltype(size<2>(TileShapeQK{}))>());
  using SmemLayoutAtomV = decltype(cutlass::gemm::collective::detail::sm120_rr_smem_selector<Element, decltype(size<2>(TileShapePV{}))>());
  
  using SmemLayoutQ = decltype(tile_to_shape(SmemLayoutAtomQ{}, select<0, 2>(TileShapeQK{})));
  using SmemLayoutK = decltype(tile_to_shape(SmemLayoutAtomK{}, 
                   make_shape(shape<1>(TileShapeQK{}), shape<2>(TileShapeQK{}), Int<Stages::value>{})));
  using SmemLayoutV = decltype(tile_to_shape(SmemLayoutAtomV{}, 
                   make_shape(shape<1>(TileShapePV{}), shape<2>(TileShapePV{}), Int<Stages::value>{})));

  using MainloopPipeline = cutlass::PipelineTmaAsync<Stages::value>;
  using MainloopPipelineQ = cutlass::PipelineTmaAsync<StagesQ::value>;

  using PipelineState  = typename cutlass::PipelineState<MainloopPipeline::Stages>;
  using PipelineStateQ  = typename cutlass::PipelineState<MainloopPipelineQ::Stages>;

  using TileShapeOut = TileShapePV;
  using TiledMmaOut = TiledMmaPV;
  using ElementOut = ElementAccumulator;

  // Scale factor shared memory layouts
  using SmemLayoutSFQ = decltype(BlkScaledConfig::deduce_smem_layoutSFQ(TiledMmaQK{}, TileShapeQK{}));
  using SmemLayoutSFK = decltype(BlkScaledConfig::deduce_smem_layoutSFKV(TiledMmaQK{}, TileShapeQK{}));
  using SmemLayoutSFV = decltype(BlkScaledConfig::deduce_smem_layoutSFKV(TiledMmaPV{}, TileShapePV{}));

  struct SharedStorage {
    cute::array_aligned<Element, cute::cosize_v<SmemLayoutQ>> smem_q;
    union {
      cute::array_aligned<Element, cute::cosize_v<SmemLayoutK>> smem_k;
      cute::array_aligned<Element, cute::cosize_v<SmemLayoutV>> smem_v;
    };
    // Scale factor storage
    cute::array_aligned<ElementScaleFactor, cute::cosize_v<SmemLayoutSFQ>> smem_SFQ;
    cute::array_aligned<ElementScaleFactor, cute::cosize_v<SmemLayoutSFK>> smem_SFK;
    cute::array_aligned<ElementScaleFactor, cute::cosize_v<SmemLayoutSFV>> smem_SFV;
  };

  struct Arguments {
    const Element* ptr_Q;
    LayoutQ dQ;
    const Element* ptr_K;
    LayoutK dK;
    const Element* ptr_V;
    LayoutV dV;
    
    // Scale factor parameters for NVFP4 GEMM operations
    const ElementScaleFactor* ptr_SFQ;
    LayoutSFQ dSFQ;
    const ElementScaleFactor* ptr_SFK;
    LayoutSFK dSFK;
    const ElementScaleFactor* ptr_SFV;
    LayoutSFV dSFV;
  };

  // TMA configuration (based on Sage3 approach)
  using GmemTiledCopy = SM90_TMA_LOAD;
  using GmemTiledCopySF = SM90_TMA_LOAD;
  
  // Define global memory layouts (simplified like Sage3)
  using StrideQKV = cute::Stride<int64_t, _1, int64_t, int64_t>;
  
  using TMA_Q = decltype(make_tma_copy(
      GmemTiledCopy{},
      make_tensor(make_gmem_ptr(static_cast<Element const*>(nullptr)), 
                  repeat_like(StrideQKV{}, int32_t(0)), StrideQKV{}),
      SmemLayoutQ{},
      select<0, 2>(TileShapeQK{}),
      _1{}));
  
  using TMA_K = decltype(make_tma_copy(
      GmemTiledCopy{},
      make_tensor(make_gmem_ptr(static_cast<Element const*>(nullptr)), 
                  repeat_like(StrideQKV{}, int32_t(0)), StrideQKV{}),
      take<0, 2>(SmemLayoutK{}),
      select<1, 2>(TileShapeQK{}),
      _1{}));
  
  using TMA_V = decltype(make_tma_copy(
      GmemTiledCopy{},
      make_tensor(make_gmem_ptr(static_cast<Element const*>(nullptr)), 
                  repeat_like(StrideQKV{}, int32_t(0)), StrideQKV{}),
      take<0, 2>(SmemLayoutV{}),
      select<1, 2>(TileShapePV{}),
      _1{}));
  
  // Scale factor TMA (simplified like Sage3)
  using TMA_SFQ = decltype(make_tma_copy<uint16_t>(
      GmemTiledCopySF{},
      make_tensor(static_cast<ElementScaleFactor const*>(nullptr), LayoutSF{}),
      SmemLayoutSFQ{},
      make_shape(shape<0>(TileShapeQK{}), shape<2>(TileShapeQK{})),
      _1{}));
  
  using TMA_SFK = decltype(make_tma_copy<uint16_t>(
      GmemTiledCopySF{},
      make_tensor(static_cast<ElementScaleFactor const*>(nullptr), LayoutSF{}),
      SmemLayoutSFK{}(_,_,cute::Int<0>{}),
      make_shape(shape<1>(TileShapeQK{}), shape<2>(TileShapeQK{})),
      _1{}));
  
  using TMA_SFV = decltype(make_tma_copy<uint16_t>(
      GmemTiledCopySF{},
      make_tensor(static_cast<ElementScaleFactor const*>(nullptr), LayoutSF{}),
      SmemLayoutSFV{}(_,_,cute::Int<0>{}),
      make_shape(shape<1>(TileShapePV{}), shape<2>(TileShapePV{})),
      _1{}));

  struct Params {
    TMA_Q tma_load_q;
    TMA_K tma_load_k;
    TMA_V tma_load_v;
    
    // Scale factor TMA
    TMA_SFQ tma_load_sfq;
    TMA_SFK tma_load_sfk;
    TMA_SFV tma_load_sfv;

    float scale_softmax;
    float scale_softmax_log2;
    float rp_dropout;
  };

  using LoadQ = cutlass::fmha::collective::CollectiveLoadTma<
    cutlass::fmha::collective::LoadKind::kQ,
    MainloopPipelineQ,
    Element,
    SmemLayoutQ,
    TMA_Q
  >;

  using LoadK = cutlass::fmha::collective::CollectiveLoadTma<
    cutlass::fmha::collective::LoadKind::kK,
    MainloopPipeline,
    Element,
    SmemLayoutK,
    TMA_K
  >;

  using LoadV = cutlass::fmha::collective::CollectiveLoadTma<
    cutlass::fmha::collective::LoadKind::kV,
    MainloopPipeline,
    Element,
    SmemLayoutV,
    TMA_V
  >;

  static_assert(size(TiledMmaQK{}) == size(TiledMmaPV{}));

  static const int MaxThreadsPerBlock = size(TiledMmaQK{});

  template<class ProblemShape>
  static bool can_implement(ProblemShape const& problem_size, Arguments const& args) {
    return true
      && (get<4>(problem_size) <= get<2>(TileShape{}))
      && ((get<4>(problem_size) % Alignment) == 0)
      && ((get<2>(problem_size) % Alignment) == 0)
    ;
  }

  template<class ProblemShape>
  static Params to_underlying_arguments(ProblemShape const& problem_size, Arguments const& args, void* workspace) {

    // Create TMA parameters directly
    auto tma_load_q = TMA_Q{};
    auto tma_load_k = TMA_K{};
    auto tma_load_v = TMA_V{};
    auto tma_load_sfq = TMA_SFQ{};
    auto tma_load_sfk = TMA_SFK{};
    auto tma_load_sfv = TMA_SFV{};

    return Params{
        tma_load_q,
        tma_load_k,
        tma_load_v,
        1.0f / (float) std::sqrt(get<4>(problem_size)),
        (float) (std::log2(std::exp(1.0)) / std::sqrt(get<4>(problem_size))),
        1.0f,
        tma_load_sfq,
        tma_load_sfk,
        tma_load_sfv,
    };
  }

  CUTLASS_DEVICE
  static void prefetch_tma_descriptors(Params const& params) {
    cute::prefetch_tma_descriptor(params.tma_load_q.get_tma_descriptor());
    cute::prefetch_tma_descriptor(params.tma_load_k.get_tma_descriptor());
    cute::prefetch_tma_descriptor(params.tma_load_v.get_tma_descriptor());
  }

  template<class BlkCoord, class ProblemShape>
  CUTLASS_DEVICE auto
  compute(
      int block_rank_in_cluster,
      BlkCoord const& blk_coord, Params const& params, ProblemShape const& problem_size,
      MainloopPipeline& pipeline, PipelineState& smem_pipe_read, PipelineState& smem_pipe_write,
      MainloopPipelineQ& pipeline_q, PipelineStateQ& smem_pipe_read_q, PipelineStateQ& smem_pipe_write_q,
      SharedStorage& storage)
  {
    int warp_idx   = cutlass::canonical_warp_idx_sync();
    int thread_idx = threadIdx.x;
      
    PipelineState smem_pipe_release = smem_pipe_read;
    [[maybe_unused]] PipelineStateQ smem_pipe_release_q = smem_pipe_read_q;


    int fusion_tile_count = Fusion{}.get_trip_count(blk_coord, TileShape{}, problem_size);

    LoadQ load_q{params.tma_load_q, pipeline_q, storage.smem_q};
    auto load_state_q = load_q.init_state(_0{}, problem_size, TileShapeQK{}, blk_coord, 1);

    LoadK load_k{params.tma_load_k, pipeline, storage.smem_k};
    auto load_state_k = load_k.init_state(block_rank_in_cluster, problem_size, TileShapeQK{}, blk_coord, fusion_tile_count);

    LoadV load_v{params.tma_load_v, pipeline, storage.smem_v};
    auto load_state_v = load_v.init_state(block_rank_in_cluster, problem_size, TileShapePV{}, blk_coord, fusion_tile_count);

    // Set predicate for the lowest lane_id in the warp
    int lane_predicate = cute::elect_one_sync();

    // Issue TmaLoads (Prologue fetches)
    if (warp_idx == 0) {
      auto q_tile_iter = cute::make_coord_iterator(1);
      int q_tile_count = 1;
      load_q.step(q_tile_iter, load_state_q, smem_pipe_write_q, lane_predicate, q_tile_count);
    }

    // Loop over K elems
    auto k_tile_iter = cute::make_coord_iterator(fusion_tile_count);

    int k_tile_count_tma = 2 * fusion_tile_count;

    uint16_t mcast_mask_b = 0;

    if (warp_idx == 0 && lane_predicate == 1) {
      // For NVFP4 block-scaled GEMM, we use standard TMA load
      // No multicast mask needed for this implementation

      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < StageCount; i++) {
        if (i % 2 == 0) {
          load_k.template step<false>(k_tile_iter, load_state_k, smem_pipe_write, lane_predicate, k_tile_count_tma, mcast_mask_b);
        } else {
          load_v.template step<true>(k_tile_iter, load_state_k, smem_pipe_write, lane_predicate, k_tile_count_tma, mcast_mask_b);
        }
      }
    }

    TiledMmaQK tiled_mma_qk;
    auto thr_mma_qk = tiled_mma_qk.get_thread_slice(thread_idx);
    
    // Mainloop setup QK
    Tensor sQ = make_tensor(make_smem_ptr(storage.smem_q.data()), SmemLayoutQ{});
    Tensor sK = make_tensor(make_smem_ptr(storage.smem_k.data()), SmemLayoutK{});

    Tensor tSsQ = thr_mma_qk.partition_A(sQ);                                   // (MMA,MMA_M,MMA_K,PIPE)
    Tensor tSsK = thr_mma_qk.partition_B(sK);                                   // (MMA,MMA_N,MMA_K,PIPE)
    Tensor tSrQ = thr_mma_qk.make_fragment_A(tSsQ);                            // (MMA,MMA_N,MMA_K,PIPE)
    Tensor tSrK = thr_mma_qk.make_fragment_B(tSsK);                            // (MMA,MMA_M,MMA_N,PIPE)
    
    // Prepare: MMA PV
    TiledMmaPV tiled_mma_pv;
    auto thr_mma_pv = tiled_mma_pv.get_thread_slice(thread_idx);
    
    // Mainloop setup PV
    Tensor sV = make_tensor(make_smem_ptr(storage.smem_v.data()), SmemLayoutV{});

    Tensor tOsV = thr_mma_pv.partition_B(sV);                                   // (MMA,MMA_N,MMA_K,PIPE)
    Tensor tOrV = thr_mma_pv.make_fragment_B(tOsV);                            // (MMA,MMA_M,MMA_N,PIPE)
  
    int k_tile_count = Fusion{}.get_unmasked_trip_count(blk_coord, TileShape{}, problem_size);

    pipeline_q.consumer_wait(smem_pipe_read_q);
 
    // mapping into QK accumulator
    Tensor cP = make_identity_tensor(take<0,2>(TileShapeQK{}));
    Tensor tPcP = thr_mma_qk.partition_C(cP);
    int m_block = get<0>(blk_coord);
    tPcP.data() = tPcP.data() + E<0>{} * m_block * get<0>(TileShapeQK{});
  
    // Allocate PV acc
    Tensor acc_pv = partition_fragment_C(tiled_mma_pv, take<0, 2>(TileShapePV{}));

    cutlass::fmha::collective::CollectiveSoftmax<ElementAccumulator, Fusion, decltype(params)> softmax{params};
    auto softmax_state = softmax.init(acc_pv, tiled_mma_pv);

    if (true)
    {
        --k_tile_count;
        // Allocate QK acc
        Tensor acc_qk = partition_fragment_C(tiled_mma_qk, take<0, 2>(TileShapeQK{}));
  
        pipeline.consumer_wait(smem_pipe_read);

        // MMA QK
        warpgroup_fence_operand(acc_qk);
        warpgroup_arrive();
  
        gemm_zero_acc(tiled_mma_qk, tSrQ(_,_,_,_0{}), tSrK(_,_,_,smem_pipe_read.index()), acc_qk);
        warpgroup_commit_batch();

        ++smem_pipe_read;
  
        // Wait for the pipeline MMAs to drain
        warpgroup_wait<0>();
        warpgroup_fence_operand(acc_qk);

        softmax.step(acc_qk, tiled_mma_qk, tPcP, softmax_state, problem_size);
  
        Tensor acc_qk_fixed = make_fragment_like<Element>(convert_c_layout_to_a_layout(acc_qk.layout(), shape<1>(typename decltype(tiled_mma_pv)::LayoutA_TV{})));
  
        Tensor acc_qk_input = make_tensor(acc_qk_fixed.data(), acc_qk.layout());
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < size(acc_qk); i++) {
            acc_qk_input(i) = static_cast<Element>(acc_qk(i));
        }
  
        pipeline.consumer_wait(smem_pipe_read);

        // MMA PV
        warpgroup_fence_operand(acc_pv);
        warpgroup_fence_operand(acc_qk_fixed);
       warpgroup_arrive();
  
        gemm_zero_acc(tiled_mma_pv, acc_qk_fixed, tOrV(_,_,_,smem_pipe_read.index()), acc_pv);
        warpgroup_commit_batch();
  
        //
        // Advance the pipe
        //

        // Advance consumer pipeline
        ++smem_pipe_read;

        pipeline.consumer_release(smem_pipe_release);
        ++smem_pipe_release;

        tPcP.data() = tPcP.data() + E<1>{} * get<1>(TileShapeQK{});
    }
  
    CUTLASS_PRAGMA_NO_UNROLL
    for ( ; k_tile_count > 0; --k_tile_count)
    {
        // Allocate QK acc
        Tensor acc_qk = partition_fragment_C(tiled_mma_qk, take<0, 2>(TileShapeQK{}));
  
        pipeline.consumer_wait(smem_pipe_read);

        // MMA QK
        warpgroup_fence_operand(acc_qk);
        warpgroup_arrive();

        gemm_zero_acc(tiled_mma_qk, tSrQ(_,_,_,_0{}), tSrK(_,_,_,smem_pipe_read.index()), acc_qk);
        warpgroup_commit_batch();

        ++smem_pipe_read;

        if (warp_idx == 0) {
          load_k.template step<false>(k_tile_iter, load_state_k, smem_pipe_write, lane_predicate, k_tile_count_tma, mcast_mask_b);
        }
  
        // Wait for the pipeline MMAs to drain
        warpgroup_wait<0>();
        warpgroup_fence_operand(acc_qk);
        warpgroup_fence_operand(acc_pv);

        softmax.template step_interleave_begin<false>(acc_qk, tiled_mma_qk, tPcP, softmax_state, acc_pv, tiled_mma_pv, problem_size);

        pipeline.consumer_release(smem_pipe_release);

        ++smem_pipe_release;

        pipeline.consumer_wait(smem_pipe_read);

        // MMA PV  
        auto layout_qk_input = convert_c_layout_to_a_layout(acc_qk.layout(), shape<1>(typename decltype(tiled_mma_pv)::LayoutA_TV{}));
  
        Tensor acc_qk_input = make_tensor(acc_qk.data(), layout_qk_input);
  
        static_assert(decltype(size<1>(layout_qk_input) == _1{})::value);
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < size<2>(tOrV); i++) {
          Tensor acc_qk_element = make_fragment_like<Element>(layout_qk_input(_, _0{}, _0{}));
          Tensor acc_qk_element_mk = tensor_op_mk_v(tiled_mma_pv, acc_qk_element);
          Tensor acc_qk_input_mk = tensor_op_mk_v(tiled_mma_pv, acc_qk_input(_, _0{}, i));
          softmax.step_interleave_step(acc_qk_input_mk, softmax_state);
          CUTLASS_PRAGMA_UNROLL
          for (int j = 0; j < size(acc_qk_element_mk); j++) {
            acc_qk_element_mk(j) = static_cast<Element>(acc_qk_input_mk(j));
          }
          warpgroup_arrive();
          CUTLASS_PRAGMA_UNROLL
          for (int j = 0; j < size<1>(tOrV); j++) {
            cute::gemm(tiled_mma_pv, acc_qk_element, tOrV(_,j,i,smem_pipe_read.index()), acc_pv(_,_0{},j));
          }
        }
        warpgroup_commit_batch();
  
        // Wait for the pipeline MMAs to drain
        pipeline.consumer_release(smem_pipe_release);
        ++smem_pipe_release;

        ++smem_pipe_read;

        if (warp_idx == 0) {
          load_v.template step<true>(k_tile_iter, load_state_v, smem_pipe_write, lane_predicate, k_tile_count_tma, mcast_mask_b);
        }

        tPcP.data() = tPcP.data() + E<1>{} * get<1>(TileShapeQK{});
    }

    k_tile_count += Fusion{}.get_masked_trip_count(blk_coord, TileShape{}, problem_size);

    CUTLASS_PRAGMA_NO_UNROLL
    for ( ; k_tile_count > 0; --k_tile_count)
    {
        // Allocate QK acc
        Tensor acc_qk = partition_fragment_C(tiled_mma_qk, take<0, 2>(TileShapeQK{}));
  
        pipeline.consumer_wait(smem_pipe_read);

        // MMA QK
        warpgroup_fence_operand(acc_qk);
        warpgroup_arrive();

        gemm_zero_acc(tiled_mma_qk, tSrQ(_,_,_,_0{}), tSrK(_,_,_,smem_pipe_read.index()), acc_qk);
        warpgroup_commit_batch();

        ++smem_pipe_read;

        if (warp_idx == 0) {
          load_k.template step<false>(k_tile_iter, load_state_k, smem_pipe_write, lane_predicate, k_tile_count_tma, mcast_mask_b);
        }
  
        // Wait for the pipeline MMAs to drain
        warpgroup_wait<0>();
        warpgroup_fence_operand(acc_qk);
        warpgroup_fence_operand(acc_pv);

        softmax.step_interleave_begin(acc_qk, tiled_mma_qk, tPcP, softmax_state, acc_pv, tiled_mma_pv, problem_size);

        pipeline.consumer_release(smem_pipe_release);

        ++smem_pipe_release;

        pipeline.consumer_wait(smem_pipe_read);

        // MMA PV  
        auto layout_qk_input = convert_c_layout_to_a_layout(acc_qk.layout(), shape<1>(typename decltype(tiled_mma_pv)::LayoutA_TV{}));
  
        Tensor acc_qk_input = make_tensor(acc_qk.data(), layout_qk_input);
  
        static_assert(decltype(size<1>(layout_qk_input) == _1{})::value);
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < size<2>(tOrV); i++) {
          Tensor acc_qk_element = make_fragment_like<Element>(layout_qk_input(_, _0{}, _0{}));
          Tensor acc_qk_element_mk = tensor_op_mk_v(tiled_mma_pv, acc_qk_element);
          Tensor acc_qk_input_mk = tensor_op_mk_v(tiled_mma_pv, acc_qk_input(_, _0{}, i));
          softmax.step_interleave_step(acc_qk_input_mk, softmax_state);
          CUTLASS_PRAGMA_UNROLL
          for (int j = 0; j < size(acc_qk_element_mk); j++) {
            acc_qk_element_mk(j) = static_cast<Element>(acc_qk_input_mk(j));
          }
          warpgroup_arrive();
          CUTLASS_PRAGMA_UNROLL
          for (int j = 0; j < size<1>(tOrV); j++) {
            cute::gemm(tiled_mma_pv, acc_qk_element, tOrV(_,j,i,smem_pipe_read.index()), acc_pv(_,_0{},j));
          }
        }
        warpgroup_commit_batch();
  
        // Wait for the pipeline MMAs to drain
        pipeline.consumer_release(smem_pipe_release);
        ++smem_pipe_release;

        ++smem_pipe_read;

        if (warp_idx == 0) {
          load_v.template step<true>(k_tile_iter, load_state_v, smem_pipe_write, lane_predicate, k_tile_count_tma, mcast_mask_b);
        }

        tPcP.data() = tPcP.data() + E<1>{} * get<1>(TileShapeQK{});
    }

    // Wait for the pipeline MMAs to drain
    warpgroup_wait<0>();
    warpgroup_fence_operand(acc_pv);

    Tensor lse = softmax.tail(softmax_state, acc_pv, tiled_mma_pv);

    return make_tuple(acc_pv, lse);
  }
};

}  // namespace cutlass::fmha::collective

