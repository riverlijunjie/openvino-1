// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// GatedResidual UP_PROJ on a oneDNN micro-GEMM:
//   mixed[t, d] = (1 / C) * sum_c sigmoid(sum_l T[t, l] * W_up[c * D + d, l]) * Rbar[t, c * D + d]
// Every work-group owns a (d, token) tile of the output and runs one micro-GEMM per branch on it (A = the W_up rows of the branch,
// B = T). The sigmoid, the product with Rbar and the sum over the branches are applied to the accumulator tile in registers,
// so neither the read gate nor the per-branch partial mixes ever reach memory.
// W_up is f16, or (GR_UPM_Q) i4 / u4 / i8 / u8 with f16 scales [G, C*D] and, with GR_UPM_ZP, zero points of the weight type, which the
// micro-GEMM applies while it loads A.
// JIT: GR_BRANCHES, GR_HIDDEN, GR_LOWRANK, SUBGROUP_SIZE, USE_SLM (the micro-GEMM needs SLM), GR_UPM_BLOCK_RB (D is a multiple of the
// subgroup tile: Rbar is read and the result written in blocks of whole tile rows), GR_UPM_Q, GR_UPM_ZP.
// gws = [ceil(D / wg_tile_m) * sg_per_wg_m * SUBGROUP_SIZE, ceil(tokens / wg_tile_n) * sg_per_wg_n, 1], lws = [sg_per_wg_m * SUBGROUP_SIZE, sg_per_wg_n, 1].

#include "include/batch_headers/common.cl"
#include "include/batch_headers/generic_vector_ops.cl"
#include "include/batch_headers/tile_ops.cl"
#define DECORATOR grup
#include "expert_gemm_common.cl"

#define GR_UPM_CD (GR_BRANCHES * GR_HIDDEN)
#define GR_UPM_INV_C (1.0f / (float)GR_BRANCHES)
#define GR_UPM_LOG2E 1.4426950408889634f
#define GR_UPM_SIGMOID(x) native_recip(1.0f + native_exp2(-(x) * GR_UPM_LOG2E))

#if GR_UPM_Q
#    define GR_UPM_WT uchar
#    if GR_UPM_ZP
#        define GR_UPM_Q_ARGS , const global half* w_scale, const global uchar* w_zp
#        define GR_UPM_Q_PARAMS , w_scale, w_zp, GR_UPM_CD
#    else
#        define GR_UPM_Q_ARGS , const global half* w_scale
#        define GR_UPM_Q_PARAMS , w_scale, GR_UPM_CD
#    endif
#else
#    define GR_UPM_WT half
#    define GR_UPM_Q_ARGS
#    define GR_UPM_Q_PARAMS
#endif

#if GR_UPM_BLOCK_RB
// Rbar / mixed tiles of the subgroup tile, one row of UGEMM_SG_TILE_M contiguous halves per token.
DECLARE_2D_TILE(gr_rb_tile_type, half, SUBGROUP_SIZE, UGEMM_SG_TILE_M, 1, 1, UGEMM_SG_TILE_N)
DECLARE_2D_TILE_BLOCK_OPS(gr_rb_tile_type, half, SUBGROUP_SIZE, UGEMM_SG_TILE_M, 1, 1, UGEMM_SG_TILE_N)
#    define GR_UPM_RB(t, i0, j) tile_access(t, i0, j, SUBGROUP_SIZE, UGEMM_SG_TILE_M, 1, 1)
#endif
#define GR_UPM_ACC(t, i0, j) tile_access(t, i0, j, SUBGROUP_SIZE, UGEMM_C_TYPE_BLOCK0, UGEMM_C_TYPE_BLOCK1, UGEMM_C_TYPE_NBLOCK0)

__attribute__((intel_reqd_sub_group_size(SUBGROUP_SIZE))) KERNEL(gr_up_micro)(const global half* t_in,
                                                                             const global GR_UPM_WT* w_up GR_UPM_Q_ARGS,
                                                                             const global half* rbar,
                                                                             global half* mixed,
                                                                             uint num_tokens
#ifdef USE_SLM
                                                                             ,
                                                                             local int* slm
#endif
) {
    const uint sg_i = sub_group_broadcast(get_local_id(0) / SUBGROUP_SIZE, 0);
    const uint sg_j = sub_group_broadcast(get_local_id(1), 0);

    const uint wg_i0 = get_group_id(0) * UGEMM_WG_TILE_M;
    const uint wg_j0 = get_group_id(1) * UGEMM_WG_TILE_N;
    const uint sg_i0 = wg_i0 + sg_i * UGEMM_SG_TILE_M;
    const uint sg_j0 = wg_j0 + sg_j * UGEMM_SG_TILE_N;

    if (wg_j0 >= num_tokens) {
        return;
    }

    UGEMM_C_TYPE mix;
    tile_fill(mix, 0.0f);

#if GR_UPM_BLOCK_RB
    const bool in_range = sg_i0 < GR_HIDDEN;
#else
    const int sglid = get_sub_group_local_id();
    const int br = UGEMM_C_TYPE_BLOCK0;
    const int nbr = UGEMM_C_TYPE_NBLOCK0;
    const int bc = UGEMM_C_TYPE_BLOCK1;
    const int nbc = UGEMM_C_TYPE_NBLOCK1;
    const int sg = SUBGROUP_SIZE;
#endif

    __attribute__((opencl_unroll_hint(1))) for (uint c = 0; c < GR_BRANCHES; c++) {
        // the branches are row blocks of one [C*D, L] matrix: the micro-GEMM addresses them by row offset
#ifdef USE_SLM
        UGEMM_C_TYPE acc = UG()(w_up, GR_LOWRANK, t_in, GR_LOWRANK, GR_UPM_CD, num_tokens, GR_LOWRANK, c * GR_HIDDEN + wg_i0, wg_j0, 0, sg_i, sg_j, slm GR_UPM_Q_PARAMS);
#else
        UGEMM_C_TYPE acc = UG()(w_up, GR_LOWRANK, t_in, GR_LOWRANK, GR_UPM_CD, num_tokens, GR_LOWRANK, c * GR_HIDDEN + wg_i0, wg_j0, 0, sg_i, sg_j, 0 GR_UPM_Q_PARAMS);
#endif
#if GR_UPM_BLOCK_RB
        if (in_range) {
            gr_rb_tile_type rb;
            tile_load_block(&rb, rbar + c * GR_HIDDEN, num_tokens, GR_UPM_CD, sg_i0, sg_j0);
            unroll_for (int j = 0; j < UGEMM_SG_TILE_N; j++) {
                unroll_for (int i0 = 0; i0 < UGEMM_SG_TILE_M; i0 += SUBGROUP_SIZE) {
                    GR_UPM_ACC(mix, i0, j) += GR_UPM_SIGMOID(GR_UPM_ACC(acc, i0, j)) * convert_float(GR_UPM_RB(rb, i0, j));
                }
            }
        }
#else
        unroll_for (int j = 0; j < bc * nbc; j++) {
            if (sg_j0 + j < num_tokens) {
                const global half* rbar_row = rbar + (size_t)(sg_j0 + j) * GR_UPM_CD + c * GR_HIDDEN + sg_i0;
                unroll_for (int i0 = 0; i0 < br * nbr; i0 += sg) {
                    const int i = i0 + sglid;
                    if (sg_i0 + i < GR_HIDDEN) {
                        const int ri = i0 / br + nbr * (j / bc);
                        const int rj = (i0 % br) / sg + (j % bc) * (br / sg);
                        mix.x[ri][rj] += GR_UPM_SIGMOID(acc.x[ri][rj]) * convert_float(rbar_row[i]);
                    }
                }
            }
        }
#endif
    }

#if GR_UPM_BLOCK_RB
    if (in_range) {
        gr_rb_tile_type rb;
        unroll_for (int j = 0; j < UGEMM_SG_TILE_N; j++) {
            unroll_for (int i0 = 0; i0 < UGEMM_SG_TILE_M; i0 += SUBGROUP_SIZE) {
                GR_UPM_RB(rb, i0, j) = convert_half_rte(GR_UPM_ACC(mix, i0, j) * GR_UPM_INV_C);
            }
        }
        tile_store_block(rb, mixed, num_tokens, GR_HIDDEN, sg_i0, sg_j0);
    }
#else
    unroll_for (int j = 0; j < bc * nbc; j++) {
        if (sg_j0 + j < num_tokens) {
            global half* out_row = mixed + (size_t)(sg_j0 + j) * GR_HIDDEN + sg_i0;
            unroll_for (int i0 = 0; i0 < br * nbr; i0 += sg) {
                const int i = i0 + sglid;
                if (sg_i0 + i < GR_HIDDEN) {
                    const int ri = i0 / br + nbr * (j / bc);
                    const int rj = (i0 % br) / sg + (j % bc) * (br / sg);
                    out_row[i] = convert_half_rte(mix.x[ri][rj] * GR_UPM_INV_C);
                }
            }
        }
    }
#endif
}
