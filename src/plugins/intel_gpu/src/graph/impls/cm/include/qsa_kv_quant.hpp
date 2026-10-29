// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Quantized main-KV cache helpers shared by the SparsePA kernels. The page layout is documented in
// intel_gpu/op/sparse_kv_quant.hpp: per (block, kv head) page, `page_size` token-major data rows of
// QSA_X_ROW_BYTES bytes (4-bit values packed two per byte, low nibble = even channel) followed by the scale /
// zero-point parameters at QSA_X_PARAM_OFFSET.
//
// JIT constants per cache X in {K, V}: QSA_X_MODE (0 = f16, 1 = by token, 2 = by channel), QSA_X_BITS (8 or 4),
// QSA_X_SYM (biased symmetric codes, no zero point stored), QSA_X_GROUP (tokens per BY_CHANNEL group),
// QSA_X_ROW_BYTES, QSA_X_PAGE_BYTES, QSA_X_PARAM_OFFSET.
//
// A classic include guard, see qsa_gr_common.hpp for why `#pragma once` cannot be used.
#ifndef ATTR
#define ATTR [[type("svmptr_t")]]
#endif

#ifndef QSA_KV_QUANT_HPP
#define QSA_KV_QUANT_HPP

#include "qsa_common.hpp"

namespace qsa_kvq {

// Smallest scale kept, so that 1 / scale stays finite in f16 / f32.
constexpr float MIN_SCALE = 1.0f / 16384.0f;

CM_INLINE uint page_base(uint physical_page, uint kv_head, uint kv_heads, uint page_bytes) {
    return (physical_page * kv_heads + kv_head) * page_bytes;
}

/// Maps work item `wg` of a grid sized `tokens / group_size + 2 * num_seqs` onto the (sequence, token group) it
/// re-quantizes: sequence `i` owns the groups touched by its new tokens.
CM_INLINE bool
resolve_group_wg(const int32_t* subsequence_begins ATTR, const int32_t* past_lens ATTR, uint num_seqs, uint wg, uint group_size, uint& seq, uint& group) {
    uint acc = 0;
    for (uint i = 0; i < num_seqs; i++) {
        const uint q_len = (uint)subsequence_begins[i + 1] - (uint)subsequence_begins[i];
        if (q_len == 0) {
            continue;
        }
        const uint past = (uint)past_lens[i];
        const uint first = past / group_size;
        const uint n = (past + q_len - 1) / group_size - first + 1;
        if (wg < acc + n) {
            seq = i;
            group = first + (wg - acc);
            return true;
        }
        acc += n;
    }
    return false;
}

/// Loads N bytes (N = 4 * power of two <= 64 dwords, or a multiple of 256) at a 4-byte aligned byte offset.
template <int N>
CM_INLINE void load_bytes(vector_ref<uint8_t, N> dst, const uint8_t* p ATTR, uint byte_off) {
    constexpr int DW = N / 4;
    static_assert(N % 4 == 0 && ((DW & (DW - 1)) == 0 || DW % 64 == 0), "unsupported byte count");
    auto d32 = dst.template format<uint32_t>();
    if constexpr (DW <= 64) {
        d32 = cm_ptr_load<uint32_t, DW>((const uint32_t*)p, byte_off);
    } else {
#pragma unroll
        for (int i = 0; i < DW / 64; i++) {
            d32.template select<64, 1>(i * 64) = cm_ptr_load<uint32_t, 64>((const uint32_t*)p, byte_off + i * 256);
        }
    }
}

template <int N>
CM_INLINE void store_bytes(uint8_t* p ATTR, uint byte_off, vector_ref<uint8_t, N> src) {
    constexpr int DW = N / 4;
    static_assert(N % 4 == 0 && ((DW & (DW - 1)) == 0 || DW % 64 == 0), "unsupported byte count");
    auto s32 = src.template format<uint32_t>();
    if constexpr (DW <= 64) {
        cm_ptr_store<uint32_t, DW>((uint32_t*)p, byte_off, s32);
    } else {
#pragma unroll
        for (int i = 0; i < DW / 64; i++) {
            cm_ptr_store<uint32_t, 64>((uint32_t*)p, byte_off + i * 256, s32.template select<64, 1>(i * 64));
        }
    }
}

/// Unpacks a row of D codes (8-bit: one per byte; 4-bit: low nibble = even channel) into halves.
template <int D, int BITS>
CM_INLINE void unpack_codes(vector_ref<half, D> out, vector_ref<uint8_t, D * BITS / 8> raw) {
    if constexpr (BITS == 8) {
        out = raw;
    } else {
        vector<uint16_t, D / 2> wide = raw;
        vector<half, D / 2> lo = wide & 0xF;
        vector<half, D / 2> hi = wide >> 4;
        out.template select<D / 2, 2>(0) = lo;
        out.template select<D / 2, 2>(1) = hi;
    }
}

/// Number of bytes of a packed row of D codes.
template <int D, int BITS>
constexpr int row_bytes_of() {
    return D * BITS / 8;
}

template <int BITS>
constexpr float zero_code_sym() {
    return BITS == 8 ? 128.0f : 8.0f;
}

/// Token-parameter dword: scale in the low half, zero point in the high half.
CM_INLINE uint pack_token_params(half scale, half zp) {
    vector<half, 2> h;
    h[0] = scale;
    h[1] = zp;
    return h.format<uint32_t>()[0];
}

/// Quantizes one row of D halves per token. Returns the (f16-rounded) scale / zero point and the packed codes.
/// asym: x ~ (q - zp) * scale over a range extended to include 0; sym: biased codes around zero_code_sym().
template <int D, int BITS, int SYM>
CM_INLINE void quantize_row(vector_ref<half, D> row, vector_ref<uint8_t, D * BITS / 8> packed, half& scale_h, half& zp_h) {
    constexpr float QMAX = BITS == 8 ? 255.0f : 15.0f;
    vector<float, D> x = row;
    float mn = cm_reduced_min<float>(x);
    float mx = cm_reduced_max<float>(x);
    float scale = 1.0f;
    float zp = 0.0f;
    if constexpr (SYM) {
        const float amax = mx > -mn ? mx : -mn;
        scale = amax / (BITS == 8 ? 127.0f : 7.0f);
        zp = zero_code_sym<BITS>();
    } else {
        mn = mn < 0.0f ? mn : 0.0f;
        mx = mx > 0.0f ? mx : 0.0f;
        scale = (mx - mn) / QMAX;
        zp = scale > 0.0f ? -mn / scale : 0.0f;
    }
    scale = scale > MIN_SCALE ? scale : MIN_SCALE;
    if constexpr (!SYM) {
        zp = -mn / scale;
    }
    scale_h = (half)scale;
    zp_h = (half)zp;
    vector<float, 1> inv_v = (float)scale_h;
    inv_v = cm_inv(inv_v);
    const float inv = inv_v[0];
    const float zpf = (float)zp_h;

    vector<float, D> y = x * inv + zpf;
    y = cm_max<float>(y, 0.0f);
    y = cm_min<float>(y, QMAX);
    y += 0.5f;
    vector<uint8_t, D> q = y;
    if constexpr (BITS == 8) {
        packed = q;
    } else {
        vector<uint16_t, D / 2> lo = q.template select<D / 2, 2>(0);
        vector<uint16_t, D / 2> hi = q.template select<D / 2, 2>(1);
        packed = lo | (hi << 4);
    }
}

/// Dequantizes one token row in place: x = (q - zp) * scale.
template <int D>
CM_INLINE void dequant_token(vector_ref<half, D> row, half scale, half zp) {
    row = (row - zp) * scale;
}

/// Loads one BY_TOKEN row of D channels of `tok` (page-local) into halves.
template <int D, int BITS, int SYM, int ROW_BYTES, int PARAM_OFFSET>
CM_INLINE void load_row_token(vector_ref<half, D> out, const uint8_t* cache ATTR, uint page_base, uint tok) {
    vector<uint8_t, ROW_BYTES> raw;
    load_bytes<ROW_BYTES>(raw, cache, page_base + tok * ROW_BYTES);
    unpack_codes<D, BITS>(out, raw);
    vector<uint32_t, 1> pd = cm_ptr_load<uint32_t, 1>((const uint32_t*)cache, page_base + PARAM_OFFSET + tok * 4);
    vector<half, 2> ph = pd.template format<half>();
    half zp = ph[1];
    if constexpr (SYM) {
        zp = (half)zero_code_sym<BITS>();
    }
    out = (out - zp) * ph[0];
}

/// Loads the per-channel scale / zero point vectors of the BY_CHANNEL group record at `rec` (byte offset).
template <int D, int BITS, int SYM>
CM_INLINE void load_group_params(vector_ref<half, D> scale, vector_ref<half, D> zp, const uint8_t* cache ATTR, uint rec) {
    qsa_gr::load_hv<D>(scale, (const half*)(cache + rec), 0);
    if constexpr (SYM) {
        zp = (half)zero_code_sym<BITS>();
    } else {
        qsa_gr::load_hv<D>(zp, (const half*)(cache + rec + D * 2), 0);
    }
}

/// Loads one BY_CHANNEL row of D channels of `tok` (page-local) into halves.
template <int D, int BITS, int SYM, int ROW_BYTES, int PARAM_OFFSET, int GROUP>
CM_INLINE void load_row_channel(vector_ref<half, D> out, const uint8_t* cache ATTR, uint page_base, uint tok) {
    vector<uint8_t, ROW_BYTES> raw;
    load_bytes<ROW_BYTES>(raw, cache, page_base + tok * ROW_BYTES);
    unpack_codes<D, BITS>(out, raw);
    vector<half, D> scale, zp;
    load_group_params<D, BITS, SYM>(scale, zp, cache, page_base + PARAM_OFFSET + (tok / GROUP) * (D * (SYM ? 2 : 4)));
    out = (out - zp) * scale;
}

#ifdef CM_HAS_LSC_UNTYPED_2D

/// Loads `rows` token rows (<= ROWS, starting at page-local `tok0`) x C channels (starting at channel `col0`) of a
/// quantized page with one 2D block message and dequantizes them to halves; rows >= `rows` are zero.
/// MODE 1 = BY_TOKEN parameters (one per row), MODE 2 = BY_CHANNEL (all rows must lie in one GROUP).
/// D is the full head size, which sets the parameter stride of a BY_CHANNEL group record.
template <int ROWS, int C, int BITS, int SYM, int ROW_BYTES, int PARAM_OFFSET, int MODE, int GROUP, int D>
CM_INLINE void load_tile(matrix_ref<half, ROWS, C> out, const uint8_t* cache ATTR, uint page_base, uint tok0, uint rows, uint col0) {
    constexpr int CB = C * BITS / 8;
    matrix<uint8_t, ROWS, CB> raw;
    lsc::block_2d_desc<uint8_t, 1, ROWS, CB> desc(const_cast<uint8_t*>(cache) + page_base + tok0 * ROW_BYTES, rows - 1, ROW_BYTES - 1, ROW_BYTES - 1, 0, 0);
    cm_load<lsc::Normal>(raw.template format<uint8_t>(), desc.set_block_x(col0 * BITS / 8));
    if constexpr (BITS == 8) {
        out = raw;
    } else {
        matrix<uint16_t, ROWS, CB> wide = raw;
        matrix<half, ROWS, CB> lo = wide & 0xF;
        matrix<half, ROWS, CB> hi = wide >> 4;
        out.template select<ROWS, 1, CB, 2>(0, 0) = lo;
        out.template select<ROWS, 1, CB, 2>(0, 1) = hi;
    }
    if constexpr (MODE == 1) {
        vector<uint32_t, ROWS> pd = cm_ptr_load<uint32_t, ROWS>((const uint32_t*)cache, page_base + PARAM_OFFSET + tok0 * 4);
        vector<half, 2 * ROWS> ph = pd.template format<half>();
#    pragma unroll
        for (int r = 0; r < ROWS; r++) {
            half zp = ph[2 * r + 1];
            if constexpr (SYM) {
                zp = (half)zero_code_sym<BITS>();
            }
            out.row(r) = (out.row(r) - zp) * ph[2 * r];
        }
    } else {
        const uint rec = page_base + PARAM_OFFSET + (tok0 / GROUP) * (D * (SYM ? 2 : 4));
        vector<half, C> scale, zp;
        qsa_gr::load_hv<C>(scale, (const half*)(cache + rec), col0);
        if constexpr (SYM) {
            zp = (half)zero_code_sym<BITS>();
        } else {
            qsa_gr::load_hv<C>(zp, (const half*)(cache + rec + D * 2), col0);
        }
#    pragma unroll
        for (int r = 0; r < ROWS; r++) {
            out.row(r) = (out.row(r) - zp) * scale;
        }
    }
#    pragma unroll
    for (int r = 0; r < ROWS; r++) {
        if (r >= (int)rows) {
            out.row(r) = 0.0f;
        }
    }
}

#endif  // CM_HAS_LSC_UNTYPED_2D

// ---------------------------------------------------------------------------
// Dense (SparseSDPA) compressed K/V: KVCacheCompressed output = i8 data [B, Hkv, S, D] + planar per-token
// f16 scale and i8 zero point [B, Hkv, S, 1]; x = (q - zp) * scale. `row` below is the flat token row index
// ((b * Hkv + h) * pitch + pos) in the data tensor (`D` bytes per row) and in the parameter tensors.
// ---------------------------------------------------------------------------

/// Element `e` of a half array without assuming 4-byte alignment of `e`.
CM_INLINE half load_half_elem(const half* p ATTR, uint e) {
    vector<uint32_t, 1> d = cm_ptr_load<uint32_t, 1>((const uint32_t*)p, (e >> 1) * 4);
    const uint w = d[0];
    vector<uint16_t, 1> u = (e & 1) ? (w >> 16) : (w & 0xFFFF);
    return u.template format<half>()[0];
}

/// Element `e` of an int8 array as half.
CM_INLINE half load_i8_elem(const int8_t* p ATTR, uint e) {
    vector<uint32_t, 1> d = cm_ptr_load<uint32_t, 1>((const uint32_t*)p, (e >> 2) * 4);
    const uint w = d[0];
    int v = (int)((w >> ((e & 3) * 8)) & 0xFF);
    if (v >= 128) {
        v -= 256;
    }
    return (half)(float)v;
}

/// Loads and dequantizes one dense token row of D channels. `row` indexes the data tensor, `prow` the scale / zero-point
/// tensors (their sequence padding is allocated independently of the data's).
template <int D>
CM_INLINE void load_row_dense(vector_ref<half, D> out, const uint8_t* data ATTR, const half* scale ATTR, const int8_t* zp ATTR, uint row, uint prow) {
    vector<uint8_t, D> raw;
    load_bytes<D>(raw, data, row * D);
    vector<int8_t, D> sraw = raw.template format<int8_t>();
    out = sraw;
    out = (out - load_i8_elem(zp, prow)) * load_half_elem(scale, prow);
}

#ifdef CM_HAS_LSC_UNTYPED_2D

/// Dense counterpart of load_tile: `rows` token rows from flat data row `row0` (flat parameter row `prow0`), channels [col0, col0 + C).
template <int ROWS, int C, int D>
CM_INLINE void load_tile_dense(matrix_ref<half, ROWS, C> out,
                               const uint8_t* data ATTR,
                               const half* scale ATTR,
                               const int8_t* zp ATTR,
                               uint row0,
                               uint prow0,
                               uint rows,
                               uint col0) {
    matrix<uint8_t, ROWS, C> raw;
    lsc::block_2d_desc<uint8_t, 1, ROWS, C> desc(const_cast<uint8_t*>(data) + row0 * D, rows - 1, D - 1, D - 1, 0, 0);
    cm_load<lsc::Normal>(raw.template format<uint8_t>(), desc.set_block_x(col0));
    matrix<int8_t, ROWS, C> sraw = raw.template format<int8_t, ROWS, C>();
    out = sraw;
#    pragma unroll
    for (int r = 0; r < ROWS; r++) {
        if (r < (int)rows) {
            out.row(r) = (out.row(r) - load_i8_elem(zp, prow0 + r)) * load_half_elem(scale, prow0 + r);
        } else {
            out.row(r) = 0.0f;
        }
    }
}

#endif  // CM_HAS_LSC_UNTYPED_2D

}  // namespace qsa_kvq

#endif  // QSA_KV_QUANT_HPP
