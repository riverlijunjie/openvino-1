// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/common.cl"

#define VLOAD CAT(vload, VEC_BLK_SIZE)
#define VSTORE CAT(vstore, VEC_BLK_SIZE)
#define INPUT_VEC_TYPE  MAKE_VECTOR_TYPE(INPUT0_TYPE, VEC_BLK_SIZE)
#define OUTPUT_VEC_TYPE MAKE_VECTOR_TYPE(OUTPUT_TYPE, VEC_BLK_SIZE)

#ifndef FUSE_MUL_WG
#define ACC_TYPE float
#else
#define ACC_TYPE OUTPUT_TYPE
#endif

#define ACC_VEC_TYPE MAKE_VECTOR_TYPE(ACC_TYPE, VEC_BLK_SIZE)
#define TO_ACC_VEC_TYPE CAT(convert_, ACC_VEC_TYPE)
#define TO_OUTPUT_VEC_TYPE CAT(convert_, OUTPUT_VEC_TYPE)

KERNEL(moe_scatter_reduction_ref)(
    OPTIONAL_SHAPE_INFO_ARG
    const __global INPUT0_TYPE* input,
    const __global INPUT1_TYPE* experts_per_token,
    const __global INPUT2_TYPE* expert_weights,
    const __global INPUT3_TYPE* tokens_per_expert,
    const __global INPUT4_TYPE* experts_start_offset,
    const __global INPUT5_TYPE* tokens_len_per_expert,
    const __global INPUT6_TYPE* experts_ids,
#ifdef SET_ACTUAL_USED_EXPERTS_NUM
    const __global INPUT6_TYPE* used_expert_num,
#endif
    __global OUTPUT_TYPE* output
)
{
    const uint token_group_id = (uint)get_group_id(0);
    const uint threads_index = (uint)get_local_id(0);

    ACC_VEC_TYPE output_vec[BATCHES_PER_THREAD];
    // start_offset_idx[i] = n : info for i-th expert in this thread is in the nth slot of the mask
    __local uint start_offset_index[ACTIVE_EXPERTS];
    __local uint expert_input_offsets[ACTIVE_EXPERTS];

    // Initialize start_offset_index to an invalid sentinel
    if (threads_index < ACTIVE_EXPERTS) {
        start_offset_index[threads_index] = (uint)UINT_MAX;
        expert_input_offsets[threads_index] = (uint)UINT_MAX;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (threads_index < ACTIVE_EXPERTS) {
        INPUT1_TYPE expert_id = experts_per_token[token_group_id * ACTIVE_EXPERTS  + threads_index];
#ifdef SET_ACTUAL_USED_EXPERTS_NUM
        int actual_used_expert_num = used_expert_num[0];
        for (int i = 0; i < actual_used_expert_num; i++) {
#else
        for (int i = 0; i < INPUT6_BATCH_NUM; i++) {
#endif
            if (experts_ids[i] == expert_id) {
                start_offset_index[threads_index] = i;
                break;
            }
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    // Search for input offsets
    for (uint i = 0; i < ACTIVE_EXPERTS; i++) {
        if (start_offset_index[i] == (uint)UINT_MAX)
            continue;

        INPUT5_TYPE token_len = tokens_len_per_expert[start_offset_index[i]];
        INPUT4_TYPE expert_offset = experts_start_offset[start_offset_index[i]];

        // Hybrid search: use single thread for short sequences to benefit from early exit,
        // and parallel search for long sequences to utilize memory bandwidth.
        if (token_len < 256) {
            if (threads_index == 0) {
                for (uint tid = 0; tid < token_len; tid++) {
                    if (tokens_per_expert[expert_offset + tid] == token_group_id) {
                        expert_input_offsets[i] = expert_offset + tid;
                        break;
                    }
                }
            }
        } else {
            for (uint tid = threads_index; tid < token_len; tid += get_local_size(0)) {
                if (tokens_per_expert[expert_offset + tid] == token_group_id) {
                    expert_input_offsets[i] = expert_offset + tid;
                }
            }
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    uint dest_index = token_group_id * HIDDEN_SIZE;
    uint output_pos = dest_index + threads_index * VEC_BLK_SIZE * BATCHES_PER_THREAD;

    for (uint i = 0; i < BATCHES_PER_THREAD; i++) {
        output_vec[i] = (ACC_VEC_TYPE)(0.0f);
    }

    // Sort the processing order to match the reference implementation (Ascending Expert ID)
    // start_offset_index maps the local TopK slot to the index in the global sorted Active Expert list.
    // Iterating in ascending order of start_offset_index ensures we accumulate Expanded Expert results in Expert ID order.
    uint private_start_indices[ACTIVE_EXPERTS];
    for (uint i = 0; i < ACTIVE_EXPERTS; i++) {
        private_start_indices[i] = start_offset_index[i];
    }

    for (uint step = 0; step < ACTIVE_EXPERTS; step++) {
        // Find the next expert in the sorted order
        uint min_val = (uint)UINT_MAX;
        uint min_idx = (uint)UINT_MAX;

        for (uint i = 0; i < ACTIVE_EXPERTS; i++) {
            if (private_start_indices[i] < min_val) {
                min_val = private_start_indices[i];
                min_idx = i;
            }
        }

        if (min_idx == (uint)UINT_MAX)
            break;

        // Mark as processed
        private_start_indices[min_idx] = (uint)UINT_MAX;
        uint i = min_idx;

        uint input_offset = expert_input_offsets[i];

        // If no matching token was found, skip accumulation for this expert
        if (input_offset == (uint)UINT_MAX)
            continue;

#ifndef FUSE_MUL_WG
        INPUT2_TYPE expert_weight = expert_weights[token_group_id * ACTIVE_EXPERTS  + i];
#endif

        for (uint j = 0; j < BATCHES_PER_THREAD; j++) {
            const uint input_pos = input_offset * HIDDEN_SIZE + j * VEC_BLK_SIZE + threads_index * VEC_BLK_SIZE * BATCHES_PER_THREAD;
            INPUT_VEC_TYPE input_data = VLOAD(0, &input[input_pos]);
            ACC_VEC_TYPE input_acc = TO_ACC_VEC_TYPE(input_data);
#ifndef FUSE_MUL_WG
            input_acc *= (ACC_TYPE)expert_weight;
#endif
            output_vec[j] += input_acc;
        }
    }

    for (uint v = 0; v < BATCHES_PER_THREAD; v++) {
        const uint out_pos = output_pos + v * VEC_BLK_SIZE;
        VSTORE(TO_OUTPUT_VEC_TYPE(output_vec[v]), 0, &output[out_pos]);
    }
}
