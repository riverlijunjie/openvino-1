// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/common.cl"

#define VLOAD CAT(vload, VEC_BLK_SIZE)
#define VSTORE CAT(vstore, VEC_BLK_SIZE)

KERNEL(moe_gather_ref)(
    OPTIONAL_SHAPE_INFO_ARG
    const __global INPUT0_TYPE* input,
    const __global INPUT1_TYPE* token_indices,
    __global OUTPUT_TYPE* output
#ifdef OUTPUT_ROUTING_WEIGHTS
    ,const __global INPUT2_TYPE* routing_weights
    ,__global OUTPUT2_TYPE* sorted_routing_weights
    ,const __global INPUT3_TYPE* topk_ids
    ,const __global INPUT4_TYPE* expert_ids
    ,const __global INPUT5_TYPE* start_offsets
#ifdef SET_ACTUAL_USED_EXPERTS_NUM
    ,const __global INPUT4_TYPE* used_expert_num
#endif
#endif
)
{
    const uint token_group_id = (uint)get_group_id(0);
    const uint threads_index = (uint)get_local_id(0);

#if UNALIGNED_ELEMENTS > 0
    if ((threads_index == get_local_size(0) - 1) && (UNALIGNED_ELEMENTS > 0)) {
        for (uint i = 0; i < UNALIGNED_ELEMENTS; i++) {
            const INPUT1_TYPE token_index = token_indices[token_group_id] * HIDDEN_SIZE;

            const uint dest_index = token_group_id * HIDDEN_SIZE;

            const uint input_pos = token_index + threads_index * VEC_BLK_SIZE * BATCHES_PER_THREAD + i;
            const uint output_pos = dest_index + threads_index * VEC_BLK_SIZE * BATCHES_PER_THREAD + i;
            output[output_pos] = input[input_pos];
        }
    } else {
#endif
        for (uint i = 0; i < BATCHES_PER_THREAD; i++) {
            const INPUT1_TYPE token_index = token_indices[token_group_id] * HIDDEN_SIZE  +  i * VEC_BLK_SIZE;

            const uint dest_index = token_group_id * HIDDEN_SIZE + i * VEC_BLK_SIZE;

            const uint input_pos = token_index + threads_index * VEC_BLK_SIZE * BATCHES_PER_THREAD;
            const uint output_pos = dest_index + threads_index * VEC_BLK_SIZE * BATCHES_PER_THREAD;
            VSTORE(VLOAD(0, &input[input_pos]), 0, &output[output_pos]);
        }
#if UNALIGNED_ELEMENTS > 0
    }
#endif

#ifdef OUTPUT_ROUTING_WEIGHTS
    if (threads_index == 0) {
        int token_id = token_indices[token_group_id];
        // find out which k-th expert it is
        int expert_id = 0;
#ifdef SET_ACTUAL_USED_EXPERTS_NUM
        int actual_used_expert_num = used_expert_num[0];
        int last_expert_idx = 0;
        for (int i = 0; i < actual_used_expert_num; i++) {
             int start_offset = start_offsets[i];
             // we assume that the start_offset is sorted asc
             if (token_group_id >= start_offset) {
                 // check next
                 if (i == actual_used_expert_num - 1 || token_group_id < start_offsets[i+1]) {
                     expert_id = expert_ids[i];
                     break;
                 }
             }
        }
        
        // search in topk_ids to find the weight
        // topk_ids: [token_num, TOP_K]
        // routing_weights: [token_num, TOP_K]
        // sorted_routing_weights: [total_tokens]
        const __global INPUT3_TYPE* current_token_topk = topk_ids + token_id * TOP_K;
        const __global INPUT2_TYPE* current_token_weights = routing_weights + token_id * TOP_K;
        
        OUTPUT2_TYPE w = 0;
        for (int k = 0; k < TOP_K; k++) {
            if (current_token_topk[k] == expert_id) {
                w = current_token_weights[k];
                break;
            }
        }
        
        sorted_routing_weights[token_group_id] = w;
#endif
    }
#endif
}
