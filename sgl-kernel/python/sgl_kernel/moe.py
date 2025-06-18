from typing import Any, Dict, Optional

import torch
from math import prod
from typing import List, Optional, Tuple

def moe_align_block_size(
    topk_ids,
    num_experts,
    block_size,
    sorted_token_ids,
    experts_ids,
    num_tokens_post_pad,
    token_cnts_buffer,
    cumsum_buffer,
):
    torch.ops.sgl_kernel.moe_align_block_size.default(
        topk_ids,
        num_experts,
        block_size,
        sorted_token_ids,
        experts_ids,
        num_tokens_post_pad,
        token_cnts_buffer,
        cumsum_buffer,
    )


def topk_softmax(
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    token_expert_indices: torch.Tensor,
    gating_output: float,
) -> None:
    torch.ops.sgl_kernel.topk_softmax.default(
        topk_weights, topk_ids, token_expert_indices, gating_output
    )


def moe_fused_gate(
    input_tensor,
    bias,
    num_expert_group,
    topk_group,
    topk,
    num_fused_shared_experts=0,
    routed_scaling_factor=0,
):
    # This fused kernel function is used to select topk expert in a hierarchical 2-layer fashion
    # it split group of expert into num_expert_group, and use top2 expert weight sum in each group
    # as the group weight to select expert groups and then select topk experts within the selected groups
    # the #experts is decided by the input tensor shape and we currently only support power of 2 #experts
    # and #experts should be divisible by num_expert_group. #expert/num_expert_group <= 32 is limited for now.
    # for non-supported case, we suggest to use the biased_grouped_topk func in sglang.srt.layers.moe.topk
    # num_fused_shared_experts: if > 0, the last several experts will be replaced with shared experts
    # routed_scaling_factor: if > 0, the shared experts will be scaled by this factor
    return torch.ops.sgl_kernel.moe_fused_gate.default(
        input_tensor,
        bias,
        num_expert_group,
        topk_group,
        topk,
        num_fused_shared_experts,
        routed_scaling_factor,
    )


def ep_moe_pre_reorder(
    input_tensor,
    gateup_input,
    src2dst,
    topk_ids,
    a1_scales,
    start_expert_id,
    end_expert_id,
    topk,
    use_per_token_if_dynamic,
):
    return torch.ops.sgl_kernel.ep_moe_pre_reorder.default(
        input_tensor,
        gateup_input,
        src2dst,
        topk_ids,
        a1_scales,
        start_expert_id,
        end_expert_id,
        topk,
        use_per_token_if_dynamic,
    )


def ep_moe_silu_and_mul(
    gateup_output,
    down_input,
    reorder_topk_ids,
    scales,
    start_expert_id,
    end_expert_id,
):
    return torch.ops.sgl_kernel.ep_moe_silu_and_mul.default(
        gateup_output,
        down_input,
        reorder_topk_ids,
        scales,
        start_expert_id,
        end_expert_id,
    )


def ep_moe_post_reorder(
    down_output,
    output,
    src2dst,
    topk_ids,
    topk_weights,
    start_expert_id,
    end_expert_id,
    topk,
):
    return torch.ops.sgl_kernel.ep_moe_post_reorder.default(
        down_output,
        output,
        src2dst,
        topk_ids,
        topk_weights,
        start_expert_id,
        end_expert_id,
        topk,
    )


def fp8_blockwise_scaled_grouped_mm(
    output,
    a_ptrs,
    b_ptrs,
    out_ptrs,
    a_scales_ptrs,
    b_scales_ptrs,
    a,
    b,
    scales_a,
    scales_b,
    stride_a,
    stride_b,
    stride_c,
    layout_sfa,
    layout_sfb,
    problem_sizes,
    expert_offsets,
    workspace,
):
    torch.ops.sgl_kernel.fp8_blockwise_scaled_grouped_mm.default(
        output,
        a_ptrs,
        b_ptrs,
        out_ptrs,
        a_scales_ptrs,
        b_scales_ptrs,
        a,
        b,
        scales_a,
        scales_b,
        stride_a,
        stride_b,
        stride_c,
        layout_sfa,
        layout_sfb,
        problem_sizes,
        expert_offsets,
        workspace,
    )


def moe_permute(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk: int,
    n_expert: int,
    expert_map: Optional[torch.Tensor] = None,
    align_block_size: int = -1,
    fill_invalid_expert: int = -1
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
           torch.Tensor]:
    """
    This function expands and permutes activation to gather uncontinuous tokens 
      for each expert.
    Parameters:
    - hidden_states (torch.Tensor): The input tensor to the MoE layer.    
    - topk_ids (torch.Tensor): topk expert route id for each token.
    - topk (int): The number of top-k experts to select.
    - n_expert (int): The number of expert.
    - expert_map (Optional[torch.Tensor]):  A tensor mapping expert indices 
        from the global expert space to the local expert space of the expert 
        parallel shard.
    - align_block_size (Optional[int]): align group gemm block size for deepgemm
    - fill_invalid_expert(int): fill expert id in m_indices for invalid expert 
      to workaround DeepGemm unsupported -1 in m_indices
    Returns:
    - permuted_hidden_states (torch.Tensor): permuted activation.
    - expert_first_token_offset (torch.Tensor): offset of the first token
       of each expert for standard grouped gemm. if enable 'align_block_size'
       expert_first_token_offset will align up to 'align_block_size'.
    - inv_permuted_idx (torch.Tensor): idx map for moe_unpermute.
    - permuted_idx (torch.Tensor): idx map from hidden to permuted_hidden.
    - m_indices: m_indices for grouped gemm in deepgemm,`m_indices[i]` records 
    the group which the j-th row of the LHS belong to.`
    """
    n_token, n_hidden = hidden_states.shape
    assert (n_hidden * hidden_states.element_size()
            ) % 16 == 0, "permue kernel need hidden dim align to 16B"
    permuted_row_size = n_token * topk
    if align_block_size is not None:
        permuted_row_size = (permuted_row_size + n_expert *
                             (align_block_size - 1) + align_block_size -
                             1) // align_block_size * align_block_size
    n_local_expert = n_expert
    if expert_map is not None:
        n_local_expert = torch.sum(expert_map != -1).item()

    permuted_hidden_states = torch.empty(
        (permuted_row_size, n_hidden),
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )
    token_expert_indices = torch.arange(0,
                                        n_token * topk,
                                        dtype=torch.int32,
                                        device=hidden_states.device).reshape(
                                            (n_token, topk))

    m_indices = torch.full((permuted_row_size, ),
                           fill_invalid_expert,
                           dtype=torch.int32,
                           device=hidden_states.device)
    expert_first_token_offset = torch.empty(n_local_expert + 1,
                                            dtype=torch.int64,
                                            device=hidden_states.device)
    # todo clamp (0, n_token * topk - 1) to avoid out of bound ?
    permuted_idx = torch.full((permuted_row_size, ),
                              n_token * topk,
                              dtype=torch.int32,
                              device=hidden_states.device)
    inv_permuted_idx = torch.empty((n_token, topk),
                                   dtype=torch.int32,
                                   device=hidden_states.device)
    torch.ops.sgl_kernel.moe_permute.default(hidden_states, topk_ids, token_expert_indices,
                                 expert_map, n_expert, n_local_expert, topk,
                                 align_block_size, permuted_hidden_states,
                                 expert_first_token_offset, inv_permuted_idx,
                                 permuted_idx, m_indices)
    return (permuted_hidden_states, expert_first_token_offset,
            inv_permuted_idx, permuted_idx, m_indices)


def moe_unpermute(
    permuted_hidden_states: torch.Tensor,
    topk_weights: torch.Tensor,
    inv_permuted_idx: torch.Tensor,
    expert_first_token_offset: torch.Tensor,
    topk: int,
) -> torch.Tensor:
    """
    This function expands and permutes activation to gathering uncontinuous 
      tokens for each expert.
    Parameters:
    - permuted_hidden_states (torch.Tensor): permuted activation.
    - topk_weights (torch.Tensor): topk expert route weight for each token.
    - inv_permuted_idx (torch.Tensor): row idx map for moe_unpermute.
    - expert_first_token_offset (torch.Tensor): offset of the first token
       of each expert for grouped gemm.
    - topk (int): The number of top-k experts to select.
    Returns:
    - hidden_states (torch.Tensor): The reduced and unpermuted activation 
      tensor.  
    """
    n_token, n_hidden = topk_weights.shape[0], permuted_hidden_states.shape[-1]
    assert (n_hidden * permuted_hidden_states.element_size()
            ) % 16 == 0, "unpermue kernel need hidden dim align to 16B"
    hidden_states = torch.empty((n_token, n_hidden),
                                dtype=permuted_hidden_states.dtype,
                                device=permuted_hidden_states.device)

    torch.ops.sgl_kernel.moe_unpermute.default(permuted_hidden_states, topk_weights,
                                   inv_permuted_idx, expert_first_token_offset,
                                   topk, hidden_states)
    return hidden_states


def _customized_moe_permute(
    curr_hidden_states: torch.Tensor,
    a1q_scale: Optional[torch.Tensor],
    curr_topk_ids: torch.Tensor,
    global_num_experts: int,
    expert_map: Optional[torch.Tensor],
    block_m: int,
):
    fill_invalid_expert = 0
    topk = curr_topk_ids.shape[1]
    tokens_in_chunk, _ = curr_hidden_states.shape
    num_tokens = topk * tokens_in_chunk
    (permuted_hidden_states, expert_first_token_offset, inv_permuted_idx,
     permuted_idx, m_indices) = moe_permute(curr_hidden_states, curr_topk_ids,
                                            topk, global_num_experts,
                                            expert_map, block_m,
                                            fill_invalid_expert)
    permuted_idx = permuted_idx.clamp(max=num_tokens - 1)
    if a1q_scale is not None:
        a1q_scale = a1q_scale[permuted_idx // topk]
    return (permuted_hidden_states, a1q_scale, permuted_idx, m_indices,
            inv_permuted_idx, expert_first_token_offset)


def _customized_moe_unpermute_and_reduce(
    curr_hidden: torch.Tensor,
    inv_perm: Optional[torch.Tensor],
    topk_weight: torch.Tensor,
    first_token_offset: torch.Tensor,
) -> torch.Tensor:
    M, topk = topk_weight.shape
    output = moe_unpermute(curr_hidden, topk_weight, inv_perm,
                           first_token_offset, topk)
    return output


def _fp8_quantize(
    A: torch.Tensor,
    A_scale: Optional[torch.Tensor],
    block_shape: Optional[List[int]],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Perform fp8 quantization on the inputs.  If a block_shape
    is provided, the output will be blocked.
    """
    from sglang.srt.layers.quantization.fp8_kernel import scaled_fp8_quant, per_token_group_quant_fp8
    if block_shape is None:
        A, A_scale = scaled_fp8_quant(A, A_scale)
    else:
        assert len(block_shape) == 2
        _, block_k = block_shape[0], block_shape[1]
        A, A_scale = per_token_group_quant_fp8(A, block_k)
        assert A.shape[-1] // block_k == A_scale.shape[-1]
    return A, A_scale


def _resize_cache(x: torch.Tensor, v: Tuple[int, ...]) -> torch.Tensor:
    """
    Shrink the given tensor and apply the given view to it.  This is
    used to resize the intermediate fused_moe caches.
    """
    assert prod(v) <= x.numel()
    return x.flatten()[:prod(v)].view(*v)


def _valid_deep_gemm(hidden_states: torch.Tensor,
                     w1: torch.Tensor,
                     w2: torch.Tensor,
                     expert_map: Optional[torch.Tensor] = None) -> bool:
    """
    Check if the given problem size is supported by the DeepGemm grouped
    gemm kernel.  All of M, N, K and the quantization block_shape must be
    aligned by `dg.get_m_alignment_for_contiguous_layout()`.
    """
    # Lazy import to avoid CUDA initialization problems.
    import deep_gemm as dg

    # Expert maps not supported yet.
    if expert_map is not None:
        return False

    align = dg.get_m_alignment_for_contiguous_layout()
    M = hidden_states.shape[0]
    _, K, N = w2.shape

    if align > M or N % align != 0 or K % align != 0:
        return False

    return (hidden_states.is_contiguous() and w1.is_contiguous()
            and w2.is_contiguous())

# Modified from https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/fused_moe/deep_gemm_moe.py
def deep_gemm_moe_fp8(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    inplace: bool = False,
    activation: str = "silu",
    global_num_experts: int = -1,
    expert_map: Optional[torch.Tensor] = None,
    a1_scale: Optional[torch.Tensor] = None,
    a2_scale: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    This function computes a a8w8-quantized Mixture of Experts (MoE) layer
    using two sets of quantized weights, w1_q and w2_q, and top-k gating
    mechanism. The matrix multiplications are implemented with DeepGemm
    grouped gemm.

    Parameters:
    - hidden_states (torch.Tensor): The input tensor to the MoE layer.
        Shape: [M, K]
    - w1 (torch.Tensor): The first set of fp8 quantized expert weights.
        Shape: [num_experts, K, 2N] (the weights are passed transposed)
    - w2 (torch.Tensor): The second set of fp8 quantized expert weights.
        Shape: [num_experts, N, K] (the weights are passed transposed)
    - w1_scale (torch.Tensor): The fp32 scale to dequantize w1_q.
        Shape: [num_experts] or [num_experts, 2N]
    - w2_scale (torch.Tensor): The fp32 scale to dequantize w2_q.
        Shape: [num_experts] or [num_experts, K]
    - topk_weights (torch.Tensor): The weights of each token->expert mapping.
    - topk_ids (torch.Tensor): The token->expert mapping for topk_weights.
    - inplace (bool): If True, perform the operation in-place.
        Defaults to False.
    - activation (str): The activation function to apply after the first
        MoE layer.
    - global_num_experts (int): The total number of experts in the global
        expert space.
    - expert_map (Optional[torch.Tensor]):  A tensor mapping expert indices
        from the global expert space to the local expert space of the expert
        parallel shard.
    - a1_scale (Optional[torch.Tensor]): The optional fp32 scale to quantize a.
        Shape: scalar or [M]
    - a2_scale (Optional[torch.Tensor]): The optional fp32 scale to
        quantize the intermediate result between the gemms.
        Shape: scalar or [M]

    Returns:
    - torch.Tensor: The bfloat16 output tensor after applying the MoE layer.
    """
    # Lazy import to avoid CUDA initialization problems.
    import deep_gemm as dg
    from sglang.srt.layers.quantization import deep_gemm_wrapper

    assert expert_map is None, "Expert maps not supported yet"

    assert hidden_states.shape[1] == w1.shape[2], "Hidden size mismatch"

    assert topk_weights.shape == topk_ids.shape, "topk shape mismatch"
    assert hidden_states.is_contiguous(), "Hidden_states must be contiguous"
    assert w1.stride(-1) == 1, "Stride of last dimension must be 1"
    assert w2.stride(-1) == 1, "Stride of last dimension must be 1"
    assert hidden_states.dtype in [
        torch.float32, torch.float16, torch.bfloat16
    ]
    assert w1.dtype == torch.float8_e4m3fn
    assert w2.dtype == torch.float8_e4m3fn
    assert w1.shape[0] == w2.shape[0], "Expert number mismatch"
    assert w1.shape[0] == w1_scale.shape[0], "w1 scales expert number mismatch"
    assert w1.shape[0] == w2_scale.shape[0], "w2 scales expert number mismatch"
    assert a1_scale is None or a1_scale.dim(
    ) == 0 or a1_scale.shape[0] == 1 or a1_scale.shape[
        0] == hidden_states.shape[0], "Input scale shape mismatch"
    assert a2_scale is None or a1_scale is None or a2_scale.shape == a1_scale.shape, "Intermediate scale shape mismatch"  # noqa: E501

    num_tokens, _ = hidden_states.shape
    E, N, _ = w1.shape
    K = w2.shape[1]
    if global_num_experts == -1:
        global_num_experts = E

    # We execute the fused_moe kernel in chunks to circumvent this issue:
    # https://github.com/vllm-project/vllm/issues/5938
    CHUNK_SIZE = 64 * 1024

    assert _valid_deep_gemm(hidden_states, w1, w2, expert_map)

    if inplace:
        out_hidden_states = hidden_states
    else:
        out_hidden_states = torch.empty_like(hidden_states)

    block_m = dg.get_m_alignment_for_contiguous_layout()
    block_shape = [block_m, block_m]

    assert w1_scale is not None
    assert w2_scale is not None

    # We attempt to transpose and align offline in Fp8MoEMethod, in which
    # case these calls will be nops.  Otherwise, they'll be performed every
    # time the layer is executed.
    w1_scale = dg.get_col_major_tma_aligned_tensor(w1_scale).contiguous()
    w2_scale = dg.get_col_major_tma_aligned_tensor(w2_scale).contiguous()

    def round_up(x: int, y: int) -> int:
        return ((x + y - 1) // y) * y

    M_sum = topk_ids.numel() + global_num_experts * (block_m - 1)
    M_sum = round_up(M_sum, block_m)

    num_chunks = (num_tokens // CHUNK_SIZE) + 1

    # We can reuse the memory between cache1 and cache3 because by the time
    # we need cache3, we're done with cache1
    workspace13 = torch.empty(M_sum * max(N, K),
                              device=hidden_states.device,
                              dtype=hidden_states.dtype)

    workspace1 = workspace13[:M_sum * N].view(M_sum, N)
    workspace2 = torch.empty((M_sum, N // 2),
                             device=hidden_states.device,
                             dtype=hidden_states.dtype)
    workspace3 = workspace13[:M_sum * K].view(M_sum, K)

    for chunk in range(num_chunks):
        begin_chunk_idx, end_chunk_idx = (chunk * CHUNK_SIZE,
                                          min((chunk + 1) * CHUNK_SIZE,
                                              num_tokens))
        curr_hidden_states = hidden_states[begin_chunk_idx:end_chunk_idx]
        tokens_in_chunk, _ = curr_hidden_states.shape

        if tokens_in_chunk == 0:
            break

        curr_topk_ids = topk_ids[begin_chunk_idx:end_chunk_idx]
        curr_topk_weights = topk_weights[begin_chunk_idx:end_chunk_idx]

        a1q_scale: Optional[torch.Tensor] = None

        qcurr_hidden_states, a1q_scale = _fp8_quantize(curr_hidden_states,
                                                       a1_scale, block_shape)

        # (qcurr_hidden_states_, a1q_scale_, sorted_token_ids_, expert_ids_,
        #  inv_perm_) = _moe_permute(qcurr_hidden_states, a1q_scale,
        #                           curr_topk_ids, global_num_experts,
        #                           expert_map=expert_map, block_m=block_m)
        (qcurr_hidden_states, a1q_scale, sorted_token_ids, expert_ids,
         inv_perm, first_token_offset) = _customized_moe_permute(
             qcurr_hidden_states, a1q_scale, curr_topk_ids, global_num_experts,
             expert_map, block_m)

        # Adjust the intermediate cache size and config for the last chunk.
        # Note that in most cases we only have one chunk so the cache size
        # and config are already set correctly and do not need to be adjusted.
        if tokens_in_chunk < CHUNK_SIZE and chunk > 0:
            curr_M = sorted_token_ids.numel()
            workspace1 = _resize_cache(workspace1, (curr_M, N))
            workspace2 = _resize_cache(workspace2, (curr_M, N // 2))
            workspace3 = _resize_cache(workspace3, (curr_M, K))

        deep_gemm_wrapper.grouped_gemm_nt_f8f8bf16_contig(
            (qcurr_hidden_states, a1q_scale), (w1, w1_scale), workspace1,
            expert_ids)

        if activation == "silu":
            torch.ops._C.silu_and_mul(workspace2, workspace1.view(-1, N))
        elif activation == "gelu":
            torch.ops._C.gelu_and_mul(workspace2, workspace1.view(-1, N))
        else:
            raise ValueError(f"Unsupported FusedMoe activation: {activation}")

        a2q_scale: Optional[torch.Tensor] = None

        qworkspace2, a2q_scale = _fp8_quantize(workspace2, a2_scale,
                                               block_shape)

        deep_gemm_wrapper.grouped_gemm_nt_f8f8bf16_contig(
            (qworkspace2, a2q_scale), (w2, w2_scale), workspace3, expert_ids)

        out_hidden_states[begin_chunk_idx:end_chunk_idx] = \
          _customized_moe_unpermute_and_reduce(workspace3.view(*workspace3.shape),
          inv_perm, curr_topk_weights, first_token_offset)

    return out_hidden_states

def prepare_moe_input(
    topk_ids,
    expert_offsets,
    problem_sizes1,
    problem_sizes2,
    input_permutation,
    output_permutation,
    num_experts,
    n,
    k,
    blockscale_offsets: Optional[torch.Tensor] = None,
):
    torch.ops.sgl_kernel.prepare_moe_input.default(
        topk_ids,
        expert_offsets,
        blockscale_offsets,
        problem_sizes1,
        problem_sizes2,
        input_permutation,
        output_permutation,
        num_experts,
        n,
        k,
    )


def apply_shuffle_mul_sum(
    input,
    output,
    permutation,
    factors,
):
    torch.ops.sgl_kernel.apply_shuffle_mul_sum.default(
        input, output, permutation, factors
    )


def cutlass_fp4_group_mm(
    a_fp4,
    b_fp4,
    a_blockscale,
    b_blockscale,
    alphas,
    out_dtype,
    device,
    params: Dict[str, Any],
):
    """
    An FP4 Blockscaled Group Gemm that takes in  a_tensors, b_tensors and runs
    the gemms for each combination based on the specified problem sizes.

    This is used as the MoE gemm during NVFP4 Quantized FusedMoE forward.
    - a/b_tensors: the NVFP4 a_ptrs and b_ptrs tensors which are quantized
                     input and expert weights.
    - a_/b_scales: The blockscales in FP8-E4M3 precision
    - ab_strides/c_strides: Strides for the a/b tensors between rows.
    - expert_offsets/sf_offsets: Indices that mark at which token index
                    each expert begins its computation. The number of tokens
                    computed with expert E is expert_offsets[E + 1] -
                    expert_offsets[E] And the sf_size per expert is
                    sf_offset[E+1] - sf_offset[E]
    - problem_sizes: MxNxK sizes of each expert's multiplication in two grouped
                     MMs used in the fused MoE operation.
    """
    m_topk = a_fp4.shape[0]
    n = b_fp4.shape[1]
    c_shape = (m_topk, n)
    c = torch.empty(c_shape, device=device, dtype=out_dtype)
    torch.ops.sgl_kernel.cutlass_fp4_group_mm.default(
        c,
        a_fp4,
        b_fp4,
        a_blockscale,
        b_blockscale,
        alphas,
        params["ab_strides"],
        params["c_strides"],
        params["problem_sizes"],
        params["expert_offsets"],
        params["blockscale_offsets"],
    )
    return c.to(dtype=out_dtype)
