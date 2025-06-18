# python3 benchmark/kernels/fused_moe_triton/benchmark_deepgemm_moe.py  --model deepseek-ai/DeepSeek-V3-0324 -tp 8 --dtype fp8_w8a8 --use-deep-gemm --trust-remote-code
import argparse
from typing import Any, TypedDict

import ray
import torch
from ray.experimental.tqdm_ray import tqdm
from transformers import AutoConfig

from sglang.srt.layers.moe.fused_moe_triton.fused_moe import *
from sglang.srt.layers.moe.topk import fused_topk
from sgl_kernel import deep_gemm_moe_fp8

class BenchmarkConfig(TypedDict):
    BLOCK_SIZE_M: int
    BLOCK_SIZE_N: int
    BLOCK_SIZE_K: int
    GROUP_SIZE_M: int
    num_warps: int
    num_stages: int


def benchmark_config(config: BenchmarkConfig,
                     num_tokens: int,
                     num_experts: int,
                     shard_intermediate_size: int,
                     hidden_size: int,
                     topk: int,
                     dtype: torch.dtype,
                     use_fp8_w8a8: bool,
                     use_int8_w8a16: bool,
                     num_iters: int = 100,
                     block_quant_shape: List[int] = None,
                     use_deep_gemm: bool = False) -> float:
    init_dtype = torch.float16 if use_fp8_w8a8 else dtype
    x = torch.randn(num_tokens, hidden_size, dtype=dtype)
    if use_int8_w8a16:
        w1 = torch.randint(-127,
                           127, (
                               num_experts,
                               shard_intermediate_size,
                               hidden_size,
                           ),
                           dtype=torch.int8)
        w2 = torch.randint(-127,
                           127, (
                               num_experts,
                               hidden_size,
                               shard_intermediate_size // 2,
                           ),
                           dtype=torch.int8)
    else:
        w1 = torch.randn(num_experts,
                         shard_intermediate_size,
                         hidden_size,
                         dtype=init_dtype)
        w2 = torch.randn(num_experts,
                         hidden_size,
                         shard_intermediate_size // 2,
                         dtype=init_dtype)
    gating_output = torch.randn(num_iters,
                                num_tokens,
                                num_experts,
                                dtype=torch.float32)

    w1_scale = None
    w2_scale = None
    a1_scale = None
    a2_scale = None
    if use_int8_w8a16:
        w1_scale = torch.randn((num_experts, 2 * shard_intermediate_size),
                               dtype=torch.float32)
        w2_scale = torch.randn((hidden_size, num_experts), dtype=torch.float32)
    if use_fp8_w8a8:
        if block_quant_shape:
            block_n, block_k = block_quant_shape[0], block_quant_shape[1]
            E = num_experts
            N = shard_intermediate_size // 2
            K = hidden_size
            factor_for_scale = 1e-2
            n_tiles_w1 = (2 * N + block_n - 1) // block_n
            n_tiles_w2 = (K + block_n - 1) // block_n
            k_tiles_w1 = (K + block_k - 1) // block_k
            k_tiles_w2 = (N + block_k - 1) // block_k
            w1_scale = torch.rand((E, n_tiles_w1, k_tiles_w1),
                                  dtype=torch.float32) * factor_for_scale
            w2_scale = torch.rand((E, n_tiles_w2, k_tiles_w2),
                                  dtype=torch.float32) * factor_for_scale
        else:
            w1_scale = torch.randn(num_experts, dtype=torch.float32)
            w2_scale = torch.randn(num_experts, dtype=torch.float32)

        a1_scale = torch.randn(1, dtype=torch.float32)
        a2_scale = torch.randn(1, dtype=torch.float32)

        w1 = w1.to(torch.float8_e4m3fn)
        w2 = w2.to(torch.float8_e4m3fn)

    input_gating = torch.empty(num_tokens, num_experts, dtype=torch.float32)

    def prepare(i: int):
        input_gating.copy_(gating_output[i])

    def run():
        from sglang.srt.layers.moe.fused_moe_triton import override_config
        with override_config(config):
            if use_deep_gemm:
                topk_weights, topk_ids = fused_topk(
                    x, input_gating, topk, False)
                deep_gemm_moe_fp8(
                    x,
                    w1,
                    w2,
                    w1_scale,
                    w2_scale,
                    topk_weights,
                    topk_ids,
                    global_num_experts=num_experts)
            else:
                fused_moe(
                    x,
                    w1,
                    w2,
                    input_gating,
                    topk,
                    renormalize=True,
                    inplace=True,
                    use_fp8_w8a8=use_fp8_w8a8,
                    use_int8_w8a16=use_int8_w8a16,
                    w1_scale=w1_scale,
                    w2_scale=w2_scale,
                    a1_scale=a1_scale,
                    a2_scale=a2_scale,
                    block_shape=block_quant_shape,
                )

    # JIT compilation & warmup
    run()
    torch.cuda.synchronize()

    # Capture 10 invocations with CUDA graph
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(10):
            run()
    torch.cuda.synchronize()

    # Warmup
    for _ in range(5):
        graph.replay()
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    latencies: list[float] = []
    for i in range(num_iters):
        prepare(i)
        torch.cuda.synchronize()

        start_event.record()
        graph.replay()
        end_event.record()
        end_event.synchronize()
        latencies.append(start_event.elapsed_time(end_event))
    avg = sum(latencies) / (num_iters * 10) * 1000  # us
    graph.reset()
    return avg


@ray.remote(num_gpus=1)
class BenchmarkWorker:

    def __init__(self, seed: int) -> None:
        torch.set_default_device("cuda")
        self.seed = seed
        # Get the device ID to allocate tensors and kernels
        # on the respective GPU. This is required for Ray to work
        # correctly with multi-GPU tuning on the ROCm platform.
        self.device_id = int(ray.get_gpu_ids()[0])

    def benchmark(
        self,
        num_tokens: int,
        num_experts: int,
        shard_intermediate_size: int,
        hidden_size: int,
        topk: int,
        dtype: torch.dtype,
        use_fp8_w8a8: bool,
        use_int8_w8a16: bool,
        block_quant_shape: List[int] = None,
        use_deep_gemm: bool = False,
    ) -> tuple[dict[str, int], float]:
        dtype_str = get_config_dtype_str(dtype,
                                         use_int8_w8a16=use_int8_w8a16,
                                         use_fp8_w8a8=use_fp8_w8a8)
        # NOTE(woosuk): The current naming convention uses w2.shape[2], which
        # is the intermediate size after silu_and_mul.
        op_config = get_moe_configs(num_experts, shard_intermediate_size // 2,
                                    dtype_str, block_n=128, block_k=128)
        if op_config is None:
            config = get_default_config(num_tokens,
                                        num_experts,
                                        shard_intermediate_size,
                                        hidden_size,
                                        topk,
                                        dtype_str,
                                        is_marlin=False)
        else:
            config = op_config[min(op_config.keys(),
                                   key=lambda x: abs(x - num_tokens))]
        kernel_time = benchmark_config(config,
                                       num_tokens,
                                       num_experts,
                                       shard_intermediate_size,
                                       hidden_size,
                                       topk,
                                       dtype,
                                       use_fp8_w8a8,
                                       use_int8_w8a16,
                                       num_iters=100,
                                       block_quant_shape=block_quant_shape,
                                       use_deep_gemm=use_deep_gemm)
        return config, kernel_time


def get_weight_block_size_safety(config, default_value=None):

    quantization_config = getattr(config, 'quantization_config', {})
    if isinstance(quantization_config, dict):
        return quantization_config.get('weight_block_size', default_value)
    return default_value


def main(args: argparse.Namespace):
    print(args)

    config = AutoConfig.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)

    if config.architectures[0] == "DbrxForCausalLM":
        E = config.ffn_config.moe_num_experts
        topk = config.ffn_config.moe_top_k
        intermediate_size = config.ffn_config.ffn_hidden_size
        shard_intermediate_size = 2 * intermediate_size // args.tp_size
    elif config.architectures[0] == "JambaForCausalLM":
        E = config.num_experts
        topk = config.num_experts_per_tok
        intermediate_size = config.intermediate_size
        shard_intermediate_size = 2 * intermediate_size // args.tp_size
    elif (config.architectures[0]
          in ("DeepseekV3ForCausalLM", "DeepseekV2ForCausalLM")):
        E = config.n_routed_experts
        topk = config.num_experts_per_tok
        intermediate_size = config.moe_intermediate_size
        shard_intermediate_size = 2 * intermediate_size // args.tp_size
    elif config.architectures[0] in ("Qwen2MoeForCausalLM",
                                     "Qwen3MoeForCausalLM"):
        E = config.num_experts
        topk = config.num_experts_per_tok
        intermediate_size = config.moe_intermediate_size
        shard_intermediate_size = 2 * intermediate_size // args.tp_size
    else:
        # Default: Mixtral.
        E = config.num_local_experts
        topk = config.num_experts_per_tok
        intermediate_size = config.intermediate_size
        shard_intermediate_size = 2 * intermediate_size // args.tp_size

    hidden_size = config.hidden_size
    dtype = config.torch_dtype
    use_fp8_w8a8 = args.dtype == "fp8_w8a8"
    use_int8_w8a16 = args.dtype == "int8_w8a16"
    block_quant_shape = get_weight_block_size_safety(config)

    if args.batch_size is None:
        batch_sizes = [128, 256, 512, 1024, 1536, 2048, 3072, 4096, 8192]
    else:
        batch_sizes = [args.batch_size]

    use_deep_gemm = bool(args.use_deep_gemm)

    ray.init()
    num_gpus = int(ray.available_resources()["GPU"])
    workers = [BenchmarkWorker.remote(args.seed) for _ in range(num_gpus)]

    def _distribute(method: str, inputs: list[Any]) -> list[Any]:
        outputs = []
        worker_idx = 0
        for input_args in inputs:
            worker = workers[worker_idx]
            worker_method = getattr(worker, method)
            output = worker_method.remote(*input_args)
            outputs.append(output)
            worker_idx = (worker_idx + 1) % num_gpus
        return ray.get(outputs)


    outputs = _distribute(
        "benchmark",
        [(batch_size, E, shard_intermediate_size, hidden_size, topk, dtype,
            use_fp8_w8a8, use_int8_w8a16, block_quant_shape, use_deep_gemm)
            for batch_size in batch_sizes])

    for batch_size, (config, kernel_time) in zip(batch_sizes, outputs):
        print(f"Batch size: {batch_size}, config: {config}")
        print(f"Kernel time: {kernel_time:.2f} us")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",
                        type=str,
                        default="mistralai/Mixtral-8x7B-Instruct-v0.1")
    parser.add_argument("--tp-size",
                        "-tp",
                        "--tensor-parallel-size",
                        type=int,
                        default=2)
    parser.add_argument("--dtype",
                        type=str,
                        choices=["auto", "fp8_w8a8", "int8_w8a16"],
                        default="auto")
    parser.add_argument("--use-deep-gemm", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, required=False)
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args()

    main(args)
