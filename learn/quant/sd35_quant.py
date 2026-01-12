import torch
import os
import time
import argparse
from diffusers import StableDiffusion3Pipeline, SD3Transformer2DModel
from transformers import T5EncoderModel
from torchao.quantization import quantize_, int8_weight_only, int4_weight_only
try:
    from torchao.quantization import Float8DynamicActivationFloat8WeightConfig
    HAS_FP8 = True
except ImportError:
    HAS_FP8 = False
import torch._inductor.config as inductor_config
# 针对 Windows 环境的稳定性优化
inductor_config.triton.cudagraphs = False
inductor_config.triton.cudagraph_trees = False  # 明确禁用这个导致溢出的树结构
if hasattr(inductor_config, "static_memory_optimization"):
    inductor_config.static_memory_optimization = False
if hasattr(inductor_config, "fx_graph_cache"):
    inductor_config.fx_graph_cache = False
# -------------------------- Configuration --------------------------
# Default values
DEFAULT_MODEL_ID = "stabilityai/stable-diffusion-3.5-medium"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_DIR = "others/checkpoints/sd35_quantized"

def get_quant_strategy(q_type):
    if q_type == "int8_wo":
        return int8_weight_only()
    elif q_type == "int4_wo":
        return int4_weight_only()
    elif q_type == "fp8":
        if not HAS_FP8:
            raise ImportError("Float8DynamicActivationFloat8WeightConfig not found in torchao. Please upgrade torchao.")
        # Requires Ada architecture (RTX 40 series) or Hopper
        return Float8DynamicActivationFloat8WeightConfig()
    else:
        raise ValueError(f"Unsupported quantization type: {q_type}")

def quantize_component(model, name, strategy, q_type):
    print(f"--- Quantizing {name} using {q_type} ---")
    start_time = time.time()
    
    # Apply quantization
    # We filter for Linear layers as they are the primary targets in Transformer/T5
    def filter_fn(module, name):
        # Quantize Linear layers but typically skip small layers or specific heads if needed
        return isinstance(module, torch.nn.Linear)

    # Some versions of torchao use quantize_, others might use different APIs
    # Here we use the functional API similar to the project's ViT example
    quantize_(model, strategy, filter_fn)
    
    end_time = time.time()
    print(f"Finished quantizing {name} in {end_time - start_time:.2f}s")
    return model

def benchmark_inference(pipe, name, num_iters=5):
    print(f"\n--- Benchmarking {name} ---")
    prompt = "A high-tech cyberpunk city with neon lights, 8k resolution, highly detailed."
    
    # Warmup
    print(f"Warming up {name}...")
    with torch.no_grad():
        for _ in range(2):
            pipe(prompt, num_inference_steps=20)
    
    torch.cuda.synchronize()
    start_time = time.time()
    
    with torch.no_grad():
        for i in range(num_iters):
            print(f"Iteration {i+1}/{num_iters}...")
            pipe(prompt, num_inference_steps=20)
    
    torch.cuda.synchronize()
    end_time = time.time()
    
    avg_time = (end_time - start_time) / num_iters
    print(f"{name} Average Inference Time (20 steps): {avg_time:.2f}s")
    return avg_time

# -------------------------- Main Process --------------------------
def run_quantization(model_id, q_type, do_inference, do_benchmark=False):
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR, exist_ok=True)

    results = {}

    if do_benchmark:
        print("\n--- Phase 0: Baseline Benchmark (BF16) ---")
        pipe_bf16 = StableDiffusion3Pipeline.from_pretrained(
            model_id, 
            torch_dtype=torch.bfloat16
        ).to(DEVICE)
        results['bf16'] = benchmark_inference(pipe_bf16, "Original BF16", num_iters=3)
        # 释放内存
        del pipe_bf16
        torch.cuda.empty_cache()

    print(f"\n--- Phase 1: Quantization ({q_type}) ---")
    print(f"Loading SD3.5 Medium components from {model_id}...")
    
    # 1. Quantize Transformer
    print("Loading Transformer...")
    transformer = SD3Transformer2DModel.from_pretrained(
        model_id, 
        subfolder="transformer", 
        torch_dtype=torch.bfloat16
    ).to(DEVICE)
    
    strategy = get_quant_strategy(q_type)
    transformer = quantize_component(transformer, "Transformer", strategy, q_type)
    
    # 2. Quantize T5 Text Encoder
    print("Loading T5 Text Encoder...")
    text_encoder_3 = T5EncoderModel.from_pretrained(
        model_id, 
        subfolder="text_encoder_3", 
        torch_dtype=torch.bfloat16
    ).to(DEVICE)
    
    text_encoder_3 = quantize_component(text_encoder_3, "T5 Encoder", strategy, q_type)

    # 3. Assemble Pipeline
    print("Assembling quantized pipeline...")
    pipe = StableDiffusion3Pipeline.from_pretrained(
        model_id,
        transformer=transformer,
        text_encoder_3=text_encoder_3,
        torch_dtype=torch.bfloat16
    )
    pipe.to(DEVICE)

    if q_type == "fp8":
        print("\nApplying torch.compile to FP8 transformer for performance...")
        # 在 Windows 上避免使用 max-autotune，因为它强制开启的特性会导致溢出
        pipe.transformer = torch.compile(
            pipe.transformer, 
            options={
                "triton.cudagraphs": False,
                "triton.cudagraph_trees": False,
                "triton.autotune_pointwise": True,
                "triton.store_cubin": True,
                "size_asserts": False,
            }
        )

    # 4. Benchmarking Quantized Model
    if do_benchmark:
        results['quant'] = benchmark_inference(pipe, f"Quantized {q_type}", num_iters=3)
        speedup = results['bf16'] / results['quant']
        print(f"\n========================================")
        print(f"Speedup Ratio: {speedup:.2f}x")
        print(f"========================================")

    # 5. Simple Inference Test
    if do_inference:
        print("Running inference test...")
        prompt = "A cinematic shot of a futuristic laboratory, glass viles with glowing liquid, intricate machinery, 8k, highly detailed."
        
        # SD3.5 Medium benefits from specific guidance scales
        with torch.no_grad():
            image = pipe(
                prompt=prompt,
                num_inference_steps=28,
                guidance_scale=4.5,
            ).images[0]
        
        test_image_path = os.path.join(SAVE_DIR, f"sd35_test_{q_type}.png")
        image.save(test_image_path)
        print(f"Inference test complete. Image saved to {test_image_path}")

    # 5. Save Quantized Model
    print(f"Saving quantized components to {SAVE_DIR}...")
    
    # 方案：使用 torch.save 保存 state_dict，因为 safetensors 目前对 torchao 的某些量化子类支持不佳
    transformer_path = os.path.join(SAVE_DIR, f"transformer_{q_type}.pt")
    text_encoder_path = os.path.join(SAVE_DIR, f"text_encoder_3_{q_type}.pt")
    
    torch.save(transformer.state_dict(), transformer_path)
    torch.save(text_encoder_3.state_dict(), text_encoder_path)
    
    # 同时尝试以非 safetensors 格式保存配置（以便以后 load_pretrained）
    try:
        transformer.save_pretrained(os.path.join(SAVE_DIR, f"transformer_{q_type}"), safe_serialization=False)
        text_encoder_3.save_pretrained(os.path.join(SAVE_DIR, f"text_encoder_3_{q_type}"), safe_serialization=False)
    except Exception as e:
        print(f"Warning: save_pretrained with safe_serialization=False failed: {e}")

    print(f"Save complete. Files located in {SAVE_DIR}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SD3.5 Medium Quantization Script")
    parser.add_argument("--model_id", type=str, default=DEFAULT_MODEL_ID, help="HuggingFace model ID")
    parser.add_argument("--type", type=str, default="int8_wo", choices=["int8_wo", "int4_wo", "fp8"], help="Quantization type")
    parser.add_argument("--infer", action="store_true", help="Run inference test after quantization")
    parser.add_argument("--benchmark", action="store_true", help="Run speed benchmark comparison")
    
    args = parser.parse_args()
    
    run_quantization(args.model_id, args.type, args.infer, args.benchmark)
