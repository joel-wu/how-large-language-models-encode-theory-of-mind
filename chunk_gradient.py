#!/usr/bin/env python3
"""
Layer chunker for squared-gradient checkpoints
(adapted from SqueezeLLM: Dense and Sparse Quantization)

- Loads a HF causal LM (LLaMA/Mistral/OPT; Qwen treated as LLaMA).
- Iterates decoder layers, collects core linear weights, and writes one .pt per layer:
  { "q_proj": Tensor, "k_proj": Tensor, ... }
- Intended to be used after gradients have been written into weights.
"""

import argparse, os, json
from typing import List, Tuple
import torch
import transformers
from tqdm import tqdm


# --------- model parsing ---------

def infer_model_type(name_or_path: str, model: transformers.PreTrainedModel) -> str:
    n = (name_or_path or "").lower()
    if "opt" in n:
        return "opt"
    if any(k in n for k in ["llama", "vicuna", "qwen", "deepseek"]):
        return "llama"
    if "mistral" in n:
        return "mistral"
    # fallback by structure
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return "llama"
    if hasattr(model, "model") and hasattr(model.model, "decoder") and hasattr(model.model.decoder, "layers"):
        return "opt"
    raise NotImplementedError("Unable to infer model type (try --model_type).")

def get_layers(model: transformers.PreTrainedModel, model_type: str):
    inner = model.model if hasattr(model, "model") else model
    if model_type in ("llama", "mistral"):
        return inner.layers
    if model_type == "opt":
        return inner.decoder.layers
    raise NotImplementedError(f"Unsupported model_type: {model_type}")

def module_names(model_type: str) -> List[str]:
    if model_type in ("llama", "mistral"):
        return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    if model_type == "opt":
        return ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]
    raise NotImplementedError

def modules_from_layer(layer, model_type: str) -> List[torch.nn.Module]:
    if model_type in ("llama", "mistral"):
        return [
            layer.self_attn.q_proj,
            layer.self_attn.k_proj,
            layer.self_attn.v_proj,
            layer.self_attn.o_proj,
            layer.mlp.gate_proj,
            layer.mlp.up_proj,
            layer.mlp.down_proj,
        ]
    if model_type == "opt":
        return [
            layer.self_attn.q_proj,
            layer.self_attn.k_proj,
            layer.self_attn.v_proj,
            layer.self_attn.out_proj,
            layer.fc1,
            layer.fc2,
        ]
    raise NotImplementedError


# --------- main ---------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Path or repo id of the checkpoint with gradients in weights")
    ap.add_argument("--output_path", required=True, help="Dir to write layer_{i}.pt files")
    ap.add_argument("--model_type", choices=["llama","mistral","opt"], default=None, help="Override model family")
    ap.add_argument("--cache_dir", default=None, help="HF cache dir")
    ap.add_argument("--device_map", default="cpu", help="HF device_map (e.g., 'cpu' or 'auto')")
    args = ap.parse_args()

    os.makedirs(args.output_path, exist_ok=True)

    model = transformers.AutoModelForCausalLM.from_pretrained(
        args.model,
        cache_dir=args.cache_dir,
        device_map=args.device_map,      # pure loading; no forward pass needed
        torch_dtype=torch.float16,       # dtype does not matter for saving tensors
        trust_remote_code=True,
    )
    mt = args.model_type or infer_model_type(args.model, model)
    layers = get_layers(model, mt)
    names = module_names(mt)

    manifest = {
        "source_model": args.model,
        "model_type": mt,
        "num_layers": len(layers),
        "modules": names,
    }

    for i, layer in tqdm(list(enumerate(layers)), desc="Chunking layers"):
        mods = modules_from_layer(layer, mt)
        blob = {}
        for lin, name in zip(mods, names):
            # move to cpu to avoid GPU pinning in saved files
            w = lin.weight.data.detach().to("cpu").contiguous()
            blob[name] = w
        torch.save(blob, os.path.join(args.output_path, f"layer_{i}.pt"))

    with open(os.path.join(args.output_path, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"[DONE] wrote {len(layers)} layer files to {args.output_path}")

if __name__ == "__main__":
    main()
