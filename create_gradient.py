#!/usr/bin/env python3
"""
Squared-gradient saver (adopted from SqueezeLLM: Dense and Sparse Quantization)

- Loads a HF causal LM (LLaMA/Mistral/OPT; Qwen treated as LLaMA layout).
- Datasets:
  * c4  : allenai/c4 small shard; random window; no padding; labels = input_ids
  * tom : local JSON/JSONL with key "txt"; pad to seqlen; supervise last non-pad token
- Registers grad hooks (g -> g^2), accumulates over nsamples, overwrites weights with grads, save_pretrained(out).
"""
import argparse, os, json, random
from typing import List, Tuple
import numpy as np
import torch
from datasets import load_dataset
import transformers
from tqdm import tqdm

def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

def parse_model_type(name: str) -> str:
    n = name.lower()
    if "opt" in n: return "opt"
    if "llama" in n or "vicuna" in n or "qwen" in n: return "llama"
    if "mistral" in n: return "mistral"
    raise NotImplementedError(f"Unknown model family: {name}")

def get_modules_from_layer(layer, mtype: str):
    if mtype in ("llama","mistral"):
        return [layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj,
                layer.self_attn.o_proj, layer.mlp.gate_proj, layer.mlp.up_proj, layer.mlp.down_proj]
    if mtype == "opt":
        return [layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj,
                layer.self_attn.out_proj, layer.fc1, layer.fc2]
    raise NotImplementedError

# ---------------- Data ----------------

def build_c4_loader(tokenizer, nsamples: int, seqlen: int, seed: int) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """Random contiguous window of length seqlen; labels = input_ids."""
    set_seed(seed)
    ds = load_dataset("allenai/c4",
                      data_files={"train": "en/c4-train.00000-of-01024.json.gz"},
                      split="train")
    loader = []
    for _ in range(nsamples):
        for _try in range(100):
            i = random.randint(0, len(ds)-1)
            text = ds[i]["text"]
            enc = tokenizer(text, return_tensors="pt", add_special_tokens=False)
            if enc.input_ids.shape[1] >= seqlen: break
        L = enc.input_ids.shape[1]
        start = 0 if L == seqlen else random.randint(0, L - seqlen)
        end = start + seqlen
        inp = enc.input_ids[:, start:end]
        labels = inp.clone()
        loader.append((inp, labels))
    return loader

def build_tom_loader(
    tokenizer,
    data_path: str,
    nsamples: int,
    seqlen: int,            # if 0: use full length; if >0: cap to that many tokens
    seed: int,
    max_ctx: int | None,    # model context limit; if not None and sample is longer, keep the tail
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Local JSON/JSONL with 'txt'; use full sample by default; supervise only the last token.
       If seqlen>0, cap to seqlen. If max_ctx is set and text is longer, keep the last max_ctx tokens.
    """
    assert data_path and os.path.exists(data_path), f"Missing --data_path: {data_path}"
    set_seed(seed)

    # load JSONL or JSON array
    recs = []
    with open(data_path, "r", encoding="utf-8") as f:
        first = f.readline()
        try:  # JSONL
            obj = json.loads(first) if first.strip() else None
            if obj and "txt" in obj and isinstance(obj["txt"], str):
                recs.append(obj)
            for line in f:
                if line.strip():
                    o = json.loads(line)
                    if "txt" in o and isinstance(o["txt"], str):
                        recs.append(o)
        except json.JSONDecodeError:  # JSON array
            f.seek(0)
            recs = [r for r in json.load(f) if "txt" in r and isinstance(r["txt"], str)]
    assert recs, "No valid records with key 'txt'"

    # deterministic subset
    idxs = list(range(len(recs)))
    random.shuffle(idxs)
    idxs = idxs[:min(nsamples, len(idxs))]

    loader = []
    for i in idxs:
        text = recs[i]["txt"]
        enc = tokenizer(text, return_tensors="pt", add_special_tokens=False)  # no pad, no truncation
        ids = enc.input_ids  # [1, L]

        # respect model context (keep tail so last token is preserved)
        if max_ctx is not None and ids.shape[1] > max_ctx:
            ids = ids[:, -max_ctx:]

        # optional manual cap via --seqlen (>0 means cap)
        if seqlen and seqlen > 0 and ids.shape[1] > seqlen:
            # keep tail so the final token of the sample remains included
            ids = ids[:, -seqlen:]

        # build labels: only last token supervised
        labels = torch.full_like(ids, -100)
        if ids.shape[1] > 0:
            labels[0, -1] = ids[0, -1]

        loader.append((ids, labels))
    return loader


# ---------------- Main ----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF model path/name")
    ap.add_argument("--dataset", choices=["c4","tom"], required=True)
    ap.add_argument("--data_path", default=None, help="Path for dataset=tom")
    ap.add_argument("--nsamples", type=int, default=100)
    ap.add_argument("--seqlen", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache_dir", required=True)
    args = ap.parse_args()

    set_seed(args.seed)

    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = transformers.AutoModelForCausalLM.from_pretrained(
        args.model, cache_dir=args.cache_dir, device_map="auto",
        trust_remote_code=True, torch_dtype=torch.float16,
    )
    model.eval()

    mtype = parse_model_type(args.model)
    inner = model.model if hasattr(model, "model") else model
    layers = inner.layers if mtype in ("llama","mistral") else inner.decoder.layers

    # read model context limit if available (keeps tail when trimming)
    max_ctx = getattr(model.config, "max_position_embeddings", None)

    # grad hook: g -> g^2
    def sq(g): return g.square()
    for ly in layers:
        for mod in get_modules_from_layer(ly, mtype):
            mod.weight.register_hook(sq)

    # data
    if args.dataset == "c4":
        loader = build_c4_loader(tokenizer, args.nsamples, args.seqlen, args.seed)
    else:
        loader = build_tom_loader(tokenizer, args.data_path, args.nsamples, args.seqlen, args.seed, max_ctx)

    # accumulate
    model.zero_grad(set_to_none=True)
    for inp, labels in tqdm(loader, desc="Accumulating grad^2"):
        inp = inp.to(model.device); labels = labels.to(model.device)
        out = model(input_ids=inp, labels=labels)
        out.loss.backward()

    # overwrite weights with grads
    for ly in layers:
        for mod in get_modules_from_layer(ly, mtype):
            if mod.weight.grad is not None:
                mod.weight.data = mod.weight.grad

    os.makedirs(args.out, exist_ok=True)
    print(f"[INFO] Saving squared gradients to {args.out}")
    model.save_pretrained(args.out)
    print("[DONE]")

if __name__ == "__main__":
    main()
