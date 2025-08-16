#!/usr/bin/env python3
"""
ToM + Perplexity evaluation runner.

- Loads a HF CausalLM (repo id or local path).
- Applies a TOM-vs-C4 gradient-based mask per m in a sweep; saves masked models.
- Evaluates perplexity on WikiText-2 (raw v1, test split).
- Evaluates ToM (S1/S2/S3) with vLLM and writes CSVs.
"""

import argparse, os, gc, re, json, math, importlib.util
from typing import List, Tuple
from pathlib import Path

import numpy as np
import torch
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams


# ---------------- Perplexity ----------------

def eval_perplexity(model, tokenizer, seqlen: int = 2048) -> float:
    """Perplexity on WikiText-2 (raw v1), test split."""
    losses = []
    model.config.use_cache = False

    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(ds["text"])
    enc = tokenizer(text, return_tensors="pt")
    # enc.input_ids: [1, L]
    total = enc.input_ids.numel()
    nsamples = total // seqlen
    if nsamples == 0:
        return float("nan")

    enc.input_ids = enc.input_ids[:, :nsamples * seqlen].reshape(nsamples, seqlen)

    for idx in tqdm(torch.utils.data.DataLoader(torch.arange(nsamples), batch_size=2),
                    desc="Perplexity"):
        with torch.no_grad():
            batch = enc.input_ids[idx].to(model.device)
            out = model(input_ids=batch, labels=batch)
            losses.append(out.loss.item())

    eval_loss = float(torch.tensor(losses).mean())
    return math.exp(eval_loss)


# ---------------- ToM prompt builders (S1/S2/S3) ----------------

def build_s1_txt(tsk, typ="txt", reverse=False):
    if typ in ["txt_open"]:
        if tsk["NUMBER"] == 1:
            txt = tsk["txt"] + " XNAM connects to the CX and explores its contents. XNAM reads the label on the CX."
        elif tsk["NUMBER"] == 17:
            txt = tsk["txt"] + " XNAM puts one in a CD player and listens to the songs. XNAM can can clearly hear that it is full of S2 music."
        else:
            txt = tsk["txt"] + " XNAM opens the CX and looks inside. XNAM reads the label."
    else:
        if typ in ["txt_correctlabel"]:
            if tsk["NUMBER"] == 1:
                txt = tsk["txt_correctlabel"] + " XNAM does not connect to the CX and does not explore its contents. XNAM reads the label on the CX. "
            elif tsk["NUMBER"] == 17:
                txt = tsk["txt_correctlabel"] + " XNAM does not open the box and does not listen to any of the CDs. XNAM reads the label."
            else:
                txt = tsk["txt_correctlabel"] + " XNAM does not open the CX and does not look inside. XNAM reads the label."
        if typ in ["txt_informed", "txt"]:
            if tsk["NUMBER"] == 1:
                txt = tsk["txt"] + " XNAM does not connect to the CX and does not explore its contents. XNAM reads the label on the CX. "
            elif tsk["NUMBER"] == 17:
                txt = tsk["txt"] + " XNAM does not open the box and does not listen to any of the CDs. XNAM reads the label."
            else:
                txt = tsk["txt"] + " XNAM does not open the CX and does not look inside. XNAM reads the label."
        if typ in ["txt_informed"]:
            txt = txt + ' A cousin calls XNAM and tells OBJ_PRO that the CX has S1 in it, and that XPRO should ignore the label that says "S2". XNAM believes OBJ_PRO cousin.'

    txt = (txt.replace("S1", tsk["o1"]).replace("S2", tsk["o2"]).replace("CX", tsk["c"])
              .replace("XNAM", tsk["xnam"]).replace("XPRO", tsk["xpro"])
              .replace("OBJ_PRO", tsk["obj_pro"]).replace("POS_PRO", tsk["pos_pro"])
              .replace(". h", ". H").replace(". s", ". S"))

    q1 = tsk["xpro"].capitalize() + " opens the " + tsk["c"] + " and looks inside. " + tsk["xpro"].capitalize() + " can clearly see that it is full of"
    q2 = tsk["xnam"] + " calls a friend to tell them that " + tsk["xpro"] + " has just found a " + tsk["c"] + " full of"
    if tsk["NUMBER"] == 17:
        q1 = tsk["xpro"].capitalize() + " puts one in a CD player and listens to the songs. " + tsk["xpro"].capitalize() + " can clearly hear that it is full of"
    if tsk["NUMBER"] == 14:
        q2 = q2.replace("found", "received")
    if tsk["NUMBER"] == 12:
        q2 = q2.replace("found", "received")

    if reverse:
        txt = txt.replace(tsk["o1"], "####").replace(tsk["o2"], tsk["o1"]).replace("####", tsk["o2"])

    expected = [tsk["o1"], tsk["o2"]]
    if typ in ["txt_transparent", "txt_correctlabel", "txt_informed", "txt_open", "txt_present"]:
        expected = [tsk["o1"], tsk["o1"]]
    elif typ in ["txt_notransfer", "txt_stayed"]:
        expected = [tsk["o2"], tsk["o2"]]
    if reverse:
        expected = [w.replace(tsk["o1"], "####").replace(tsk["o2"], tsk["o1"]).replace("####", tsk["o2"]) for w in expected]

    expected_tokens = [w.split()[0] for w in expected]
    typ2key = {"txt": "false belief", "txt_correctlabel": "correct label", "txt_informed": "informed protagonist", "txt_open": "open container"}
    base_key = typ2key.get(typ, "false belief")
    result_key = f"{base_key} {'(T)' if reverse else '(F)'}"
    return txt, q1, q2, expected_tokens, result_key

def build_s2_txt(tsk, typ="txt", reverse=False):
    txt = tsk["txt"] + " " + tsk["txt_informed"] if typ == "txt_informed" else tsk.get(typ, tsk["txt"])
    q1, q2 = tsk["q1"], tsk["q2"]
    if reverse:
        txt = txt.replace(tsk["o1"], "####").replace(tsk["o2"], tsk["o1"]).replace("####", tsk["o2"])
    expected = [tsk["o1"], tsk["o2"]]
    if typ in ["txt_notransfer"]:
        expected = [tsk["o2"], tsk["o2"]]
    elif typ in ["txt_transparent", "txt_correctlabel", "txt_informed", "txt_open", "txt_present", "txt_stayed"]:
        expected = [tsk["o1"], tsk["o1"]]
    if reverse:
        expected = [w.replace(tsk["o1"], "####").replace(tsk["o2"], tsk["o1"]).replace("####", tsk["o2"]) for w in expected]
    expected_tokens = [w.split()[0] for w in expected]
    typ2key = {"txt": "false belief", "txt_notransfer": "no transfer", "txt_informed": "informed protagonist", "txt_present": "present protagonist"}
    base_key = typ2key.get(typ, "false belief")
    result_key = f"{base_key} {'(T)' if reverse else '(F)'}"
    return txt, q1, q2, expected_tokens, result_key

def build_s3_txt(tsk, typ="txt", reverse=False):
    txt = tsk["txt_reverse"] if typ == "txt_reverse" else tsk["txt"]
    expected = ["Yes"] if reverse else ["No"]
    expected_tokens = [w.split()[0] for w in expected]
    result_key = "Irony reversed" if reverse else "Irony"
    return txt, expected_tokens, result_key

def make_prompts_s1(tsks):
    types = ["txt", "txt_correctlabel", "txt_informed", "txt_open"]
    all_prompts, meta = [], []
    for i, tsk in enumerate(tsks):
        for typ in types:
            for rev in [False, True]:
                final_txt, q1, q2, exps, rkey = build_s1_txt(tsk, typ, reverse=rev)
                p1 = f"Complete the following story: {final_txt} {q1}"
                p2 = f"Complete the following story: {final_txt} {q2}"
                all_prompts += [p1, p2]
                meta += [{"task_idx": i, "result_key": rkey, "expected": exps[0]},
                         {"task_idx": i, "result_key": rkey, "expected": exps[1]}]
    return all_prompts, meta

def make_prompts_s2(tsks):
    types = ["txt", "txt_notransfer", "txt_informed", "txt_present"]
    all_prompts, meta = [], []
    for i, tsk in enumerate(tsks):
        for typ in types:
            for rev in [False, True]:
                final_txt, q1, q2, exps, rkey = build_s2_txt(tsk, typ, reverse=rev)
                p1 = f"Complete the following story: {final_txt} {q1}"
                p2 = f"Complete the following story: {final_txt} {q2}"
                all_prompts += [p1, p2]
                meta += [{"task_idx": i, "result_key": rkey, "expected": exps[0]},
                         {"task_idx": i, "result_key": rkey, "expected": exps[1]}]
    return all_prompts, meta

def make_prompts_s3(tsks):
    all_prompts, meta = [], []
    for i, tsk in enumerate(tsks):
        txt, exps, rkey = build_s3_txt(tsk, typ="txt", reverse=False)
        p = f"Read the following story: {txt} Correct Answer:"
        all_prompts.append(p); meta.append({"task_idx": i, "result_key": rkey, "expected": exps[0]})
        txt2, exps2, rkey2 = build_s3_txt(tsk, typ="txt_reverse", reverse=True)
        p2 = f"Read the following story: {txt2} Correct Answer:"
        all_prompts.append(p2); meta.append({"task_idx": i, "result_key": rkey2, "expected": exps2[0]})
    return all_prompts, meta

def generate_in_batches(prompts, llm, batch_size=32):
    out = []
    params = SamplingParams(max_tokens=5, temperature=0.6, top_p=0.9)
    for start in range(0, len(prompts), batch_size):
        outputs = llm.generate(prompts[start:start+batch_size], params)
        for o in outputs:
            out.append(o.outputs[0].text)
    return out

def _first_word(s: str) -> str:
    toks = s.strip().split()
    return re.sub(r'[.,\'";]', '', toks[0]) if toks else ""

def evaluate_unexpected(tsks, llm, batch_size=32):
    prompts, meta = make_prompts_s1(tsks)
    outs = generate_in_batches(prompts, llm, batch_size=batch_size)
    n = len(tsks)
    keys = ["false belief (F)", "false belief (T)", "correct label (F)", "correct label (T)",
            "informed protagonist (F)", "informed protagonist (T)", "open container (F)", "open container (T)"]
    res = {k: [1]*n for k in keys}
    for i, text in enumerate(outs):
        info = meta[i]; t = info["task_idx"]; k = info["result_key"]; exp = info["expected"]
        if _first_word(text) != exp: res[k][t] = 0
    return res

def evaluate_transfer(tsks, llm, batch_size=32):
    prompts, meta = make_prompts_s2(tsks)
    outs = generate_in_batches(prompts, llm, batch_size=batch_size)
    n = len(tsks)
    keys = ["false belief (F)", "false belief (T)", "no transfer (F)", "no transfer (T)",
            "informed protagonist (F)", "informed protagonist (T)", "present protagonist (F)", "present protagonist (T)"]
    res = {k: [1]*n for k in keys}
    for i, text in enumerate(outs):
        info = meta[i]; t = info["task_idx"]; k = info["result_key"]; exp = info["expected"]
        if _first_word(text) != exp: res[k][t] = 0
    return res

def evaluate_irony(tsks, llm, batch_size=32):
    prompts, meta = make_prompts_s3(tsks)
    outs = generate_in_batches(prompts, llm, batch_size=batch_size)
    n = len(tsks)
    res = {"Irony": [1]*n, "Irony reversed": [1]*n}
    for i, text in enumerate(outs):
        info = meta[i]; t = info["task_idx"]; k = info["result_key"]; exp = info["expected"]
        if _first_word(text) != exp: res[k][t] = 0
    return res


# ---------------- Task loading ----------------

def load_tasks(path: str):
    p = Path(path)
    if p.suffix.lower() == ".json":
        with open(p, "r", encoding="utf-8") as f:
            obj = json.load(f)
        return obj["s1"], obj["s2"], obj["s3"]
    if p.suffix.lower() == ".py":
        spec = importlib.util.spec_from_file_location("tasks_mod", str(p))
        mod = importlib.util.module_from_spec(spec)  # type: ignore
        assert spec and spec.loader
        spec.loader.exec_module(mod)  # type: ignore
        return mod.tsks1, mod.tsks2, mod.tsks3
    raise ValueError("Unsupported tasks file. Use .json or .py")


# ---------------- Mask application ----------------

LLAMA_LIN = ["q","k","v","o","gate","up","down"]
LLAMA_NAME = {
    "q": "self_attn.q_proj.weight",
    "k": "self_attn.k_proj.weight",
    "v": "self_attn.v_proj.weight",
    "o": "self_attn.o_proj.weight",
    "gate": "mlp.gate_proj.weight",
    "up": "mlp.up_proj.weight",
    "down": "mlp.down_proj.weight",
}
ALT_KEYS = {
    "q": "q_proj", "k": "k_proj", "v": "v_proj", "o": "o_proj",
    "gate": "gate_proj", "up": "up_proj", "down": "down_proj",
}

def _infer_family(name: str, model) -> str:
    n = (name or "").lower()
    if "opt" in n: return "opt"
    if "mistral" in n: return "mistral"
    return "llama"

def _get_layers(model, family: str):
    inner = model.model if hasattr(model, "model") else model
    if family in ("llama","mistral"): return inner.layers
    if family == "opt": return inner.decoder.layers
    raise NotImplementedError

def _get_chunk_tensor(blob: dict, key: str):
    if key in blob: return blob[key]
    alt = ALT_KEYS.get(key)
    if alt and alt in blob: return blob[alt]
    raise KeyError(f"Missing key '{key}' (or '{alt}') in chunk file.")

def apply_mask_and_save(base_model_id: str,
                        grad_tom_dir: str,
                        grad_c4_dir: str,
                        m_value: float,
                        out_dir: str,
                        cache_dir: str | None,
                        device_map: str = "auto",
                        scale: float = 0.0) -> str:
    os.makedirs(out_dir, exist_ok=True)
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id, trust_remote_code=True, torch_dtype=torch.float16,
        device_map=device_map, cache_dir=cache_dir
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
    family = _infer_family(base_model_id, model)
    layers = _get_layers(model, family)

    for i, layer in enumerate(tqdm(layers, desc=f"Mask m={m_value}")):
        tom_path = os.path.join(grad_tom_dir, f"layer_{i}.pt")
        c4_path  = os.path.join(grad_c4_dir,  f"layer_{i}.pt")
        if not (os.path.exists(tom_path) and os.path.exists(c4_path)):
            raise FileNotFoundError(f"Missing chunk files for layer {i}: {tom_path} or {c4_path}")
        g1 = torch.load(tom_path, map_location="cpu")
        g2 = torch.load(c4_path,  map_location="cpu")

        for s in LLAMA_LIN:
            pname = LLAMA_NAME[s]
            w = layer.state_dict()[pname]
            dev = w.device
            gweight1 = _get_chunk_tensor(g1, s).to(dev)
            gweight2 = _get_chunk_tensor(g2, s).to(dev)

            num_outliers = int(gweight1.numel() * float(m_value))
            if num_outliers <= 0:
                mask1 = torch.zeros_like(gweight1, dtype=torch.bool)
                mask2 = torch.zeros_like(gweight2, dtype=torch.bool)
            else:
                thr1 = gweight1.reshape(-1).topk(k=num_outliers).values[-1]
                thr2 = gweight2.reshape(-1).topk(k=num_outliers).values[-1]
                mask1 = gweight1 > thr1
                mask2 = gweight2 > thr2

            mask = mask1 & (~mask2)
            original = w
            no_outlier = original * (~mask)
            mean_no = no_outlier.mean()
            new_diff = original - mean_no
            scaled = new_diff * scale
            adjusted = scaled + mean_no
            final = no_outlier + adjusted * mask
            layer.state_dict()[pname].copy_(final)

    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    return out_dir


# ---------------- vLLM ToM evaluation ----------------

def run_tom_eval(model_path: str,
                 tasks_s1, tasks_s2, tasks_s3,
                 out_root: str, tensor_parallel_size: int,
                 max_model_len: int, batch_size: int,
                 reps: int, m_tag: str, cache_dir: str):
    llm = LLM(model=model_path, max_model_len=max_model_len,
              tensor_parallel_size=tensor_parallel_size, gpu_memory_utilization=0.8, dtype="float16",
                  download_dir=cache_dir, enforce_eager=True)
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)  # noqa: F841

    for rep in range(1, reps + 1):
        for tname in ["unexpected", "transfer", "irony"]:
            out_csv = os.path.join(out_root, "tom", str(rep), m_tag, tname, "results.csv")
            os.makedirs(os.path.dirname(out_csv), exist_ok=True)
            if os.path.isfile(out_csv):
                print(f"[skip] {out_csv} exists")
                continue

            if tname == "unexpected":
                res = evaluate_unexpected(tasks_s1, llm, batch_size=batch_size)
            elif tname == "transfer":
                res = evaluate_transfer(tasks_s2, llm, batch_size=batch_size)
            else:
                res = evaluate_irony(tasks_s3, llm, batch_size=batch_size)

            pd.DataFrame(res).to_csv(out_csv, index=True)
            total = sum(sum(v) for v in res.values())
            print(f"[ToM] rep={rep} {tname} total={total} -> {out_csv}")

    del llm
    gc.collect()
    torch.cuda.empty_cache()


# ---------------- CLI ----------------

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--grad_tom_chunks", required=True)
    ap.add_argument("--grad_c4_chunks", required=True)
    ap.add_argument("--tom_tasks", required=True, help=".json with keys s1/s2/s3 or a .py exporting tsks1/2/3")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--cache_dir", required=True)
    ap.add_argument("--tensor_parallel_size", type=int, default=4)
    ap.add_argument("--max_model_len", type=int, default=8192)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--scale", type=float, default=0.0)
    ap.add_argument("--device_map", default="auto")
    ap.add_argument("--ppl_seqlen", type=int, default=2048)

    m_group = ap.add_mutually_exclusive_group(required=True)
    m_group.add_argument("--m_list", type=str, help="comma-separated, e.g. 0,2e-5,5e-5")
    m_group.add_argument("--m_start", type=float, help="range start")
    ap.add_argument("--m_end", type=float, default=None)
    ap.add_argument("--m_step", type=float, default=None)
    return ap.parse_args()


def main():
    args = parse_args()
    tasks_s1, tasks_s2, tasks_s3 = load_tasks(args.tom_tasks)

    if args.m_list:
        m_values = [float(x) for x in args.m_list.split(",")]
    else:
        assert args.m_end is not None and args.m_step is not None
        m_values = list(np.arange(args.m_start, args.m_end + args.m_step, args.m_step))

    os.makedirs(args.out_dir, exist_ok=True)
    ppl_csv = os.path.join(args.out_dir, "perplexity_results.csv")
    ppl_rows = []
    base_model_id = args.model

    for m in m_values:
        m_tag = f"{m}"
        print(f"\n=== m={m_tag} ===")
        masked_dir = os.path.join(args.out_dir, "models", f"m_{m_tag}")
        if m == 0.0:
            masked_model_path = base_model_id
        else:
            masked_model_path = apply_mask_and_save(
                base_model_id=base_model_id,
                grad_tom_dir=args.grad_tom_chunks,
                grad_c4_dir=args.grad_c4_chunks,
                m_value=m,
                out_dir=masked_dir,
                cache_dir=args.cache_dir,
                device_map=args.device_map,
                scale=args.scale,
            )

        print("[ppl] loading model...")
        model = AutoModelForCausalLM.from_pretrained(
            masked_model_path, trust_remote_code=True,
            torch_dtype=torch.float16, device_map=args.device_map,
            cache_dir=args.cache_dir
        )
        tok = AutoTokenizer.from_pretrained(masked_model_path, trust_remote_code=True)
        tok.pad_token = tok.eos_token
        ppl = eval_perplexity(model, tok, seqlen=args.ppl_seqlen)
        print(f"[ppl] m={m_tag} perplexity={ppl:.4f}")
        ppl_rows.append({"m": m, "perplexity": ppl})
        del model
        gc.collect()
        torch.cuda.empty_cache()

        run_tom_eval(masked_model_path, tasks_s1, tasks_s2, tasks_s3,
                     out_root=args.out_dir,
                     tensor_parallel_size=args.tensor_parallel_size,
                     max_model_len=args.max_model_len,
                     batch_size=args.batch_size, reps=args.reps, m_tag=m_tag, cache_dir=args.cache_dir)

    pd.DataFrame(ppl_rows).to_csv(ppl_csv, index=False)
    print(f"[DONE] Perplexity summary -> {ppl_csv}")


if __name__ == "__main__":
    main()
