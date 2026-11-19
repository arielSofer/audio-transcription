"""Kill-test for "diffusion LM + interlaced deep layers" (an agent's idea, run here on a real model).

Idea: a masked-diffusion LM decodes a block in S denoising steps, each step is a full forward over
the whole sequence. If the DEEP layers' contribution barely changes between steps, only the shallow
layers have to run every step; the deep layers (whose weights would be streamed from an SSD) are
re-run only every k steps and their last contribution ("deep delta" = h_L - h_m) is reused.

Two measurements on dllm-hub/Qwen3-0.6B-diffusion-mdlm-v0.1 (loaded with the STANDARD Qwen3 classes;
this model is Qwen3 with bidirectional attention, so we pass an all-visible 4D mask and never execute
the repo's custom code):
  1. how much the deep delta changes between steps (relative L2), split by position type
  2. generation quality when the deep half is refreshed only every k steps:
       token agreement with the always-refresh decode, and per-token NLL of the generated text under an
       independent autoregressive model (Qwen3-0.6B), plus the fraction of layer evaluations saved.

usage: python diffusion_deepcache.py --dllm DIR --ar DIR --out results/diffusion_deepcache.json
"""
from __future__ import annotations

import argparse
import json
import math
import time

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

PROMPTS = [
    "What is the capital of Australia, and what is it famous for?",
    "Write a Python function that returns the n-th Fibonacci number.",
    "Explain in two sentences why the sky is blue.",
    "List three benefits of regular exercise.",
    "Translate 'good morning, how are you?' into French and Spanish.",
    "What is 17 times 23? Show the steps.",
    "Give me a short tip for learning a new language.",
    "Summarize the plot of Romeo and Juliet in two sentences.",
]
EOS = 151645  # <|im_end|>


def layer_out(layer, h, mask, pos, pe):
    out = layer(h, attention_mask=mask, position_ids=pos, position_embeddings=pe)
    return out[0] if isinstance(out, tuple) else out


@torch.no_grad()
def forward_full(model, ids, mask, m):
    mdl = model.model
    h = mdl.embed_tokens(ids)
    pos = torch.arange(ids.shape[1]).unsqueeze(0)
    pe = mdl.rotary_emb(h, pos)
    h_m = None
    for i, layer in enumerate(mdl.layers):
        if i == m:
            h_m = h
        h = layer_out(layer, h, mask, pos, pe)
    logits = model.lm_head(mdl.norm(h))
    return logits[0], h_m[0], h[0]


@torch.no_grad()
def forward_shallow(model, ids, mask, m, delta):
    mdl = model.model
    h = mdl.embed_tokens(ids)
    pos = torch.arange(ids.shape[1]).unsqueeze(0)
    pe = mdl.rotary_emb(h, pos)
    for layer in mdl.layers[:m]:
        h = layer_out(layer, h, mask, pos, pe)
    h = h[0] + delta
    return model.lm_head(mdl.norm(h)), None


@torch.no_grad()
def generate(model, prompt_ids, mask_id, new_tokens, steps, m, k, record=None):
    """Low-confidence-first unmasking, temperature 0. Deep half refreshed every k steps (k=1: never stale)."""
    n_layers = len(model.model.layers)
    L0 = prompt_ids.shape[1]
    L = L0 + new_tokens
    x = torch.cat([prompt_ids, torch.full((1, new_tokens), mask_id)], 1)
    mask4d = torch.zeros(1, 1, L, L)                     # additive mask, all positions visible (bidirectional)
    per_step = new_tokens // steps
    delta, cost = None, 0
    for s in range(steps):
        if k == 1 or s % k == 0 or delta is None:
            logits, h_m, h_L = forward_full(model, x, mask4d, m)
            delta = h_L - h_m
            cost += n_layers
        else:
            logits, _ = forward_shallow(model, x, mask4d, m, delta)
            cost += m
        masked = x[0] == mask_id
        if record is not None:
            record.append(dict(delta=delta.clone(), masked=masked.clone()))
        probs = F.softmax(logits.float(), -1)
        conf, x0 = probs.max(-1)
        conf = torch.where(masked, conf, torch.full_like(conf, -1.0))
        idx = conf.topk(per_step).indices
        x[0, idx] = x0[idx]
    return x[0, L0:], cost / (steps * n_layers)


def stability(records, L0):
    """Relative change of the deep delta between step s and s+lag, by position type at step s+lag."""
    out = {}
    for lag in (1, 2, 4, 8):
        acc = {"prompt": [], "still_masked": [], "newly_unmasked": [], "already_unmasked": []}
        cos = {k: [] for k in acc}
        for rec in records:
            for s in range(len(rec) - lag):
                d0, d1 = rec[s]["delta"], rec[s + lag]["delta"]
                m0, m1 = rec[s]["masked"], rec[s + lag]["masked"]
                rel = (d1 - d0).norm(dim=-1) / d0.norm(dim=-1).clamp_min(1e-9)
                cs = F.cosine_similarity(d1, d0, dim=-1)
                cats = {"prompt": torch.arange(len(m0)) < L0,
                        "still_masked": m0 & m1,
                        "newly_unmasked": m0 & ~m1,
                        "already_unmasked": ~m0 & (torch.arange(len(m0)) >= L0)}
                for k, sel in cats.items():
                    if sel.any():
                        acc[k].append(rel[sel].mean().item())
                        cos[k].append(cs[sel].mean().item())
        out[lag] = {k: dict(rel_change=sum(v) / len(v), cosine=sum(cos[k]) / len(cos[k])) for k, v in acc.items() if v}
    return out


@torch.no_grad()
def ar_nll(ar, prompt_ids, gen_ids):
    """Per-token NLL of the generated tokens (cut at the first <|im_end|>) under an autoregressive model."""
    g = gen_ids.tolist()
    if EOS in g:
        g = g[: g.index(EOS) + 1]
    g = torch.tensor(g).unsqueeze(0)
    ids = torch.cat([prompt_ids, g], 1)
    logits = ar(ids).logits[0, :-1].float()
    tgt = ids[0, 1:]
    lp = F.cross_entropy(logits, tgt, reduction="none")
    return lp[prompt_ids.shape[1] - 1:].mean().item(), g.shape[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dllm", required=True)
    ap.add_argument("--ar", required=True)
    ap.add_argument("--out", default="results/diffusion_deepcache.json")
    ap.add_argument("--new-tokens", type=int, default=32)
    ap.add_argument("--steps", type=int, default=16)
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    tok = AutoTokenizer.from_pretrained(a.dllm)
    model = AutoModelForCausalLM.from_pretrained(a.dllm, dtype=torch.float32).eval()
    ar = AutoModelForCausalLM.from_pretrained(a.ar, dtype=torch.float32).eval()
    mask_id = tok.mask_token_id
    n_layers = len(model.model.layers)

    prompts = []
    for q in PROMPTS:
        text = tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False,
                                       add_generation_prompt=True, enable_thinking=False)
        prompts.append(tok(text, return_tensors="pt", add_special_tokens=False).input_ids)

    # ---- sanity: manual layer-by-layer forward == model forward, and the mask really is bidirectional ----
    x = torch.cat([prompts[0], torch.full((1, 8), mask_id)], 1)
    L = x.shape[1]
    full_mask = torch.zeros(1, 1, L, L)
    ref = model(x, attention_mask=full_mask).logits[0]
    mine, _, _ = forward_full(model, x, full_mask, n_layers // 2)
    x2 = x.clone()
    x2[0, -1] = tok("a", add_special_tokens=False).input_ids[0]
    ref2 = model(x2, attention_mask=full_mask).logits[0]
    causal = model(x).logits[0]
    causal2 = model(x2).logits[0]
    print(f"manual vs model max|diff| = {(ref - mine).abs().max().item():.2e}", flush=True)
    print(f"bidirectional: changing the LAST token changes logits at position 0 by {(ref[0] - ref2[0]).abs().max().item():.3f} "
          f"(causal: {(causal[0] - causal2[0]).abs().max().item():.3f})", flush=True)

    # ---- baseline decode (always refresh) with recording ----
    records, base_out, base_nll = [], [], []
    t0 = time.time()
    for i, p in enumerate(prompts):
        rec = []
        g, _ = generate(model, p, mask_id, a.new_tokens, a.steps, n_layers // 2, 1, record=rec)
        records.append(rec)
        base_out.append(g)
        base_nll.append(ar_nll(ar, p, g)[0])
        if i < 3:
            print(f"[baseline {i}] {tok.decode(g, skip_special_tokens=True)!r}", flush=True)
    print(f"baseline done in {time.time() - t0:.0f}s, mean AR-NLL {sum(base_nll) / len(base_nll):.3f}", flush=True)
    res = {"baseline_ar_nll": sum(base_nll) / len(base_nll), "configs": []}
    per = [stability([r], prompts[i].shape[1]) for i, r in enumerate(records)]   # needs each prompt's length
    stab = {}
    for lag in (1, 2, 4, 8):
        stab[lag] = {}
        for cat in ("prompt", "still_masked", "newly_unmasked", "already_unmasked"):
            vals = [d[lag][cat] for d in per if lag in d and cat in d[lag]]
            if vals:
                stab[lag][cat] = dict(rel_change=sum(v["rel_change"] for v in vals) / len(vals),
                                      cosine=sum(v["cosine"] for v in vals) / len(vals))
    res["stability"] = stab
    print("\ndeep-delta relative change between step s and s+lag (lower = more reusable):")
    for lag, d in stab.items():
        print(f"  lag {lag}: " + "  ".join(f"{c}={v['rel_change']:.3f}" for c, v in d.items()), flush=True)

    # ---- quality with a stale deep half ----
    for m in (n_layers // 2, (3 * n_layers) // 4):
        for k in (2, 4):
            agree, nlls, cost = [], [], 0.0
            t0 = time.time()
            for i, p in enumerate(prompts):
                g, c = generate(model, p, mask_id, a.new_tokens, a.steps, m, k)
                agree.append((g == base_out[i]).float().mean().item())
                nlls.append(ar_nll(ar, p, g)[0])
                cost += c / len(prompts)
            cfg = dict(split_layer=m, refresh_every=k, layer_evals_vs_full=round(cost, 3),
                       token_agreement_with_baseline=sum(agree) / len(agree),
                       ar_nll=sum(nlls) / len(nlls), ar_nll_baseline=res["baseline_ar_nll"])
            res["configs"].append(cfg)
            print(json.dumps(cfg), f"({time.time() - t0:.0f}s)", flush=True)
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
