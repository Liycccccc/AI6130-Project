from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, Iterable, List, Optional

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from prompts import get_prompt


def iter_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def write_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def is_chat_tokenizer(tokenizer) -> bool:
    if not hasattr(tokenizer, "apply_chat_template"):
        return False
    # Some tokenizers expose apply_chat_template but don't ship a chat_template,
    # which raises: "tokenizer.chat_template is not set".
    tpl = getattr(tokenizer, "chat_template", None)
    return bool(tpl)


def build_model_inputs(
    tokenizer,
    system: Optional[str],
    instruction: str,
    model_id: str,
) -> str:
    # Prefer native chat templates when available (e.g., TinyLlama-Chat, Qwen-Instruct).
    if is_chat_tokenizer(tokenizer):
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": instruction})
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    # Fallback: plain text "chat-like" format (works for both base & instruct LMs).
    if system:
        return (
            "### System\n"
            f"{system}\n\n"
            "### User\n"
            f"{instruction}\n\n"
            "### Assistant\n"
        )
    return f"{instruction}\n\nAnswer:"


@torch.inference_mode()
def generate_one(
    model,
    tokenizer,
    prompt_text: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> str:
    inputs = tokenizer(prompt_text, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    do_sample = temperature > 0
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature if do_sample else None,
        top_p=top_p if do_sample else None,
        pad_token_id=tokenizer.eos_token_id,
    )
    gen = out[0, inputs["input_ids"].shape[1] :]
    return tokenizer.decode(gen, skip_special_tokens=True).strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", required=True)
    ap.add_argument("--input_jsonl", required=True)
    ap.add_argument("--output_jsonl", required=True)
    ap.add_argument("--prompt_style", required=True, choices=["zero", "cot", "constrained"])
    ap.add_argument("--max_samples", type=int, default=0, help="0 = all")
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--log_every", type=int, default=10, help="Print progress every N samples")
    ap.add_argument("--dtype", default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    args = ap.parse_args()

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map.get(args.dtype, "auto")

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch_dtype,
        device_map="auto",
    )
    model.eval()

    rows_out: List[Dict[str, Any]] = []
    t0 = time.time()

    total = args.max_samples if args.max_samples and args.max_samples > 0 else None
    seen = 0
    for ex in tqdm(
        iter_jsonl(args.input_jsonl),
        desc="infer",
        total=total,
        file=sys.stdout,
        dynamic_ncols=True,
        mininterval=1.0,
        smoothing=0.0,
        leave=True,
    ):
        if total is not None and seen >= total:
            break
        seen += 1
        if args.log_every > 0 and (seen == 1 or seen % args.log_every == 0):
            ex_id = ex.get("id")
            ds = ex.get("dataset")
            sp = ex.get("split")
            print(f"[{seen}/{total if total is not None else '?'}] model={args.model_id} dataset={ds} split={sp} id={ex_id}", flush=True)
        spec = get_prompt(ex, args.prompt_style)
        prompt_text = build_model_inputs(tokenizer, spec.system, spec.instruction, args.model_id)
        gen_text = generate_one(
            model=model,
            tokenizer=tokenizer,
            prompt_text=prompt_text,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        rows_out.append(
            {
                "id": ex.get("id"),
                "dataset": ex.get("dataset"),
                "split": ex.get("split"),
                "answer_type": ex.get("answer_type"),
                "gold": ex.get("answer"),
                "prompt_id": spec.prompt_id,
                "prompt_style": spec.style,
                "model_id": args.model_id,
                "raw_generation": gen_text,
                "ts": time.time(),
            }
        )

    write_jsonl(args.output_jsonl, rows_out)
    dt = time.time() - t0
    print(f"Wrote {len(rows_out)} predictions to {args.output_jsonl} in {dt:.1f}s")


if __name__ == "__main__":
    main()

