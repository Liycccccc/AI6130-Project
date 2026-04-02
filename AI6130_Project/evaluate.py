from __future__ import annotations

import argparse
import json
import os
import re
from typing import Any, Dict, Iterable, Optional

import pandas as pd


NUM_RE = re.compile(r"[-+]?\d+(?:,\d{3})*(?:\.\d+)?")
MC_RE = re.compile(r"\b([ABCDE])\b", re.IGNORECASE)


def iter_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _try_parse_constrained_json(text: str) -> Optional[str]:
    text = text.strip()
    if not text:
        return None
    # common: model adds extra text; grab first {...} block
    m = re.search(r"\{[\s\S]*?\}", text)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except Exception:
        return None
    if isinstance(obj, dict) and "answer" in obj:
        v = obj["answer"]
        if v is None:
            return None
        return str(v).strip()
    return None


def extract_answer(response_text: str, ground_truth: Any, prompt_style: str) -> tuple[Any, Any]:
    """
    Follow user's extraction logic:
    - If ground_truth is one of A-E: extract option letter (last match among patterns), else "NONE"
    - Else: extract numeric answer with prioritized patterns, else fallback last number, else inf
    Additionally, if prompt_style == "constrained", try JSON {"answer": ...} first.
    Returns (pred, gold) where pred/gold are both:
    - str for MC (e.g., "A") or "NONE"
    - float for numeric (or inf when extraction fails)
    """
    response_text = (response_text or "").strip()

    if prompt_style == "constrained":
        v = _try_parse_constrained_json(response_text)
        if v is not None:
            response_text = v.strip()

    gt_str = str(ground_truth).strip().upper()

    if gt_str in ["A", "B", "C", "D", "E"]:
        patterns = [
            r"(?i)(?:answer|option)(?:\s+is)?(?:\s*:)?\s*([A-E])\b",
            r"(?i)the correct answer is\s*([A-E])\b",
            r"(?i)therefore,.*?\b([A-E])\b",
            r"(?:^|\s)([A-E])(?:\s|$|\.|\,|:)",
        ]
        for pattern in patterns:
            matches = re.findall(pattern, response_text)
            if matches:
                return matches[-1].upper(), gt_str
        return "NONE", gt_str

    # numeric
    text = response_text.replace(",", "").replace("**", "").replace("$", "")
    patterns = [
        r"(?i)answer is\s*[:=]*\s*(-?\d+\.?\d*)",
        r"(?i)answer:\s*(-?\d+\.?\d*)",
    ]

    try:
        label_num = float(str(ground_truth).replace(",", "").replace("$", ""))
    except ValueError:
        label_num = float("inf")

    for pattern in patterns:
        matches = re.findall(pattern, text)
        if matches:
            try:
                predict_num = float(matches[-1])
                return predict_num, label_num
            except ValueError:
                continue

    pred = re.findall(r"-?\d+\.?\d*", text)
    if not pred:
        return float("inf"), label_num
    try:
        predict_num = float(pred[-1])
        return predict_num, label_num
    except ValueError:
        return float("inf"), label_num


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_jsonl", required=True)
    ap.add_argument("--out_csv", required=True)
    args = ap.parse_args()

    rows = []
    for r in iter_jsonl(args.pred_jsonl):
        answer_type = (r.get("answer_type") or "").strip()
        prompt_style = (r.get("prompt_style") or "").strip()
        pred, gold = extract_answer(r.get("raw_generation", ""), r.get("gold"), prompt_style)
        if isinstance(pred, float) and isinstance(gold, float):
            ok = (pred != float("inf")) and (gold != float("inf")) and (pred == gold)
        else:
            ok = (pred != "NONE") and (gold != "") and (str(pred).upper() == str(gold).upper())
        rows.append(
            {
                **r,
                "pred": pred,
                "gold_norm": gold,
                "correct": bool(ok),
            }
        )

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    df.to_csv(args.out_csv, index=False, encoding="utf-8")

    # Summary print
    if len(df) == 0:
        print("No rows.")
        return

    summary = (
        df.groupby(["model_id", "prompt_id", "dataset", "split"])["correct"]
        .mean()
        .reset_index()
        .rename(columns={"correct": "accuracy"})
        .sort_values(["model_id", "prompt_id", "dataset", "split"])
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

