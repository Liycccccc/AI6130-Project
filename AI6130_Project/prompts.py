from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PromptSpec:
    prompt_id: str
    style: str  # "zero_shot" | "cot" | "constrained"
    system: str | None
    instruction: str


def _base_instruction(example: dict) -> str:
    q = (example.get("question") or "").strip()
    dataset = (example.get("dataset") or "").strip()
    answer_type = (example.get("answer_type") or "").strip()

    header = []
    if dataset:
        header.append(f"Dataset: {dataset}")
    if answer_type:
        header.append(f"Answer type: {answer_type}")
    header_txt = "\n".join(header)

    if header_txt:
        return f"{header_txt}\n\nQuestion:\n{q}"
    return f"Question:\n{q}"


def get_prompt(example: dict, style: str) -> PromptSpec:
    """
    Returns a prompt spec with (optional) system message and a single user instruction.
    """
    style = style.strip().lower()
    base = _base_instruction(example)

    if style in {"zero", "zero-shot", "zeroshot", "zero_shot"}:
        return PromptSpec(
            prompt_id="zero_shot_v1",
            style="zero_shot",
            system="You are a helpful assistant that solves math and reasoning problems.",
            instruction=(
                f"{base}\n\n"
                "Give the final answer only."
            ),
        )

    if style in {"cot", "chain-of-thought", "chain_of_thought"}:
        return PromptSpec(
            prompt_id="cot_v1",
            style="cot",
            system="You are a careful assistant. Think step-by-step to solve the problem.",
            instruction=(
                f"{base}\n\n"
                "Solve step-by-step, then give the final answer on a new line starting with:\n"
                "Answer: "
            ),
        )

    if style in {"constrained", "format", "json"}:
        return PromptSpec(
            prompt_id="constrained_json_v1",
            style="constrained",
            system="You are a helpful assistant. Follow the output format strictly.",
            instruction=(
                f"{base}\n\n"
                "Return ONLY a single-line JSON object with exactly one key \"answer\".\n"
                "Rules:\n"
                "- If the answer is a number, output it without extra words.\n"
                "- If it is multiple-choice, output only the option letter (A/B/C/D/E).\n"
                "Output format example:\n"
                "{\"answer\":\"42\"}\n"
            ),
        )

    raise ValueError(f"Unknown style: {style}")

