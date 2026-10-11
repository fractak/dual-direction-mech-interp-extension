"""Dataset loaders for safety and epistemic direction extraction.

Loads from HuggingFace Hub datasets:
  - Safety: AdvBench (harmful) + Alpaca (benign)
  - Epistemic: SQuAD v2 unanswerable + TriviaQA answerable (no-context)
"""
from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PromptSample:
    """A single prompt with metadata for direction extraction."""

    id: str
    prompt: str
    group: str           # 'harmful', 'benign', 'unanswerable', 'answerable', 'incapable', 'capable'
    direction_type: str  # 'safety' | 'epistemic' | 'capability'
    source: str          # dataset name
    metadata: Dict[str, Any] = field(default_factory=dict)


def load_safety_prompts(
    limit: int = 200,
    seed: int = 42,
) -> Dict[str, List[PromptSample]]:
    """Load harmful (AdvBench) and benign (Alpaca) prompts for safety direction.

    Returns: {'harmful': [...], 'benign': [...]}
    """
    from datasets import load_dataset

    # Load harmful prompts from AdvBench
    advbench = load_dataset("ivnle/advbench_harmful_behaviors", split="train")
    harmful_prompts = [
        PromptSample(
            id=f"advbench_{i}",
            prompt=row["instruction"],
            group="harmful",
            direction_type="safety",
            source="advbench",
        )
        for i, row in enumerate(advbench)
    ]
    logger.info("Loaded %d harmful prompts from AdvBench", len(harmful_prompts))

    # Load benign prompts from Alpaca
    alpaca = load_dataset("tatsu-lab/alpaca", split="train")
    benign_prompts = [
        PromptSample(
            id=f"alpaca_{i}",
            prompt=row["instruction"],
            group="benign",
            direction_type="safety",
            source="alpaca",
        )
        for i, row in enumerate(alpaca)
        if row["instruction"].strip()  # skip empty instructions
    ]
    logger.info("Loaded %d benign prompts from Alpaca", len(benign_prompts))

    # Balance and limit
    rng = random.Random(seed)
    half = limit // 2

    if len(harmful_prompts) > half:
        harmful_prompts = rng.sample(harmful_prompts, half)
    if len(benign_prompts) > half:
        benign_prompts = rng.sample(benign_prompts, half)

    return {"harmful": harmful_prompts, "benign": benign_prompts}


def load_epistemic_prompts(
    limit: int = 200,
    seed: int = 42,
) -> Dict[str, List[PromptSample]]:
    """Load unanswerable (SQuAD v2) and answerable (TriviaQA) prompts.

    Returns: {'unanswerable': [...], 'answerable': [...]}

    For unanswerable: SQuAD v2 questions with empty answer spans.
    For answerable: TriviaQA questions (no-context variant, model must use parametric knowledge).
    """
    from datasets import load_dataset

    # Load unanswerable questions from SQuAD v2
    squad = load_dataset("rajpurkar/squad_v2", "squad_v2", split="validation")
    unanswerable_prompts = []
    for i, row in enumerate(squad):
        answers = row.get("answers", {})
        answer_texts = answers.get("text", []) if isinstance(answers, dict) else []
        if len(answer_texts) == 0 or all(t == "" for t in answer_texts):
            unanswerable_prompts.append(PromptSample(
                id=f"squad2_unanswerable_{i}",
                prompt=_format_qa_prompt(row["question"], row.get("context", "")),
                group="unanswerable",
                direction_type="epistemic",
                source="squad_v2",
                metadata={"context": row.get("context", ""), "question": row["question"]},
            ))
    logger.info("Loaded %d unanswerable prompts from SQuAD v2", len(unanswerable_prompts))

    # Load answerable questions from TriviaQA (no context — tests parametric knowledge)
    trivia = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split="validation")
    answerable_prompts = []
    for i, row in enumerate(trivia):
        answer = row.get("answer", {})
        aliases = answer.get("aliases", []) if isinstance(answer, dict) else []
        if aliases:
            answerable_prompts.append(PromptSample(
                id=f"triviaqa_{i}",
                prompt=_format_qa_prompt_nocontext(row["question"]),
                group="answerable",
                direction_type="epistemic",
                source="trivia_qa",
                metadata={"answer_aliases": aliases, "question": row["question"]},
            ))
    logger.info("Loaded %d answerable prompts from TriviaQA", len(answerable_prompts))

    # Balance and limit
    rng = random.Random(seed)
    half = limit // 2

    if len(unanswerable_prompts) > half:
        unanswerable_prompts = rng.sample(unanswerable_prompts, half)
    if len(answerable_prompts) > half:
        answerable_prompts = rng.sample(answerable_prompts, half)

    return {"unanswerable": unanswerable_prompts, "answerable": answerable_prompts}


def _format_qa_prompt(question: str, context: str) -> str:
    """Format a QA prompt with context (SQuAD-style)."""
    if context:
        return (
            f"Based on the following passage, answer the question. "
            f"If the answer cannot be determined from the passage, say "
            f"\"I don't know\" or \"The answer is not in the passage.\"\n\n"
            f"Passage: {context[:1000]}\n\n"
            f"Question: {question}"
        )
    return f"Answer the following question: {question}"


def _format_qa_prompt_nocontext(question: str) -> str:
    """Format a QA prompt without context (parametric knowledge test)."""
    return f"Answer the following question concisely: {question}"



def load_capability_prompts(
    limit: int = 200,
    seed: int = 42,
    split: str = "all",
) -> Dict[str, List[PromptSample]]:
    """Load matched capability-impossible and capability-possible prompts.

    split:
        'train' -> 16 held-in task families / 80 pairs
        'eval'  -> 4 held-out task families / 20 pairs
        'all'   -> original full 100-pair dataset

    Returns: {'incapable': [...], 'capable': [...]}
    """
    import json
    from pathlib import Path

    filenames = {
        "all": "capability_pairs.json",
        "train": "capability_train_pairs.json",
        "eval": "capability_eval_pairs.json",
        "expanded_eval": "capability_eval_expanded.json",
    }

    if split not in filenames:
        raise ValueError(
            f"Unknown capability split {split!r}; "
            f"expected one of {sorted(filenames)}"
        )

    data_path = (
        Path(__file__).resolve().parents[3]
        / "data"
        / filenames[split]
    )

    if not data_path.exists():
        raise FileNotFoundError(f"Capability dataset not found: {data_path}")

    pairs = json.loads(data_path.read_text())

    rng = random.Random(seed)
    num_pairs = min(limit // 2, len(pairs))

    if num_pairs < len(pairs):
        pairs = rng.sample(pairs, num_pairs)

    incapable_prompts = []
    capable_prompts = []

    for i, pair in enumerate(pairs):
        pair_id = pair.get("id", str(i))

        incapable_prompts.append(
            PromptSample(
                id=f"capability_incapable_{pair_id}",
                prompt=pair["incapable"],
                group="incapable",
                direction_type="capability",
                source=f"matched_capability_pairs_{split}",
                metadata={"pair_id": pair_id, "split": split},
            )
        )

        capable_prompts.append(
            PromptSample(
                id=f"capability_capable_{pair_id}",
                prompt=pair["capable"],
                group="capable",
                direction_type="capability",
                source=f"matched_capability_pairs_{split}",
                metadata={"pair_id": pair_id, "split": split},
            )
        )

    logger.info(
        "Loaded capability split=%s: %d matched pairs (%d prompts)",
        split,
        len(pairs),
        len(incapable_prompts) + len(capable_prompts),
    )

    return {
        "incapable": incapable_prompts,
        "capable": capable_prompts,
    }


def load_all_prompts(
    safety_limit: int = 200,
    epistemic_limit: int = 200,
    capability_limit: int = 50,
    seed: int = 42,
) -> Dict[str, List[PromptSample]]:
    """Load all prompt groups for all direction types."""
    safety = load_safety_prompts(limit=safety_limit, seed=seed)
    epistemic = load_epistemic_prompts(limit=epistemic_limit, seed=seed)
    capability = load_capability_prompts(limit=capability_limit, seed=seed)

    return {
        **safety,
        **epistemic,
        **capability,
    }

def prompts_to_text_list(samples: Sequence[PromptSample]) -> List[str]:
    """Extract plain text prompts from PromptSample list."""
    return [s.prompt for s in samples]
