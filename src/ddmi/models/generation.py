"""Text generation utilities."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, List, Optional

from ddmi.models.positions import POSITION_CONTROL_MODES, locate_positions

import torch


@dataclass(frozen=True)
class TextGenerationConfig:
    """Configuration for text generation."""

    max_new_tokens: int = 128
    temperature: float = 0.0
    do_sample: bool = False
    max_input_length: Optional[int] = 512
    enable_thinking: Optional[bool] = None
    record_token_ids: bool = False
    position_control_mode: Optional[str] = None


def generate_text(
    model: Any,
    tokenizer: Any,
    prompt: str,
    config: Optional[TextGenerationConfig] = None,
    chat_template: bool = True,
    return_metadata: bool = False,
    prefill_callback: Optional[Callable[[int], None]] = None,
):
    """Generate text from a prompt. Returns only the generated portion."""
    if config is None:
        config = TextGenerationConfig()

    if chat_template and hasattr(tokenizer, "apply_chat_template"):
        messages = [{"role": "user", "content": prompt}]
        template_options = {}
        if config.enable_thinking is not None:
            template_options["enable_thinking"] = config.enable_thinking
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, **template_options
        )
    else:
        text = prompt

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=config.max_input_length,
    )
    device = _resolve_device(model)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    input_len = inputs["input_ids"].shape[1]
    full_input_len = None
    if return_metadata:
        full_input_len = tokenizer(text, return_tensors="pt", truncation=False)["input_ids"].shape[1]

    position_metadata = None
    if config.position_control_mode is not None:
        if config.position_control_mode not in (*POSITION_CONTROL_MODES, "observe"):
            raise ValueError("Unknown position-control mode")
        offset_batch = tokenizer(text, return_tensors="pt", truncation=False,
                                 return_offsets_mapping=True)
        ids = inputs["input_ids"][0].tolist()
        if offset_batch["input_ids"][0].tolist() != ids:
            raise ValueError("Position controls forbid truncation or changed tokenization")
        offsets = offset_batch["offset_mapping"][0].tolist()
        positions = locate_positions(text, prompt, ids, offsets)
        key = POSITION_CONTROL_MODES.get(config.position_control_mode)
        selected = positions[key] if key is not None else None
        if selected is not None:
            if prefill_callback is None:
                raise ValueError("Position-control hook callback is missing")
            prefill_callback(selected)
        elif prefill_callback is not None:
            raise ValueError("Observe must not select an edited position")
        position_metadata = {
            "indices": positions, "selected_index": selected,
            "user_content_char_span": [text.index(prompt), text.index(prompt)+len(prompt)],
            "neighborhoods": {
                name: [{"index": j, "id": ids[j],
                        "token": tokenizer.convert_ids_to_tokens(ids[j])}
                       for j in range(max(0,index-2), min(len(ids),index+3))]
                for name,index in positions.items()
            },
        }

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=config.max_new_tokens,
            temperature=config.temperature if config.do_sample else None,
            do_sample=config.do_sample,
            pad_token_id=tokenizer.pad_token_id,
            use_cache=True,
        )

    generated = output[0, input_len:]
    decoded = tokenizer.decode(generated, skip_special_tokens=True)
    if return_metadata:
        metadata = {
            "response": decoded,
            "input_tokens": input_len,
            "untruncated_input_tokens": full_input_len,
            "input_truncated": full_input_len > input_len,
            "generated_tokens": len(generated),
            "hit_token_limit": len(generated) == config.max_new_tokens,
        }
        if config.record_token_ids:
            metadata["prefill_token_ids"] = inputs["input_ids"][0].tolist()
            metadata["generated_token_ids"] = generated.tolist()
        if position_metadata is not None:
            metadata["position_control"] = position_metadata
        return metadata
    return decoded


def generate_batch(
    model: Any,
    tokenizer: Any,
    prompts: List[str],
    config: Optional[TextGenerationConfig] = None,
    chat_template: bool = True,
) -> List[str]:
    """Generate text for multiple prompts (sequentially for memory safety)."""
    return [
        generate_text(model, tokenizer, p, config=config, chat_template=chat_template)
        for p in prompts
    ]


def _resolve_device(model: Any) -> torch.device:
    """Resolve model device."""
    if hasattr(model, "device"):
        return model.device
    for param in model.parameters():
        if param.device.type != "meta":
            return param.device
    return torch.device("cpu")
