"""Names and identities of the two checkpoints in the paper.

The original checkpoint is already instruction-tuned. It is not a base model.
The adapter was trained on both safety and general instruction examples.
"""

MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
CHECKPOINTS = {
    "original_instruct": {
        "model_id": MODEL_ID,
        "label": "Original Instruct",
        "description": "Llama 3.1 8B Instruct, without the additional adapter",
        "requires_adapter": False,
    },
    "safety_lora": {
        "model_id": MODEL_ID,
        "label": "Instruct + safety LoRA",
        "description": "Llama 3.1 8B Instruct with the study's additional LoRA adapter",
        "requires_adapter": True,
    },
}
