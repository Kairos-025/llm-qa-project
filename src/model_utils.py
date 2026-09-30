"""
Loads the language model and generates answers in batches.
Every method (baseline, prompting, RAG, fine-tuned) uses these same two
functions, so the only thing that changes between methods is the prompt
or the adapter, never the model setup.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

DEFAULT_MODEL = "microsoft/Phi-3-mini-4k-instruct"


def load_model(model_id: str = DEFAULT_MODEL, adapter_path: str = None):
    """Load the model in 4-bit (the same precision QLoRA fine-tuning uses,
    so all four methods run on an identical base model)."""
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,  # T4 GPUs do not support bfloat16
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.padding_side = "left"  # required for batched generation
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb,
        device_map="auto",
        dtype=torch.float16,
    )
    if adapter_path:  # used in Phase 6 to load the fine-tuned LoRA adapter
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return model, tokenizer


@torch.no_grad()
def generate_batch(model, tokenizer, messages_list, max_new_tokens: int = 48):
    """messages_list: one chat (list of {"role", "content"} dicts) per question.
    Returns one answer string per question. Greedy decoding, so results are
    reproducible run to run."""
    prompts = [
        tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
        for m in messages_list
    ]
    enc = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
    out = model.generate(
        **enc,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
    )
    new_tokens = out[:, enc["input_ids"].shape[1]:]
    return [t.strip() for t in tokenizer.batch_decode(new_tokens, skip_special_tokens=True)]
