# This script exists just to load models faster
import functools
import os

import torch
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer, OPTForCausalLM

from _settings import MODEL_PATH


def _resolve_pretrained_path(model_name):
    if os.path.exists(model_name):
        return model_name
    if model_name.startswith('facebook/opt-'):
        local_opt = os.path.join(MODEL_PATH, model_name.split('/')[-1])
        if os.path.exists(local_opt):
            return local_opt
    local_path = os.path.join(MODEL_PATH, model_name)
    if os.path.exists(local_path):
        return local_path
    return model_name


@functools.lru_cache()
def _load_pretrained_model(model_name, device, torch_dtype=torch.float16):
    if str(device).startswith('cpu') and torch_dtype == torch.float16:
        torch_dtype = torch.float32

    resolved_name = _resolve_pretrained_path(model_name)
    trust_remote_code = model_name == 'falcon-7b' or 'falcon' in str(resolved_name).lower()

    if model_name in {'microsoft/deberta-large-mnli', 'roberta-large-mnli'}:
        model = AutoModelForSequenceClassification.from_pretrained(resolved_name)
    elif model_name.startswith('facebook/opt-') and os.path.exists(resolved_name):
        model = OPTForCausalLM.from_pretrained(resolved_name, torch_dtype=torch_dtype)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            resolved_name,
            cache_dir=None,
            trust_remote_code=trust_remote_code,
            torch_dtype=torch_dtype,
        )
    model.to(device)
    return model


@functools.lru_cache()
def _load_pretrained_tokenizer(model_name, use_fast=False):
    resolved_name = _resolve_pretrained_path(model_name)
    trust_remote_code = model_name == 'falcon-7b' or 'falcon' in str(resolved_name).lower()
    tokenizer = AutoTokenizer.from_pretrained(
        resolved_name,
        use_fast=use_fast,
        trust_remote_code=trust_remote_code,
    )

    if model_name in {'llama-7b-hf', 'llama-13b-hf', 'llama2-7b-hf'}:
        tokenizer.eos_token_id = 2
        tokenizer.bos_token_id = 1
        tokenizer.eos_token = tokenizer.decode(tokenizer.eos_token_id)
        tokenizer.bos_token = tokenizer.decode(tokenizer.bos_token_id)

    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer
