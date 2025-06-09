import copy
import time
from contextlib import contextmanager
from math import exp
from typing import List, Literal

import numpy as np
import torch
from torch import nn
from transformers import (AutoModel, AutoModelForCausalLM, AutoTokenizer,
                          BertModel, BertTokenizer, BertTokenizerFast,
                          Gemma2ForCausalLM, LlamaForCausalLM,
                          Qwen2ForCausalLM, RobertaModel, RobertaTokenizer,
                          RobertaTokenizerFast)

from token_mapper import get_select_tokens_bpe, get_select_tokens_wordpiece


# FOR Our Method
def adaptive_ratio(
    position: int,
    other_scores: List[float],
    target_position: int = 0,
):
    if type(other_scores) is np.ndarray:
        other_scores = other_scores.tolist()
    if position > len(other_scores):
        return 1.0
    target_score = other_scores[position - 1]
    # remove target score
    other_scores = other_scores[: position - 1] + other_scores[position:]
    other_scores.sort(reverse=True)
    safe = target_score - other_scores[max(0, target_position - 1)]
    diff = 0.0
    diff_cnt = 0
    for i in range(len(other_scores) - 1):
        diff += other_scores[i] - other_scores[i + 1]
        diff_cnt += 1
    diff /= diff_cnt
    safe /= diff
    # sigmoid
    return 1.0 - 1.0 / (1.0 + exp(-safe))


# FOR MCG
def get_embedding_layer(model):
    if (
        isinstance(model, LlamaForCausalLM)
        or isinstance(model, Gemma2ForCausalLM)
        or isinstance(model, Qwen2ForCausalLM)
    ):
        return model.model.embed_tokens.weight
    elif isinstance(model, (BertModel, RobertaModel)):
        return model.embeddings.word_embeddings.weight
    else:
        raise ValueError(f"model type: {type(model)} not implemented")


def get_embeddings(model, input_ids):
    if (
        isinstance(model, LlamaForCausalLM)
        or isinstance(model, Gemma2ForCausalLM)
        or isinstance(model, Qwen2ForCausalLM)
    ):
        return model.model.embed_tokens(input_ids)
    elif isinstance(model, (BertModel, RobertaModel)):
        return model.embeddings.word_embeddings(input_ids)
    else:
        raise ValueError(f"model type: {type(model)} not implemented")


def loss_calculator_wrapper(target: torch.Tensor, model_type: str):
    # Establish the Loss Calculation Closure
    def loss_calculator_decoder(logits: torch.Tensor):
        loss_func = nn.CrossEntropyLoss(reduction="none")
        current_target = target.repeat([logits.shape[0], 1])
        loss = loss_func(logits.transpose(1, 2), current_target)
        return loss.mean(dim=-1)

    def loss_calculator_encoder(embedding: torch.Tensor):
        loss_func = nn.CosineEmbeddingLoss(reduction="none")
        current_target = target.repeat([embedding.shape[0], 1])
        loss = loss_func(
            embedding,
            current_target,
            torch.ones(embedding.shape[0], device=embedding.device),
        )
        return loss

    if model_type == "encoder":
        return loss_calculator_encoder
    elif model_type == "decoder":
        return loss_calculator_decoder
    else:
        raise ValueError(f"model type: {model_type} not implemented")


# For Phantom
def loss_calculator_phantom_wrapper(
    in_set: torch.Tensor, out_set: torch.Tensor, model_type: str
):
    # Establish the Loss Calculation Closure
    def loss_calculator_decoder(logits: torch.Tensor):
        loss_func = nn.CrossEntropyLoss(reduction="none")
        current_target = in_set.repeat([logits.shape[0], 1])
        loss = loss_func(logits.transpose(1, 2), current_target)
        return loss.mean(dim=-1)

    def loss_calculator_encoder(embedding: torch.Tensor):
        loss_func = nn.CosineEmbeddingLoss(reduction="none")
        current_target = in_set.repeat([embedding.shape[0], 1]) - out_set.repeat(
            [embedding.shape[0], 1]
        )
        loss = loss_func(
            embedding,
            current_target,
            torch.ones(embedding.shape[0], device=embedding.device),
        )
        return loss

    if model_type == "encoder":
        return loss_calculator_encoder
    elif model_type == "decoder":
        return loss_calculator_decoder
    else:
        raise ValueError(f"model type: {model_type} not implemented")


# FOR GC
def clear_cache():
    import gc

    gc.collect()


def clear_cache_decorator(func):
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        clear_cache()
        return result

    return wrapper


# FOR Tensorboard
GLOBAL_WRITER = None


def setup_global_writer(tag):
    """
    Set global tensorboard writer
    """
    global GLOBAL_WRITER
    from torch.utils.tensorboard import SummaryWriter

    GLOBAL_WRITER = SummaryWriter(tag)


def get_global_writer():
    """
    Get global tensorboard writer
    """
    return GLOBAL_WRITER


@contextmanager
def timeit(round, tag):
    start_time = time.time()
    try:
        yield
    finally:
        GLOBAL_WRITER.add_scalar("time/" + tag, time.time() - start_time, round)


# BASIC Util
GLOBAL_POOLING_TYPE = "mean"


def set_global_pooling_type(pooling_type: Literal["mean", "cls"]):
    global GLOBAL_POOLING_TYPE
    GLOBAL_POOLING_TYPE = pooling_type


def pooling(last_hidden_states, attention_mask=None):
    if GLOBAL_POOLING_TYPE == "mean":
        if attention_mask is None:
            attention_mask = torch.ones(
                last_hidden_states.shape[:-1], device=last_hidden_states.device
            )
        masked_embeddings = last_hidden_states * attention_mask.unsqueeze(-1)
        sum_embeddings = masked_embeddings.sum(dim=1)
        sum_mask = attention_mask.sum(dim=1).unsqueeze(-1)
        return sum_embeddings / sum_mask
    elif GLOBAL_POOLING_TYPE == "cls":
        return last_hidden_states[:, 0, :]
    else:
        raise ValueError(f"Pooling type: {GLOBAL_POOLING_TYPE} not implemented")


def load_model(model_path, tokenizer_path=None, model_type="encoder", **kwargs):
    if model_type == "encoder":
        model = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
            device_map="auto",
            low_cpu_mem_usage=True,
            use_cache=False,
            torch_dtype=torch.float16,
            **kwargs,
        ).eval()
    elif model_type == "decoder":
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            # load_in_8bit=True,
            device_map="auto",
            low_cpu_mem_usage=True,
            use_cache=False,
            torch_dtype=torch.float16,
            **kwargs,
        ).eval()
    else:
        raise ValueError(f"model type: {type(model)} not implemented")

    tokenizer_path = model_path if tokenizer_path is None else tokenizer_path
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        trust_remote_code=True,
        clean_up_tokenization_spaces=False,
        use_fast=True,
    )
    tokenizer.padding_side = "left"

    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


@clear_cache_decorator
def generate_batch(
    model, tokenizer, all_messages, stop, max_new_tokens=256, batch_size=4
):
    if len(all_messages) == 0:
        return []
    all_message_with_chat_template = []
    for messages in all_messages:
        message_with_chat_template = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        all_message_with_chat_template.append(message_with_chat_template)
    results = []
    for i in range(0, len(all_messages), batch_size):
        clear_cache()
        batch_input = tokenizer(
            all_message_with_chat_template[i : i + batch_size],
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        ).to(model.device)
        # Greedy Decoding
        generate_config = copy.deepcopy(model.generation_config)
        generate_config.max_new_tokens = max_new_tokens
        generate_config.stop_strings = stop
        generate_config.num_beams = 1
        generate_config.do_sample = False
        generate_config.top_p = None
        generate_config.top_k = None
        generate_config.temperature = None
        batch_output_ids = model.generate(
            **batch_input,
            tokenizer=tokenizer,
            generation_config=generate_config,
            pad_token_id=tokenizer.pad_token_id,
        )
        results.extend(
            tokenizer.batch_decode(
                batch_output_ids[:, batch_input["input_ids"].shape[1] :]
            )
        )
    return results


# FOR OUR METHOD
def get_transfer_matrix(
    llm_tokenizer, retriever_tokenizer, transfer_matrix_path, ascii_only=True
):

    transfer_matrix = np.load(transfer_matrix_path)

    llm_tokens = get_select_tokens_bpe(
        llm_tokenizer,
        ascii_only=ascii_only,
    )
    if isinstance(retriever_tokenizer, (BertTokenizer, BertTokenizerFast)):
        retriever_tokens = get_select_tokens_wordpiece(
            retriever_tokenizer, ascii_only=ascii_only
        )
    elif isinstance(retriever_tokenizer, (RobertaTokenizer, RobertaTokenizerFast)):
        retriever_tokens = get_select_tokens_bpe(
            retriever_tokenizer, ascii_only=ascii_only
        )
    else:
        raise NotImplementedError

    retriever_tokens = [
        retriever_token
        for retriever_token in retriever_tokens
        if not retriever_token.startswith("##")
    ]

    llm_input_ids = llm_tokenizer.convert_tokens_to_ids(llm_tokens)
    retriever_input_ids = retriever_tokenizer.convert_tokens_to_ids(retriever_tokens)

    print(f"transfer_matrix shape: {transfer_matrix.shape}")
    return (
        llm_input_ids,
        retriever_input_ids,
        transfer_matrix,
    )
