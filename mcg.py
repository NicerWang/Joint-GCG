from typing import List

import numpy as np
import torch
from tqdm import trange

from utils import clear_cache, get_embedding_layer, get_embeddings, pooling


def token_gradients(
    model,
    input_ids,
    optimize_slice,
    cal_loss,
    loss_slice=None,
    model_type: str = "encoder",
):
    embed_weights = get_embedding_layer(model)
    # For each word in the vocabulary, generate a gradient.
    # Shape [input position, vocabulary]
    one_hot = torch.zeros(
        input_ids[optimize_slice].shape[0],
        embed_weights.shape[0],
        device=model.device,
        dtype=embed_weights.dtype,
    )
    # Convert to One-Hot encoding
    one_hot.scatter_(
        1,
        input_ids[optimize_slice].unsqueeze(1),
        torch.ones(one_hot.shape[0], 1, device=model.device, dtype=embed_weights.dtype),
    )
    one_hot.requires_grad_()
    # Recalculate to utilize autograd for gradient calculation
    input_embeds = (one_hot @ embed_weights).unsqueeze(0)
    # Concatenating with the previous embedding
    embeds = get_embeddings(model, input_ids.unsqueeze(0)).detach()
    full_embeds = torch.cat(
        [
            embeds[:, : optimize_slice.start, :],
            input_embeds,
            embeds[:, optimize_slice.stop :, :],
        ],
        dim=1,
    )
    if model_type == "encoder":
        full_embeds = full_embeds[:, :512, :]

    del embed_weights, input_embeds, embeds
    clear_cache()
    if model_type == "encoder":
        last_hidden_state = model(inputs_embeds=full_embeds).last_hidden_state
        pooling_result = pooling(last_hidden_state)
        loss = cal_loss(pooling_result)
    elif model_type == "decoder":
        logits = model(inputs_embeds=full_embeds).logits
        loss = cal_loss(logits[:, loss_slice, :])
    else:
        raise ValueError(f"model type: {model_type} not implemented")

    loss.backward()

    grad = one_hot.grad.clone()
    grad = grad / grad.norm(dim=-1, keepdim=True)
    grad = grad.detach().cpu()

    return grad


def sample_replacements(
    optimization_tokens,
    coordinate_grad,
    batch_size,
    select_token_ids,
    topk,
    epoch,
    c_min=2,
):
    """
    Sampled replaceable token (MCG), reducing the value of topK can accelerate convergence speed.

    Args:
        epoch (int): current epoch
        c_min (int): Minimum c value
    """
    c = int(max(coordinate_grad.shape[0] / (2 ** (epoch + 1)), c_min))
    # Remove disabled Tokens in coordinate_grad
    mask = torch.zeros(coordinate_grad.shape[1], dtype=torch.bool)
    mask[select_token_ids] = True
    coordinate_grad[:, ~mask] = np.infty
    # For each position, sample TopK replaceable Tokens; the total number of replacement methods is the number of TopK positions.
    top_indices = (-coordinate_grad).topk(topk, dim=1).indices
    optimization_tokens = optimization_tokens.to(coordinate_grad.device)
    original_control_tokens = optimization_tokens.repeat(batch_size, 1)
    # Use torch to randomly generate (batch_size, c) different random numbers, indicating the optimization position
    # Note, the sampling location here is with replacement sampling.
    new_token_pos = torch.randint(
        0, optimization_tokens.shape[0], (batch_size, c), device=coordinate_grad.device
    )
    # Use torch to randomly generate (batch_size, c, 1) different random numbers, specify the replacement token
    new_token_topk_pos = torch.randint(
        0, topk, (batch_size, c, 1), device=coordinate_grad.device
    )
    # Extract and replace tokens.
    new_token_val = torch.gather(top_indices[new_token_pos], 2, new_token_topk_pos)
    # Apply the extracted tokens
    new_optimization_tokens = original_control_tokens.scatter_(
        1, new_token_pos, new_token_val.squeeze(-1)
    )
    return new_optimization_tokens


def cal_losses(
    model,
    tokenizer,
    input_ids,
    cal_loss,
    optimize_slice: slice,
    candidates: List[str],
    batch_size: int,
    loss_slice: slice = None,
    model_type: str = "encoder",
):
    def forward_encoder_only(model, input_ids, attention_mask):
        """
        Forward reasoning, obtain logits and concatenate
        """
        losses = []
        for i in trange(0, input_ids.shape[0], batch_size):
            batch_input_ids = input_ids[i : i + batch_size]
            if attention_mask is not None:
                batch_attention_mask = attention_mask[i : i + batch_size]
            else:
                batch_attention_mask = None
            losses.append(
                cal_loss(
                    pooling(
                        model(
                            input_ids=batch_input_ids,
                            attention_mask=batch_attention_mask,
                        ).last_hidden_state,
                        batch_attention_mask,
                    )
                )
            )
            del batch_input_ids, batch_attention_mask
            clear_cache()
        return torch.concat(losses, dim=-1)

    def forward_decoder_only(model, input_ids, attention_mask):
        """
        Forward reasoning, obtain logits, calculate loss and return
        """
        losses = []
        for i in range(0, input_ids.shape[0], batch_size):
            batch_input_ids = input_ids[i : i + batch_size]
            if attention_mask is not None:
                batch_attention_mask = attention_mask[i : i + batch_size]
            else:
                batch_attention_mask = None
            current_logits = model(
                input_ids=batch_input_ids, attention_mask=batch_attention_mask
            ).logits
            losses.append(cal_loss(current_logits[:, loss_slice, :]))
            del batch_input_ids, batch_attention_mask, current_logits
            clear_cache()
        return torch.concat(losses, dim=-1)

    test_ids = []
    max_len = 0
    for candidate in candidates:
        candidate_ids = torch.tensor(
            tokenizer(candidate, add_special_tokens=False).input_ids,
            device=model.device,
        )
        max_len = max(candidate_ids.shape[0], max_len)
        test_ids.append(candidate_ids)
    nested_ids = torch.nested.nested_tensor(test_ids)
    test_ids = torch.nested.to_padded_tensor(
        nested_ids, tokenizer.pad_token_id, output_size=(len(test_ids), max_len)
    ).to(model.device)
    locs = (
        torch.arange(optimize_slice.start, optimize_slice.stop)
        .repeat(test_ids.shape[0], 1)
        .to(model.device)
    )
    ids = torch.scatter(
        input_ids.unsqueeze(0).repeat(test_ids.shape[0], 1).to(model.device),
        1,
        locs,
        test_ids,
    )
    attn_mask = (ids != tokenizer.pad_token_id).type(ids.dtype)
    if model_type == "encoder":
        ids = ids[:, :512]
        attn_mask = attn_mask[:, :512]
    del locs, test_ids, nested_ids
    clear_cache()
    if model_type == "encoder":
        losses = forward_encoder_only(
            model=model, input_ids=ids, attention_mask=attn_mask
        )
    elif model_type == "decoder":
        losses = forward_decoder_only(
            model=model, input_ids=ids, attention_mask=attn_mask
        )
    else:
        raise ValueError(f"Model type: {model_type} not implemented")
    return losses


def filter_candidates(tokenizer, candidates, filter_candidate, space_sep_tokens):
    """
    If the length of the tokenizer changes after changing the token, discard it.
    """
    filtered_candidates = set()
    for i in range(candidates.shape[0]):
        if space_sep_tokens:
            elements = []
            for j in candidates[i]:
                elements.append(tokenizer.decode(j, skip_special_tokens=True).strip())
            decoded_str = " ".join(elements)
        else:
            decoded_str = tokenizer.decode(candidates[i], skip_special_tokens=True)
        decoded_str = " " + decoded_str.strip()
        if filter_candidate:
            if (
                tokenizer(decoded_str, add_special_tokens=False).input_ids
                == candidates[i].tolist()
            ):
                filtered_candidates.add(decoded_str)
        else:
            filtered_candidates.add(decoded_str)
    return list(filtered_candidates)
