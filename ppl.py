import pandas as pd
import torch


def calculate_perplexity_wo_load(text, model, tokenizer):
    encodings = tokenizer(text, return_tensors="pt")
    seq_len = encodings.input_ids.size(1)

    trg_len = seq_len
    input_ids = encodings.input_ids.to(model.device)
    target_ids = input_ids.clone()
    target_ids[:, :-trg_len] = -100

    with torch.no_grad():
        outputs = model(input_ids, labels=target_ids)

        # loss is calculated using CrossEntropyLoss which averages over valid labels
        # N.B. the model only calculates loss over trg_len - 1 labels, because it internally shifts the labels
        # to the left by 1.
        neg_log_likelihood = outputs.loss

    ppl = torch.exp(neg_log_likelihood)
    return ppl
