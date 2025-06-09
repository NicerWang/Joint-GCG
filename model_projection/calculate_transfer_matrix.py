import argparse
import gc
import os

import lightning as lt
import numpy as np
import torch
from tqdm import tqdm, trange
from train_autoencoder import Autoencoder
from transformers import AutoModel, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Tool Attack Model Configuration")
    parser.add_argument(
        "--device", type=str, default="cuda", help="The device to use for computation"
    )
    parser.add_argument(
        "--ret_model_path",
        type=str,
        default="/fastio/wanghw/models/msmarco-roberta-base-ance-firstp",
        help="Path to the retriever model",
    )
    parser.add_argument(
        "--llm_model_path",
        type=str,
        default="/fastio/wanghw/models/Qwen2.5-7B-Instruct",
        help="Path to the LLM model",
    )
    parser.add_argument(
        "--ae_model_path", type=str, required=True, help="Path to the autoencoder model"
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="/fastio/wanghw/transfer_matrix/tmp/qwen2.5-ance",
        help="Path to save the transfer matrix",
    )
    return parser.parse_args()


args = parse_args()
device = torch.device(args.device)

if not os.path.exists(args.save_path):
    os.makedirs(args.save_path)

# 加载模型和Tokenizer

ret_tokenizer = AutoTokenizer.from_pretrained(args.ret_model_path, use_fast=True)
ret_model = AutoModel.from_pretrained(args.ret_model_path, device_map="cpu")
llm_tokenizer = AutoTokenizer.from_pretrained(args.llm_model_path, use_fast=True)
llm_model = AutoModel.from_pretrained(args.llm_model_path, device_map="cpu")

print("Model loaded")


print("Projecting llama embeddings...")

model = Autoencoder.load_from_checkpoint(
    args.ae_model_path,
    strict=False,
    llm_shape=llm_model.embed_tokens.weight.shape[1],
    ret_shape=ret_model.embeddings.word_embeddings.weight.shape[1],
)

full_llm_emb = llm_model.embed_tokens.weight.to(model.device)
full_llm_proj = model.forward_encoder(full_llm_emb)


np.save(
    os.path.join(args.save_path, "full_llm_proj.npy"),
    full_llm_proj.cpu().detach().numpy(),
)
print("Projection done")

print("Calculating transfer matrix...")
full_bert_emb = ret_model.embeddings.word_embeddings.weight

full_llm_proj = full_llm_proj.to(device)
full_bert_emb = full_bert_emb.to(device)


def process_single_bert_embedding_torch(bert_embedding, llama_embeddings_T):
    """Helper function to process a single BERT embedding using PyTorch on GPU.
       Given a BERT embedding, calculate the coefficients for the LLaMA embeddings that approximate it.
    Args:
        bert_embedding (torch.Tensor): Tensor of shape (768,) representing a BERT embedding on GPU.
        llama_embeddings_T (torch.Tensor): Tensor of shape (768, 32000) representing LLaMA embeddings on GPU.
    Returns:
        torch.Tensor: Tensor of shape (32000,) containing the coefficients for the LLaMA embeddings.
    """
    solution, residual, rank, singular = torch.linalg.lstsq(
        llama_embeddings_T, bert_embedding.unsqueeze(1)
    )
    return solution.squeeze(1), residual, rank, singular


def calculate_linear_composition_torch(bert_embeddings, llama_embeddings):
    """
    Calculate the linear composition of LLaMA embeddings to approximate BERT embeddings using GPU.
    Args:
        bert_embeddings (numpy.ndarray): Array of shape (30522, 4096) representing BERT embeddings.
        llama_embeddings (numpy.ndarray): Array of shape (32000, 4096) representing LLaMA embeddings.
        device (torch.device): Device for GPU/CPU computation.
    Returns:
        numpy.ndarray: Array of shape (30522, 32000) where each row contains the coefficients
                       for the LLaMA embeddings that best approximate the corresponding BERT embedding.
    """
    # Convert bert_embeddings and llama_embeddings to torch tensors and move to device
    bert_embeddings_torch = torch.tensor(
        bert_embeddings, dtype=torch.float32, device=device
    )
    llama_embeddings_torch = torch.tensor(
        llama_embeddings.T, dtype=torch.float32, device=device
    )

    results = []
    residuals = []
    ranks = []
    singulars = []
    for i in trange(0, bert_embeddings_torch.shape[0]):
        result, residual, rank, singular = process_single_bert_embedding_torch(
            bert_embeddings_torch[i], llama_embeddings_torch
        )
        results.append(
            result.cpu().numpy()
        )  # Move result back to CPU for final collection
        residuals.append(residual.cpu().numpy())
        ranks.append(rank.cpu().numpy())
        singulars.append(singular.cpu().numpy())
        del result
        del residual
        del rank
        del singular
        torch.cuda.empty_cache()
        gc.collect()

    return np.vstack(results), np.array(residuals), np.array(ranks), np.array(singulars)


# coefficients_matrix = calculate_linear_composition_torch(full_bert_emb, full_llm_proj)
coefficients_matrix, residuals, ranks, singulars = calculate_linear_composition_torch(
    full_bert_emb, full_llm_proj
)
print("Transfer matrix calculated")
np.save(os.path.join(args.save_path, "transfer_matrix.npy"), coefficients_matrix)
np.save(os.path.join(args.save_path, "residuals.npy"), residuals)
np.save(os.path.join(args.save_path, "ranks.npy"), ranks)
np.save(os.path.join(args.save_path, "singulars.npy"), singulars)
