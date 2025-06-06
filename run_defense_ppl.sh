set +e

export CUDA_VISIBLE_DEVICES=0

python entry_rag.py --model_llm qwen --model_retriever contriever --asset_type msmarco --type ppl

