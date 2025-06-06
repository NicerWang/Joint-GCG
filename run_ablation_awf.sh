set +e

export CUDA_VISIBLE_DEVICES=0


python entry_rag.py --model_llm llama --model_retriever contriever --asset_type msmarco --type ablation1

python entry_rag.py --model_llm llama --model_retriever contriever --asset_type hotpotqa --type ablation1

python entry_rag.py --model_llm llama --model_retriever contriever --asset_type nq --type ablation1

