set +e

export CUDA_VISIBLE_DEVICES=0

# base experiments
python entry_rag.py --model_llm qwen --model_retriever contriever --asset_type hotpotqa --type base
python entry_rag.py --model_llm qwen --model_retriever contriever --asset_type hotpotqa --type liar
python entry_rag.py --model_llm qwen --model_retriever contriever --asset_type hotpotqa --type v2

python entry_rag.py --model_llm qwen --model_retriever contriever --asset_type nq --type base
python entry_rag.py --model_llm qwen --model_retriever contriever --asset_type nq --type liar
python entry_rag.py --model_llm qwen --model_retriever contriever --asset_type nq --type v2

python entry_rag.py --model_llm qwen --model_retriever contriever --asset_type msmarco --type base
python entry_rag.py --model_llm qwen --model_retriever contriever --asset_type msmarco --type liar
python entry_rag.py --model_llm qwen --model_retriever contriever --asset_type msmarco --type v2

python entry_rag.py --model_llm llama --model_retriever contriever --asset_type hotpotqa --type base
python entry_rag.py --model_llm llama --model_retriever contriever --asset_type hotpotqa --type liar
python entry_rag.py --model_llm llama --model_retriever contriever --asset_type hotpotqa --type v2

python entry_rag.py --model_llm llama --model_retriever contriever --asset_type nq --type base
python entry_rag.py --model_llm llama --model_retriever contriever --asset_type nq --type liar
python entry_rag.py --model_llm llama --model_retriever contriever --asset_type nq --type v2

python entry_rag.py --model_llm llama --model_retriever contriever --asset_type msmarco --type base
python entry_rag.py --model_llm llama --model_retriever contriever --asset_type msmarco --type liar
python entry_rag.py --model_llm llama --model_retriever contriever --asset_type msmarco --type v2

python entry_rag.py --model_llm qwen --model_retriever bge --asset_type hotpotqa --type base
python entry_rag.py --model_llm qwen --model_retriever bge --asset_type hotpotqa --type liar
python entry_rag.py --model_llm qwen --model_retriever bge --asset_type hotpotqa --type v2

python entry_rag.py --model_llm qwen --model_retriever bge --asset_type nq --type base
python entry_rag.py --model_llm qwen --model_retriever bge --asset_type nq --type liar
python entry_rag.py --model_llm qwen --model_retriever bge --asset_type nq --type v2

python entry_rag.py --model_llm qwen --model_retriever bge --asset_type msmarco --type base
python entry_rag.py --model_llm qwen --model_retriever bge --asset_type msmarco --type liar
python entry_rag.py --model_llm qwen --model_retriever bge --asset_type msmarco --type v2

python entry_rag.py --model_llm llama --model_retriever bge --asset_type nq --type base
python entry_rag.py --model_llm llama --model_retriever bge --asset_type nq --type liar
python entry_rag.py --model_llm llama --model_retriever bge --asset_type nq --type v2

python entry_rag.py --model_llm llama --model_retriever bge --asset_type hotpotqa --type base
python entry_rag.py --model_llm llama --model_retriever bge --asset_type hotpotqa --type liar
python entry_rag.py --model_llm llama --model_retriever bge --asset_type hotpotqa --type v2

python entry_rag.py --model_llm llama --model_retriever bge --asset_type msmarco --type base
python entry_rag.py --model_llm llama --model_retriever bge --asset_type msmarco --type liar
python entry_rag.py --model_llm llama --model_retriever bge --asset_type msmarco --type v2