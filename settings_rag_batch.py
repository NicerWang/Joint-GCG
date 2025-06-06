from dataclasses import dataclass

from attack_rag_batch import attack_base as attack_base_rag_batch
from attack_rag_batch import attack_joint as attack_joint_rag_batch

# [NOT INCLUDEDED] Please download them:
MODEL_LLM = ""  # Llama3-8B-Instruct path
MODEL_RETRIEVER = ""  # contriever-msmarco path
# [NOT INCLUDEDED] Please generate or download them by following instructions:
# Download from https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/
CORPUS_PATH = "attack_data/msmarco/corpus.jsonl"
# [NOT INCLUDEDED] Please generate them by following instructions:
# TODO:
TRANSFER_MATRIX_PATH = ""  # contriever-llama3 transfer matrix path (.npy)


@dataclass
class PhantomJointWithInitialFakeCorpus:
    method: callable = attack_joint_rag_batch
    tag: str = "prag_v2"
    model_llm: str = MODEL_LLM
    model_retriever: str = MODEL_RETRIEVER
    corpus_jsonl_path: str = CORPUS_PATH
    transfer_matrix_path: str = TRANSFER_MATRIX_PATH
    assets_path: str = "attack_data/phantom_msmarco/trigger-all-withinit"
    queries_top50_hits_json_path: str = (
        "attack_data/phantom_msmarco/phantom-msmarco-samples_contriever-msmarco_ground-truth.json"
    )


@dataclass
class PhantomBaseline:
    method: callable = attack_base_rag_batch
    tag: str = "prag_base"
    model_llm: str = MODEL_LLM
    model_retriever: str = MODEL_RETRIEVER
    corpus_jsonl_path: str = CORPUS_PATH
    transfer_matrix_path: str = TRANSFER_MATRIX_PATH
    assets_path: str = "attack_data/phantom_msmarco/trigger-all"
    queries_top50_hits_json_path: str = (
        "attack_data/phantom_msmarco/phantom-msmarco-samples_contriever-msmarco_ground-truth.json"
    )
