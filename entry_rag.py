from dataclasses import asdict
from typing import Literal

import fire

import settings_rag
from utils import set_global_pooling_type

class_mapping = {
    "v2": "PoisionedRAGJoint",
    "base": "PoisionedRAGBaseline",
    "liar": "LIARRAGBaseline",
    "ablation1": "PoisionedRAGAblation1",
    "ablation2": "PoisionedRAGAblation2",
    "ablation3": "PoisionedRAGAblation3",
    "blackbox": "PoisionedRAGJoint",
    "ppl": "PoisionedRAGDefensePPL",
}

ablation1_ratios = [0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875]

assets_mapping = {
    "hotpotqa": "attack_data/poisoned_rag_hotpotqa",
    "msmarco": "attack_data/poisoned_rag_msmarco",
    "nq": "attack_data/poisoned_rag_nq",
}

corpus_mapping = {
    # [NOT INCLUDEDED] Please generate or download them by following instructions:
    # Download from https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/
    "hotpotqa": "attack_data/hotpotqa/corpus.jsonl",
    "msmarco": "attack_data/msmarco/corpus.jsonl",
    "nq": "attack_data/nq/corpus.jsonl",
}

fake_corpus_mapping = {
    "contriever": "attack_data/poisoned_rag_msmarco/msmarco_fakecorpus_contriever_ground-truth.json",
    "bge": "attack_data/poisoned_rag_msmarco/msmarco_fakecorpus_bge_ground-truth.json",
}

# Pre-generated search results file
partial_retrieval_results_mapping = {
    "contriever": {
        "hotpotqa": "attack_data/poisoned_rag_hotpotqa/hotpotqa_contriever_ground-truth.json",
        "msmarco": "attack_data/poisoned_rag_msmarco/msmarco-100_contriever-ms_ground-truth.json",
        "nq": "attack_data/poisoned_rag_nq/nq_contriever_ground-truth.json",
    },
    "bge": {
        "hotpotqa": "attack_data/poisoned_rag_hotpotqa/hotpotqa_bge_ground-truth.json",
        "nq": "attack_data/poisoned_rag_nq/nq_bge_ground-truth.json",
        "msmarco": "attack_data/poisoned_rag_nq/msmarco_bge_ground-truth.json",
    },
}

model_llm_mapping = {
    # [NOT INCLUDEDED] Please download them:
    "qwen": "",  # Qwen2.5-7B-Instruct path
    "llama": "",  # Llama3-8B-Instruct path
}

model_retriever_mapping = {
    # [NOT INCLUDEDED] Please download them:
    "bge": "",  # bge-base-en-v1.5 path
    "contriever": "",  # contriever-msmarco path
}

transfer_matrix_mapping = {
    # [NOT INCLUDEDED] Please generate them by following instructions:
    # TODO:
    "bge": {
        "qwen": "",  # bge-qwen2.5 transfer matrix path (.npy)
        "llama": "",  # bge-llama3 transfer matrix path (.npy)
    },
    "contriever": {
        "qwen": "",  # contriever-qwen2.5 transfer matrix path (.npy)
        "llama": "",  # contriever-llama3 transfer matrix path (.npy)
    },
}


def main(
    model_llm: Literal["qwen", "llama"],
    model_retriever: Literal["bge", "contriever"],
    asset_type: Literal["hotpotqa", "msmarco", "nq"],
    type: Literal["v2", "base", "liar", "ablation1", "ablation2", "blackbox", "ppl"],
):

    class_name = class_mapping[type]
    attack_obj = getattr(settings_rag, class_name)()
    method = attack_obj.method
    params = asdict(attack_obj)
    params.pop("method")

    if model_retriever == "bge":
        set_global_pooling_type("cls")

    model_llm_path = model_llm_mapping[model_llm]
    model_retriever_path = model_retriever_mapping[model_retriever]
    transfer_matrix_path = transfer_matrix_mapping[model_retriever][model_llm]
    assets_path = assets_mapping[asset_type]
    corpus_jsonl_path = corpus_mapping[asset_type]
    queries_top50_hits_json_path = partial_retrieval_results_mapping[model_retriever][
        asset_type
    ]
    params["model_llm"] = model_llm_path
    params["model_retriever"] = model_retriever_path
    params["assets_path"] = assets_path
    params["corpus_jsonl_path"] = corpus_jsonl_path
    params["queries_top50_hits_json_path"] = queries_top50_hits_json_path

    log_path_bath = f"logs/{model_llm}-{model_retriever}/{asset_type}/{type}"
    params["log_path"] = log_path_bath

    if type != "liar":
        params["transfer_matrix_path"] = transfer_matrix_path

    if type == "ablation1":
        for ratio in ablation1_ratios:
            params["fixed_ratio"] = ratio
            params["log_path"] = log_path_bath + f"/{ratio}"
            method(**params)
    elif type == "blackbox":
        assert asset_type == "msmarco", "Blackbox attack only supports msmarco"
        params["queries_top50_hits_json_path"] = fake_corpus_mapping[model_retriever]
        params["log_path"] = log_path_bath + "/fake_corpus"
        method(**params)
    else:
        method(**params)


if __name__ == "__main__":
    fire.Fire(main)
