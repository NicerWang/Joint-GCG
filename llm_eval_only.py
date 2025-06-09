import json
import os
from typing import Literal

import fire
import numpy as np
from openai import OpenAI

from entry_rag import (
    corpus_mapping,
    model_llm_mapping,
    model_retriever_mapping,
    partial_retrieval_results_mapping,
)
from prompt.rag_prompt import RAGAttacker
from utils import generate_batch, load_model, set_global_pooling_type

# DATA JSON SCHEMA
# {
#    "query": "",
#    "correct_answer": "",
#    "incorrect_answer": "",
#    "adv_text": ""
# }

os.environ["OPENAI_API_KEY"] = ""
os.environ["OPENAI_BASE_URL"] = ""
llm_model = None
llm_tokenizer = None


def init_llm(llm: Literal["qwen", "llama", "gpt"]):
    global llm_model, llm_tokenizer
    infer_func = None
    if llm == "qwen":
        llm_model, llm_tokenizer = load_model(
            model_llm_mapping["qwen"], model_type="decoder"
        )

        def qwen_infer_func(all_messages):
            return generate_batch(
                llm_model, llm_tokenizer, all_messages, stop=None, batch_size=16
            )

        infer_func = qwen_infer_func
    elif llm == "llama":
        llm_model, llm_tokenizer = load_model(
            model_llm_mapping["llama"], model_type="decoder"
        )

        def llama_infer_func(all_messages):
            return generate_batch(
                llm_model, llm_tokenizer, all_messages, stop=None, batch_size=16
            )

        infer_func = llama_infer_func
    elif llm == "gpt":
        client = OpenAI()

        def gpt_infer_func(all_messages):
            results = []
            for messages in all_messages:
                need_retry = True
                while need_retry:
                    need_retry = False
                    try:
                        result = (
                            client.chat.completions.create(
                                model="gpt-4o",
                                messages=messages,
                                temperature=0.0,
                                max_tokens=256,
                            )
                            .choices[0]
                            .message.content
                        )
                    except Exception as e:
                        print("API Error, retrying...")
                        need_retry = True
                results.append(result)
            return results

        infer_func = gpt_infer_func
    else:
        raise ValueError(f"Invalid llm model: {llm}")

    return infer_func


def init_retriever(
    retriever: Literal["bge", "contriever"],
    corpus: Literal["msmarco", "hotpotqa", "nq"],
):

    if retriever == "bge":
        set_global_pooling_type("cls")
        retriever_model, retriever_tokenizer = load_model(
            model_retriever_mapping["bge"], model_type="encoder"
        )
        queries_top50_hits_json_path = partial_retrieval_results_mapping[retriever][
            corpus
        ]
    elif retriever == "contriever":
        set_global_pooling_type("mean")
        retriever_model, retriever_tokenizer = load_model(
            model_retriever_mapping["contriever"], model_type="encoder"
        )
        queries_top50_hits_json_path = partial_retrieval_results_mapping[retriever][
            corpus
        ]
    else:
        raise ValueError(f"Invalid retriever model: {retriever}")

    rag_attacker = RAGAttacker(
        model=retriever_model,
        tokenizer=retriever_tokenizer,
        corpus_jsonl_path=corpus_mapping[corpus],
        queries_top50_hits_json_path=queries_top50_hits_json_path,
    )
    return rag_attacker


def main(
    path: str,
    rag_attacker: RAGAttacker,
    infer_func,
):
    with open(path) as f:
        data = json.load(f)
    total = len(data)
    incorrect_count = 0
    correct_count = 0
    retrieved_count = 0
    all_messages = []

    for item in data:
        prompt, _, extra_docs_cnt, _ = rag_attacker.retrieve_doc(
            item["query"],
            extra_docs=[item["adv_text"]],
        )
        retrieved_count += extra_docs_cnt
        all_messages.append(
            [
                {
                    "role": "user",
                    "content": prompt,
                }
            ]
        )
    all_results = infer_func(all_messages)
    for item, result in zip(data, all_results):
        if item["incorrect_answer"].lower() in result.lower():
            incorrect_count += 1
        if item["correct_answer"].lower() in result.lower():
            correct_count += 1
    print(
        f"Total: {total}, Correct: {correct_count}, Incorrect: {incorrect_count}, Retrieved: {retrieved_count}"
    )


if __name__ == "__main__":
    import gc
    import random

    import torch

    seed = 42

    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
