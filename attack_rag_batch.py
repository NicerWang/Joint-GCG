import copy
import os

import numpy as np
import torch

from attack_manager import DecoderAttackManager, EncoderAttackManager
from prompt.rag_prompt import RAGAttacker
from rag.phantom import PhantomBaselineAttackCore, PhantomJointAttackCore
from token_mapper import *
from utils import (get_global_writer, get_transfer_matrix, load_model,
                   setup_global_writer)


def attack_joint(
    tag: str,
    model_llm: str,
    model_retriever: str,
    assets_path: str,
    transfer_matrix_path: str,
    corpus_jsonl_path: str,
    queries_top50_hits_json_path: str,
    log_path: str,
    n_samples: int = 256,
    topk: int = 32,
    epochs: int = 32,
    tag_length: int = 8,
    ascii_only: bool = True,
    start_cluster_idx: int = 0,
    max_new_tokens: int = 256,
    infer_batch_size: int = 2,
    retriever_optimize_steps: int = 0,
    initial_fake_corpus_without_tag_cmd: str = None,
):
    hparams = copy.deepcopy(locals())

    import json

    data_pairs = json.load(open(f"{assets_path}/data.json", "r"))
    attack_objectives = data_pairs["attack_objective"]
    data_pairs = data_pairs["data"]

    if not isinstance(attack_objectives, list):
        attack_objectives = [attack_objectives]

    import random

    seed = 42

    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    retriever_model, retriever_tokenizer = load_model(
        model_retriever, model_type="encoder"
    )
    llm_model, llm_tokenizer = load_model(model_llm, model_type="decoder")

    select_input_ids_llm, select_input_ids_retriever, transform_matrix = (
        get_transfer_matrix(
            llm_tokenizer,
            retriever_tokenizer,
            transfer_matrix_path=transfer_matrix_path,
            ascii_only=ascii_only,
        )
    )
    decoder_attacker = DecoderAttackManager(
        model=llm_model,
        tokenizer=llm_tokenizer,
        adv_tag=None,
        n_samples=n_samples,
        select_token_ids=select_input_ids_llm,
        topk=topk,
        model_id="llama3" if "Meta-Llama-3" in model_llm else None,
    )
    encoder_attacker = EncoderAttackManager(
        model=retriever_model,
        tokenizer=retriever_tokenizer,
        adv_tag=None,
        n_samples=n_samples,
        select_token_ids=select_input_ids_retriever,
        topk=topk,
    )

    rag_attacker = RAGAttacker(
        model=retriever_model,
        tokenizer=retriever_tokenizer,
        corpus_jsonl_path=corpus_jsonl_path,
        queries_top50_hits_json_path=queries_top50_hits_json_path,
    )

    tb_pefix = os.path.join(log_path, "tb")
    log_prefix = os.path.join(log_path, "log")
    os.makedirs(tb_pefix, exist_ok=True)

    for idx, data_pair in enumerate(data_pairs):
        for obj in attack_objectives:
            try:
                initial_fake_corpus_without_tag_cmd = data_pair[
                    "initial_fake_corpus_without_tag_cmd"
                ]
                print(
                    f"Using initial_fake_corpus from data.json: {initial_fake_corpus_without_tag_cmd}"
                )
            except KeyError:
                initial_fake_corpus_without_tag_cmd = None

            if idx < start_cluster_idx:
                continue
            setup_global_writer(
                os.path.join(tb_pefix, f"{tag}.cluster_{idx}_obj_{obj}")
            )
            get_global_writer().add_hparams(hparams, metric_dict={})

            in_retriever_targets = rag_attacker._normalized_embedding(
                data_pair["in_query"]
            )
            if len(data_pair["out_query"]) == 0:
                out_retriever_targets = torch.zeros_like(in_retriever_targets)
            else:
                out_retriever_targets = rag_attacker._normalized_embedding(
                    data_pair["out_query"]
                )
            retriever_target = in_retriever_targets.mean(
                dim=0
            ) - out_retriever_targets.mean(dim=0)
            encoder_attacker.set_target(retriever_target)

            adv_tag = " " + " ".join(["!"] * tag_length)
            decoder_attacker.reset(adv_tag=adv_tag)
            encoder_attacker.reset(adv_tag=adv_tag)
            attack_core = PhantomJointAttackCore(
                tag=f"{tag}/cluster_{idx}",
                llm_model=llm_model,
                llm_tokenizer=llm_tokenizer,
                transfer_matrix=transform_matrix,
                adv_tag=adv_tag,
                rag_attacker=rag_attacker,
                queries=data_pair["in_query"],
                test_queries=data_pair["test_query"],
                trigger=data_pair["trigger"],
                attack_objective=obj,
                encoder_attacker=encoder_attacker,
                decoder_attacker=decoder_attacker,
                max_new_tokens=max_new_tokens,
                batch_size=infer_batch_size,
                optimize_shape=(tag_length, transform_matrix.shape[1]),
                verbose=1,
                log_path=log_prefix,
                retriever_optimize_steps=retriever_optimize_steps,
                initial_fake_corpus_without_tag_cmd=initial_fake_corpus_without_tag_cmd,
            )
            for epoch in range(epochs + 1):
                os.system("clear")
                print(f"Attack #{idx}/{len(data_pairs)}")
                print(f"Current: {epoch}/{epochs}")
                success = attack_core.step(
                    epoch, eval_only=True if epoch == epochs else False
                )
                if success:
                    break


def attack_base(
    tag: str,
    model_llm: str,
    model_retriever: str,
    assets_path: str,
    corpus_jsonl_path: str,
    queries_top50_hits_json_path: str,
    log_path: str,
    n_samples: int = 256,
    topk: int = 32,
    epochs: int = 256 + 32,
    tag_length_retriever: int = 128,
    tag_length_llm: int = 8,
    ascii_only: bool = True,
    start_cluster_idx: int = 0,
    max_new_tokens: int = 256,
    infer_batch_size: int = 2,
    retriever_optimize_steps: int = 256,
):
    hparams = copy.deepcopy(locals())

    import json

    data_pairs = json.load(open(f"{assets_path}/data.json", "r"))
    attack_objectives = data_pairs["attack_objective"]
    data_pairs = data_pairs["data"]

    if not isinstance(attack_objectives, list):
        attack_objectives = [attack_objectives]

    import random

    seed = 42

    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    retriever_model, retriever_tokenizer = load_model(
        model_retriever, model_type="encoder"
    )
    llm_model, llm_tokenizer = load_model(model_llm, model_type="decoder")

    decoder_attacker = DecoderAttackManager(
        model=llm_model,
        tokenizer=llm_tokenizer,
        adv_tag=None,
        n_samples=n_samples,
        select_token_ids=llm_tokenizer.convert_tokens_to_ids(
            get_select_tokens_llama3(llm_tokenizer, ascii_only=ascii_only)
        ),
        topk=topk,
    )
    encoder_attacker = EncoderAttackManager(
        model=retriever_model,
        tokenizer=retriever_tokenizer,
        adv_tag=None,
        n_samples=n_samples,
        select_token_ids=retriever_tokenizer.convert_tokens_to_ids(
            get_select_tokens_bert(
                retriever_tokenizer,
                pre_hook=wordpiece_suffix_filter,
                ascii_only=ascii_only,
            )
        ),
        topk=topk,
    )

    rag_attacker = RAGAttacker(
        model=retriever_model,
        tokenizer=retriever_tokenizer,
        corpus_jsonl_path=corpus_jsonl_path,
        queries_top50_hits_json_path=queries_top50_hits_json_path,
    )

    tb_pefix = os.path.join(log_path, "tb")
    log_prefix = os.path.join(log_path, "log")
    os.makedirs(tb_pefix, exist_ok=True)

    for idx, data_pair in enumerate(data_pairs):
        for obj in attack_objectives:
            if idx < start_cluster_idx:
                continue
            setup_global_writer(
                os.path.join(tb_pefix, f"{tag}.cluster_{idx}_obj_{obj}")
            )
            get_global_writer().add_hparams(hparams, metric_dict={})

            in_retriever_targets = rag_attacker._normalized_embedding(
                data_pair["in_query"]
            )
            if len(data_pair["out_query"]) == 0:
                out_retriever_targets = torch.zeros_like(in_retriever_targets)
            else:
                out_retriever_targets = rag_attacker._normalized_embedding(
                    data_pair["out_query"]
                )
            retriever_target = in_retriever_targets.mean(
                dim=0
            ) - out_retriever_targets.mean(dim=0)
            encoder_attacker.set_target(retriever_target)

            adv_tag_retriever = " " + " ".join(["!"] * tag_length_retriever)
            adv_tag_llm = " " + " ".join(["?"] * tag_length_llm)
            decoder_attacker.reset(adv_tag=adv_tag_llm)
            encoder_attacker.reset(adv_tag=adv_tag_retriever)
            # PoisionedRAG propose to put adv_tag_llm in the front of adv_tag_retriever

            attack_core = PhantomBaselineAttackCore(
                tag=f"{tag}/cluster_{idx}",
                llm_model=llm_model,
                llm_tokenizer=llm_tokenizer,
                adv_tag_llm=adv_tag_llm,
                adv_tag_retriever=adv_tag_retriever,
                rag_attacker=rag_attacker,
                queries=data_pair["in_query"],
                test_queries=data_pair["test_query"],
                encoder_attacker=encoder_attacker,
                decoder_attacker=decoder_attacker,
                trigger=data_pair["trigger"],
                attack_objective=obj,
                max_new_tokens=max_new_tokens,
                batch_size=infer_batch_size,
                verbose=1,
                log_path=log_prefix,
                retriever_optimize_steps=retriever_optimize_steps,
            )
            for epoch in range(epochs + 1):
                os.system("clear")
                print(f"Attack #{idx}/{len(data_pairs)}")
                print(f"Current: {epoch}/{epochs}")
                success = attack_core.step(
                    epoch, eval_only=True if epoch == epochs else False
                )
                if success:
                    break
