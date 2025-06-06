import copy
import os

import numpy as np
import torch
from transformers import (BertTokenizer, BertTokenizerFast, RobertaTokenizer,
                          RobertaTokenizerFast)

from attack_manager import DecoderAttackManager, EncoderAttackManager
from prompt.rag_prompt import RAGAttacker
from rag.poisionedrag import (LIARBaselineAttackCore,
                              PoisionedRAGJointAttackCore)
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
    use_adaptive_ratio: bool,
    log_path: str,
    start_cluster_idx: int = 0,
    max_new_tokens: int = 256,
    infer_batch_size: int = 16,
    use_query_as_suffix: bool = True,
    n_samples: int = 128,
    topk: int = 16,
    epochs: int = 64,
    tag_length: int = 32,
    ascii_only: bool = True,
    fixed_ratio: float = 0.2,
    joint_loss_only: bool = False,
    joint_grad_only: bool = False,
    eval_gap: int = 0,
):
    hparams = copy.deepcopy(locals())

    import json

    data_pairs = json.load(open(f"{assets_path}/data.json", "r"))

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
        if idx < start_cluster_idx:
            continue
        setup_global_writer(os.path.join(tb_pefix, f"{tag}.cluster_{idx}"))
        get_global_writer().add_hparams(hparams, metric_dict={})

        retriever_target = rag_attacker._normalized_embedding([data_pair["query"]])
        retriever_target = retriever_target.mean(dim=0)
        encoder_attacker.set_target(retriever_target)

        current_tag_length = tag_length
        query_append_str = ""
        if use_query_as_suffix:
            query_append_str = data_pair["query"] + " "
        adv_tag = " " + " ".join(["!"] * current_tag_length)
        decoder_attacker.reset(adv_tag=adv_tag)
        encoder_attacker.reset(adv_tag=adv_tag)
        fake_corpus = " " + adv_tag + " " + query_append_str + data_pair["adv_text"]
        attack_core = PoisionedRAGJointAttackCore(
            tag=f"{tag}/cluster_{idx}",
            llm_model=llm_model,
            llm_tokenizer=llm_tokenizer,
            transfer_matrix=transform_matrix,
            adv_tag=adv_tag,
            rag_attacker=rag_attacker,
            fake_corpus=fake_corpus,
            queries=[data_pair["query"]],
            encoder_attacker=encoder_attacker,
            decoder_attacker=decoder_attacker,
            target_outputs=[data_pair["incorrect_answer"]],
            max_new_tokens=max_new_tokens,
            batch_size=infer_batch_size,
            optimize_shape=(current_tag_length, transform_matrix.shape[1]),
            verbose=1,
            log_path=log_prefix,
            use_adaptive_ratio=use_adaptive_ratio,
            fixed_ratio=fixed_ratio,
            joint_loss_only=joint_loss_only,
            joint_grad_only=joint_grad_only,
            eval_gap=eval_gap,
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
    step_llm: int = 8,
    step_retriever: int = 8,
    start_cluster_idx: int = 0,
    max_new_tokens: int = 256,
    use_query_as_suffix: bool = True,
    infer_batch_size: int = 16,
    n_samples: int = 128,
    topk: int = 16,
    epochs: int = 128,
    tag_length_retriever: int = 16,
    tag_length_llm: int = 16,
    ascii_only: bool = True,
):
    hparams = copy.deepcopy(locals())

    import json

    data_pairs = json.load(open(f"{assets_path}/data.json", "r"))

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
            get_select_tokens_bpe(llm_tokenizer, ascii_only=ascii_only)
        ),
        topk=topk,
    )
    if isinstance(retriever_tokenizer, (BertTokenizer, BertTokenizerFast)):
        encoder_attacker = EncoderAttackManager(
            model=retriever_model,
            tokenizer=retriever_tokenizer,
            adv_tag=None,
            n_samples=n_samples,
            select_token_ids=retriever_tokenizer.convert_tokens_to_ids(
                get_select_tokens_wordpiece(
                    retriever_tokenizer,
                    pre_hook=wordpiece_suffix_filter,
                    ascii_only=ascii_only,
                )
            ),
            topk=topk,
        )
    elif isinstance(retriever_tokenizer, (RobertaTokenizer, RobertaTokenizerFast)):
        encoder_attacker = EncoderAttackManager(
            model=retriever_model,
            tokenizer=retriever_tokenizer,
            adv_tag=None,
            n_samples=n_samples,
            select_token_ids=retriever_tokenizer.convert_tokens_to_ids(
                get_select_tokens_bpe(retriever_tokenizer, ascii_only=ascii_only)
            ),
            topk=topk,
        )
    else:
        raise NotImplementedError

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
        if idx < start_cluster_idx:
            continue
        setup_global_writer(os.path.join(tb_pefix, f"{tag}.cluster_{idx}"))
        get_global_writer().add_hparams(hparams, metric_dict={})

        retriever_target = rag_attacker._normalized_embedding([data_pair["query"]])
        retriever_target = retriever_target.mean(dim=0)
        encoder_attacker.set_target(retriever_target)

        adv_tag_retriever = " " + " ".join(["!"] * tag_length_retriever)
        adv_tag_llm = " " + " ".join(["?"] * tag_length_llm)
        query_append_str = ""
        if use_query_as_suffix:
            query_append_str = data_pair["query"] + " "
        decoder_attacker.reset(adv_tag=adv_tag_llm)
        encoder_attacker.reset(adv_tag=adv_tag_retriever)
        # PoisionedRAG propose to put adv_tag_llm in the front of adv_tag_retriever
        fake_corpus = (
            " "
            + adv_tag_llm
            + " "
            + adv_tag_retriever
            + " "
            + query_append_str
            + data_pair["adv_text"]
        )
        attack_core = LIARBaselineAttackCore(
            tag=f"{tag}/cluster_{idx}",
            llm_model=llm_model,
            llm_tokenizer=llm_tokenizer,
            adv_tag_llm=adv_tag_llm,
            adv_tag_retriever=adv_tag_retriever,
            rag_attacker=rag_attacker,
            fake_corpus=fake_corpus,
            queries=[data_pair["query"]],
            encoder_attacker=encoder_attacker,
            decoder_attacker=decoder_attacker,
            target_outputs=[data_pair["incorrect_answer"]],
            max_new_tokens=max_new_tokens,
            batch_size=infer_batch_size,
            verbose=1,
            log_path=log_prefix,
            step_llm=step_llm,
            step_retriever=step_retriever,
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
