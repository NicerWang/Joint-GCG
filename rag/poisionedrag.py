import json
from statistics import mean
from typing import Callable, List

from attack_manager import DecoderAttackManager, EncoderAttackManager
from core import BaselineAttackCore, JointAttackCore
from rag.base import RAGMixin
from utils import adaptive_ratio, clear_cache, get_global_writer


# set use_adaptive_ratio to False and fixed_ratio to 0.0 for PoisionedRAG baseline
class PoisionedRAGJointAttackCore(JointAttackCore, RAGMixin):
    def __init__(
        self,
        tag,
        llm_model,
        llm_tokenizer,
        transfer_matrix,
        adv_tag: str,
        rag_attacker,
        fake_corpus: str,
        queries: List[str],
        encoder_attacker: EncoderAttackManager,
        decoder_attacker: DecoderAttackManager,
        target_outputs: List[str],
        max_new_tokens: int,
        batch_size: int,
        optimize_shape: tuple,
        candidate_filter: Callable = None,
        verbose: int = 0,
        log_path: str = None,
        use_adaptive_ratio: bool = False,
        fixed_ratio: float = 0.0,
        joint_loss_only: bool = False,
        joint_grad_only: bool = False,
        eval_gap: int = 4,  # set to 0 to disable evaluation
    ):
        self.tag = tag
        self.llm_model = llm_model
        self.llm_tokenizer = llm_tokenizer
        self.rag_attacker = rag_attacker
        self.fake_corpus = fake_corpus
        self.target_outputs = target_outputs
        self.max_new_tokens = max_new_tokens
        self.batch_size = batch_size
        self.queries = queries
        self.encoder_attacker = encoder_attacker
        self.decoder_attacker = decoder_attacker
        self.optimize_shape = optimize_shape
        self.adv_tag = adv_tag
        self.transfer_matrix = transfer_matrix
        self.verbose = verbose
        self.log_path = log_path
        self.use_adaptive_ratio = use_adaptive_ratio
        self.fixed_ratio = fixed_ratio
        self.joint_loss_only = joint_loss_only
        self.joint_grad_only = joint_grad_only
        self.eval_gap = eval_gap
        self.candidate_filter = candidate_filter

        self.prev_loss_retriever = None
        self.prev_loss_llm = None
        self.writer = get_global_writer()

    def step(self, epoch, eval_only=False):
        clear_cache()
        self.log_data_json = {}
        self.log_data_pkl = {}
        if self.eval_gap == 0 or epoch % self.eval_gap != 0:
            _, all_messages_with_target, data_dict, similaritys = (
                self.calculate_asr_batch(self.queries, calculate_asr=False)
            )
        else:
            generated_messages, all_messages_with_target, data_dict, similaritys = (
                self.calculate_asr_batch(self.queries)
            )
            if self.verbose > 0:
                self.log_data_json["generate"] = generated_messages
                self.log_data_json["target"] = self.target_outputs

            self.writer.add_text("generate", json.dumps(generated_messages), epoch)

        self.record_data(
            data_dict,
            epoch,
        )

        if "poisioned" in data_dict and data_dict["poisioned"] > 0:
            return True

        if eval_only:
            return True

        if len(all_messages_with_target) == 0:
            # PoisionedRAG baseline do not have retriever optimization
            if not self.use_adaptive_ratio:
                return True
            ratio = 1
            self.step_retriever(epoch + 1)
        else:
            if self.use_adaptive_ratio:
                ratio = mean(
                    [adaptive_ratio(*similarity) for similarity in similaritys]
                )
            else:
                ratio = self.fixed_ratio
            self.step_llm(epoch + 1, ratio, all_messages_with_target)

        self.writer.add_scalar("ratio", ratio, epoch)

        if self.verbose > 0:
            self.log_data_json["ratio"] = ratio
            self.log_data_json["fake_corpus"] = self.fake_corpus
            self.log_data_json["adv_tag"] = self.adv_tag
            self.log_data_json["all_messages_with_target"] = all_messages_with_target
            self.log_data_json["data_dict"] = data_dict
            import os
            import pickle

            log_folder = os.path.join(self.log_path, self.tag)
            os.makedirs(log_folder, exist_ok=True)
            with open(os.path.join(log_folder, f"{epoch}.json"), "w") as f:
                json.dump(self.log_data_json, f, ensure_ascii=False, indent=2)
            if self.verbose > 1:
                with open(os.path.join(log_folder, f"{epoch}.pkl"), "wb") as f:
                    pickle.dump(self.log_data_pkl, f, 2)
            del self.log_data_json, self.log_data_pkl
        return False


class LIARBaselineAttackCore(BaselineAttackCore, RAGMixin):
    def __init__(
        self,
        tag,
        llm_model,
        llm_tokenizer,
        adv_tag_llm: str,
        adv_tag_retriever: str,
        rag_attacker,
        fake_corpus: str,
        queries: List[str],
        encoder_attacker: EncoderAttackManager,
        decoder_attacker: DecoderAttackManager,
        target_outputs: List[str],
        max_new_tokens: int,
        batch_size: int,
        candidate_filter: Callable = None,
        verbose: int = 0,
        log_path: str = None,
        step_llm: int = 20,
        step_retriever: int = 10,
        eval_gap: int = 4,  # set to 0 to disable evaluation
    ):
        self.tag = tag
        self.llm_model = llm_model
        self.llm_tokenizer = llm_tokenizer
        self.rag_attacker = rag_attacker
        self.fake_corpus = fake_corpus
        self.target_outputs = target_outputs
        self.max_new_tokens = max_new_tokens
        self.batch_size = batch_size
        self.queries = queries
        self.encoder_attacker = encoder_attacker
        self.decoder_attacker = decoder_attacker
        self.adv_tag_llm = adv_tag_llm
        self.adv_tag_retriever = adv_tag_retriever
        self.verbose = verbose
        self.log_path = log_path
        self.llm_step = step_llm
        self.retriever_step = step_retriever
        self.eval_gap = eval_gap
        self.candidate_filter = candidate_filter

        self.writer = get_global_writer()

    def step(self, epoch, eval_only=False):
        clear_cache()
        self.log_data_json = {}
        self.log_data_pkl = {}
        if self.eval_gap == 0 or epoch % self.eval_gap != 0:
            _, all_messages_with_target, data_dict, similaritys = (
                self.calculate_asr_batch(self.queries, calculate_asr=False)
            )
        else:
            generated_messages, all_messages_with_target, data_dict, similaritys = (
                self.calculate_asr_batch(self.queries)
            )
            if self.verbose > 0:
                self.log_data_json["generate"] = generated_messages
                self.log_data_json["target"] = self.target_outputs

            self.writer.add_text("generate", json.dumps(generated_messages), epoch)

        self.record_data(
            data_dict,
            epoch,
        )

        if "poisioned" in data_dict and data_dict["poisioned"] > 0:
            return True

        if eval_only:
            return True

        if (
            epoch % (self.retriever_step + self.llm_step) < self.retriever_step
            or len(all_messages_with_target) == 0
        ):
            self.step_retriever(epoch + 1)
        else:
            self.step_llm(epoch + 1, all_messages_with_target)

        if self.verbose > 0:
            self.log_data_json["fake_corpus"] = self.fake_corpus
            self.log_data_json["adv_tag_retriever"] = self.adv_tag_retriever
            self.log_data_json["adv_tag_llm"] = self.adv_tag_llm
            self.log_data_json["all_messages_with_target"] = all_messages_with_target
            self.log_data_json["data_dict"] = data_dict
            import os
            import pickle

            log_folder = os.path.join(self.log_path, self.tag)
            os.makedirs(log_folder, exist_ok=True)
            with open(os.path.join(log_folder, f"{epoch}.json"), "w") as f:
                json.dump(self.log_data_json, f, ensure_ascii=False, indent=2)
            if self.verbose > 1:
                with open(os.path.join(log_folder, f"{epoch}.pkl"), "wb") as f:
                    pickle.dump(self.log_data_pkl, f, 2)
            del self.log_data_json, self.log_data_pkl
        return False
