from typing import List, Literal

import torch
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizer

from mcg import (cal_losses, filter_candidates, sample_replacements,
                 token_gradients)
from utils import clear_cache_decorator, loss_calculator_wrapper, timeit


class BaseAttackManager:
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        adv_tag: str,
        n_samples: int,
        select_tokens_ids: List[int],
        topk: int,
        model_type: Literal["encoder", "decoder"],
        model_id: str = None,
    ):
        """
        Args:
            adv_tag (str): Content that can be optimized for attack
            n_samples (int): Number of samples per attack step
            select_tokens_ids (List[int]): List of available token IDs
            topk (int): Number of tokens that can be selected for each position
            model_type (Literal["encoder", "decoder"]): Type of the model
            model_id (str, optional): Model ID used for TensorBoard logs, default is None. If None, model_type will be used as the model_id

        """
        self.model = model
        self.tokenizer = tokenizer
        self.adv_tag = adv_tag
        self.n_samples = n_samples
        self.select_tokens_ids = select_tokens_ids
        self.topk = topk
        self.model_type = model_type
        self.epoch = 0
        self.optimization_tokens = None
        self.old_adv_tags = []
        if model_id is None:
            self.model_id = model_type
        else:
            self.model_id = model_id

    def post_hook(self, new_adv_tag):
        """
        Post-processing function to update the state

        Args:
            new_adv_tag (str): The newly selected content that can be optimized for attack

        """
        self.epoch += 1
        self.adv_tag = new_adv_tag
        self.optimization_tokens = None

    def reset(self, adv_tag, reset_epoch=True):
        """
        Reset the state to restart the attack

        Args:
            adv_tag (str): The new content that can be optimized
        """
        if reset_epoch:
            self.epoch = 0
        self.adv_tag = adv_tag
        self.optimization_tokens = None

    def sample_candidates(
        self, coordinate_grad, candidate_checker, space_sep_tokens=False
    ) -> List[str]:
        """
        Sample some new candidate replacements based on the gradient at each coordinate position

        Args:
            coordinate_grad (Tensor): The tensor of gradients at each coordinate position
            candidate_checker (Callable): The filter function for candidate replacements, with parameters (candidates: List[str], adv_tag: str)

        Returns:
            List[str]: The filtered candidate replacements
        """
        with torch.no_grad():
            with timeit(self.epoch, f"sample_replacements/{self.model_type}"):
                sampled_tokens = sample_replacements(
                    self.optimization_tokens,
                    coordinate_grad,
                    self.n_samples * 8,
                    self.select_tokens_ids,
                    self.topk,
                    self.epoch,
                )
        filtered_candidates = filter_candidates(
            self.tokenizer,
            sampled_tokens,
            filter_candidate=True,
            space_sep_tokens=space_sep_tokens,
        )
        assert len(filtered_candidates) > 0, "Sampled candidate replacements are empty"

        if candidate_checker:
            filtered_candidates = candidate_checker(filtered_candidates, self.adv_tag)
        filtered_candidates = filtered_candidates[: self.n_samples]
        assert len(filtered_candidates) > 0, "Sampled candidate replacements are empty"
        return filtered_candidates

    @clear_cache_decorator
    def cal_loss(
        self,
        candidates,
        input_ids,
        optimize_slice,
        target=None,
        loss_slice=None,
        batch_size=8,
    ):
        """
        Batch calculate the loss of candidates and return the loss tensor
        """
        if self.model_type == "encoder":
            batch_size = 128
        with timeit(self.epoch, f"{self.model_type}/cal_losses"):
            with torch.no_grad():
                losses = cal_losses(
                    self.model,
                    self.tokenizer,
                    input_ids,
                    loss_calculator_wrapper(
                        self.target if target is None else target, self.model_type
                    ),
                    optimize_slice,
                    candidates,
                    batch_size,
                    loss_slice,
                    model_type=self.model_type,
                )
        return losses


class DecoderAttackManager(BaseAttackManager):

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        adv_tag: str,
        n_samples: int,
        select_token_ids: List[int],
        topk: int,
        model_id: str = None,
    ):
        """
        Args:
            adv_tag (str): The content that can be optimized for attack
            n_samples (int): The number of samples per attack step
            select_token_ids (List[int]): A list of available token IDs
            topk (int): The number of tokens that can be selected for each position
            model_id (str, optional): Model ID, used for tensorboard logging, defaults to None. If None, model_type is used as model_id
        """
        super().__init__(
            model,
            tokenizer,
            adv_tag,
            n_samples,
            select_token_ids,
            topk,
            model_type="decoder",
            model_id=model_id,
        )

    @clear_cache_decorator
    def step(self, messages: List[str]):
        # generate target and loss_slice
        full_prompt = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=False, tokenize=False
        )
        clear_prompt = self.tokenizer.apply_chat_template(
            messages[:-1], add_generation_prompt=True, tokenize=False
        )

        str_optimize_slice = slice(
            full_prompt.find(self.adv_tag),
            full_prompt.find(self.adv_tag) + len(self.adv_tag),
        )
        assert str_optimize_slice.start != -1, "adv_tag not found"

        full_ids = self.tokenizer(full_prompt, add_special_tokens=False).input_ids
        clear_ids = self.tokenizer(clear_prompt, add_special_tokens=False).input_ids
        # Normally, all should be reduced by 1, reducing by 2 is to remove the end token
        loss_slice = slice(len(clear_ids) - 1, len(full_ids) - 2)

        # Convert the front and back segments to IDs according to str_optimize_slice
        input_sequence_before = full_prompt[: str_optimize_slice.start]
        input_sequence_after = full_prompt[str_optimize_slice.stop :]
        optimize_sequence = full_prompt[str_optimize_slice]
        input_sequence_ids_before = self.tokenizer(
            input_sequence_before, add_special_tokens=False
        ).input_ids
        input_sequence_ids_after = self.tokenizer(
            input_sequence_after, add_special_tokens=False
        ).input_ids

        # build optimize_slice and input_ids
        optimize_sequence_ids = self.tokenizer(
            optimize_sequence, add_special_tokens=False
        ).input_ids
        optimize_slice = slice(
            len(input_sequence_ids_before),
            len(input_sequence_ids_before) + len(optimize_sequence_ids),
        )
        input_ids = (
            input_sequence_ids_before + optimize_sequence_ids + input_sequence_ids_after
        )

        assert (
            input_ids == self.tokenizer(full_prompt, add_special_tokens=False).input_ids
        ), "The split position has caused damage to the token."

        input_ids = torch.tensor(input_ids).to(self.model.device)

        # For all samples, optimization_tokens (within each step) are the same, only need to be processed once.
        if self.optimization_tokens is None:
            self.optimization_tokens = input_ids[optimize_slice]
        target = torch.tensor(
            full_ids[len(clear_ids) : len(full_ids) - 1], device=self.model.device
        )
        # Calculate the gradient at each coordinate location
        with timeit(self.epoch, f"{self.model_id}/token_gradients"):
            coordinate_grad = token_gradients(
                self.model,
                input_ids,
                optimize_slice,
                loss_calculator_wrapper(target, self.model_type),
                loss_slice,
                model_type=self.model_type,
            )
        return coordinate_grad, input_ids, optimize_slice, target, loss_slice

    @clear_cache_decorator
    def batch_step(self, all_messages: List[List[str]]):
        llm_coorinate_grad = 0
        llm_optimize_data = []
        for messages in tqdm(all_messages, desc="Accumulate Coordinate Grads"):
            coordinate_grad, input_ids, optimize_slice, target, loss_slice = self.step(
                messages
            )
            llm_coorinate_grad += coordinate_grad
            llm_optimize_data.append((input_ids, optimize_slice, target, loss_slice))
        return llm_coorinate_grad, llm_optimize_data

    @clear_cache_decorator
    def batch_cal_losses(self, candidates: List[str], llm_optimize_data):
        llm_loss = 0
        for input_ids, optimize_slice, target, loss_slice in tqdm(
            llm_optimize_data, desc="Calculate Loss"
        ):
            loss = self.cal_loss(
                candidates, input_ids, optimize_slice, target, loss_slice
            )
            llm_loss += loss
        return llm_loss


class EncoderAttackManager(BaseAttackManager):
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        adv_tag: str,
        n_samples: int,
        select_token_ids: List[int],
        topk: int,
        model_id: str = None,
    ):
        """
        The Encoder model is applicable to all input sequences [CLS]input_sequence[SEP] and performs MeanPooling.

        Args:
            adv_tag (str): Content to be optimized for attack
            n_samples (int): Number of samples per attack step
            select_token_ids (List[int]): List of available token IDs
            topk (int): Number of selectable tokens per position
            model_id (str, optional): Model ID for tensorboard logging, defaults to None. If None, model_type is used as model_id
        """
        super().__init__(
            model,
            tokenizer,
            adv_tag,
            n_samples,
            select_token_ids,
            topk,
            model_type="encoder",
            model_id=model_id,
        )

    def set_target(self, target: torch.Tensor):
        self.target = target.to(self.model.device)

    @clear_cache_decorator
    def step(self, input_sequence: str):

        str_optimize_slice = slice(
            input_sequence.find(self.adv_tag),
            input_sequence.find(self.adv_tag) + len(self.adv_tag),
        )
        assert str_optimize_slice.start != -1, "Not found adv_tag"

        # According to str_optimize_slice, convert the two segments before and after into id
        input_sequence_before = input_sequence[: str_optimize_slice.start]
        input_sequence_after = input_sequence[str_optimize_slice.stop :]
        optimize_sequence = input_sequence[str_optimize_slice]
        input_sequence_ids_before = self.tokenizer(
            input_sequence_before, add_special_tokens=False
        ).input_ids
        input_sequence_ids_after = self.tokenizer(
            input_sequence_after, add_special_tokens=False
        ).input_ids

        # Develop the functions `optimize_slice` and `input_ids`.
        optimize_sequence_ids = self.tokenizer(
            optimize_sequence, add_special_tokens=False
        ).input_ids
        optimize_slice = slice(
            len(input_sequence_ids_before) + 1,
            len(input_sequence_ids_before) + 1 + len(optimize_sequence_ids),
        )
        input_ids = (
            [self.tokenizer.cls_token_id]
            + input_sequence_ids_before
            + optimize_sequence_ids
            + input_sequence_ids_after
            + [self.tokenizer.sep_token_id]
        )

        assert len(input_ids) == len(
            self.tokenizer(input_sequence, add_special_tokens=True).input_ids
        ), "The split position has caused damage to the token."

        input_ids = torch.tensor(input_ids).to(self.model.device)

        # For all samples, optimization_tokens (within each step) only need to be processed once.
        if self.optimization_tokens is None:
            self.optimization_tokens = input_ids[optimize_slice]

        # Calculate the gradient at each coordinate location
        with timeit(self.epoch, f"{self.model_id}/token_gradients"):
            coordinate_grad = token_gradients(
                self.model,
                input_ids,
                optimize_slice,
                loss_calculator_wrapper(self.target, self.model_type),
                model_type=self.model_type,
            )
        return coordinate_grad, input_ids, optimize_slice
