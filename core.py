import torch

from token_mapper import TokenGradJoiner


class BasicAttackCore:
    def record_data(
        self,
        data_dict,
        epoch,
    ):
        for key, value in data_dict.items():
            self.writer.add_scalar(f"data/{key}", value, epoch)

    def choose_candidate(self, candidates, loss, greater_is_better=False):
        if greater_is_better:
            best_id = loss.argmax().item()
        else:
            best_id = loss.argmin().item()
        new_candidate = candidates[best_id]
        return new_candidate, best_id


class JointAttackCore(BasicAttackCore):
    def normalize(self, x):
        return (x - x.mean().item()) / (x.max().item() - x.min().item())

    def build_offset_mapping(self):
        retriever_tokenized = self.encoder_attacker.tokenizer.encode_plus(
            self.adv_tag,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        llm_tokenized = self.decoder_attacker.tokenizer.encode_plus(
            self.adv_tag,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )

        if self.decoder_attacker.model_id == "llama3":
            # Fix the offset mapping of Llama3 Model
            input_ids = llm_tokenized["input_ids"]
            offset_mapping = llm_tokenized["offset_mapping"]

            for i, (input_id, (start, end)) in enumerate(
                zip(input_ids, offset_mapping)
            ):
                token = self.decoder_attacker.tokenizer.convert_ids_to_tokens(input_id)
                if input_id not in self.decoder_attacker.tokenizer.all_special_ids:
                    offset_mapping[i] = (start, start + len(token))

            llm_offset_mapping = offset_mapping

        else:
            llm_offset_mapping = llm_tokenized["offset_mapping"]

        for idx, (start, end) in enumerate(llm_offset_mapping):
            if self.adv_tag[start] == " ":
                llm_offset_mapping[idx] = (start + 1, end)

        return retriever_tokenized.offset_mapping, llm_offset_mapping

    def step_retriever(self, epoch):
        doc = self.get_doc()

        self.decoder_attacker.optimization_tokens = torch.tensor(
            self.decoder_attacker.tokenizer(self.adv_tag, add_special_tokens=False)[
                "input_ids"
            ]
        )
        retriever_coordinate_grad, retriever_input_ids, retriever_optimize_slice = (
            self.encoder_attacker.step(doc)
        )

        # Gradient Aggregation
        transfered_retriever_coordinate_grad = (
            retriever_coordinate_grad @ self.transfer_matrix
        )

        retriever_offset_mapping, llm_offset_mapping = self.build_offset_mapping()
        llama_to_bert_mapping = TokenGradJoiner.get_llama_to_bert_mapping(
            bert_offsets=retriever_offset_mapping, llama_offsets=llm_offset_mapping
        )
        all_coorinate_grad = torch.zeros(self.optimize_shape)
        for i in range(self.optimize_shape[0]):
            mappings = llama_to_bert_mapping[i]
            for bert_coordinate, weight in mappings:
                all_coorinate_grad[i] += (
                    transfered_retriever_coordinate_grad[bert_coordinate] * weight
                )

        candidates = self.decoder_attacker.sample_candidates(
            all_coorinate_grad, self.candidate_filter
        )
        if self.prev_loss_retriever is None:
            candidates.append(
                self.adv_tag
            )  # In the first step, delay the calculation of the initial loss.
        self.writer.add_scalar("candidate_num", len(candidates), epoch)
        retriever_loss = self.encoder_attacker.cal_loss(
            candidates, retriever_input_ids, retriever_optimize_slice
        )

        if self.verbose > 0:
            self.log_data_json["prev_loss_retriever"] = self.prev_loss_retriever

        if self.prev_loss_retriever is None:
            self.prev_loss_retriever = retriever_loss[-1].item()
            self.writer.add_scalar("loss/encoder", self.prev_loss_retriever, epoch - 1)
            self.writer.add_text("fake_corpus", doc, epoch - 1)

        candidate, best_id = self.choose_candidate(candidates, retriever_loss)
        self.prev_loss_retriever = retriever_loss[best_id].item()
        self.writer.add_scalar("loss/encoder", self.prev_loss_retriever, epoch)

        self.update_doc(self.adv_tag, candidate)
        self.decoder_attacker.post_hook(candidate)
        self.encoder_attacker.post_hook(candidate)
        self.writer.add_text("fake_corpus", self.get_doc(), epoch)
        self.adv_tag = candidate

        if self.verbose > 1:
            self.log_data_pkl["retriever_loss"] = retriever_loss.cpu().numpy()
            self.log_data_pkl["retriever_grad_norm"] = (
                retriever_coordinate_grad.cpu().numpy()
            )
            self.log_data_pkl["best_id"] = best_id
            self.log_data_pkl["candidates"] = candidates

    def step_llm(self, epoch, ratio, all_messages):
        doc = self.get_doc()
        # LLM optimization
        llm_coorinate_grad_raw, llm_optimize_data = self.decoder_attacker.batch_step(
            all_messages
        )
        llm_coorinate_grad = self.normalize(llm_coorinate_grad_raw)

        # Retrievr optimization
        (
            retriever_coordinate_grad_raw,
            retriever_input_ids,
            retriever_optimize_slice,
        ) = self.encoder_attacker.step(doc)
        retriever_coordinate_grad = self.normalize(retriever_coordinate_grad_raw)

        if self.joint_loss_only:
            all_coorinate_grad = llm_coorinate_grad
        else:
            # Gradient Aggregation
            transfered_retriever_coordinate_grad = (
                retriever_coordinate_grad @ self.transfer_matrix
            )
            retriever_offset_mapping, llm_offset_mapping = self.build_offset_mapping()
            llama_to_bert_mapping = TokenGradJoiner.get_llama_to_bert_mapping(
                bert_offsets=retriever_offset_mapping, llama_offsets=llm_offset_mapping
            )
            all_coorinate_grad = torch.zeros(self.optimize_shape)
            for i in range(self.optimize_shape[0]):
                all_coorinate_grad[i] = llm_coorinate_grad[i] * (1 - ratio)
                mappings = llama_to_bert_mapping[i]
                for bert_coordinate, weight in mappings:
                    all_coorinate_grad[i] += (
                        transfered_retriever_coordinate_grad[bert_coordinate]
                        * weight
                        * ratio
                    )

        # sample candidates
        candidates = self.decoder_attacker.sample_candidates(
            all_coorinate_grad, self.candidate_filter
        )
        if self.prev_loss_retriever is None or self.prev_loss_llm is None:
            candidates.append(
                self.adv_tag
            )  # In the first step, delay the calculation of the initial loss.
        self.writer.add_scalar("candidate_num", len(candidates), epoch)

        # LLM Loss
        llm_loss = self.decoder_attacker.batch_cal_losses(candidates, llm_optimize_data)
        llm_loss = llm_loss / len(all_messages)

        # Retriever Loss
        retriever_loss = self.encoder_attacker.cal_loss(
            candidates, retriever_input_ids, retriever_optimize_slice
        )
        if self.verbose > 0:
            self.log_data_json["prev_loss_llm"] = self.prev_loss_llm
            self.log_data_json["prev_loss_retriever"] = self.prev_loss_retriever

        if self.prev_loss_retriever is None:
            self.prev_loss_retriever = retriever_loss[-1].item()
            self.writer.add_scalar("loss/encoder", self.prev_loss_retriever, epoch - 1)
            self.writer.add_text("fake_corpus", doc, epoch - 1)
        if self.prev_loss_llm is None:
            self.prev_loss_llm = llm_loss[-1].item()
            self.writer.add_scalar("loss/decoder", self.prev_loss_llm, epoch - 1)
            self.writer.add_text("fake_corpus", doc, epoch - 1)

        # Calculate the decline rate
        loss_decrease_llm = self.prev_loss_llm - llm_loss
        loss_decrease_retriever = self.prev_loss_retriever - retriever_loss

        if self.joint_grad_only:
            loss_decrease = self.normalize(loss_decrease_llm)
        else:
            loss_decrease = (1 - ratio) * self.normalize(
                loss_decrease_llm
            ) + ratio * self.normalize(loss_decrease_retriever)

        candidate, best_id = self.choose_candidate(candidates, loss_decrease, True)

        self.prev_loss_llm = llm_loss[best_id].item()
        self.writer.add_scalar("loss/decoder", self.prev_loss_llm, epoch)
        self.prev_loss_retriever = retriever_loss[best_id].item()
        self.writer.add_scalar("loss/encoder", self.prev_loss_retriever, epoch)

        self.update_doc(self.adv_tag, candidate)
        self.decoder_attacker.post_hook(candidate)
        self.encoder_attacker.post_hook(candidate)
        self.writer.add_text("fake_corpus", self.get_doc(), epoch)
        self.adv_tag = candidate

        if self.verbose > 1:
            self.log_data_pkl["llm_loss"] = llm_loss.cpu().numpy()
            self.log_data_pkl["retriever_loss"] = retriever_loss.cpu().numpy()
            self.log_data_pkl["llm_grad"] = llm_coorinate_grad_raw.cpu().numpy()
            self.log_data_pkl["retriever_grad"] = (
                retriever_coordinate_grad_raw.cpu().numpy()
            )
            self.log_data_pkl["llm_grad_norm"] = llm_coorinate_grad.cpu().numpy()
            self.log_data_pkl["retriever_grad_norm"] = (
                retriever_coordinate_grad.cpu().numpy()
            )
            self.log_data_pkl["llm_decrease"] = loss_decrease_llm.cpu().numpy()
            self.log_data_pkl["retriever_decrease"] = (
                loss_decrease_retriever.cpu().numpy()
            )
            self.log_data_pkl["transfered_retriever_coordinate_grad"] = (
                transfered_retriever_coordinate_grad.cpu().numpy()
            )
            self.log_data_pkl["loss_decrease"] = loss_decrease
            self.log_data_pkl["best_id"] = best_id
            self.log_data_pkl["candidates"] = candidates


class BaselineAttackCore(BasicAttackCore):
    def step_llm(self, epoch, all_messages):
        llm_coorinate_grad, llm_optimize_data = self.decoder_attacker.batch_step(
            all_messages
        )
        candidates = self.decoder_attacker.sample_candidates(
            llm_coorinate_grad,
            self.candidate_filter,
        )
        if epoch == 1:
            candidates.append(self.adv_tag_retriever)
        self.writer.add_scalar("candidate_num", len(candidates), epoch)
        llm_loss = self.decoder_attacker.batch_cal_losses(candidates, llm_optimize_data)
        if epoch == 1:
            self.writer.add_scalar("loss/decoder", llm_loss[-1].item(), 0)
            self.writer.add_text("fake_corpus", self.get_doc(), 0)
        candidate, best_id = self.choose_candidate(candidates, llm_loss)
        self.writer.add_scalar("loss/decoder", llm_loss[best_id].item(), epoch)
        self.writer.add_text("fake_corpus", self.get_doc(), epoch)
        self.update_doc(self.adv_tag_llm, candidate)
        self.decoder_attacker.post_hook(candidate)
        self.adv_tag_llm = candidate

    def step_retriever(self, epoch):
        doc = self.get_doc()
        retriever_coordinate_grad, retriever_input_ids, retriever_optimize_slice = (
            self.encoder_attacker.step(doc)
        )
        candidates = self.encoder_attacker.sample_candidates(
            retriever_coordinate_grad,
            self.candidate_filter,
        )
        if epoch == 1:
            candidates.append(self.adv_tag_retriever)
        self.writer.add_scalar("candidate_num", len(candidates), epoch)
        retriever_loss = self.encoder_attacker.cal_loss(
            candidates, retriever_input_ids, retriever_optimize_slice
        )
        if epoch == 1:
            self.writer.add_scalar("loss/encoder", retriever_loss[-1].item(), 0)
            self.writer.add_text("fake_corpus", doc, 0)

        candidate, best_id = self.choose_candidate(candidates, retriever_loss)
        self.writer.add_scalar("loss/encoder", retriever_loss[best_id].item(), epoch)
        self.writer.add_text("fake_corpus", self.get_doc(), epoch)
        self.update_doc(self.adv_tag_retriever, candidate)
        self.encoder_attacker.post_hook(candidate)
        self.adv_tag_retriever = candidate
