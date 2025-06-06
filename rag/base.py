from utils import generate_batch


class RAGMixin:
    def calculate_asr_batch(
        self, queries, targets=None, calculate_asr=True, return_success_idx=False
    ):
        if targets is None:
            targets = self.target_outputs
        data_dict = {"retrieved": 0, "poisioned": 0}
        positions = []
        similaritys = []

        all_messages = []
        all_messages_with_target = []
        all_retrieved_idxs = []
        for idx, query in enumerate(queries):
            (system_message, similarity, retrieved_extra_cnt, position) = (
                self.rag_attacker.retrieve_doc(
                    query,
                    extra_docs=[self.fake_corpus],
                )
            )
            position = position[0] + 1 if len(position) > 0 else 6
            positions.append(position)
            similaritys.append((position, similarity))
            if retrieved_extra_cnt != 0:
                data_dict["retrieved"] += 1
                messages = [
                    {
                        "role": "user",
                        "content": system_message,
                    }
                ]
                all_messages.append(messages)
                all_messages_with_target.append(
                    messages + [{"role": "assistant", "content": targets[idx]}]
                )
                all_retrieved_idxs.append(idx)

        data_dict["mean_position"] = sum(positions) / len(positions)

        if not calculate_asr:
            data_dict.pop("poisioned")
            return None, all_messages_with_target, data_dict, similaritys

        success_idxs = []
        generated_messages = generate_batch(
            self.llm_model,
            self.llm_tokenizer,
            all_messages,
            stop=None,
            max_new_tokens=self.max_new_tokens,
            batch_size=self.batch_size,
        )
        for generated_message, idx in zip(generated_messages, all_retrieved_idxs):
            if generated_message.lower().find(targets[idx].lower()) != -1:
                data_dict["poisioned"] += 1
                if return_success_idx:
                    success_idxs.append(idx)
        if return_success_idx:
            return (
                generated_messages,
                all_messages_with_target,
                success_idxs,
                all_retrieved_idxs,
            )

        return (generated_messages, all_messages_with_target, data_dict, similaritys)

    def get_doc(self):
        return self.fake_corpus

    def update_doc(self, old_tag, new_tag):
        self.fake_corpus = self.fake_corpus.replace(old_tag, new_tag)
