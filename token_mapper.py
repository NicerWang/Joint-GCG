# Return value: Mapped Token, whether it is a word (can appear independently)
def tiktoken_bpe_2_wordpiece(token):
    if token.startswith("Ġ"):
        token = token.replace("Ġ", "")
        return token, False
    else:
        token = "##" + token
        return token, True


def wordpiece_2_tiktoken_bpe(token):
    if token.startswith("##"):
        token = token.replace("##", "")
        return token, True
    else:
        token = "Ġ" + token
        return token, False


def sentence_piece_bpe_2_wordpiece(token):
    if token.startswith("▁"):
        token = token.replace("▁", "")
        return token, False
    else:
        token = "##" + token
        return token, True


def wordpiece_2_sentence_piece_bpe(token):
    if token.startswith("##"):
        token = token.replace("##", "")
        return token, True
    else:
        token = "▁" + token
        return token, False


def wordpiece_suffix_filter(token):
    if token.startswith("##"):
        return token, True
    else:
        return token, False


def tiktoken_bpe_suffix_filter(token):
    if not token.startswith("Ġ"):
        return token.replace("Ġ", ""), False
    else:
        return token, True


def sentence_piece_bpe_suffix_filter(token):
    if not token.startswith("▁"):
        return token.replace("▁", ""), False
    else:
        return token, True


def get_select_tokens_bpe(
    tokenizer, *, pre_hook=None, ascii_only=True, return_transformed_token=False
):
    """
    Obtain replaceable tokens
    """
    select_tokens = []

    def is_ascii(s):
        if s.startswith("Ġ"):
            s = s[1:]
        return s.isascii() and s.isprintable()

    def is_alphabet(s):
        if s.startswith("Ġ") and len(s) > 1:
            s = s[1]
        else:
            s = s[0]
        return s.isalpha()

    def is_start_with_space(s):
        return s.startswith("Ġ")

    for token in tokenizer.get_vocab().keys():
        _token = None
        if pre_hook is not None:
            _token, skip = pre_hook(token)
            if skip:
                continue
        # Non-ASCII token
        if ascii_only and not is_ascii(_token if _token is not None else token):
            continue
        # Remove tokens with special characters as the first letter.
        if not is_alphabet(_token if _token is not None else token):
            continue
        if not is_start_with_space(_token if _token is not None else token):
            continue
        # "Non-special token"
        if token in tokenizer.all_special_tokens:
            continue
        # Remove possible special characters
        special_char = False
        for char in ["'", '"', "\\"]:
            if char in str((r"%r" % token).strip())[1:-1]:
                special_char = True
        if special_char:
            continue
        # Everything is normal, the token is available.
        select_tokens.append(
            _token if (return_transformed_token and _token is not None) else token
        )
    print(f"Select tokens BPE: {len(select_tokens)}")
    return select_tokens


def get_select_tokens_wordpiece(
    tokenizer, *, pre_hook=None, ascii_only=True, return_transformed_token=False
):
    """
    Obtain replaceable tokens
    """
    select_tokens = []

    def is_ascii(s):
        return s.isascii() and s.isprintable()

    for token in tokenizer.get_vocab().keys():
        _token = None
        if pre_hook is not None:
            _token, skip = pre_hook(token)
            if skip:
                continue
        # Non-ASCII token
        if ascii_only and not is_ascii(_token if _token is not None else token):
            continue
        # "Non-special token"
        if token in tokenizer.all_special_tokens or (
            token.startswith("[") and token.endswith("]")
        ):
            continue
        # Remove possible special characters
        special_char = False
        for char in ["'", '"', "\\"]:
            if char in str((r"%r" % token).strip())[1:-1]:
                special_char = True
        if special_char:
            continue
        # Everything is normal, the token is available.
        select_tokens.append(
            _token if (return_transformed_token and _token is not None) else token
        )
    print(f"Select tokens WordPiece: {len(select_tokens)}")
    return select_tokens


# Maintain compatibility
get_select_tokens_bert = get_select_tokens_wordpiece
get_select_tokens_llama3 = get_select_tokens_bpe


class TokenGradJoiner:
    @staticmethod
    def get_llama_to_bert_mapping(bert_offsets, llama_offsets):
        """
        Generate a weighted mapping for each LLaMa token to corresponding BERT tokens based on character overlap.

        Args:
        - bert_offsets: List of tuples representing (start, end) character positions of BERT tokens.
        - llama_offsets: List of tuples representing (start, end) character positions of LLaMa tokens.

        Returns:
        - llama_to_bert_mapping: A list of lists where each entry corresponds to a LLaMa token,
        and contains tuples of BERT token index and the weight (based on character overlap).
        """
        llama_to_bert_mapping = []

        # Iterate through LLaMa token offsets
        for llama_start, llama_end in llama_offsets:
            aligned_bert_tokens = []

            # Calculate the total length of the LLaMa token (for normalization)
            llama_token_length = llama_end - llama_start

            # Find BERT tokens that overlap with the LLaMa token range
            for bert_idx, (bert_start, bert_end) in enumerate(bert_offsets):
                # Calculate overlap between the LLaMa and BERT token
                overlap_start = max(llama_start, bert_start)
                overlap_end = min(llama_end, bert_end)
                overlap_length = max(
                    0, overlap_end - overlap_start
                )  # Ensure non-negative

                if overlap_length > 0:
                    # Calculate weight as the ratio of the overlap length to the total LLaMa token length
                    weight = overlap_length / llama_token_length
                    aligned_bert_tokens.append((bert_idx, weight))

            # Append the weighted mapping for the current LLaMa token
            llama_to_bert_mapping.append(aligned_bert_tokens)

        return llama_to_bert_mapping

    @staticmethod
    def join_token_grads(ret_token_grads, llama_token_grads, llama_to_bert_mapping):
        """
        Combine LLaMa token gradients with BERT token gradients using the provided mapping and weights.
        Args:
        - llama_token_grads: List of gradients for LLaMa tokens, needs to be projected to BERT token space.
        - ret_token_grads: List of gradients for BERT tokens.
        - llama_to_bert_mapping: A list of lists where each entry corresponds to a LLaMa token,
          and contains tuples of BERT token index and the weight (based on character overlap).
        Returns:
        - fused_grads: A list of gradients for the fused tokens (LLaMa + weighted BERT gradients).
        """
        fused_grads = []

        for llama_idx, llama_mappings in enumerate(llama_to_bert_mapping):
            llama_grad = llama_token_grads[llama_idx]
            fused_grad = llama_grad.clone()

            for bert_idx, weight in llama_mappings:
                bert_grad = ret_token_grads[bert_idx]
                fused_grad += weight * bert_grad

            fused_grads.append(fused_grad)

        return fused_grads
