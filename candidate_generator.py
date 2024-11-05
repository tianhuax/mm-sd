import torch
from typing import Optional, Tuple, Dict
import copy
from transformers.cache_utils import Cache
from transformers import DynamicCache
import inspect


class AssistedCandidateGenerator():
    def __init__(
        self,
        input_ids,
        assistant_model,
        generation_config,
        model_kwargs,
        original_inputs,  # Add this parameter
    ):
        # Make sure all data at the same device as assistant model
        self.device = assistant_model.device
        input_ids = input_ids.to(self.device)

        # Prepare the assistant and the starting number of candidate tokens
        self.assistant_model = assistant_model
        self.num_assistant_tokens = assistant_model.generation_config.num_assistant_tokens
        self.assistant_confidence_threshold = assistant_model.generation_config.assistant_confidence_threshold

        # Set eos in assistant same as in target model
        self.assistant_model.generation_config.eos_token_id = generation_config.eos_token_id

        # Prepare the relevant kwargs for the assistant model
        assistant_kwargs = {}
        excluded_keys = (
            "encoder_outputs",
            "assistant_encoder_outputs",
            "past_key_values",
            "max_new_tokens",  # exclude to prevent duplication
            "min_new_tokens",  # exclude to prevent duplication
        )
        for key, value in model_kwargs.to_dict().items():  
            if key not in excluded_keys:
                assistant_kwargs[key] = (
                    value.detach().to(self.device) if isinstance(value, torch.Tensor) else copy.deepcopy(value)
                )
        self.assistant_kwargs = assistant_kwargs
        self.input_ids_key = "input_ids"

        # Prepare generation-related options.
        self.generation_config = copy.deepcopy(generation_config)

        self.generation_config.return_dict_in_generate = True
        self.generation_config.output_scores = True
        self.generation_config.assistant_confidence_threshold = self.assistant_confidence_threshold
        # this flag allow us set the confidence stopping criteria for assistant model generation.
        self.generation_config.is_assistant = True

        # avoid unnecessary warnings that min_length is larger than max_new_tokens
        # remove the `MinLengthLogitsProcessor` if exists (NOTE: no need to check for `MinNewTokensLogitsProcessor`)
        self.main_model_min_length = self.generation_config.min_length
        self.generation_config.min_length = 0
        self.generation_config.min_new_tokens = None

        # We need to roll back the cache in assisted generation, only DynamicCache is supported
        self.generation_config.cache_implementation = None

        # Store the original inputs
        self.original_inputs = original_inputs

        # Store input_ids for tracking
        self.input_ids = input_ids
        
    def get_candidates(self, input_ids: torch.LongTensor) -> Tuple[torch.LongTensor, Optional[torch.FloatTensor]]:
        """
        Fetches the candidates to be tried for the current input.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. [What are input IDs?](../glossary#input-ids)

        Return:
            `torch.LongTensor` of shape `(batch_size, candidate_length)` containing the candidate sequences to be
            assessed by the model and a `torch.FloatTensor` of shape `(batch_size, candidate_length,
            vocabulary_size)` containing the logits associated to each candidate.
        """
        input_ids = input_ids.to(self.device)

        # Don't generate more than `max_length - 1` candidates since the target model generates one extra token.
        new_cur_len = input_ids.shape[-1]
        max_new_tokens = min(int(self.num_assistant_tokens), self.generation_config.max_length - new_cur_len - 1)
        min_new_tokens = max(min(max_new_tokens, self.main_model_min_length - new_cur_len), 0)
        if max_new_tokens == 0:
            return input_ids, None

        # 1. If it is not the first round of candidate generation, prepare the inputs based on the input_ids length
        # (which implicitly contains the number of accepted candidates from the previous round)
        has_past_key_values = self.assistant_kwargs.get("past_key_values", None) is not None
        if has_past_key_values:
            new_cache_size = new_cur_len - 1
            self.assistant_kwargs["past_key_values"] = self.crop_past_key_values(
                self.assistant_kwargs["past_key_values"], new_cache_size - 1
            )  # the assistant does not have the token after the last match, hence the -1

            self.assistant_kwargs = self.prepare_attention_mask(
                self.assistant_kwargs, new_cur_len
            )
            self.assistant_kwargs = self.prepare_token_type_ids(self.assistant_kwargs, new_cur_len)

        # 2. Forecast next N tokens using the assistant model.
        assistant_generation_kwargs = {
            self.input_ids_key: input_ids,
            "min_new_tokens": min_new_tokens,
            "max_new_tokens": max_new_tokens,
            "generation_config": self.generation_config,
            # Add the additional required keys from original inputs
            "attention_mask": self.original_inputs.get("attention_mask"),
            "pixel_values": self.original_inputs.get("pixel_values"),
            "image_grid_thw": self.original_inputs.get("image_grid_thw"),
        }

        # Ensure we're passing all necessary kwargs
        for key, value in self.original_inputs.items():
            if key not in assistant_generation_kwargs and key not in self.assistant_kwargs:
                assistant_generation_kwargs[key] = value

        update_dict = {
            "return_dict_in_generate": True,
            "output_scores": True,
        }
        self.assistant_kwargs.update(**update_dict)
        # need to ensure that assistant_generation_kwargs also contains dict_keys(['input_ids', 'attention_mask', 'pixel_values', 'image_grid_thw'])
        assistant_output = self.assistant_model.generate(**assistant_generation_kwargs, **self.assistant_kwargs)

        # 3. Update variables for the next round of candidate generation
        self.assistant_kwargs["past_key_values"] = assistant_output.past_key_values

        # 4. Prepare variables for output
        candidate_logits = torch.stack(assistant_output.scores, dim=1)
        candidate_ids = assistant_output.sequences
        return candidate_ids, candidate_logits

    # TODO: modify this for custom implementation of better speculative decoding strategy
    def update_candidate_strategy(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, num_matches: int):
        """
        Updates the candidate generation strategy based on the outcomes.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. [What are input IDs?](../glossary#input-ids)
            scores (`torch.FloatTensor` of shape `(batch_size, candidate_length, config.vocab_size)`):
                Prediction scores of a language modeling head. These can be logits for each vocabulary when not using
                beam search or log softmax for each vocabulary token when using beam search
            num_matches (`int`):
                The number of matches between the candidate sequences and the model predictions.
        """
        # Adjust the max number of assistant tokens to use in the next iteration. This is a simple heuristic,
        # probably can be improved -- we want to balance the benefits of getting assistant tokens correct with the
        # cost of forecasting incorrect assistant tokens.
        if self.assistant_model.generation_config.num_assistant_tokens_schedule in {
            "heuristic",
            "heuristic_transient",
        }:
            if num_matches == int(self.num_assistant_tokens):
                self.num_assistant_tokens += 2.0
            else:
                self.num_assistant_tokens = max(1.0, self.num_assistant_tokens - 1.0)

    def crop_past_key_values(self, past_key_values, max_length):
        """ Crop the past key values to the max length in the DynamicCache """
        past_key_values.crop(max_length)
        return past_key_values
    
    def prepare_attention_mask(self, model_kwargs, new_length):
        """Expands or crops the model's mask for decoding purposes, to the defined length"""

        mask_key = "attention_mask"
        if mask_key not in model_kwargs.to_dict():
            return model_kwargs

        mask = model_kwargs[mask_key]
        mask_length_diff = new_length - mask.shape[1]

        if mask_length_diff < 0:
            model_kwargs[mask_key] = mask[:, :mask_length_diff]
        elif mask_length_diff > 0:
            model_kwargs[mask_key] = torch.cat([mask, mask.new_ones((mask.shape[0], mask_length_diff))], dim=-1)

        return model_kwargs
    
    def prepare_token_type_ids(self, model_kwargs, new_length):
        """Expands or crops the model's token_type_ids for decoding purposes, to the defined length"""
        if "token_type_ids" not in model_kwargs.to_dict() or model_kwargs["token_type_ids"] is None:
            return model_kwargs

        token_type_ids = model_kwargs["token_type_ids"]
        final_token_type = token_type_ids[:, -1].unsqueeze(-1)
        type_length_diff = new_length - token_type_ids.shape[1]

        if type_length_diff < 0:
            token_type_ids = token_type_ids[:, :type_length_diff]
        elif type_length_diff > 0:
            token_type_copies = final_token_type.repeat(1, type_length_diff)
            model_kwargs["token_type_ids"] = torch.cat([model_kwargs["token_type_ids"], token_type_copies], dim=-1)
        return model_kwargs

    def run_speculative_decoding(self, input_ids, generation_config, model_kwargs):
        # init values
        do_sample = generation_config.do_sample
        output_attentions = generation_config.output_attentions
        output_hidden_states = generation_config.output_hidden_states

        # keep track of which sequences are already finished
        model_kwargs = self.get_initial_cache_position(input_ids, model_kwargs)

        start_from_empty_dynamic_cache = False
        tmp_model_kwargs = model_kwargs.to_dict()
        past_key_values = tmp_model_kwargs.get("past_key_values", None)
        if isinstance(past_key_values, DynamicCache):
            if past_key_values.get_seq_length() == 0:
                start_from_empty_dynamic_cache = True

        cur_len = input_ids.shape[-1]

        #  1. Fetch candidate sequences from a `CandidateGenerator`
        candidate_input_ids, candidate_logits = self.get_candidates(input_ids)

        if candidate_logits is not None:
            candidate_logits = candidate_logits.to(self.device)

        candidate_length = candidate_input_ids.shape[1] - input_ids.shape[1]
        is_done_candidate = None

        # 2. Use the original model to obtain the next token logits given the candidate sequence. We obtain
        # `candidate_length + 1` relevant logits from this process: in the event that all candidates are correct,
        # we use this forward pass to also pick the subsequent logits in the original model.

        # 2.1. Prepare the model inputs
        candidate_kwargs = copy.copy(model_kwargs)
        candidate_kwargs = self.prepare_attention_mask(
            candidate_kwargs, candidate_input_ids.shape[1]
        )
        candidate_kwargs = self.prepare_token_type_ids(candidate_kwargs, candidate_input_ids.shape[1])
        if "cache_position" in candidate_kwargs.to_dict():
            candidate_kwargs.cache_position = torch.cat(
                (
                    candidate_kwargs.cache_position,
                    torch.arange(cur_len, cur_len + candidate_length, device=input_ids.device, dtype=torch.long),
                ),
                dim=0,
            )

        model_inputs = self.prepare_inputs_for_generation(candidate_input_ids, **candidate_kwargs.to_dict())
        if "num_logits_to_keep" in model_inputs:
            model_inputs["num_logits_to_keep"] = candidate_length + 1

        # 2.2. Run a forward pass on the candidate sequence
        # prepare variable output controls (note: some models won't accept all output controls)
        model_inputs.update({"output_attentions": output_attentions} if output_attentions else {})
        model_inputs.update({"output_hidden_states": output_hidden_states} if output_hidden_states else {})

        tmp_model_inputs = model_inputs
        keys_to_keep = [
            "input_ids",
            "attention_mask",
            "position_ids",
            "past_key_values",
            "inputs_embeds",
            "labels",
            "use_cache",
            "output_attentions",
            "output_hidden_states",
            "return_dict",
            "pixel_values",
            "pixel_values_videos",
            "image_grid_thw",
            "video_grid_thw",
            "rope_deltas"
        ]
        tmp_model_inputs = {k: v for k, v in tmp_model_inputs.items() if k in keys_to_keep}
        outputs = self.assistant_model(**tmp_model_inputs)

        # 2.3. Process the new logits
        # .float() is needed to retain precision for later logits manipulations
        new_logits = outputs.logits[:, -candidate_length - 1 :].float()  # excludes the input prompt if present
        next_token_logits = new_logits.clone()

        # 3. Select the accepted tokens. There are two possible cases:
        # Case 1: `do_sample=True` and we have logits for the candidates (originally from speculative decoding)
        # 👉 Apply algorithm 1 from the speculative decoding paper (https://arxiv.org/pdf/2211.17192.pdf).
        if do_sample and candidate_logits is not None:
            valid_tokens, n_matches = self.speculative_sampling(
                candidate_input_ids,
                candidate_logits,
                candidate_length,
                new_logits,
                is_done_candidate,
            )

        # Case 2: all other cases (originally from assisted generation) 👉 Compare the tokens selected from the
        # original model logits with the candidate tokens. We can keep the candidate tokens until the first
        # mismatch, or until the max length is reached.
        else:
            if do_sample:
                probs = new_logits.softmax(dim=-1)
                selected_tokens = torch.multinomial(probs[0, :, :], num_samples=1).squeeze(1)[None, :]
            else:
                selected_tokens = new_logits.argmax(dim=-1)

            candidate_new_tokens = candidate_input_ids[:, cur_len:]
            n_matches = ((~(candidate_new_tokens == selected_tokens[:, :-1])).cumsum(dim=-1) < 1).sum()

            # Ensure we don't generate beyond max_len or an EOS token
            if is_done_candidate and n_matches == candidate_length:
                n_matches -= 1
            valid_tokens = selected_tokens[:, : n_matches + 1]

            # 4. Update variables according to the number of matching assistant tokens. Remember: the token generated
            # by the model after the last candidate match is also valid, as it is generated from a correct sequence.
            # Because of this last token, assisted generation search reduces to a normal greedy search/sample if there
            # is no match.

            # 4.1. Get the valid continuation, after the matching tokens
            input_ids = torch.cat((input_ids, valid_tokens), dim=-1)
            new_cur_len = input_ids.shape[-1]

            # 4.2. Discard past key values relative to unused assistant tokens
            new_cache_size = new_cur_len - 1
            outputs.past_key_values = self.crop_past_key_values(outputs.past_key_values, new_cache_size)

            # 5. Update the candidate generation strategy if needed
            self.update_candidate_strategy(input_ids, new_logits, n_matches)

            model_kwargs = self.update_model_kwargs_for_generation(
                outputs,
                model_kwargs,
                num_new_tokens=n_matches + 1,
            )

        if (
            hasattr(self, "assistant_model")
            and self.assistant_model.generation_config.num_assistant_tokens_schedule == "heuristic"
        ):
            self.assistant_model.generation_config.num_assistant_tokens = self.num_assistant_tokens
        
        # Always return the input_ids
        return input_ids


    def speculative_sampling(
        candidate_input_ids,
        candidate_logits,
        candidate_length,
        new_logits,
        is_done_candidate,
    ):
        """
        Applies sampling as in the speculative decoding paper (https://arxiv.org/pdf/2211.17192.pdf, algorithm 1). Returns
        the selected tokens, as well as the number of candidate matches.

        NOTE: Unless otherwise stated, the variable names match those in the paper.
        """
        new_candidate_input_ids = candidate_input_ids[:, -candidate_length:]
        # Gets the probabilities from the logits. q_i and p_i denote the assistant and model probabilities of the tokens
        # selected by the assistant, respectively.
        q = candidate_logits.softmax(dim=-1)
        q_i = q[:, torch.arange(candidate_length), new_candidate_input_ids].squeeze(0, 1)
        p = new_logits.softmax(dim=-1)
        p_i = p[:, torch.arange(candidate_length), new_candidate_input_ids].squeeze(0, 1)
        probability_ratio = p_i / q_i

        # When probability_ratio > 1 (i.e. q_i(x) < p_i(x), or "assistant probability of the candidate token is smaller
        # than the model probability for the same token"), keep the token. Otherwise reject with p = 1 - probability_ratio
        # (= keep with p = probability_ratio). Keep all the tokens until the first rejection
        r_i = torch.rand_like(probability_ratio)
        is_accepted = r_i <= probability_ratio
        n_matches = ((~is_accepted).cumsum(dim=-1) < 1).sum()  # this is `n` in algorithm 1

        # Ensure we don't generate beyond max_len or an EOS token (not in algorithm 1, but needed for correct behavior)
        if is_done_candidate and n_matches == candidate_length:
            # Output length is assumed to be `n_matches + 1`. Since we won't generate another token with the target model
            # due to acceptance on EOS we fix `n_matches`
            n_matches -= 1
            valid_tokens = new_candidate_input_ids[:, : n_matches + 1]
        else:
            # Next token selection: if there is a rejection, adjust the distribution from the main model before sampling.
            gamma = candidate_logits.shape[1]
            p_n_plus_1 = p[:, n_matches, :]
            if n_matches < gamma:
                q_n_plus_1 = q[:, n_matches, :]
                p_prime = torch.clamp((p_n_plus_1 - q_n_plus_1), min=0)
                p_prime.div_(p_prime.sum())
            else:
                p_prime = p_n_plus_1
            t = torch.multinomial(p_prime, num_samples=1).squeeze(1)[None, :]

            # The selected tokens include the matches (if any) plus the next sampled tokens
            if n_matches > 0:
                valid_tokens = torch.cat((new_candidate_input_ids[:, :n_matches], t), dim=-1)
            else:
                valid_tokens = t

        return valid_tokens, n_matches

    def get_initial_cache_position(self, input_ids, model_kwargs):
        """Calculates `cache_position` for the pre-fill stage based on `input_ids` and optionally past length"""

        cache_position = torch.ones_like(input_ids[0, :], dtype=torch.int64).cumsum(0) - 1

        past_length = 0
        tmp_model_kwargs = model_kwargs.to_dict()
        if tmp_model_kwargs.get("past_key_values") is not None:
            cache = tmp_model_kwargs["past_key_values"]
            past_length = 0
            if not isinstance(cache, Cache):
                past_length = cache[0][0].shape[2]
            elif hasattr(cache, "get_seq_length") and cache.get_seq_length() is not None:
                past_length = cache.get_seq_length()

            cache_position = cache_position[past_length:]
            
        model_kwargs.cache_position = cache_position
        return model_kwargs

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        past_key_values: Optional[Cache] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        """
        Prepare the model inputs for generation. In includes operations like computing the 4D attention mask or
        slicing inputs given the existing cache.

        See the forward pass in the model documentation for expected arguments (different models might have different
        requirements for e.g. `past_key_values`). This function should work as is for most LLMs.
        """

        # 1. Handle BC:
        model_inputs = {}
        model_inputs["cache_position"] = cache_position

        # 2. Generic cache-dependent input preparation
        # If we have cache: let's slice `input_ids` through `cache_position`, to keep only the unprocessed tokens
        # Exception 1: when passing input_embeds, input_ids may be missing entries
        # Exception 2: some generation methods do special slicing of input_ids, so we don't need to do it here
        if past_key_values is not None:
            model_inputs["past_key_values"] = past_key_values
            if input_ids.shape[1] != cache_position.shape[0]:  # Default case (the "else", a no op, is Exception 2)
                input_ids = input_ids[:, cache_position]

        # 3. Prepare base model inputs
        input_ids_key = "input_ids"
        model_inputs[input_ids_key] = input_ids.clone(memory_format=torch.contiguous_format)
        model_inputs["inputs_embeds"] = None

        # 4. Create missing `position_ids` on the fly
        if (
            attention_mask is not None
            and kwargs.get("position_ids") is None
            and "position_ids" in set(inspect.signature(self.forward).parameters.keys())
        ):
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            kwargs["position_ids"] = position_ids  # placed in kwargs for further processing (see below)

        # 5. Slice model inputs if it's an input that should have the same length as `input_ids`
        for model_input_name in ["position_ids", "token_type_ids"]:
            model_input = kwargs.get(model_input_name)
            if model_input is not None:
                if past_key_values:
                    model_input = model_input[:, -input_ids.shape[1] :]
                    model_input = model_input.clone(memory_format=torch.contiguous_format)
                model_inputs[model_input_name] = model_input

        if attention_mask is not None:
            model_inputs["attention_mask"] = attention_mask

        # 6. Forward ALL kwargs that are uninitialized (e.g. `use_cache`).
        for key, value in kwargs.items():
            if key not in model_inputs:
                model_inputs[key] = value

        # 7. Remove unexpected `generate` inputs (TODO @joao: fix trainer and examples)
        model_inputs.pop("labels", None)
        return model_inputs

    def update_model_kwargs_for_generation(
        self,
        outputs,
        model_kwargs,
        num_new_tokens=1,
    ):
        # update past_key_values keeping its naming used in model code
        cache_name, cache = self.extract_past_from_model_output(outputs)
        model_kwargs.update(**{cache_name: cache})
        if getattr(outputs, "state", None) is not None:
            model_kwargs["state"] = outputs.state

        # update token_type_ids with last value
        if "token_type_ids" in model_kwargs.to_dict():
            token_type_ids = model_kwargs["token_type_ids"]
            model_kwargs["token_type_ids"] = torch.cat([token_type_ids, token_type_ids[:, -1].unsqueeze(-1)], dim=-1)

        # update attention mask
        if "attention_mask" in model_kwargs.to_dict():
            attention_mask = model_kwargs["attention_mask"]
            model_kwargs["attention_mask"] = torch.cat(
                [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1
            )

        # TODO: figure out where cache_position comes from lol
        if model_kwargs.to_dict().get("use_cache", True):
            model_kwargs.update(**{"cache_position": model_kwargs.to_dict()["cache_position"][-1:] + num_new_tokens})
        else:
            past_positions = model_kwargs.pop("cache_position")
            new_positions = torch.arange(
                past_positions[-1] + 1, past_positions[-1] + num_new_tokens + 1, dtype=past_positions.dtype
            ).to(past_positions.device)
            model_kwargs["cache_position"] = torch.cat((past_positions, new_positions))
        return model_kwargs
    
    def extract_past_from_model_output(self, outputs):
        past_key_values = None
        cache_name = "past_key_values"
        if "past_key_values" in outputs:
            past_key_values = outputs.past_key_values
        elif "mems" in outputs:
            past_key_values = outputs.mems
        elif "past_buckets_states" in outputs:
            past_key_values = outputs.past_buckets_states
        elif "cache_params" in outputs:
            past_key_values = outputs.cache_params
            cache_name = "cache_params"

        return cache_name, past_key_values


