"""
Attribution notice: boilerplate is ported over from HF transformers source code

All speculative decoding functionality is custom defined
"""

import torch
from transformers import DynamicCache, GenerationConfig
import copy
from candidate_generator import AssistedCandidateGenerator

class Generation: 

    def __init__(self, target_model, draft_model, processor, kwargs):
        self.target_model = target_model
        self.draft_model = draft_model
        self.processor = processor
        self.kwargs = kwargs

        self.generation_config = self.prepare_generation_config()
        self.model_kwargs = copy.deepcopy(self.generation_config)

    def generate(self, inputs):
        
        # return self.target_model.generate(**inputs, **self.kwargs)
        # return self.draft_model.generate(**inputs, **self.kwargs)

        # extract the input tensor
        inputs_tensor, model_input_name = self.prepare_model_inputs(inputs)
        batch_size = inputs_tensor.shape[0]
        self.device = inputs_tensor.device

        # prepare special tokens for the generation config
        self.prepare_special_tokens(self.device)   
        
        # prepare 'input_ids', which will be used for auto-regressive generation
        input_ids = inputs_tensor if model_input_name == "input_ids" else self.model_kwargs.pop("input_ids")

        # prepare generation lengths
        input_ids_length = input_ids.shape[-1]
        self.generation_config.max_length = self.generation_config.max_new_tokens + input_ids_length
        self.generation_config.min_length = 0

        # prepare the cache
        max_cache_length = self.generation_config.max_length
        self.prepare_cache_for_generation()

        # create the candidate generator
        candidate_generator = AssistedCandidateGenerator(
            input_ids, 
            self.draft_model, 
            self.generation_config, 
            self.model_kwargs, 
        )

        # run speculative decoding
        result = candidate_generator.run_speculative_decoding(
            input_ids,
            generation_config=self.generation_config,
            model_kwargs=self.model_kwargs,
        )

        return result

    def prepare_generation_config(self):

        # manually define Qwen2 GenerationConfig and update with kwargs
        config = GenerationConfig(
            attn_implementation="flash_attention_2",
            bos_token_id=151643,
            do_sample=False,
            eos_token_id=[
                151645,
                151643
            ],
            pad_token_id=151643,
            # temperature=0.0,
            # top_k=1,
            # top_p=0.001
        )
        config.update(**self.kwargs)
        return config

    def prepare_model_inputs(self, inputs): 
        model_input_name = "input_ids"
        return inputs["input_ids"], model_input_name

    def prepare_special_tokens(self, device):
        
        # Convert special tokens to tensors
        def _tensor_or_none(token, device=None):
            if token is None:
                return token

            device = device if device is not None else self.device
            if isinstance(token, torch.Tensor):
                return token.to(device)
            return torch.tensor(token, device=device, dtype=torch.long)

        bos_token_tensor = _tensor_or_none(self.generation_config.bos_token_id, device=device)
        eos_token_tensor = _tensor_or_none(self.generation_config.eos_token_id, device=device)
        pad_token_tensor = _tensor_or_none(self.generation_config.pad_token_id, device=device)
        decoder_start_token_tensor = _tensor_or_none(self.generation_config.decoder_start_token_id, device=device)

        # We can have more than one eos token. Always treat it as a 1D tensor (when it exists).
        if eos_token_tensor is not None and eos_token_tensor.ndim == 0:
            eos_token_tensor = eos_token_tensor.unsqueeze(0)

        # Update generation config with the updated special tokens tensors
        self.generation_config._bos_token_tensor = bos_token_tensor
        self.generation_config._eos_token_tensor = eos_token_tensor
        self.generation_config._pad_token_tensor = pad_token_tensor
        self.generation_config._decoder_start_token_tensor = decoder_start_token_tensor

    def prepare_cache_for_generation(self):
        # assistant model provided, need to use dynamic cache
        self.generation_config.cache_implementation = None
        self.model_kwargs.past_key_values = DynamicCache()

    