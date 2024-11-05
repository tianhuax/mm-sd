import torch
import copy
import logging
import gc

# Configure logging
logging.basicConfig(level=logging.DEBUG)
DEBUG = False
FIRST_N_TOKENS = 3
NUM_DRAFT_SAMPLES = 6


class DynamicCache:
    def __init__(self, device=torch.device("cuda"), dtype=torch.bfloat16, max_length=2048):
        """
        Initializes the DynamicCache with specified device and data type.

        Args:
            device (torch.device): The device to store the cache tensors.
            dtype (torch.dtype): The data type for the cache tensors.
        """
        self.past_key_values = []
        self.device = device
        self.dtype = dtype
        self.max_length = max_length
        self.sliding_window = max_length // 2  # Add sliding window
        logging.debug(f"Initialized DynamicCache with device: {self.device} and dtype: {self.dtype}")

    def update(self, new_past_key_values, num_tokens):
        """
        Updates the cache with new past_key_values in bfloat16.

        Args:
            new_past_key_values (list of tuples): New (key, value) tensors from the model.
            num_tokens (int): Number of tokens to update in the cache.
        """
        if not self.past_key_values:
            # Initialize with the first set of past_key_values in bfloat16
            self.past_key_values = [
                (k.detach().to(dtype=self.dtype, device=self.device),
                 v.detach().to(dtype=self.dtype, device=self.device))
                for k, v in new_past_key_values
            ]
            logging.debug("Cache initialized with first past_key_values in bfloat16.")
        else:
            for i, (new_k, new_v) in enumerate(new_past_key_values):
                new_k = new_k.detach().to(dtype=self.dtype, device=self.device)
                new_v = new_v.detach().to(dtype=self.dtype, device=self.device)
                
                # Concatenate with sliding window
                if self.past_key_values[i][0].size(-1) > self.max_length - num_tokens:
                    # Keep the last sliding_window tokens
                    self.past_key_values[i] = (
                        self.past_key_values[i][0][..., -self.sliding_window:],
                        self.past_key_values[i][1][..., -self.sliding_window:]
                    )
                
                self.past_key_values[i] = (
                    torch.cat((self.past_key_values[i][0], new_k), dim=-1),
                    torch.cat((self.past_key_values[i][1], new_v), dim=-1)
                )

                # Clear temporary tensors
                del new_k
                del new_v

        torch.cuda.empty_cache()

    def __getitem__(self, idx):
        """
        Allows subscript access to past_key_values.

        Args:
            idx (int): Index of the layer.

        Returns:
            tuple: (key, value) tensors for the specified layer in bfloat16.
        """
        return self.past_key_values[idx]

    def __len__(self):
        """
        Returns the number of layers in the cache.
        """
        return len(self.past_key_values)

    def to_list(self):
        """
        Converts the DynamicCache to a list of tuples, compatible with Transformers.

        Returns:
            list of tuples: Each tuple contains (key, value) tensors in bfloat16.
        """
        return self.past_key_values.copy()

class Generation:
    def __init__(self, target_model, draft_model, processor, kwargs):
        """
        Initializes the Generation class with target and draft models, processor, and additional kwargs.

        Args:
            target_model (PreTrainedModel): The target Hugging Face model for generation.
            draft_model (PreTrainedModel): The draft Hugging Face model for speculative sampling.
            processor (Processor): The processor associated with the models.
            kwargs (dict): Additional keyword arguments for configuration.
        """
        self.target_model = target_model
        self.draft_model = draft_model
        self.processor = processor
        self.kwargs = kwargs
        self.tokenizer = processor.tokenizer  # Ensure tokenizer is accessible and consistent

        # Determine the device from the target model
        self.device = next(target_model.parameters()).device
        logging.debug(f"Generation initialized with device: {self.device}")

        # Prepare separate generation configurations for target and draft models
        self.target_generation_config = self.prepare_generation_config()
        self.draft_generation_config = copy.deepcopy(self.target_generation_config)

        # Initialize separate model_kwargs for target and draft models without using internal cache
        self.target_model_kwargs = copy.deepcopy(self.target_generation_config.to_dict())
        self.draft_model_kwargs = copy.deepcopy(self.draft_generation_config.to_dict())

        # Disable internal caching by setting 'use_cache' to False
        self.target_model_kwargs['use_cache'] = False
        self.draft_model_kwargs['use_cache'] = False

        # Initialize external caches separately with bfloat16 precision
        self.target_cache = DynamicCache(device=self.device, dtype=torch.bfloat16)
        self.draft_cache = DynamicCache(device=self.device, dtype=torch.bfloat16)
        logging.debug("External DynamicCache instances created for target and draft models.")

    def prepare_generation_config(self):
        """
        Prepares the generation configuration from kwargs.

        Returns:
            GenerationConfig: The generation configuration object.
        """
        from transformers import GenerationConfig
        return GenerationConfig.from_dict(self.kwargs)

    def prepare_model_kwargs(self, role='target'):
        """
        Prepares model_kwargs for generation by incorporating cached past_key_values.
        Only includes parameters relevant for the model's forward pass.

        Args:
            role (str): 'target' or 'draft' to indicate which model's kwargs to prepare.

        Returns:
            dict: Prepared model_kwargs with only forward-pass relevant parameters.
        """
        if role == 'target':
            cache = self.target_cache
        elif role == 'draft':
            cache = self.draft_cache
        else:
            raise ValueError("role must be either 'target' or 'draft'.")

        # Basic kwargs that are always included
        model_kwargs = {
            'use_cache': True,  # Enable caching
            'output_attentions': False,
            'output_hidden_states': False,
            'return_dict': True
        }

        # Add past_key_values if cache exists
        if len(cache) > 0:
            model_kwargs['past_key_values'] = cache.to_list()
            if DEBUG:
                logging.debug(f"Added cache of length {len(cache)} to {role} model kwargs")
        
        return model_kwargs

    def update_cache(self, outputs, num_tokens, role='target'):
        """
        Updates the external dynamic cache with new past_key_values from the model outputs.

        Args:
            outputs (dict or ModelOutput): The outputs returned by the model's forward/generate method.
            num_tokens (int): The number of tokens to update in the cache.
            role (str): 'target' or 'draft' to indicate which model's cache to update.
        """
        if role == 'target':
            cache = self.target_cache
        elif role == 'draft':
            cache = self.draft_cache
        else:
            raise ValueError("role must be either 'target' or 'draft'.")

        # Get past_key_values from outputs
        if hasattr(outputs, 'past_key_values'):
            new_past = outputs.past_key_values
        elif isinstance(outputs, dict):
            new_past = outputs.get('past_key_values')
        else:
            new_past = None

        if new_past is not None:
            if DEBUG:
                logging.debug(f"Updating {role} cache with {num_tokens} new tokens")
            cache.update(new_past, num_tokens)

    def generate(self, inputs):
        """
        Modified generate method to properly handle sampling vs greedy decoding
        and integrate external dynamic cache.

        Args:
            inputs (dict): A dictionary containing 'input_ids', 'attention_mask', 'pixel_values', and optionally 'image_grid_thw'.

        Returns:
            torch.Tensor: The generated input_ids with appended tokens.
        """
        # Add at the start of generate method
        torch.cuda.empty_cache()  # Clear CUDA cache before starting

        input_ids = inputs['input_ids'].to(self.device, dtype=torch.long)
        attention_mask = inputs['attention_mask'].to(self.device, dtype=torch.long)
        pixel_values = inputs['pixel_values'].to(self.device, dtype=torch.float32)
        
        # Keep image_grid_thw as a tuple, don't convert to tensor
        image_grid_thw = inputs.get('image_grid_thw', None)
        
        # Get sampling flag from generation config
        do_sample = self.target_generation_config.do_sample
        max_new_tokens = self.target_generation_config.max_new_tokens
        num_first_target = FIRST_N_TOKENS
        num_draft_samples = NUM_DRAFT_SAMPLES
        tokens_generated = 0
        
        # Generate initial context with target model
        if DEBUG:
            logging.debug("Generating initial context with target model")
        
        # Prepare model_kwargs with external cache (initially empty)
        target_model_kwargs = self.prepare_model_kwargs(role='target')
        
        target_outputs = self.target_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            **target_model_kwargs
        )

        # Update external cache with the initial past_key_values
        self.update_cache(target_outputs, num_tokens=1, role='target')

        # Get first num_first_target tokens from target model
        next_tokens = []
        for _ in range(num_first_target):
            next_token_logits = target_outputs['logits'][:, -1, :]  # Only take the last token
            if do_sample:
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = next_token_logits.argmax(dim=-1, keepdim=True)
            
            # Don't concatenate to original input_ids yet
            next_tokens.append(next_token)
            
            # Update input_ids for next iteration only
            current_input_ids = torch.cat([input_ids, next_token], dim=1)
            current_attention_mask = torch.cat((
                attention_mask,
                torch.ones((attention_mask.shape[0], 1), device=self.device, dtype=torch.long)
            ), dim=1)
            
            target_outputs = self.target_model(
                input_ids=current_input_ids,
                attention_mask=current_attention_mask,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                **self.prepare_model_kwargs(role='target')
            )

            # Update external cache with the new past_key_values
            self.update_cache(target_outputs, num_tokens=1, role='target')

        # Stack tokens and update input_ids
        next_tokens = torch.cat(next_tokens, dim=1)
        input_ids = torch.cat((input_ids, next_tokens), dim=-1)
        attention_mask = torch.cat((
            attention_mask,
            torch.ones((attention_mask.shape[0], num_first_target), device=self.device, dtype=torch.long)
        ), dim=1)
        tokens_generated += num_first_target

        if DEBUG:
            logging.debug(f"Initial {num_first_target} tokens: '{self.tokenizer.batch_decode(next_tokens[0], skip_special_tokens=True)}'")

        def clear_memory():
            """Helper function to clear memory during generation"""
            torch.cuda.empty_cache()
            gc.collect()

        # Main generation loop
        while tokens_generated < max_new_tokens:  # sets hard limit on total number of tokens generated
            if DEBUG:
                logging.debug(f"\nGeneration step {tokens_generated}/{max_new_tokens}")
                logging.debug(f"GPU Memory: {torch.cuda.memory_allocated()/1024**3:.2f}GB")

            # Clear memory periodically
            if tokens_generated % 10 == 0:
                clear_memory()

            # Prepare model_kwargs with external cache for draft model
            draft_model_kwargs = self.prepare_model_kwargs(role='draft')

            # Generate K draft tokens autoregressively with draft model
            draft_tokens = []
            draft_logits = []
            current_ids = input_ids

            for _ in range(num_draft_samples):
                draft_model_kwargs = self.prepare_model_kwargs(role='draft')
                
                draft_outputs = self.draft_model(
                    input_ids=current_ids,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values,
                    image_grid_thw=image_grid_thw,
                    **draft_model_kwargs
                )
                
                # Update draft cache
                self.update_cache(draft_outputs, num_tokens=1, role='draft')
                
                # Get next token with temperature sampling
                next_logits = draft_outputs['logits'][:, -1:, :]
                next_logits = next_logits / 0.7  # Add temperature
                
                if do_sample:
                    probs = torch.softmax(next_logits.squeeze(1), dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = next_logits.argmax(dim=-1, keepdim=True)
                
                draft_tokens.append(next_token)
                draft_logits.append(next_logits)
                
                # Update current_ids for next iteration
                next_token = next_token.squeeze(-1)
                current_ids = torch.cat([current_ids, next_token], dim=1)
                attention_mask = torch.cat([
                    attention_mask,
                    torch.ones((attention_mask.shape[0], 1), device=self.device, dtype=torch.long)
                ], dim=1)

            # Stack all draft tokens and logits
            draft_tokens = torch.cat(draft_tokens, dim=1)  # Shape: [batch_size, num_draft_samples]
            draft_logits = torch.cat(draft_logits, dim=1)  # Shape: [batch_size, num_draft_samples, vocab_size]
            
            if DEBUG:
                logging.debug(f"Draft tokens: '{self.tokenizer.batch_decode(draft_tokens[0], skip_special_tokens=True)}'")

            # Get target logits for the entire draft sequence at once   
            draft_tokens = draft_tokens.squeeze(-1)
            target_outputs = self.target_model(
                input_ids=torch.cat([input_ids, draft_tokens], dim=1),
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                **self.prepare_model_kwargs(role='target')
            )

            # Update external cache with target model outputs
            self.update_cache(target_outputs, num_tokens=draft_tokens.size(1), role='target')

            # Get logits for the draft positions
            # Shape: [batch_size, num_draft_samples, vocab_size]
            target_logits = target_outputs['logits'][:, -(num_draft_samples):, :]
            # BONUS: Get the extra target logit in case all tokens are accepted
            extra_target_logits = target_outputs['logits'][:, -1:, :]

            if DEBUG:
                logging.debug(f"Target logits shape: {target_logits.shape}")
                logging.debug(f"Extra target logit shape: {extra_target_logits.shape}")

            # Do speculative sampling to accept/reject tokens
            valid_tokens_padded, n_matches = self.speculative_sampling(
                draft_logits=draft_logits,
                target_logits=target_logits,
                candidate_new_tokens=draft_tokens,
                extra_target_logits=extra_target_logits,
                do_sample=do_sample
            )

            # Update input_ids and attention_mask with accepted tokens
            input_ids = torch.cat((input_ids, valid_tokens_padded), dim=-1)
            attention_mask = torch.cat((
                attention_mask,
                torch.ones((attention_mask.shape[0], valid_tokens_padded.shape[1]), device=self.device, dtype=torch.long)
            ), dim=1)
            
            tokens_generated += n_matches

            # Check for EOS token
            if (valid_tokens_padded == self.tokenizer.eos_token_id).any():
                break

            # After processing draft tokens, clear them from memory
            del draft_tokens
            del draft_logits
            clear_memory()

            # After processing target outputs, clear them
            del target_outputs
            clear_memory()

        return input_ids

    def speculative_sampling(self, draft_logits, target_logits, candidate_new_tokens, extra_target_logits=None, do_sample=False):
        """
        Performs speculative sampling to decide which draft tokens to accept.

        Args:
            draft_logits (torch.Tensor): Logits from the draft model. Shape: [batch_size, num_draft_samples, vocab_size]
            target_logits (torch.Tensor): Logits from the target model. Shape: [batch_size, num_draft_samples, vocab_size]
            candidate_new_tokens (torch.Tensor): Draft tokens proposed. Shape: [batch_size, num_draft_samples]
            extra_target_logits (torch.Tensor, optional): Extra logits from the target model for an additional token. Shape: [batch_size, 1, vocab_size]
            do_sample (bool): Whether to use sampling or greedy decoding.

        Returns:
            tuple: (accepted_tokens, number_of_accepted_tokens)
        """
        # Constants for better token acceptance
        temperature = 0.7
        acceptance_threshold = 0.3

        # Process logits
        draft_logits = draft_logits / temperature
        target_logits = target_logits / temperature
        
        batch_size = candidate_new_tokens.size(0)
        seq_len = candidate_new_tokens.size(1)
        accepted_tokens = []

        for t in range(seq_len):
            current_token = candidate_new_tokens[:, t:t+1]
            
            if do_sample:
                draft_probs = torch.softmax(draft_logits[:, t], dim=-1)
                target_probs = torch.softmax(target_logits[:, t], dim=-1)
                
                p_t = draft_probs.gather(-1, current_token)
                q_t = target_probs.gather(-1, current_token)
                
                # Modified acceptance criteria
                acceptance_ratio = q_t / (p_t + 1e-10)
                accepted = acceptance_ratio >= acceptance_threshold
                
                if not accepted.any():
                    # Sample from target with temperature
                    new_token = torch.multinomial(target_probs, num_samples=1)
                    accepted_tokens.append(new_token)
                    if DEBUG:
                        logging.debug(f"Rejected at position {t} - Sampled new token: '{self.tokenizer.decode(new_token[0])}'")
                    break
            else:
                # Modified greedy case
                target_token = target_logits[:, t].argmax(dim=-1, keepdim=True)
                draft_token_prob = torch.softmax(draft_logits[:, t], dim=-1).gather(-1, current_token)
                target_token_prob = torch.softmax(target_logits[:, t], dim=-1).gather(-1, target_token)
                
                accepted = (current_token == target_token) or (draft_token_prob >= acceptance_threshold * target_token_prob)
                
                if not accepted.any():
                    accepted_tokens.append(target_token)
                    if DEBUG:
                        logging.debug(f"Rejected at position {t} - Using target token: '{self.tokenizer.decode(target_token[0])}'")
                    break

            accepted_tokens.append(current_token)
            if DEBUG:
                logging.debug(f"Token {t} accepted: '{self.tokenizer.decode(current_token[0])}'")

        # BONUS: If all K tokens are accepted and have extra target logits, add one more token
        if len(accepted_tokens) == seq_len and extra_target_logits is not None:
            if do_sample:
                extra_probs = torch.softmax(extra_target_logits.squeeze(1), dim=-1)
                extra_token = torch.multinomial(extra_probs, num_samples=1)
            else:
                extra_token = extra_target_logits.argmax(dim=-1, keepdim=True)
            
            extra_token = extra_token.squeeze(-1)
            accepted_tokens.append(extra_token.to(dtype=torch.long))
            if DEBUG:
                logging.debug(f"All {seq_len} tokens accepted! Adding extra token: '{self.tokenizer.batch_decode(extra_token[0], skip_special_tokens=True)}'")

        # If no tokens were accepted, return empty tensor
        if not accepted_tokens:
            return torch.empty((batch_size, 0), dtype=torch.long, device=self.device), 0

        accepted_tokens = torch.cat(accepted_tokens, dim=1)
        if DEBUG:
            logging.debug(f"Total accepted tokens: {accepted_tokens.size(1)}")
        return accepted_tokens, accepted_tokens.size(1)

