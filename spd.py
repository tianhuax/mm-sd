
import torch
from transformers import GenerationConfig
import copy
from candidate_generator import DynamicCache  # Ensure this import is correct based on your project structure
import logging
from torch.cuda.amp import autocast  # For mixed precision

# Setup logging
logging.basicConfig(level=logging.DEBUG if False else logging.INFO)

# Global debug flag
DEBUG = False

class DynamicCache:
    def __init__(self):
        # Initialize an empty list to store past key-values for each layer
        self.past_key_values = []

    def update(self, new_past_key_values, num_tokens):
        """
        Updates the cache with new past_key_values.

        Args:
            new_past_key_values (list of tuples): New (key, value) tensors from the model.
            num_tokens (int): Number of tokens to update in the cache.
        """
        if not self.past_key_values:
            # Initialize with the first set of past_key_values
            self.past_key_values = [t.detach() for t in new_past_key_values]
            if DEBUG:
                logging.debug("Cache initialized with first past_key_values.")
        else:
            for i, (new_k, new_v) in enumerate(new_past_key_values):
                # Concatenate along the sequence dimension (usually dim=-1)
                self.past_key_values[i] = torch.cat((self.past_key_values[i], new_k.detach()), dim=-1)
                self.past_key_values[i] = torch.cat((self.past_key_values[i], new_v.detach()), dim=-1)
            if DEBUG:
                logging.debug(f"Cache updated with {num_tokens} new tokens.")

    def __getitem__(self, idx):
        """
        Allows subscript access to past_key_values.

        Args:
            idx (int): Index of the layer.

        Returns:
            tuple: (key, value) tensors for the specified layer.
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
            list of tuples: Each tuple contains (key, value) tensors.
        """
        return self.past_key_values.copy()


class Generation: 

    def __init__(self, target_model, draft_model, processor, kwargs):
        self.target_model = target_model
        self.draft_model = draft_model
        self.processor = processor
        self.kwargs = kwargs
        self.tokenizer = processor.tokenizer  # Ensure tokenizer is accessible and consistent

        # Prepare separate generation configurations for target and draft models
        self.target_generation_config = self.prepare_generation_config()
        self.draft_generation_config = copy.deepcopy(self.target_generation_config)

        # Initialize separate model_kwargs for target and draft models
        self.target_model_kwargs = copy.deepcopy(self.target_generation_config.to_dict())
        self.draft_model_kwargs = copy.deepcopy(self.draft_generation_config.to_dict())

        # Initialize caches separately
        self.prepare_cache_for_generation(role='target')
        self.prepare_cache_for_generation(role='draft')

    def prepare_generation_config(self):
        """
        Prepares the GenerationConfig object based on the provided kwargs.

        Returns:
            GenerationConfig: Configured generation configuration.
        """
        # Define default generation parameters
        default_generation_kwargs = {
            'max_new_tokens': 50,
            'temperature': 1.0,
            'top_k': 50,
            'top_p': 0.95,
            'do_sample': False,  # Set to False for greedy decoding
            'num_beams': 1,
            'repetition_penalty': 1.0,
            'length_penalty': 1.0,
            'early_stopping': False,
            'pad_token_id': self.tokenizer.eos_token_id,
            'eos_token_id': self.tokenizer.eos_token_id,
            'bos_token_id': self.tokenizer.bos_token_id,
            'use_cache': True,
            # Add other default generation parameters as needed
        }
        
        # Update default parameters with any user-provided kwargs
        default_generation_kwargs.update(self.kwargs)
        
        # Initialize GenerationConfig with the combined parameters
        generation_config = GenerationConfig(**default_generation_kwargs)
        
        if DEBUG:
            logging.debug(f"GenerationConfig initialized with parameters: {generation_config}")
        
        return generation_config

    def prepare_cache_for_generation(self, role='target'):
        """
        Prepares the dynamic cache for generation by initializing past_key_values.
        This method ensures that the cache is properly set up before generation begins.

        Args:
            role (str): 'target' or 'draft' to indicate which model's cache to prepare.
        """
        if role == 'target':
            model_kwargs = self.target_model_kwargs
        elif role == 'draft':
            model_kwargs = self.draft_model_kwargs
        else:
            raise ValueError("role must be either 'target' or 'draft'.")

        # Check if 'past_key_values' is already present
        if 'past_key_values' not in model_kwargs or model_kwargs['past_key_values'] is None:
            # Initialize DynamicCache and assign to 'past_key_values'
            model_kwargs['past_key_values'] = DynamicCache()
            if DEBUG:
                logging.debug(f"DynamicCache initialized and assigned to '{role}_past_key_values'.")
        elif DEBUG:
            logging.debug(f"'{role}_past_key_values' already exists in model_kwargs.")

    def update_cache(self, outputs, num_tokens, role='target'):
        """
        Updates the dynamic cache with new past_key_values from the model outputs.

        Args:
            outputs (dict or ModelOutput): The outputs returned by the model's forward/generate method.
            num_tokens (int): The number of tokens to update in the cache.
            role (str): 'target' or 'draft' to indicate which model's cache to update.
        """
        if role == 'target':
            model_kwargs = self.target_model_kwargs
        elif role == 'draft':
            model_kwargs = self.draft_model_kwargs
        else:
            raise ValueError("role must be either 'target' or 'draft'.")

        new_past = outputs.get('past_key_values')
        if new_past:
            if DEBUG:
                logging.debug(f"Updating {role} cache with {num_tokens} new tokens.")
            model_kwargs['past_key_values'].update(new_past, num_tokens)
            if DEBUG:
                logging.debug(f"{role.capitalize()} cache successfully updated with new past_key_values.")
        else:
            if DEBUG:
                logging.warning(f"No 'past_key_values' found in the outputs to update the {role} cache.")

    def prepare_model_kwargs(self, role='target'):
        """
        Prepares model_kwargs for generation by incorporating cached past_key_values.
        Only includes 'past_key_values' if the cache is not empty.

        Args:
            role (str): 'target' or 'draft' to indicate which model's kwargs to prepare.

        Returns:
            dict: Prepared model_kwargs.
        """
        if role == 'target':
            cache = self.target_model_kwargs.get('past_key_values')
            base_kwargs = self.target_model_kwargs
        elif role == 'draft':
            cache = self.draft_model_kwargs.get('past_key_values')
            base_kwargs = self.draft_model_kwargs
        else:
            raise ValueError("role must be either 'target' or 'draft'.")

        model_kwargs = copy.deepcopy(base_kwargs)

        # Convert cache to list if it's a DynamicCache with entries
        if isinstance(cache, DynamicCache):
            if len(cache) > 0:
                model_kwargs['past_key_values'] = cache.to_list()
            else:
                model_kwargs.pop('past_key_values', None)

        return model_kwargs

    def speculative_sampling(self, draft_logits, target_logits, candidate_new_tokens, threshold=0.1):
        """
        Applies sampling based on probability ratios to accept or reject candidate tokens.

        Args:
            draft_logits (torch.Tensor): Logits from the draft model. Shape: (batch_size, candidate_length, vocab_size)
            target_logits (torch.Tensor): Logits from the target model. Shape: (batch_size, candidate_length, vocab_size)
            candidate_new_tokens (torch.Tensor): Candidate tokens generated by the draft model. Shape: (batch_size, candidate_length)
            threshold (float): Confidence threshold for acceptance.

        Returns:
            valid_tokens_padded (torch.Tensor): Tokens that are accepted.
            n_matches_max (int): Maximum number of tokens that matched across the batch.
        """
        # Calculate probabilities
        q = torch.softmax(draft_logits, dim=-1)  # Draft model probabilities
        p = torch.softmax(target_logits, dim=-1)  # Target model probabilities

        # Verify that candidate_new_tokens are within vocab_size
        vocab_size = p.size(-1)
        assert torch.all((candidate_new_tokens >= 0) & (candidate_new_tokens < vocab_size)), \
            "Candidate tokens contain invalid token indices."

        # Gather the probabilities for the candidate tokens
        p_i = p.gather(-1, candidate_new_tokens.unsqueeze(-1)).squeeze(-1)  # (batch_size, candidate_length)
        q_i = q.gather(-1, candidate_new_tokens.unsqueeze(-1)).squeeze(-1)  # (batch_size, candidate_length)

        if DEBUG:
            # Log shapes and some values for debugging
            logging.debug(f"p_i shape: {p_i.shape}")
            logging.debug(f"q_i shape: {q_i.shape}")
            logging.debug(f"candidate_new_tokens shape: {candidate_new_tokens.shape}")

            # Aggregate token information for deeper insights
            tokens = candidate_new_tokens.tolist()[0]
            p_values = p_i.tolist()[0]
            q_values = q_i.tolist()[0]
            for token, p_val, q_val in zip(tokens, p_values, q_values):
                token_str = self.tokenizer.decode([token])
                ratio = p_val / (q_val + 1e-10)  # Prevent division by zero
                logging.debug(f"Token ID: {token}, Token: '{token_str}', p_i: {p_val:.6f}, q_i: {q_val:.6f}, Ratio: {ratio:.6f}")

        # Probability ratios optimized to avoid division
        is_above_threshold = p_i >= threshold * q_i

        if DEBUG:
            # Log probability ratios
            logging.debug(f"Probability ratios sample: {is_above_threshold[:3]}")

        # Extract valid tokens based on acceptance using tensor masking
        mask = is_above_threshold
        valid_tokens = candidate_new_tokens[mask]
        n_matches_max = valid_tokens.size(0)

        if valid_tokens.numel() > 0:
            valid_tokens_padded = valid_tokens.unsqueeze(0)  # Shape: (1, num_valid_tokens)
        else:
            valid_tokens_padded = torch.empty((1, 0), dtype=torch.long, device=candidate_new_tokens.device)

        if DEBUG:
            logging.debug(f"valid_tokens_padded shape: {valid_tokens_padded.shape}")
            logging.debug(f"n_matches: tensor([{n_matches_max}], device='{candidate_new_tokens.device}')")

        return valid_tokens_padded, n_matches_max



    def generate_with_profiling(self, inputs):
        """
        Generates tokens with simplified profiling to ensure logs are saved.

        Args:
            inputs (dict): Input dictionary containing necessary tensors like 'input_ids', 'attention_mask', etc.

        Returns:
            torch.Tensor: Generated input_ids with appended tokens.
        """
        from torch.profiler import profile, ProfilerActivity, tensorboard_trace_handler, record_function

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],  # Include CUDA if using GPU
            on_trace_ready=tensorboard_trace_handler('./profile_logs'),
            record_shapes=False,
            profile_memory=False,
            with_stack=False
        ) as prof:
            with record_function("generate"):
                output = self.generate(inputs)
                # Ensure that prof.step() is called within the generate method if it contains loops
                # If generate is a single step, prof.step() can be called here
            prof.step()
        print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
        return output

    def generate(self, inputs):
        """
        Generates tokens using speculative sampling with draft and target models.

        Args:
            inputs (dict): Input dictionary containing necessary tensors like 'input_ids', 'attention_mask', etc.

        Returns:
            torch.Tensor: Generated input_ids with appended tokens.
        """

        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        pixel_values = inputs['pixel_values']
        image_grid_thw = inputs['image_grid_thw']  # Ensure this is correctly set

        # Log the presence of image_grid_thw
        if image_grid_thw is None:
            if DEBUG:
                logging.error("image_grid_thw is None. It must be provided for the model.")
            raise ValueError("image_grid_thw must be provided.")

        # Initialize variables
        max_new_tokens = self.target_generation_config.max_new_tokens
        tokens_generated = 0
        max_fallback_attempts = 5  # Define maximum fallback attempts
        fallback_attempts = 0

        while tokens_generated < max_new_tokens and fallback_attempts < max_fallback_attempts:
            # Prepare draft model kwargs with max_new_tokens=3 and convert DynamicCache to list if not empty
            draft_kwargs = self.prepare_model_kwargs(role='draft')
            draft_kwargs['max_new_tokens'] = 3  # Number of candidates

            # Use mixed precision for draft model generation
            with autocast(enabled=DEBUG and torch.cuda.is_available()):
                draft_outputs = self.draft_model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values,
                    image_grid_thw=image_grid_thw,
                    **draft_kwargs  # 'past_key_values' included only if cache is not empty
                )

            candidate_new_tokens = draft_outputs[:, input_ids.shape[1]:]

            # Get draft logits
            with autocast(enabled=DEBUG and torch.cuda.is_available()):
                draft_logits = self.draft_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values,
                    image_grid_thw=image_grid_thw,
                    past_key_values=self.draft_model_kwargs['past_key_values'].to_list() if isinstance(self.draft_model_kwargs['past_key_values'], DynamicCache) and len(self.draft_model_kwargs['past_key_values']) > 0 else None
                )['logits']

            # Get target logits
            with autocast(enabled=DEBUG and torch.cuda.is_available()):
                target_outputs = self.target_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values,
                    image_grid_thw=image_grid_thw,
                    past_key_values=self.target_model_kwargs['past_key_values'].to_list() if isinstance(self.target_model_kwargs['past_key_values'], DynamicCache) and len(self.target_model_kwargs['past_key_values']) > 0 else None
                )
            target_logits = target_outputs['logits']

            # Speculative Sampling
            valid_tokens_padded, n_matches = self.speculative_sampling(
                draft_logits=draft_logits,
                target_logits=target_logits,
                candidate_new_tokens=candidate_new_tokens,
                threshold=0.1  # Adjust as needed
            )

            if n_matches == 0:
                if DEBUG:
                    logging.debug("No tokens accepted from speculative sampling. Sampling individually from target model.")
                # Prepare target model kwargs with max_new_tokens=1 and convert DynamicCache to list if not empty
                target_kwargs = self.prepare_model_kwargs(role='target')
                target_kwargs['max_new_tokens'] = 1  # Generate one token

                # Use mixed precision for target model fallback generation
                with autocast(enabled=DEBUG and torch.cuda.is_available()):
                    fallback_outputs = self.target_model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        pixel_values=pixel_values,
                        image_grid_thw=image_grid_thw,
                        **target_kwargs  # 'past_key_values' included only if cache is not empty
                    )
                fallback_token = fallback_outputs[:, input_ids.shape[1]:]
                if DEBUG:
                    logging.debug(f"Fallback token shape: {fallback_token.shape}")

                # Check for EOS token
                if torch.any(fallback_token == self.target_model.config.eos_token_id):
                    fallback_attempts += 1
                    if DEBUG:
                        logging.debug(f"EOS token generated during fallback. Attempt {fallback_attempts} of {max_fallback_attempts}. Retrying.")
                    if fallback_attempts >= max_fallback_attempts:
                        if DEBUG:
                            logging.warning("Maximum fallback attempts reached. Terminating generation.")
                        break
                    else:
                        continue  # Retry fallback sampling

                # Update input_ids and attention_mask with the fallback token
                input_ids = torch.cat((input_ids, fallback_token), dim=-1)
                attention_mask = torch.cat((
                    attention_mask,
                    torch.ones((attention_mask.shape[0], 1), device=attention_mask.device)
                ), dim=1)
                tokens_generated += 1

                # Update cache with fallback token
                self.update_cache(target_outputs, 1, role='target')

                continue  # Proceed to the next iteration

            # Update input_ids and attention_mask with valid_tokens
            input_ids = torch.cat((input_ids, valid_tokens_padded), dim=-1)
            attention_mask = torch.cat((
                attention_mask,
                torch.ones((attention_mask.shape[0], valid_tokens_padded.shape[1]), device=attention_mask.device)
            ), dim=1)
            tokens_generated += valid_tokens_padded.shape[1]

            # Update cache with new tokens
            self.update_cache(target_outputs, valid_tokens_padded.shape[1], role='target')

            # Check for EOS token in valid_tokens
            if torch.any(valid_tokens_padded == self.target_model.config.eos_token_id):
                if DEBUG:
                    logging.debug("EOS token generated. Terminating generation.")
                break

        return input_ids