import torch
from transformers import GenerationConfig
import copy
from candidate_generator import DynamicCache  # Ensure this import is correct based on your project structure
import logging
from torch.cuda.amp import autocast  # For mixed precision

# Setup logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Global debug flag
DEBUG = True
num_draft_samples = 2 # why does increasing this help when rejection should be higher? 
num_first_target = 3

class DynamicCache:
    def __init__(self, device=torch.device("cpu")):
        # Initialize an empty list to store past key-values for each layer
        self.past_key_values = []
        self.device = device
        logging.debug(f"Initialized DynamicCache with device: {device}")

    def update(self, new_past_key_values, num_tokens):
        """
        Updates the cache with new past_key_values.

        Args:
            new_past_key_values (list of tuples): New (key, value) tensors from the model.
            num_tokens (int): Number of tokens to update in the cache.
        """
        if not self.past_key_values:
            # Initialize with the first set of past_key_values
            self.past_key_values = [t.detach().to(self.device) for t in new_past_key_values]
            logging.debug("Cache initialized with first past_key_values.")
        else:
            for i, (new_k, new_v) in enumerate(new_past_key_values):
                # Concatenate along the sequence dimension (usually dim=-1)
                self.past_key_values[i] = torch.cat((self.past_key_values[i], new_k.detach().to(self.device)), dim=-1)
                self.past_key_values[i] = torch.cat((self.past_key_values[i], new_v.detach().to(self.device)), dim=-1)
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

        # Determine the device from the target model
        self.device = next(target_model.parameters()).device
        logging.debug(f"Generation initialized with device: {self.device}")

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
        Updated generation config to better handle sampling parameters
        """
        default_generation_kwargs = {
            'max_new_tokens': 50,
            'do_sample': False,  # Default to greedy decoding
            'temperature': 1.0,
            'top_k': 50,
            'top_p': 0.95,
            'num_beams': 1,
            'repetition_penalty': 1.0,
            'length_penalty': 1.0,
            'early_stopping': False,
            'pad_token_id': self.tokenizer.eos_token_id,
            'eos_token_id': self.tokenizer.eos_token_id,
            'bos_token_id': self.tokenizer.bos_token_id,
            'use_cache': True,
        }
        
        # Update with user kwargs
        default_generation_kwargs.update(self.kwargs)
        
        # Ensure sampling parameters are consistent
        if default_generation_kwargs['do_sample']:
            assert default_generation_kwargs['temperature'] > 0, "Temperature must be positive for sampling"
            assert default_generation_kwargs['top_k'] > 0, "top_k must be positive for sampling"
            assert 0 < default_generation_kwargs['top_p'] <= 1, "top_p must be between 0 and 1"
        
        logging.debug(f"Generation config prepared with kwargs: {default_generation_kwargs}")
        return GenerationConfig(**default_generation_kwargs)

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
            # Initialize DynamicCache with the correct device and assign to 'past_key_values'
            model_kwargs['past_key_values'] = DynamicCache(device=self.device)
            logging.debug(f"DynamicCache initialized and assigned to '{role}_past_key_values'.")
        else:
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

        # Handle different output formats
        if isinstance(outputs, dict):
            new_past = outputs.get('past_key_values')
        else:
            # Handle ModelOutput object
            new_past = getattr(outputs, 'past_key_values', None)

        if new_past is not None:
            logging.debug(f"Updating {role} cache with {num_tokens} new tokens.")
            model_kwargs['past_key_values'].update(new_past, num_tokens)
            logging.debug(f"{role.capitalize()} cache successfully updated with new past_key_values.")
        else:
            logging.warning(f"No 'past_key_values' found in the outputs to update the {role} cache.")

    def prepare_model_kwargs(self, role='target'):
        """
        Preparates model_kwargs for generation by incorporating cached past_key_values.
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
                logging.debug(f"Prepared {role} model_kwargs with cache of length {len(cache)}")
            else:
                model_kwargs.pop('past_key_values', None)
                logging.debug(f"Prepared {role} model_kwargs without cache (empty cache)")

        return model_kwargs

    
    # def speculative_sampling_with_profiling(self, draft_logits, target_logits, candidate_new_tokens, threshold=0.1):
    #     """
    #     Applies speculative sampling with profiling on the speculative_sampling method.

    #     Args:
    #         draft_logits (torch.Tensor): Logits from the draft model.
    #         target_logits (torch.Tensor): Logits from the target model.
    #         candidate_new_tokens (torch.Tensor): Candidate tokens from the draft model.
    #         threshold (float): Confidence threshold for acceptance.

    #     Returns:
    #         torch.Tensor: Accepted tokens.
    #         int: Number of matches.
    #     """
    #     from torch.profiler import profile, ProfilerActivity, tensorboard_trace_handler, record_function
    #     import os

    #     # Create profile_logs directory if it doesn't exist
    #     os.makedirs("profile_logs", exist_ok=True)

    #     with profile(
    #         activities=[
    #             ProfilerActivity.CPU,
    #             ProfilerActivity.CUDA
    #         ],
    #         record_shapes=True,
    #         profile_memory=True,
    #         with_stack=True,
    #         with_flops=True,
    #         on_trace_ready=tensorboard_trace_handler("./profile_logs")
    #     ) as prof:
    #         with record_function("speculative_sampling"):
    #             result = self.speculative_sampling(draft_logits, target_logits, candidate_new_tokens, threshold)
    #         prof.step()

    #     # Print profiling results
    #     print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
        
    #     return result

    def speculative_sampling(self, draft_logits, target_logits, candidate_new_tokens, extra_target_logits=None, do_sample=False):
        """
        Implements speculative sampling following the paper's algorithm.
        If all K tokens are accepted, also returns the extra target token.
        """
        draft_logits = draft_logits.to(self.device)
        target_logits = target_logits.to(self.device)
        candidate_new_tokens = candidate_new_tokens.to(self.device)
        extra_target_logits = extra_target_logits.to(self.device) if extra_target_logits is not None else None
        batch_size = candidate_new_tokens.size(0)
        seq_len = candidate_new_tokens.size(1)

        if DEBUG:
            logging.debug(f"\nStarting speculative sampling for {seq_len} tokens")

        accepted_tokens = []
        
        # this loop here can definitely be parallelized (dunno how much it'll help tho)
        for t in range(seq_len):
            current_token = candidate_new_tokens[:, t:t+1]
            
            # Sampling case
            if do_sample:
                # Get probabilities for the proposed token
                draft_probs = torch.softmax(draft_logits[:, t], dim=-1)
                target_probs = torch.softmax(target_logits[:, t], dim=-1)
                
                # Get probabilities for the current token
                p_t = draft_probs.gather(-1, current_token)
                q_t = target_probs.gather(-1, current_token)
                
                # Accept/reject based on probability ratio
                r = torch.rand_like(p_t, device=self.device)
                acceptance_ratio = torch.minimum(torch.ones_like(p_t), q_t / (p_t + 1e-10))
                accepted = r < acceptance_ratio

                # If the token was rejected, sample from target distribution
                if not accepted.any():
                    new_token = torch.multinomial(target_probs, num_samples=1)
                    accepted_tokens.append(new_token)
                    if DEBUG:
                        logging.debug(f"Rejected at position {t} - Sampled new token: '{self.tokenizer.decode(new_token[0])}'")
                    break
            else:
                # Greedy case
                target_token = target_logits[:, t].argmax(dim=-1, keepdim=True)
                accepted = (current_token == target_token)

                # If the token was rejected, use target token
                if not accepted.any():
                    accepted_tokens.append(target_token)
                    if DEBUG:
                        logging.debug(f"Rejected at position {t} - Using target token: '{self.tokenizer.decode(target_token[0])}'")
                    break

            # Token was accepted
            accepted_tokens.append(current_token)
            if DEBUG:
                logging.debug(f"Token {t} accepted: '{self.tokenizer.decode(current_token[0])}'")

        # BONUS: If we accepted all K tokens and have extra target logits, add one more token
        if len(accepted_tokens) == seq_len and extra_target_logits is not None:
            if do_sample:
                extra_probs = torch.softmax(extra_target_logits.squeeze(1), dim=-1)
                extra_token = torch.multinomial(extra_probs, num_samples=1)
            else:
                extra_token = extra_target_logits.argmax(dim=-1, keepdim=True)
            
            accepted_tokens.append(extra_token)
            if DEBUG:
                logging.debug(f"All {seq_len} tokens accepted! Adding extra token: '{self.tokenizer.decode(extra_token[0])}'")

        # If no tokens were accepted, return empty tensor
        if not accepted_tokens:
            return torch.empty((batch_size, 0), dtype=torch.long, device=self.device), 0

        accepted_tokens = torch.cat(accepted_tokens, dim=1)
        if DEBUG:
            logging.debug(f"Total accepted tokens: {accepted_tokens.size(1)}")
        return accepted_tokens, accepted_tokens.size(1)



    # def generate_with_profiling(self, inputs):
    #     """
    #     Generates tokens with simplified profiling to ensure logs are saved.

    #     Args:
    #         inputs (dict): Input dictionary containing necessary tensors like 'input_ids', 'attention_mask', etc.

    #     Returns:
    #         torch.Tensor: Generated input_ids with appended tokens.
    #     """
    #     from torch.profiler import profile, ProfilerActivity, tensorboard_trace_handler, record_function

    #     # Move all input tensors to the correct device
    #     input_ids = inputs['input_ids'].to(self.device)
    #     attention_mask = inputs['attention_mask'].to(self.device)
    #     pixel_values = inputs['pixel_values'].to(self.device)
    #     image_grid_thw = inputs['image_grid_thw']
    #     if image_grid_thw is not None:
    #         image_grid_thw = image_grid_thw.to(self.device)

    #     inputs = {
    #         'input_ids': input_ids,
    #         'attention_mask': attention_mask,
    #         'pixel_values': pixel_values,
    #         'image_grid_thw': image_grid_thw
    #     }
        
    #     with profile(
    #         activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],  # Include CUDA if using GPU
    #         on_trace_ready=tensorboard_trace_handler('./profile_logs'),
    #         record_shapes=False,
    #         profile_memory=False,
    #         with_stack=False
    #     ) as prof:
    #         with record_function("generate"):
    #             output = self.generate(inputs)
    #             # Ensure that prof.step() is called within the generate method if it contains loops
    #             # If generate is a single step, prof.step() can be called here
    #         prof.step()
    #     print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
    #     return output

    def generate(self, inputs):
        """
        Modified generate method to properly handle sampling vs greedy decoding
        """
        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)
        pixel_values = inputs['pixel_values'].to(self.device)
        image_grid_thw = inputs['image_grid_thw']
        if image_grid_thw is not None:
            image_grid_thw = image_grid_thw.to(self.device)

        # Get sampling flag from generation config
        do_sample = self.target_generation_config.do_sample
        max_new_tokens = self.target_generation_config.max_new_tokens
        tokens_generated = 0

        # Generate initial context with target model
        if DEBUG:
            logging.debug("Generating initial context with target model")
        
        target_outputs = self.target_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            use_cache=True
        )

        # Get first num_first_target tokens from target model
        next_tokens = []
        for _ in range(num_first_target):
            next_token_logits = target_outputs['logits'][:, -1, :]
            if do_sample:
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = next_token_logits.argmax(dim=-1, keepdim=True)
            next_tokens.append(next_token)
            
            # Update input and get new target outputs
            input_ids = torch.cat([input_ids, next_token], dim=1)
            target_outputs = self.target_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                use_cache=True
            )

        # Stack tokens and update input_ids
        next_tokens = torch.cat(next_tokens, dim=1)
        input_ids = torch.cat((input_ids, next_tokens), dim=-1)
        attention_mask = torch.cat((
            attention_mask,
            torch.ones((attention_mask.shape[0], num_first_target), device=self.device)
        ), dim=1)
        tokens_generated += num_first_target

        if DEBUG:
            logging.debug(f"Initial {num_first_target} tokens: '{self.tokenizer.decode(next_tokens[0])}'")


        #===================================== TODO: correct up to here =====================================

        # Main generation loop
        while tokens_generated < max_new_tokens: # sets hard limit on total number tokens generated
            if DEBUG:
                logging.debug(f"\nGeneration step {tokens_generated}/{max_new_tokens}")

            # Generate K draft tokens autoregressively with draft model
            draft_tokens = []
            draft_logits = []
            current_ids = input_ids

            for _ in range(num_draft_samples):
                draft_outputs = self.draft_model(
                    input_ids=current_ids,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values,
                    image_grid_thw=image_grid_thw,
                    use_cache=True
                )
                
                # Get logits for next token
                next_logits = draft_outputs['logits'][:, -1:, :]  # Shape: [batch_size, 1, vocab_size]
                
                # Select next token (greedy or sampling)
                if do_sample:
                    probs = torch.softmax(next_logits.squeeze(1), dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1).unsqueeze(1)
                else:
                    next_token = next_logits.argmax(dim=-1, keepdim=True)
                
                draft_tokens.append(next_token)
                draft_logits.append(next_logits)
                next_token = next_token.squeeze(-1)
                current_ids = torch.cat([current_ids, next_token], dim=1)

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
                use_cache=True
            )

            # Get logits for the draft positions (offset by 1)
            # Shape: [batch_size, num_draft_samples, vocab_size]
            target_logits = target_outputs['logits'][:, -(num_draft_samples+1):-1, :]
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
                torch.ones((attention_mask.shape[0], valid_tokens_padded.shape[1]), device=self.device)
            ), dim=1)
            
            tokens_generated += n_matches

            # Check for EOS token
            if (valid_tokens_padded == self.tokenizer.eos_token_id).any():
                break

        return input_ids






