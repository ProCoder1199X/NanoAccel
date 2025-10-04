"""
Speculative decoding implementation for NanoAccel.
"""

import time
from typing import Dict, List, Optional, Tuple, Union
import torch
import logging

logger = logging.getLogger(__name__)


class SpeculativeDecoding:
    """
    Speculative decoding implementation for faster text generation.
    
    Uses a smaller draft model to generate multiple tokens in parallel,
    then verifies them with the main model for improved throughput.
    """
    
    def __init__(
        self,
        draft_model_name: str = "EleutherAI/pythia-70m",
        gamma: int = 4,
        early_exit_threshold: float = 0.9,
        use_efficiency_cores: bool = True
    ):
        """
        Initialize speculative decoding.
        
        Args:
            draft_model_name: HuggingFace model identifier for draft model
            gamma: Number of speculative tokens to generate
            early_exit_threshold: Threshold for early exit in draft generation
            use_efficiency_cores: Whether to use efficiency cores for draft model
        """
        self.draft_model_name = draft_model_name
        self.gamma = gamma
        self.early_exit_threshold = early_exit_threshold
        self.use_efficiency_cores = use_efficiency_cores
        
        # Model components
        self.draft_model = None
        self.draft_tokenizer = None
        
        # Statistics
        self.stats = {
            "draft_tokens_generated": 0,
            "accepted_tokens": 0,
            "rejected_tokens": 0,
            "acceptance_rate": 0.0,
            "speedup_factor": 0.0
        }
    
    def load_draft_model(self):
        """Load the draft model and tokenizer."""
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            
            logger.info(f"Loading draft model: {self.draft_model_name}")
            self.draft_tokenizer = AutoTokenizer.from_pretrained(self.draft_model_name)
            self.draft_model = AutoModelForCausalLM.from_pretrained(self.draft_model_name).to("cpu")
            
            logger.info(f"Successfully loaded draft model: {self.draft_model_name}")
            
        except Exception as e:
            logger.error(f"Error loading draft model: {e}")
            raise
    
    def generate_draft_tokens(
        self,
        input_tokens: List[int],
        main_tokenizer,
        max_draft_tokens: Optional[int] = None
    ) -> List[int]:
        """
        Generate draft tokens using the draft model.
        
        Args:
            input_tokens: Input token sequence
            main_tokenizer: Tokenizer from the main model
            max_draft_tokens: Maximum number of draft tokens to generate
            
        Returns:
            List of generated draft tokens
        """
        if not self.draft_model or not self.draft_tokenizer:
            raise RuntimeError("Draft model not loaded. Call load_draft_model() first.")
        
        max_tokens = max_draft_tokens or self.gamma
        draft_tokens = []
        
        # Use recent context for draft generation
        recent_tokens = input_tokens[-32:] if len(input_tokens) > 32 else input_tokens
        recent_text = main_tokenizer.convert_tokens_to_string(
            main_tokenizer.convert_ids_to_tokens(recent_tokens)
        )
        
        # Convert to draft model's token space
        draft_input = self.draft_tokenizer(recent_text, return_tensors="pt").to("cpu")
        
        with torch.no_grad():
            for _ in range(max_tokens):
                draft_logits = self.draft_model(**draft_input).logits[0, -1]
                probs = torch.softmax(draft_logits, dim=-1)
                
                # Early exit if confidence is high
                if probs.max() > self.early_exit_threshold:
                    break
                
                # Sample next token
                next_token = probs.argmax().item()
                draft_tokens.append(next_token)
                
                # Update input for next iteration
                draft_input["input_ids"] = torch.cat([
                    draft_input["input_ids"],
                    torch.tensor([[next_token]])
                ], dim=1)
        
        self.stats["draft_tokens_generated"] += len(draft_tokens)
        return draft_tokens
    
    def verify_draft_tokens(
        self,
        input_tokens: List[int],
        draft_tokens: List[int],
        main_model,
        main_tokenizer
    ) -> Tuple[List[int], int]:
        """
        Verify draft tokens using the main model.
        
        Args:
            input_tokens: Original input tokens
            draft_tokens: Draft tokens to verify
            main_model: Main language model
            main_tokenizer: Tokenizer from the main model
            
        Returns:
            Tuple of (accepted_tokens, num_accepted)
        """
        if not draft_tokens:
            return [], 0
        
        # Prepare input with draft tokens
        full_sequence = input_tokens + draft_tokens
        full_text = main_tokenizer.convert_tokens_to_string(
            main_tokenizer.convert_ids_to_tokens(full_sequence)
        )
        
        verify_input = main_tokenizer(full_text, return_tensors="pt").to("cpu")
        
        with torch.no_grad():
            main_logits = main_model(**verify_input).logits
        
        # Verify each draft token
        accepted_tokens = []
        num_accepted = 0
        
        for i, draft_token in enumerate(draft_tokens):
            # Get the position in the logits corresponding to this draft token
            logit_pos = len(input_tokens) + i
            
            if logit_pos < main_logits.shape[1]:
                predicted_token = main_logits[0, logit_pos].argmax().item()
                
                if predicted_token == draft_token:
                    accepted_tokens.append(draft_token)
                    num_accepted += 1
                else:
                    # Stop at first rejection
                    break
            else:
                break
        
        # Update statistics
        self.stats["accepted_tokens"] += num_accepted
        self.stats["rejected_tokens"] += len(draft_tokens) - num_accepted
        
        # Update acceptance rate
        total_draft_tokens = self.stats["draft_tokens_generated"]
        if total_draft_tokens > 0:
            self.stats["acceptance_rate"] = self.stats["accepted_tokens"] / total_draft_tokens
        
        return accepted_tokens, num_accepted
    
    def speculative_generate(
        self,
        prompt: str,
        main_model,
        main_tokenizer,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_p: float = 0.9,
        top_k: int = 50,
        do_sample: bool = True
    ) -> Dict[str, Union[str, float, int]]:
        """
        Generate text using speculative decoding.
        
        Args:
            prompt: Input prompt
            main_model: Main language model
            main_tokenizer: Tokenizer from the main model
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            do_sample: Whether to use sampling
            
        Returns:
            Dictionary containing generated text and performance metrics
        """
        start_time = time.time()
        
        # Initialize token sequence
        inputs = main_tokenizer(prompt, return_tensors="pt").to("cpu")
        tokens = inputs["input_ids"][0].tolist()
        initial_length = len(tokens)
        
        with torch.no_grad():
            while len(tokens) - initial_length < max_new_tokens:
                # Generate draft tokens
                draft_tokens = self.generate_draft_tokens(tokens, main_tokenizer)
                
                if draft_tokens:
                    # Verify draft tokens
                    accepted_tokens, num_accepted = self.verify_draft_tokens(
                        tokens, draft_tokens, main_model, main_tokenizer
                    )
                    
                    # Add accepted tokens
                    tokens.extend(accepted_tokens)
                    
                    # Generate additional token if needed and we rejected some drafts
                    if num_accepted < len(draft_tokens) and len(tokens) - initial_length < max_new_tokens:
                        # Get the rejected position and generate from there
                        reject_pos = initial_length + len(tokens) - initial_length
                        context_text = main_tokenizer.convert_tokens_to_string(
                            main_tokenizer.convert_ids_to_tokens(tokens[:reject_pos])
                        )
                        
                        context_input = main_tokenizer(context_text, return_tensors="pt").to("cpu")
                        main_logits = main_model(**context_input).logits[0, -1]
                        
                        if do_sample:
                            # Apply sampling parameters
                            main_logits = main_logits / temperature
                            
                            # Top-k filtering
                            if top_k > 0:
                                top_k_logits, _ = torch.topk(main_logits, top_k)
                                main_logits[main_logits < top_k_logits[-1]] = float('-inf')
                            
                            # Top-p filtering
                            if top_p < 1.0:
                                sorted_logits, sorted_indices = torch.sort(main_logits, descending=True)
                                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                                sorted_indices_to_remove = cumulative_probs > top_p
                                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                                sorted_indices_to_remove[..., 0] = 0
                                indices_to_remove = sorted_indices_to_remove.scatter(0, sorted_indices, sorted_indices_to_remove)
                                main_logits[indices_to_remove] = float('-inf')
                            
                            probs = torch.softmax(main_logits, dim=-1)
                            next_token = torch.multinomial(probs, 1).item()
                        else:
                            next_token = main_logits.argmax().item()
                        
                        tokens.append(next_token)
                else:
                    # Fallback to standard generation
                    context_text = main_tokenizer.convert_tokens_to_string(
                        main_tokenizer.convert_ids_to_tokens(tokens)
                    )
                    context_input = main_tokenizer(context_text, return_tensors="pt").to("cpu")
                    main_logits = main_model(**context_input).logits[0, -1]
                    next_token = main_logits.argmax().item()
                    tokens.append(next_token)
        
        end_time = time.time()
        generation_time = end_time - start_time
        
        # Calculate final statistics
        tokens_generated = len(tokens) - initial_length
        tokens_per_second = tokens_generated / generation_time if generation_time > 0 else 0
        
        # Estimate speedup factor
        if self.stats["accepted_tokens"] > 0:
            self.stats["speedup_factor"] = (
                1 + (self.stats["acceptance_rate"] * self.gamma)
            )
        
        generated_text = main_tokenizer.decode(tokens, skip_special_tokens=True)
        
        return {
            "text": generated_text,
            "tokens_generated": tokens_generated,
            "generation_time": generation_time,
            "tokens_per_second": tokens_per_second,
            "method": "speculative",
            "acceptance_rate": self.stats["acceptance_rate"],
            "estimated_speedup": self.stats["speedup_factor"]
        }
    
    def get_stats(self) -> Dict[str, Union[float, int]]:
        """Get speculative decoding statistics."""
        return self.stats.copy()
    
    def reset_stats(self):
        """Reset statistics."""
        self.stats = {
            "draft_tokens_generated": 0,
            "accepted_tokens": 0,
            "rejected_tokens": 0,
            "acceptance_rate": 0.0,
            "speedup_factor": 0.0
        }
