"""
Core NanoAccel functionality for optimized LLM inference.
"""

import os
import time
import platform
import logging
from typing import Dict, Optional, Tuple, Union, List
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from torch.cuda.amp import autocast

from .utils import detect_cpu_features, optimize_cpu_scheduling
from .quantization import QuantizationConfig
from .speculative import SpeculativeDecoding

logger = logging.getLogger(__name__)


class NanoAccel:
    """
    Main NanoAccel class for optimized LLM inference on CPU.
    
    Features:
    - Ultra-low-bit quantization (1-4 bit) for memory efficiency
    - Speculative decoding for faster token generation
    - Memory offloading and CPU scheduling optimizations
    - Compatible with models like Gemma 2B, Llama 3.2 1B/3B
    """
    
    def __init__(
        self,
        model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        quant_config: Optional[QuantizationConfig] = None,
        mixed_precision: bool = False,
        cpu_optimization: bool = True,
        verbose: bool = True
    ):
        """
        Initialize NanoAccel with the specified configuration.
        
        Args:
            model_name: HuggingFace model identifier
            quant_config: Quantization configuration
            mixed_precision: Enable mixed precision inference
            cpu_optimization: Enable CPU-specific optimizations
            verbose: Enable verbose logging
        """
        self.model_name = model_name
        self.quant_config = quant_config or QuantizationConfig()
        self.mixed_precision = mixed_precision
        self.cpu_optimization = cpu_optimization
        self.verbose = verbose
        
        # Initialize components
        self.tokenizer = None
        self.model = None
        self.draft_model = None
        self.draft_tokenizer = None
        self.cpu_info = None
        self.speculative_decoder = None
        
        # Performance tracking
        self.inference_stats = {
            "total_tokens": 0,
            "total_time": 0.0,
            "inference_count": 0
        }
        
        if self.verbose:
            self._setup_logging()
    
    def _setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    def load_model(self) -> Tuple[AutoTokenizer, AutoModelForCausalLM]:
        """
        Load the main model with optimizations.
        
        Returns:
            Tuple of (tokenizer, model)
        """
        try:
            if self.verbose:
                logger.info(f"Loading model: {self.model_name}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            # Detect CPU capabilities
            self.cpu_info = detect_cpu_features()
            if self.verbose:
                logger.info(f"CPU: {self.cpu_info['cores']} cores, AVX2: {self.cpu_info['avx2']}")
            
            # Configure data type
            dtype = torch.bfloat16 if self.mixed_precision else torch.float32
            
            # Load model with quantization if specified
            if self.quant_config.enabled:
                quant_config = self.quant_config.get_bitsandbytes_config(dtype)
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    quantization_config=quant_config,
                    device_map="cpu",
                    torch_dtype=dtype
                )
                if self.verbose:
                    logger.info(f"Loaded {self.model_name} with {self.quant_config.quant_type} quantization")
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=dtype
                )
                if self.verbose:
                    logger.info(f"Loaded {self.model_name} without quantization")
            
            # Apply optimizations
            if self.cpu_optimization:
                self._apply_cpu_optimizations()
            
            # Compile model for better performance
            if hasattr(torch, 'compile'):
                self.model = torch.compile(self.model)
            
            self.model.to("cpu")
            return self.tokenizer, self.model
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def load_draft_model(self, draft_name: str = "EleutherAI/pythia-70m"):
        """
        Load a smaller draft model for speculative decoding.
        
        Args:
            draft_name: HuggingFace model identifier for draft model
        """
        try:
            if self.verbose:
                logger.info(f"Loading draft model: {draft_name}")
            
            # Initialize speculative decoder
            self.speculative_decoder = SpeculativeDecoding(draft_model_name=draft_name)
            self.speculative_decoder.load_draft_model()
            
            self.draft_tokenizer = self.speculative_decoder.draft_tokenizer
            self.draft_model = self.speculative_decoder.draft_model
            
            if self.verbose:
                logger.info(f"Loaded draft model: {draft_name}")
                
        except Exception as e:
            logger.error(f"Error loading draft model: {e}")
            raise
    
    def _apply_cpu_optimizations(self):
        """Apply CPU-specific optimizations."""
        if not self.cpu_info:
            return
            
        try:
            # Pin to performance cores (first half)
            num_cores = self.cpu_info["cores"]
            performance_cores = list(range(num_cores // 2))
            os.sched_setaffinity(0, performance_cores)
            
            if self.verbose:
                logger.info(f"Pinned to performance cores: {performance_cores}")
                
        except (OSError, AttributeError) as e:
            if self.verbose:
                logger.warning(f"Could not set CPU affinity: {e}")
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_p: float = 0.9,
        top_k: int = 50,
        do_sample: bool = True,
        use_speculative: bool = False,
        gamma: int = 4,
        early_exit_threshold: float = 0.9
    ) -> Dict[str, Union[str, float, int]]:
        """
        Generate text using the loaded model.
        
        Args:
            prompt: Input prompt text
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            do_sample: Whether to use sampling
            use_speculative: Whether to use speculative decoding
            gamma: Number of speculative tokens to generate
            early_exit_threshold: Threshold for early exit in speculative decoding
            
        Returns:
            Dictionary containing generated text and performance metrics
        """
        if not self.model or not self.tokenizer:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        start_time = time.time()
        
        try:
            if use_speculative and self.draft_model and self.draft_tokenizer:
                generated_text = self._speculative_generate(
                    prompt, max_new_tokens, temperature, top_p, top_k, 
                    do_sample, gamma, early_exit_threshold
                )
            else:
                generated_text = self._standard_generate(
                    prompt, max_new_tokens, temperature, top_p, top_k, do_sample
                )
            
            end_time = time.time()
            generation_time = end_time - start_time
            
            # Update statistics
            tokens_generated = len(self.tokenizer.encode(generated_text)) - len(self.tokenizer.encode(prompt))
            self.inference_stats["total_tokens"] += tokens_generated
            self.inference_stats["total_time"] += generation_time
            self.inference_stats["inference_count"] += 1
            
            tokens_per_second = tokens_generated / generation_time if generation_time > 0 else 0
            
            if self.verbose:
                logger.info(f"Generated {tokens_generated} tokens in {generation_time:.2f}s "
                          f"({tokens_per_second:.2f} tokens/sec)")
            
            return {
                "text": generated_text,
                "tokens_generated": tokens_generated,
                "generation_time": generation_time,
                "tokens_per_second": tokens_per_second,
                "method": "speculative" if use_speculative else "standard"
            }
            
        except Exception as e:
            logger.error(f"Error during generation: {e}")
            raise
    
    def _standard_generate(
        self,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        do_sample: bool
    ) -> str:
        """Standard text generation without speculative decoding."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cpu")
        tokens = inputs["input_ids"][0].tolist()
        
        with torch.no_grad(), autocast(enabled=self.mixed_precision, dtype=torch.bfloat16):
            for _ in range(max_new_tokens):
                input_tensor = self.tokenizer(
                    self.tokenizer.convert_tokens_to_string(
                        self.tokenizer.convert_ids_to_tokens(tokens)
                    ),
                    return_tensors="pt"
                ).to("cpu")
                
                logits = self.model(**input_tensor).logits[0, -1]
                
                if do_sample:
                    # Apply temperature
                    logits = logits / temperature
                    
                    # Top-k filtering
                    if top_k > 0:
                        top_k_logits, _ = torch.topk(logits, top_k)
                        logits[logits < top_k_logits[-1]] = float('-inf')
                    
                    # Top-p (nucleus) filtering
                    if top_p < 1.0:
                        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                        sorted_indices_to_remove = cumulative_probs > top_p
                        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                        sorted_indices_to_remove[..., 0] = 0
                        indices_to_remove = sorted_indices_to_remove.scatter(0, sorted_indices, sorted_indices_to_remove)
                        logits[indices_to_remove] = float('-inf')
                    
                    probs = torch.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, 1).item()
                else:
                    next_token = logits.argmax().item()
                
                tokens.append(next_token)
        
        return self.tokenizer.decode(tokens, skip_special_tokens=True)
    
    def _speculative_generate(
        self,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        do_sample: bool,
        gamma: int,
        early_exit_threshold: float
    ) -> str:
        """Speculative decoding implementation."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cpu")
        tokens = inputs["input_ids"][0].tolist()
        
        with torch.no_grad(), autocast(enabled=self.mixed_precision, dtype=torch.bfloat16):
            for _ in range(max_new_tokens):
                # Generate draft tokens
                draft_preds = []
                recent_tokens = tokens[-32:] if len(tokens) > 32 else tokens
                
                # Pin draft model to efficiency cores
                try:
                    efficiency_cores = list(range(self.cpu_info["cores"] // 2, self.cpu_info["cores"]))
                    os.sched_setaffinity(0, efficiency_cores)
                except (OSError, AttributeError):
                    pass
                
                draft_input = self.draft_tokenizer(
                    self.tokenizer.convert_tokens_to_string(
                        self.tokenizer.convert_ids_to_tokens(recent_tokens)
                    ),
                    return_tensors="pt"
                ).to("cpu")
                
                for _ in range(gamma):
                    draft_logits = self.draft_model(**draft_input).logits[0, -1]
                    probs = torch.softmax(draft_logits, dim=-1)
                    
                    if probs.max() > early_exit_threshold:
                        break
                    
                    if do_sample:
                        next_token = torch.multinomial(probs, 1).item()
                    else:
                        next_token = probs.argmax().item()
                    
                    draft_preds.append(next_token)
                    draft_input["input_ids"] = torch.cat([
                        draft_input["input_ids"],
                        torch.tensor([[next_token]])
                    ], dim=1)
                
                # Switch back to performance cores for main model
                try:
                    performance_cores = list(range(self.cpu_info["cores"] // 2))
                    os.sched_setaffinity(0, performance_cores)
                except (OSError, AttributeError):
                    pass
                
                # Verify draft tokens with main model
                verify_input = self.tokenizer(
                    self.tokenizer.convert_tokens_to_string(
                        self.tokenizer.convert_ids_to_tokens(tokens + draft_preds)
                    ),
                    return_tensors="pt"
                ).to("cpu")
                
                main_logits = self.model(**verify_input).logits
                
                # Accept verified tokens
                accepted = 0
                for i, draft_token in enumerate(draft_preds):
                    if i < main_logits.shape[1] - len(tokens):
                        predicted_token = main_logits[0, -len(draft_preds) + i].argmax().item()
                        if predicted_token == draft_token:
                            accepted += 1
                        else:
                            break
                    else:
                        break
                
                tokens.extend(draft_preds[:accepted])
                
                # Generate additional token if needed
                if accepted < gamma and len(tokens) - len(inputs["input_ids"][0]) < max_new_tokens:
                    if accepted < len(draft_preds):
                        next_token = main_logits[0, -gamma + accepted].argmax().item()
                    else:
                        next_token = main_logits[0, -1].argmax().item()
                    tokens.append(next_token)
        
        return self.tokenizer.decode(tokens, skip_special_tokens=True)
    
    def get_stats(self) -> Dict[str, Union[float, int]]:
        """Get inference statistics."""
        stats = self.inference_stats.copy()
        if stats["inference_count"] > 0:
            stats["average_tokens_per_second"] = stats["total_tokens"] / stats["total_time"]
            stats["average_generation_time"] = stats["total_time"] / stats["inference_count"]
        return stats
    
    def reset_stats(self):
        """Reset inference statistics."""
        self.inference_stats = {
            "total_tokens": 0,
            "total_time": 0.0,
            "inference_count": 0
        }
