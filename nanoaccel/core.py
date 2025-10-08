"""
Enhanced Core NanoAccel functionality with advanced optimizations.
"""

import os
import time
import platform
import logging
from typing import Dict, Optional, Tuple, Union, List, Callable
from dataclasses import dataclass
from enum import Enum
import threading
from queue import Queue
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    BitsAndBytesConfig,
    StoppingCriteria,
    StoppingCriteriaList
)
from torch.cuda.amp import autocast
import numpy as np

from .utils import detect_cpu_features, optimize_cpu_scheduling
from .quantization import QuantizationConfig, DynamicQuantizer
from .speculative import SpeculativeDecoding
from .cache_manager import KVCacheManager
from .profiler import PerformanceProfiler
from .batch_processor import BatchProcessor

logger = logging.getLogger(__name__)


class InferenceMode(Enum):
    """Inference execution modes."""
    STANDARD = "standard"
    SPECULATIVE = "speculative"
    STREAMING = "streaming"
    BATCH = "batch"


@dataclass
class GenerationConfig:
    """Configuration for text generation."""
    max_new_tokens: int = 50
    min_new_tokens: int = 0
    temperature: float = 1.0
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.0
    length_penalty: float = 1.0
    do_sample: bool = True
    num_beams: int = 1
    early_stopping: bool = False
    pad_token_id: Optional[int] = None
    eos_token_id: Optional[int] = None
    use_cache: bool = True


class CustomStoppingCriteria(StoppingCriteria):
    """Custom stopping criteria for generation."""
    
    def __init__(self, stop_sequences: List[str], tokenizer):
        self.stop_sequences = stop_sequences
        self.tokenizer = tokenizer
    
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        decoded = self.tokenizer.decode(input_ids[0])
        return any(seq in decoded for seq in self.stop_sequences)


class NanoAccel:
    """
    Enhanced NanoAccel class with advanced optimizations.
    
    New Features:
    - Dynamic quantization with adaptive precision
    - Advanced KV cache management with compression
    - Streaming generation with callbacks
    - Batch processing for throughput optimization
    - Performance profiling and monitoring
    - Model warmup and compilation
    - Thread-safe operations
    - Context window management
    - Token budget control
    - Custom stopping criteria
    """
    
    def __init__(
        self,
        model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        quant_config: Optional[QuantizationConfig] = None,
        mixed_precision: bool = False,
        cpu_optimization: bool = True,
        enable_profiling: bool = False,
        enable_caching: bool = True,
        max_cache_size_mb: int = 512,
        compile_model: bool = True,
        num_threads: Optional[int] = None,
        verbose: bool = True
    ):
        """Initialize Enhanced NanoAccel."""
        self.model_name = model_name
        self.quant_config = quant_config or QuantizationConfig()
        self.mixed_precision = mixed_precision
        self.cpu_optimization = cpu_optimization
        self.enable_profiling = enable_profiling
        self.enable_caching = enable_caching
        self.max_cache_size_mb = max_cache_size_mb
        self.compile_model = compile_model
        self.num_threads = num_threads
        self.verbose = verbose
        
        # Initialize components
        self.tokenizer = None
        self.model = None
        self.draft_model = None
        self.draft_tokenizer = None
        self.cpu_info = None
        self.speculative_decoder = None
        
        # Advanced components
        self.kv_cache_manager = None
        self.profiler = None
        self.batch_processor = None
        self.dynamic_quantizer = None
        
        # Thread safety
        self._lock = threading.RLock()
        self._generation_queue = Queue()
        
        # Performance tracking
        self.inference_stats = {
            "total_tokens": 0,
            "total_time": 0.0,
            "inference_count": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "warmup_completed": False
        }
        
        if self.verbose:
            self._setup_logging()
        
        # Initialize profiler
        if self.enable_profiling:
            self.profiler = PerformanceProfiler()
    
    def _setup_logging(self):
        """Setup enhanced logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('nanoaccel.log')
            ]
        )
    
    def load_model(self) -> Tuple[AutoTokenizer, AutoModelForCausalLM]:
        """Load model with enhanced optimizations."""
        with self._lock:
            try:
                if self.verbose:
                    logger.info(f"Loading model: {self.model_name}")
                
                # Detect CPU capabilities
                self.cpu_info = detect_cpu_features()
                if self.verbose:
                    logger.info(f"CPU: {self.cpu_info['cores']} cores, "
                              f"AVX2: {self.cpu_info['avx2']}, "
                              f"AVX512: {self.cpu_info['avx512']}")
                
                # Set optimal thread count
                if self.num_threads:
                    torch.set_num_threads(self.num_threads)
                else:
                    # Use half of available cores for optimal performance
                    torch.set_num_threads(max(1, self.cpu_info['cores'] // 2))
                
                # Load tokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name,
                    use_fast=True,
                    trust_remote_code=False
                )
                
                # Ensure tokenizer has pad token
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                
                # Configure data type
                dtype = torch.bfloat16 if self.mixed_precision else torch.float32
                
                # Load model with optimizations
                if self.quant_config.enabled:
                    quant_config = self.quant_config.get_bitsandbytes_config(dtype)
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.model_name,
                        quantization_config=quant_config,
                        device_map="cpu",
                        torch_dtype=dtype,
                        low_cpu_mem_usage=True,
                        trust_remote_code=False
                    )
                    if self.verbose:
                        logger.info(f"Loaded with {self.quant_config.quant_type} quantization")
                else:
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.model_name,
                        torch_dtype=dtype,
                        low_cpu_mem_usage=True,
                        trust_remote_code=False
                    )
                
                # Apply CPU optimizations
                if self.cpu_optimization:
                    self._apply_cpu_optimizations()
                
                # Initialize dynamic quantizer
                if self.quant_config.enabled:
                    self.dynamic_quantizer = DynamicQuantizer(self.quant_config)
                
                # Initialize KV cache manager
                if self.enable_caching:
                    self.kv_cache_manager = KVCacheManager(
                        max_size_mb=self.max_cache_size_mb,
                        compression_enabled=True
                    )
                
                # Initialize batch processor
                self.batch_processor = BatchProcessor(
                    model=self.model,
                    tokenizer=self.tokenizer,
                    max_batch_size=8
                )
                
                # Compile model for better performance
                if self.compile_model and hasattr(torch, 'compile'):
                    try:
                        self.model = torch.compile(self.model, mode="reduce-overhead")
                        if self.verbose:
                            logger.info("Model compiled successfully")
                    except Exception as e:
                        logger.warning(f"Model compilation failed: {e}")
                
                self.model.to("cpu")
                self.model.eval()
                
                # Warmup
                self._warmup_model()
                
                if self.verbose:
                    logger.info("Model loaded successfully")
                
                return self.tokenizer, self.model
                
            except Exception as e:
                logger.error(f"Error loading model: {e}")
                raise
    
    def _warmup_model(self, warmup_steps: int = 3):
        """Warmup model for optimal performance."""
        if self.verbose:
            logger.info("Warming up model...")
        
        try:
            warmup_prompt = "Hello, this is a warmup prompt."
            for _ in range(warmup_steps):
                self.generate(
                    prompt=warmup_prompt,
                    max_new_tokens=10,
                    do_sample=False
                )
            
            self.inference_stats["warmup_completed"] = True
            if self.verbose:
                logger.info("Model warmup completed")
                
        except Exception as e:
            logger.warning(f"Model warmup failed: {e}")
    
    def load_draft_model(self, draft_name: str = "EleutherAI/pythia-70m"):
        """Load draft model for speculative decoding."""
        with self._lock:
            try:
                if self.verbose:
                    logger.info(f"Loading draft model: {draft_name}")
                
                self.speculative_decoder = SpeculativeDecoding(
                    draft_model_name=draft_name,
                    cpu_info=self.cpu_info
                )
                self.speculative_decoder.load_draft_model()
                
                self.draft_tokenizer = self.speculative_decoder.draft_tokenizer
                self.draft_model = self.speculative_decoder.draft_model
                
                if self.verbose:
                    logger.info(f"Draft model loaded successfully")
                    
            except Exception as e:
                logger.error(f"Error loading draft model: {e}")
                raise
    
    def _apply_cpu_optimizations(self):
        """Apply comprehensive CPU optimizations."""
        if not self.cpu_info:
            return
        
        try:
            # Pin to performance cores
            num_cores = self.cpu_info["cores"]
            performance_cores = list(range(num_cores // 2))
            os.sched_setaffinity(0, performance_cores)
            
            # Enable MKL optimizations if available
            if hasattr(torch, 'set_num_interop_threads'):
                torch.set_num_interop_threads(max(1, num_cores // 4))
            
            # Set MKL environment variables
            os.environ['MKL_NUM_THREADS'] = str(num_cores // 2)
            os.environ['OMP_NUM_THREADS'] = str(num_cores // 2)
            os.environ['MKL_DYNAMIC'] = 'FALSE'
            
            # Enable TF32 for better performance on supported hardware
            if hasattr(torch.backends.cuda, 'matmul'):
                torch.backends.cuda.matmul.allow_tf32 = True
            
            if self.verbose:
                logger.info(f"CPU optimizations applied: {num_cores} cores, "
                          f"pinned to {performance_cores}")
                
        except (OSError, AttributeError) as e:
            if self.verbose:
                logger.warning(f"Could not apply all CPU optimizations: {e}")
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 50,
        min_new_tokens: int = 0,
        temperature: float = 1.0,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.0,
        length_penalty: float = 1.0,
        do_sample: bool = True,
        num_beams: int = 1,
        use_speculative: bool = False,
        gamma: int = 4,
        early_exit_threshold: float = 0.9,
        stop_sequences: Optional[List[str]] = None,
        streaming_callback: Optional[Callable[[str], None]] = None,
        **kwargs
    ) -> Dict[str, Union[str, float, int]]:
        """
        Enhanced text generation with advanced features.
        
        Args:
            prompt: Input prompt text
            max_new_tokens: Maximum number of tokens to generate
            min_new_tokens: Minimum number of tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            repetition_penalty: Penalty for repeating tokens
            length_penalty: Penalty for sequence length
            do_sample: Whether to use sampling
            num_beams: Number of beams for beam search
            use_speculative: Whether to use speculative decoding
            gamma: Number of speculative tokens
            early_exit_threshold: Threshold for early exit
            stop_sequences: List of sequences that trigger stopping
            streaming_callback: Callback function for streaming generation
            **kwargs: Additional generation parameters
            
        Returns:
            Dictionary containing generated text and performance metrics
        """
        if not self.model or not self.tokenizer:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        with self._lock:
            if self.enable_profiling:
                self.profiler.start_profiling("generation")
            
            start_time = time.time()
            
            try:
                # Prepare generation config
                gen_config = GenerationConfig(
                    max_new_tokens=max_new_tokens,
                    min_new_tokens=min_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    repetition_penalty=repetition_penalty,
                    length_penalty=length_penalty,
                    do_sample=do_sample,
                    num_beams=num_beams,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
                
                # Setup stopping criteria
                stopping_criteria = None
                if stop_sequences:
                    stopping_criteria = StoppingCriteriaList([
                        CustomStoppingCriteria(stop_sequences, self.tokenizer)
                    ])
                
                # Choose generation method
                if streaming_callback:
                    generated_text = self._streaming_generate(
                        prompt, gen_config, streaming_callback, stopping_criteria
                    )
                elif use_speculative and self.draft_model:
                    generated_text = self._speculative_generate(
                        prompt, gen_config, gamma, early_exit_threshold, stopping_criteria
                    )
                elif num_beams > 1:
                    generated_text = self._beam_search_generate(
                        prompt, gen_config, stopping_criteria
                    )
                else:
                    generated_text = self._standard_generate(
                        prompt, gen_config, stopping_criteria
                    )
                
                end_time = time.time()
                generation_time = end_time - start_time
                
                # Calculate metrics
                tokens_generated = len(self.tokenizer.encode(generated_text)) - \
                                 len(self.tokenizer.encode(prompt))
                tokens_per_second = tokens_generated / generation_time if generation_time > 0 else 0
                
                # Update statistics
                self.inference_stats["total_tokens"] += tokens_generated
                self.inference_stats["total_time"] += generation_time
                self.inference_stats["inference_count"] += 1
                
                if self.enable_profiling:
                    self.profiler.stop_profiling("generation")
                
                result = {
                    "text": generated_text,
                    "tokens_generated": tokens_generated,
                    "generation_time": generation_time,
                    "tokens_per_second": tokens_per_second,
                    "method": self._get_generation_method(use_speculative, num_beams, streaming_callback)
                }
                
                if self.verbose:
                    logger.info(f"Generated {tokens_generated} tokens in {generation_time:.2f}s "
                              f"({tokens_per_second:.2f} tokens/sec)")
                
                return result
                
            except Exception as e:
                logger.error(f"Error during generation: {e}")
                if self.enable_profiling:
                    self.profiler.stop_profiling("generation")
                raise
    
    def _get_generation_method(self, speculative: bool, num_beams: int, streaming: bool) -> str:
        """Get the generation method name."""
        if streaming:
            return "streaming"
        elif speculative:
            return "speculative"
        elif num_beams > 1:
            return "beam_search"
        else:
            return "standard"
    
    def _standard_generate(
        self,
        prompt: str,
        config: GenerationConfig,
        stopping_criteria: Optional[StoppingCriteriaList] = None
    ) -> str:
        """Optimized standard generation."""
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True).to("cpu")
        
        with torch.no_grad(), autocast(enabled=self.mixed_precision, dtype=torch.bfloat16):
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=config.max_new_tokens,
                min_new_tokens=config.min_new_tokens,
                temperature=config.temperature,
                top_p=config.top_p,
                top_k=config.top_k,
                repetition_penalty=config.repetition_penalty,
                length_penalty=config.length_penalty,
                do_sample=config.do_sample,
                pad_token_id=config.pad_token_id,
                eos_token_id=config.eos_token_id,
                use_cache=config.use_cache,
                stopping_criteria=stopping_criteria
            )
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    def _streaming_generate(
        self,
        prompt: str,
        config: GenerationConfig,
        callback: Callable[[str], None],
        stopping_criteria: Optional[StoppingCriteriaList] = None
    ) -> str:
        """Generate text with streaming output."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cpu")
        tokens = inputs["input_ids"][0].tolist()
        generated_text = prompt
        
        with torch.no_grad(), autocast(enabled=self.mixed_precision, dtype=torch.bfloat16):
            for _ in range(config.max_new_tokens):
                input_tensor = self.tokenizer(
                    self.tokenizer.convert_tokens_to_string(
                        self.tokenizer.convert_ids_to_tokens(tokens)
                    ),
                    return_tensors="pt"
                ).to("cpu")
                
                logits = self.model(**input_tensor).logits[0, -1]
                
                # Apply repetition penalty
                if config.repetition_penalty != 1.0:
                    for token_id in set(tokens):
                        logits[token_id] /= config.repetition_penalty
                
                # Sample next token
                if config.do_sample:
                    logits = logits / config.temperature
                    
                    if config.top_k > 0:
                        top_k_logits, _ = torch.topk(logits, config.top_k)
                        logits[logits < top_k_logits[-1]] = float('-inf')
                    
                    if config.top_p < 1.0:
                        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                        sorted_indices_to_remove = cumulative_probs > config.top_p
                        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                        sorted_indices_to_remove[..., 0] = 0
                        indices_to_remove = sorted_indices_to_remove.scatter(0, sorted_indices, sorted_indices_to_remove)
                        logits[indices_to_remove] = float('-inf')
                    
                    probs = torch.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, 1).item()
                else:
                    next_token = logits.argmax().item()
                
                tokens.append(next_token)
                
                # Decode and call callback
                new_text = self.tokenizer.decode(tokens, skip_special_tokens=True)
                callback(new_text[len(generated_text):])
                generated_text = new_text
                
                # Check stopping criteria
                if next_token == config.eos_token_id:
                    break
                if stopping_criteria and stopping_criteria(torch.tensor([tokens]), None):
                    break
        
        return generated_text
    
    def _beam_search_generate(
        self,
        prompt: str,
        config: GenerationConfig,
        stopping_criteria: Optional[StoppingCriteriaList] = None
    ) -> str:
        """Generate text using beam search."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cpu")
        
        with torch.no_grad(), autocast(enabled=self.mixed_precision, dtype=torch.bfloat16):
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=config.max_new_tokens,
                min_new_tokens=config.min_new_tokens,
                num_beams=config.num_beams,
                temperature=config.temperature,
                repetition_penalty=config.repetition_penalty,
                length_penalty=config.length_penalty,
                early_stopping=config.early_stopping,
                pad_token_id=config.pad_token_id,
                eos_token_id=config.eos_token_id,
                use_cache=config.use_cache,
                stopping_criteria=stopping_criteria
            )
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    def _speculative_generate(
        self,
        prompt: str,
        config: GenerationConfig,
        gamma: int,
        early_exit_threshold: float,
        stopping_criteria: Optional[StoppingCriteriaList] = None
    ) -> str:
        """Enhanced speculative generation."""
        return self.speculative_decoder.speculative_generate(
            prompt=prompt,
            main_model=self.model,
            main_tokenizer=self.tokenizer,
            max_new_tokens=config.max_new_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k,
            do_sample=config.do_sample,
            gamma=gamma,
            early_exit_threshold=early_exit_threshold
        )["text"]
    
    def batch_generate(
        self,
        prompts: List[str],
        **kwargs
    ) -> List[Dict[str, Union[str, float, int]]]:
        """Generate text for multiple prompts in batch."""
        if not self.batch_processor:
            raise RuntimeError("Batch processor not initialized")
        
        return self.batch_processor.process_batch(prompts, **kwargs)
    
    def get_stats(self) -> Dict[str, Union[float, int]]:
        """Get comprehensive inference statistics."""
        stats = self.inference_stats.copy()
        
        if stats["inference_count"] > 0:
            stats["average_tokens_per_second"] = stats["total_tokens"] / stats["total_time"]
            stats["average_generation_time"] = stats["total_time"] / stats["inference_count"]
        
        if self.kv_cache_manager:
            cache_stats = self.kv_cache_manager.get_stats()
            stats.update(cache_stats)
        
        if self.enable_profiling and self.profiler:
            stats["profiling_data"] = self.profiler.get_summary()
        
        return stats
    
    def reset_stats(self):
        """Reset all statistics."""
        self.inference_stats = {
            "total_tokens": 0,
            "total_time": 0.0,
            "inference_count": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "warmup_completed": self.inference_stats.get("warmup_completed", False)
        }
        
        if self.kv_cache_manager:
            self.kv_cache_manager.clear()
        
        if self.enable_profiling and self.profiler:
            self.profiler.reset()
    
    def save_model(self, path: str):
        """Save model to disk."""
        if not self.model:
            raise RuntimeError("No model loaded")
        
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        logger.info(f"Model saved to {path}")
    
    def cleanup(self):
        """Cleanup resources."""
        if self.kv_cache_manager:
            self.kv_cache_manager.clear()
        
        if self.model:
            del self.model
            self.model = None
        
        if self.draft_model:
            del self.draft_model
            self.draft_model = None
        
        torch.cuda.empty_cache()
        logger.info("Resources cleaned up")
