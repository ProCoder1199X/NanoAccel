# NanoAccel Optimization Guide

Complete guide to maximizing performance on low-end hardware.

## Table of Contents
1. [Quick Wins](#quick-wins)
2. [Quantization Strategies](#quantization-strategies)
3. [Memory Optimization](#memory-optimization)
4. [CPU Optimization](#cpu-optimization)
5. [Cache Management](#cache-management)
6. [Speculative Decoding](#speculative-decoding)
7. [Batch Processing](#batch-processing)
8. [Production Deployment](#production-deployment)

## Quick Wins

### Enable All Optimizations
```python
from nanoaccel import NanoAccel, QuantizationConfig

nanoaccel = NanoAccel(
    model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    quant_config=QuantizationConfig(
        enabled=True,
        quant_type="int4"  # Most aggressive
    ),
    mixed_precision=True,      # +15% speed
    cpu_optimization=True,     # +20% speed
    enable_caching=True,       # +50% on repeated queries
    compile_model=True,        # +20% speed
    num_threads=4              # Optimal for most CPUs
)

nanoaccel.load_model()
nanoaccel.load_draft_model()  # Enable speculative decoding

# Generate with all optimizations
result = nanoaccel.generate(
    prompt="Your prompt here",
    use_speculative=True,
    gamma=6
)
```

**Expected Results:**
- 70-80% memory reduction
- 3-5x speed improvement
- 95%+ quality retention

## Quantization Strategies

### Choosing Quantization Level

```python
# INT8: Best quality-performance balance
quant_config = QuantizationConfig(
    enabled=True,
    quant_type="int8"
)
# Memory: -66% | Speed: +30% | Quality: 99%

# INT4: Aggressive optimization
quant_config = QuantizationConfig(
    enabled=True,
    quant_type="int4"
)
# Memory: -82% | Speed: +80% | Quality: 97-98%

# INT2: Maximum compression (experimental)
quant_config = QuantizationConfig(
    enabled=True,
    quant_type="int2"
)
# Memory: -90% | Speed: +100% | Quality: 90-95%
```

### Layer-Wise Quantization

For fine control over model quality:

```python
from nanoaccel import NanoAccel, QuantizationConfig
from nanoaccel.quantization import DynamicQuantizer

nanoaccel = NanoAccel(
    model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    quant_config=QuantizationConfig(enabled=True, quant_type="int4")
)
nanoaccel.load_model()

# Calibrate for optimal layer-wise quantization
sample_texts = [
    "The quick brown fox jumps over the lazy dog.",
    "Machine learning is transforming technology.",
    "Explain quantum computing in simple terms."
]

sample_inputs = nanoaccel.tokenizer(
    sample_texts,
    return_tensors="pt",
    padding=True
)["input_ids"]

nanoaccel.dynamic_quantizer.calibrate(
    nanoaccel.model,
    sample_inputs
)

# View layer sensitivities
layer_info = nanoaccel.dynamic_quantizer.get_layer_info()
for layer, info in layer_info.items():
    print(f"{layer}: {info['recommended_bits']}-bit recommended")
```

### Mixed Precision Strategy

```python
# Automatic mixed precision
nanoaccel = NanoAccel(
    model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    quant_config=QuantizationConfig(
        enabled=True,
        quant_type="int4",
        compute_dtype=torch.bfloat16  # Use BF16 for computations
    ),
    mixed_precision=True  # Automatic AMP
)

# Manual precision control
with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
    result = nanoaccel.generate(...)
```

## Memory Optimization

### Reduce Memory Footprint

```python
# Strategy 1: Aggressive quantization
quant_config = QuantizationConfig(
    enabled=True,
    quant_type="int4",
    chunk_size=1024  # Smaller chunks = less memory
)

# Strategy 2: Reduce cache size
nanoaccel = NanoAccel(
    quant_config=quant_config,
    max_cache_size_mb=256,  # Reduce from default 512MB
    enable_caching=True
)

# Strategy 3: Limit context length
result = nanoaccel.generate(
    prompt="...",
    max_new_tokens=100  # Shorter = less memory
)

# Strategy 4: Clear cache periodically
nanoaccel.reset_stats()  # Clears cache
```

### Monitor Memory Usage

```python
import psutil
import os

def print_memory_usage():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    print(f"RSS: {mem_info.rss / 1024**3:.2f} GB")
    print(f"VMS: {mem_info.vms / 1024**3:.2f} GB")

# Before loading
print_memory_usage()

# After loading
nanoaccel.load_model()
print_memory_usage()

# After generation
result = nanoaccel.generate(...)
print_memory_usage()

# After cleanup
nanoaccel.cleanup()
print_memory_usage()
```

### Memory Budget Calculator

```python
def calculate_memory_budget(
    model_params_b: float,  # Billions
    quant_bits: int = 8,
    sequence_length: int = 2048,
    batch_size: int = 1
):
    """Calculate estimated memory usage."""
    
    # Model weights
    bytes_per_param = quant_bits / 8
    model_memory_gb = model_params_b * bytes_per_param
    
    # KV cache
    kv_cache_gb = (model_params_b * 0.1 * sequence_length * 
                   batch_size * 4) / 1024**3
    
    # Activations and overhead
    overhead_gb = model_memory_gb * 0.3
    
    total_gb = model_memory_gb + kv_cache_gb + overhead_gb
    
    return {
        "model": model_memory_gb,
        "kv_cache": kv_cache_gb,
        "overhead": overhead_gb,
        "total": total_gb
    }

# Example
budget = calculate_memory_budget(
    model_params_b=1.1,  # TinyLlama
    quant_bits=4,
    sequence_length=2048
)
print(f"Estimated memory: {budget['total']:.2f} GB")
```

## CPU Optimization

### Thread Configuration

```python
import os
import torch

# Optimal thread count
num_physical_cores = os.cpu_count() // 2  # Assumes hyperthreading
torch.set_num_threads(num_physical_cores)

# MKL optimizations
os.environ['MKL_NUM_THREADS'] = str(num_physical_cores)
os.environ['OMP_NUM_THREADS'] = str(num_physical_cores)
os.environ['MKL_DYNAMIC'] = 'FALSE'

# Initialize NanoAccel with thread control
nanoaccel = NanoAccel(
    model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    num_threads=num_physical_cores,
    cpu_optimization=True
)
```

### Core Pinning

```python
import os

def pin_to_performance_cores():
    """Pin process to performance cores (P-cores)."""
    try:
        num_cores = os.cpu_count()
        # Assume first half are P-cores
        p_cores = list(range(num_cores // 2))
        os.sched_setaffinity(0, p_cores)
        print(f"Pinned to cores: {p_cores}")
    except AttributeError:
        print("Core pinning not supported on this OS")

def pin_to_efficiency_cores():
    """Pin process to efficiency cores (E-cores)."""
    try:
        num_cores = os.cpu_count()
        # Assume second half are E-cores
        e_cores = list(range(num_cores // 2, num_cores))
        os.sched_setaffinity(0, e_cores)
        print(f"Pinned to cores: {e_cores}")
    except AttributeError:
        print("Core pinning not supported on this OS")

# For main model: use P-cores
pin_to_performance_cores()
nanoaccel.load_model()

# For draft model: can use E-cores
# (Speculative decoder handles this automatically)
```

### NUMA Optimization (Multi-Socket Systems)

```python
import subprocess

def set_numa_policy():
    """Set NUMA policy for multi-socket systems."""
    try:
        # Bind to local NUMA node
        subprocess.run(['numactl', '--localalloc'], check=True)
        print("NUMA local allocation enabled")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("NUMA tools not available")

# Use before initializing NanoAccel
set_numa_policy()
nanoaccel = NanoAccel(...)
```

### CPU Feature Detection

```python
from nanoaccel.utils import detect_cpu_features

cpu_info = detect_cpu_features()

print(f"Cores: {cpu_info['cores']}")
print(f"AVX2: {cpu_info['avx2']}")
print(f"AVX512: {cpu_info['avx512']}")

# Adjust optimization based on CPU features
if cpu_info['avx512']:
    print("Excellent! AVX512 detected - optimal performance")
elif cpu_info['avx2']:
    print("Good! AVX2 detected - good performance")
else:
    print("Warning: No AVX detected - reduced performance")
    # Consider more aggressive quantization
```

## Cache Management

### Configure Cache Strategy

```python
from nanoaccel.cache_manager import KVCacheManager

# LRU (Least Recently Used) - Default
cache_manager = KVCacheManager(
    max_size_mb=512,
    eviction_strategy="lru",
    compression_enabled=True
)

# LFU (Least Frequently Used)
cache_manager = KVCacheManager(
    max_size_mb=512,
    eviction_strategy="lfu",
    compression_enabled=True
)

# Hybrid (considers both frequency and recency)
cache_manager = KVCacheManager(
    max_size_mb=512,
    eviction_strategy="hybrid",
    compression_enabled=True,
    compression_threshold_kb=100
)

nanoaccel = NanoAccel(
    model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    enable_caching=True,
    max_cache_size_mb=512
)
```

### Cache Optimization

```python
# Prefetch frequently used patterns
cache_keys = ["common_query_1", "common_query_2"]
nanoaccel.kv_cache_manager.prefetch(cache_keys)

# Compress cold entries
nanoaccel.kv_cache_manager.compress_cold_entries(
    age_threshold_seconds=60.0
)

# Optimize cache
nanoaccel.kv_cache_manager.optimize()

# Monitor cache performance
cache_stats = nanoaccel.kv_cache_manager.get_stats()
print(f"Hit rate: {cache_stats['hit_rate']:.2%}")
print(f"Compressions: {cache_stats['compressions']}")
print(f"Cache size: {cache_stats['cache_size']}")
```

### Cache Warming

```python
def warm_cache(nanoaccel, common_prompts):
    """Pre-populate cache with common queries."""
    print("Warming cache...")
    for prompt in common_prompts:
        nanoaccel.generate(
            prompt=prompt,
            max_new_tokens=10  # Just enough to populate cache
        )
    print("Cache warmed!")

common_prompts = [
    "What is",
    "How to",
    "Explain",
    "Tell me about"
]

warm_cache(nanoaccel, common_prompts)
```

## Speculative Decoding

### Optimal Configuration

```python
# Load draft model
nanoaccel.load_draft_model("EleutherAI/pythia-70m")

# Tune gamma (number of speculative tokens)
for gamma in [2, 4, 6, 8]:
    result = nanoaccel.generate(
        prompt="Test prompt",
        use_speculative=True,
        gamma=gamma,
        max_new_tokens=100
    )
    print(f"Gamma {gamma}: {result['tokens_per_second']:.2f} tok/s")

# Typical findings:
# - Gamma 4-6: Best for most cases
# - Gamma 2-3: Better for creative/diverse output
# - Gamma 6-8: Best for deterministic output
```

### Draft Model Selection

```python
# Small models (70M-160M) for fastest draft generation
draft_models = [
    "EleutherAI/pythia-70m",      # Fastest
    "EleutherAI/pythia-160m",     # Good balance
    "facebook/opt-125m",          # Alternative
]

# Benchmark draft models
for draft_model in draft_models:
    nanoaccel.load_draft_model(draft_model)
    result = nanoaccel.generate(
        prompt="Benchmark prompt",
        use_speculative=True,
        max_new_tokens=100
    )
    print(f"{draft_model}: {result['tokens_per_second']:.2f} tok/s")
```

### Acceptance Rate Tuning

```python
# Monitor acceptance rate
result = nanoaccel.generate(
    prompt="Your prompt",
    use_speculative=True,
    early_exit_threshold=0.9
)

if hasattr(nanoaccel, 'speculative_decoder'):
    stats = nanoaccel.speculative_decoder.get_stats()
    print(f"Acceptance rate: {stats['acceptance_rate']:.2%}")
    
    # Adjust threshold based on acceptance rate
    if stats['acceptance_rate'] < 0.5:
        print("Low acceptance - reduce gamma or adjust threshold")
    elif stats['acceptance_rate'] > 0.8:
        print("High acceptance - can increase gamma")
```

## Batch Processing

### Optimal Batch Size

```python
# Automatic batch size optimization
prompts = ["Prompt 1", "Prompt 2", "Prompt 3", ...]

optimal_batch_size = nanoaccel.batch_processor.optimize_batch_size(
    prompts=prompts[:10],  # Sample
    target_memory_mb=1024
)

print(f"Optimal batch size: {optimal_batch_size}")

# Use optimal batch size
nanoaccel.batch_processor.max_batch_size = optimal_batch_size
results = nanoaccel.batch_generate(prompts)
```

### Batch Processing Strategies

```python
# Strategy 1: Fixed batch processing
results = nanoaccel.batch_generate(
    prompts=prompts,
    max_new_tokens=100,
    temperature=0.7
)

# Strategy 2: Dynamic batching with grouping
from itertools import groupby

# Group by length for efficiency
prompts_by_length = sorted(
    prompts,
    key=lambda p: len(nanoaccel.tokenizer.encode(p))
)

for length, group in groupby(prompts_by_length, 
                             key=lambda p: len(nanoaccel.tokenizer.encode(p)) // 100):
    batch = list(group)
    results.extend(nanoaccel.batch_generate(batch))

# Strategy 3: Parallel processing
from concurrent.futures import ThreadPoolExecutor

def process_chunk(chunk):
    return nanoaccel.batch_generate(chunk)

with ThreadPoolExecutor(max_workers=2) as executor:
    chunk_size = 10
    chunks = [prompts[i:i+chunk_size] 
              for i in range(0, len(prompts), chunk_size)]
    results = list(executor.map(process_chunk, chunks))
```

## Production Deployment

### Best Practices

```python
import logging
from nanoaccel import NanoAccel

# 1. Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('nanoaccel_prod.log'),
        logging.StreamHandler()
    ]
)

# 2. Initialize with production settings
nanoaccel = NanoAccel(
    model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    quant_config=QuantizationConfig(
        enabled=True,
        quant_type="int4"
    ),
    mixed_precision=True,
    cpu_optimization=True,
    enable_profiling=False,  # Disable in production
    enable_caching=True,
    max_cache_size_mb=512,
    compile_model=True,
    verbose=False  # Reduce logging
)

# 3. Load and warmup
nanoaccel.load_model()
nanoaccel.load_draft_model()

# 4. Implement error handling
def safe_generate(prompt, **kwargs):
    try:
        return nanoaccel.generate(prompt, **kwargs)
    except Exception as e:
        logging.error(f"Generation failed: {e}")
        return {"text": "", "error": str(e)}

# 5. Monitor performance
import time

def monitored_generate(prompt, **kwargs):
    start = time.time()
    result = safe_generate(prompt, **kwargs)
    duration = time.time() - start
    
    logging.info(f"Generated {result.get('tokens_generated', 0)} tokens "
                f"in {duration:.2f}s")
    return result

# 6. Implement cleanup on shutdown
import atexit

def cleanup():
    logging.info("Cleaning up NanoAccel...")
    nanoaccel.cleanup()

atexit.register(cleanup)
```

### Docker Deployment

```dockerfile
# Dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Set environment variables
ENV NANOACCEL_MODEL="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
ENV NANOACCEL_QUANT_ENABLED="true"
ENV NANOACCEL_QUANT_TYPE="int4"
ENV NANOACCEL_NUM_THREADS="4"

CMD ["python", "server.py"]
```

### Monitoring and Metrics

```python
from prometheus_client import Counter, Histogram, start_http_server
import time

# Metrics
generation_counter = Counter(
    'nanoaccel_generations_total',
    'Total number of generations'
)
generation_duration = Histogram(
    'nanoaccel_generation_duration_seconds',
    'Generation duration in seconds'
)
tokens_generated = Counter(
    'nanoaccel_tokens_total',
    'Total tokens generated'
)

def instrumented_generate(prompt, **kwargs):
    generation_counter.inc()
    
    start = time.time()
    result = nanoaccel.generate(prompt, **kwargs)
    duration = time.time() - start
    
    generation_duration.observe(duration)
    tokens_generated.inc(result.get('tokens_generated', 0))
    
    return result

# Start metrics server
start_http_server(8000)
```

## Troubleshooting Common Issues

### Issue: Slow Generation

```python
# 1. Enable all optimizations
nanoaccel = NanoAccel(
    quant_config=QuantizationConfig(enabled=True, quant_type="int4"),
    mixed_precision=True,
    cpu_optimization=True,
    compile_model=True
)

# 2. Use speculative decoding
nanoaccel.load_draft_model()
result = nanoaccel.generate(..., use_speculative=True)

# 3. Check CPU usage
import psutil
print(f"CPU usage: {psutil.cpu_percent()}%")

# 4. Profile to find bottlenecks
nanoaccel.enable_profiling = True
bottlenecks = nanoaccel.profiler.get_bottlenecks()
```

### Issue: High Memory Usage

```python
# 1. Use aggressive quantization
quant_config = QuantizationConfig(
    enabled=True,
    quant_type="int4"  # or int2
)

# 2. Reduce cache size
nanoaccel = NanoAccel(max_cache_size_mb=256)

# 3. Clear cache regularly
if len(nanoaccel.kv_cache_manager.cache) > 100:
    nanoaccel.reset_stats()

# 4. Reduce context length
result = nanoaccel.generate(
    prompt=prompt[-1000:],  # Truncate prompt
    max_new_tokens=50
)
```

### Issue: Poor Quality Output

```python
# 1. Use higher precision quantization
quant_config = QuantizationConfig(
    enabled=True,
    quant_type="int8"  # Instead of int4
)

# 2. Disable mixed precision
nanoaccel = NanoAccel(mixed_precision=False)

# 3. Tune generation parameters
result = nanoaccel.generate(
    prompt=prompt,
    temperature=0.7,  # Lower for more deterministic
    top_p=0.9,
    repetition_penalty=1.2
)

# 4. Use beam search for better quality
result = nanoaccel.generate(
    prompt=prompt,
    num_beams=4,
    do_sample=False
)
```

## Advanced Tips

### 1. Context Window Management

```python
def truncate_context(text, max_tokens=1500):
    """Keep context within limits."""
    tokens = nanoaccel.tokenizer.encode(text)
    if len(tokens) > max_tokens:
        tokens = tokens[-max_tokens:]
        text = nanoaccel.tokenizer.decode(tokens)
    return text

# Use before generation
prompt = truncate_context(long_context + user_query)
```

### 2. Output Validation

```python
def validate_output(text, min_length=10, max_length=500):
    """Validate generated text."""
    if len(text) < min_length:
        return False, "Output too short"
    if len(text) > max_length:
        return False, "Output too long"
    if text.count('\n') > 10:
        return False, "Too many newlines"
    return True, "Valid"

result = nanoaccel.generate(...)
is_valid, message = validate_output(result["text"])
```

### 3. Retry Logic

```python
def generate_with_retry(prompt, max_retries=3):
    """Generate with automatic retry on failure."""
    for attempt in range(max_retries):
        try:
            result = nanoaccel.generate(prompt)
            if result.get("tokens_generated", 0) > 0:
                return result
        except Exception as e:
            logging.warning(f"Attempt {attempt + 1} failed: {e}")
            time.sleep(1)
    raise Exception("All retries failed")
```

---

**For more optimization strategies, see the main [README.md](README.md) and join our community discussions!**
