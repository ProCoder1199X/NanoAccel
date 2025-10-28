# NanoAccel: CPU-Optimized LLM Accelerator for Low-End Hardware

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI version](https://badge.fury.io/py/nanoaccel.svg)](https://badge.fury.io/py/nanoaccel)

NanoAccel is a lightweight Python library designed to accelerate inference and fine-tuning of 1B-8B parameter LLMs on low-end CPUs (e.g., i3/i5 with 8-16GB RAM), without GPUs or specialized hardware. Inspired by recent research in quantization and speculative decoding, it aims for 2-3x speedups and reduced memory footprints, making LLMs accessible on budget setups.

## 🚀 Features

- **Ultra-low-bit quantization** (1-4 bit) for memory efficiency
- **Advanced speculative decoding** with adaptive gamma adjustment for optimal performance
- **CPU scheduling optimizations** with performance/efficiency core pinning
- **Memory management** with KV cache quantization and offloading
- **Mixed precision inference** for improved performance
- **Comprehensive CLI** with system requirement checking
- **Configuration management** via YAML/JSON files and environment variables
- **Compatible models**: TinyLlama, Gemma 2B, Llama 3.2 1B/3B, Pythia, and more

## 📦 Installation

### From PyPI (Recommended)
```bash
pip install nanoaccel
```

### From Source
```bash
git clone https://github.com/ProCoder1199X/NanoAccel.git
cd NanoAccel
pip install -e .
```

### Development Installation
```bash
git clone https://github.com/ProCoder1199X/NanoAccel.git
cd NanoAccel
pip install -e ".[dev]"
```

## 🎯 Quick Start

### Basic Usage
```bash
# Simple inference
nanoaccel --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --prompt "Hello, world!"

# With quantization for memory efficiency
nanoaccel --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --quant int4 --prompt "Tell me a story"

# With speculative decoding for speed
nanoaccel --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --speculative --draft-model EleutherAI/pythia-70m
```

### Python API
```python
from nanoaccel import NanoAccel, QuantizationConfig

# Initialize with quantization
quant_config = QuantizationConfig(enabled=True, quant_type="int4")
nanoaccel = NanoAccel(
    model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    quant_config=quant_config,
    mixed_precision=True
)

# Load model
nanoaccel.load_model()

# Generate text
result = nanoaccel.generate(
    prompt="Write a short story about a robot",
    max_new_tokens=100,
    temperature=0.8
)

print(result["text"])
```

### System Requirements Check
```bash
# Check if your system can run a specific model
nanoaccel --check-requirements --model TinyLlama/TinyLlama-1.1B-Chat-v1.0

# Display CPU information
nanoaccel --cpu-info
```

## ⚙️ Configuration

### Configuration File
Create a `nanoaccel.yaml` file:

```yaml
model:
  default_model: "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
  default_draft_model: "EleutherAI/pythia-70m"

quantization:
  enabled: true
  quant_type: "int4"
  compute_dtype: "float32"

generation:
  max_tokens: 100
  temperature: 0.8
  top_p: 0.9

speculative_decoding:
  enabled: true
  gamma: 4
  early_exit_threshold: 0.9
  adaptive_gamma: true
  gamma_min: 1
  gamma_max: 8
  adaptation_window: 10

system:
  cpu_optimization: true
  mixed_precision: true
```

### Environment Variables
```bash
export NANOACCEL_MODEL="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
export NANOACCEL_QUANT_ENABLED="true"
export NANOACCEL_QUANT_TYPE="int4"
export NANOACCEL_SPECULATIVE="true"
```

## 🔧 Advanced Usage

### Custom Quantization
```python
from nanoaccel import QuantizationConfig

# Custom quantization configuration
quant_config = QuantizationConfig(
    enabled=True,
    quant_type="int8",
    compute_dtype=torch.bfloat16,
    chunk_size=2048
)
```

### Speculative Decoding
```python
# Load draft model for speculative decoding
nanoaccel.load_draft_model("EleutherAI/pythia-70m")

# Generate with speculative decoding
result = nanoaccel.generate(
    prompt="Your prompt here",
    use_speculative=True,
    gamma=6,  # Number of speculative tokens
    early_exit_threshold=0.95
)

# Advanced: Use adaptive gamma adjustment
from nanoaccel.speculative import SpeculativeDecoding

spec_decoder = SpeculativeDecoding(
    draft_model_name="EleutherAI/pythia-70m",
    gamma=4,
    adaptive_gamma=True,  # Enable adaptive adjustment
    gamma_min=1,
    gamma_max=8,
    adaptation_window=10
)

# Monitor adaptive statistics
adaptive_stats = spec_decoder.get_adaptive_stats()
print(f"Current gamma: {adaptive_stats['current_gamma']}")
print(f"Recent acceptance rates: {adaptive_stats['recent_acceptance_rates']}")
```

### Performance Monitoring
```python
# Get inference statistics
stats = nanoaccel.get_stats()
print(f"Average tokens/sec: {stats['average_tokens_per_second']:.2f}")
print(f"Total tokens generated: {stats['total_tokens']}")

# Reset statistics
nanoaccel.reset_stats()
```

## 📊 Performance

NanoAccel is optimized for low-end hardware with the following target performance:

- **Memory Usage**: 2-4x reduction with quantization
- **Speed**: 2-3x improvement with speculative decoding
- **Compatibility**: Works on CPUs without AVX2 (with reduced performance)
- **Minimum Requirements**: 8GB RAM, 2 CPU cores

### Benchmark Results (TinyLlama-1.1B)
| Configuration | Memory Usage | Tokens/sec | Speedup |
|---------------|--------------|------------|---------|
| Baseline (FP32) | 4.4 GB | 12.3 | 1.0x |
| INT8 Quantized | 2.2 GB | 15.7 | 1.3x |
| INT4 Quantized | 1.1 GB | 18.2 | 1.5x |
| Speculative (INT4) | 1.1 GB | 28.5 | 2.3x |

*Results on Intel i5-8400 (6 cores, 16GB RAM)*

## 🧪 Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=nanoaccel --cov-report=html

# Run specific test categories
pytest -m "not slow"  # Skip slow tests
pytest -m integration  # Only integration tests
```

## 🤝 Contributing

Contributions are welcome! Please read our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup
```bash
# Clone the repository
git clone https://github.com/ProCoder1199X/NanoAccel.git
cd NanoAccel

# Install development dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Run tests
pytest
```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE.txt](LICENSE.txt) file for details.

## 🙏 Acknowledgments

- Hugging Face for the Transformers library
- BitsAndBytes for quantization support
- The open-source AI community for research and inspiration

## 📚 Documentation

For detailed documentation, examples, and API reference, visit our [documentation](https://github.com/ProCoder1199X/NanoAccel#readme).

## 🐛 Issues and Support

- [Report bugs](https://github.com/ProCoder1199X/NanoAccel/issues)
- [Request features](https://github.com/ProCoder1199X/NanoAccel/issues)
- [Join discussions](https://github.com/ProCoder1199X/NanoAccel/discussions)

---

**Note**: This is an early-stage project. Performance may vary based on hardware configuration and model choice. Contributions and feedback are highly appreciated!