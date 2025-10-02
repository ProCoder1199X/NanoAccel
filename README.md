# NanoAccel: CPU-Optimized LLM Accelerator for Low-End Hardware

NanoAccel is a lightweight Python library designed to accelerate inference and fine-tuning of 1B-8B parameter LLMs on low-end CPUs (e.g., i3/i5 with 8-16GB RAM), without GPUs or specialized hardware. Inspired by recent research (e.g., quantization, speculative decoding), it aims for 2-3x speedups and reduced memory footprints, making LLMs accessible on budget setups.

## Features (Planned)
- Ultra-low-bit quantization (1-4 bit) for memory efficiency.
- Speculative decoding for faster token generation.
- Memory offloading and CPU scheduling optimizations.
- PEFT-based fine-tuning for small datasets.
- Compatible with models like Gemma 2B, Llama 3.2 1B/3B.

## Setup
1. Clone the repo: `git clone https://github.com/ProCoder1199X/NanoAccel.git`
2. Install dependencies: `pip install -r requirements.txt` (coming soon).
3. Run example: `python src/nanoaccel.py --model gemma-2b --prompt "Hello world"`

## Development Status
Early prototype. Contributions welcome!

## License
MIT License