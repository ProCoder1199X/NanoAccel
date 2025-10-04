"""
NanoAccel: CPU-Optimized LLM Accelerator for Low-End Hardware

A lightweight Python library designed to accelerate inference and fine-tuning 
of 1B-8B parameter LLMs on low-end CPUs without GPUs or specialized hardware.
"""

__version__ = "0.1.0"
__author__ = "Dheeraj Kumar"
__email__ = "dheeraj.kumar@example.com"

from .core import NanoAccel
from .utils import detect_cpu_features, quantize_kv_cache
from .quantization import QuantizationConfig
from .speculative import SpeculativeDecoding

__all__ = [
    "NanoAccel",
    "detect_cpu_features", 
    "quantize_kv_cache",
    "QuantizationConfig",
    "SpeculativeDecoding",
]