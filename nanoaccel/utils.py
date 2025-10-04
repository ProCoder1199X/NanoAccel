"""
Utility functions for NanoAccel.
"""

import os
import platform
import psutil
import logging
from typing import Dict, List, Optional, Tuple
import torch

logger = logging.getLogger(__name__)


def detect_cpu_features() -> Dict[str, any]:
    """
    Detect CPU capabilities and features for optimization.
    
    Returns:
        Dictionary containing CPU information including:
        - cores: Number of CPU cores
        - avx2: Whether AVX2 is supported
        - avx512: Whether AVX512 is supported
        - sse4: Whether SSE4 is supported
        - frequency: CPU frequency information
        - cache_size: CPU cache size information
    """
    cpu_info = {
        "cores": os.cpu_count(),
        "avx2": False,
        "avx512": False,
        "sse4": False,
        "frequency": {},
        "cache_size": {}
    }
    
    try:
        # Get CPU frequency information
        cpu_freq = psutil.cpu_freq()
        if cpu_freq:
            cpu_info["frequency"] = {
                "current": cpu_freq.current,
                "min": cpu_freq.min,
                "max": cpu_freq.max
            }
        
        # Get CPU cache information
        try:
            cache_info = psutil.cpu_cache()
            if cache_info:
                cpu_info["cache_size"] = {
                    "L1": cache_info[0] if len(cache_info) > 0 else None,
                    "L2": cache_info[1] if len(cache_info) > 1 else None,
                    "L3": cache_info[2] if len(cache_info) > 2 else None
                }
        except (AttributeError, IndexError):
            pass
        
        # Detect CPU features based on platform
        if platform.system() == "Linux":
            try:
                with open("/proc/cpuinfo", "r") as f:
                    cpuinfo = f.read()
                
                cpu_info["avx2"] = "avx2" in cpuinfo.lower()
                cpu_info["avx512"] = "avx512" in cpuinfo.lower()
                cpu_info["sse4"] = "sse4" in cpuinfo.lower()
                
            except (OSError, IOError):
                logger.warning("Could not read /proc/cpuinfo")
        elif platform.system() == "Windows":
            # On Windows, we can use WMI or registry to detect CPU features
            # For now, we'll use a simpler approach
            try:
                import subprocess
                result = subprocess.run(
                    ["wmic", "cpu", "get", "name"], 
                    capture_output=True, 
                    text=True, 
                    timeout=5
                )
                if result.returncode == 0:
                    cpu_name = result.stdout.lower()
                    cpu_info["avx2"] = "avx2" in cpu_name or "haswell" in cpu_name or "broadwell" in cpu_name
                    cpu_info["avx512"] = "avx512" in cpu_name or "skylake" in cpu_name
                    cpu_info["sse4"] = "sse4" in cpu_name or "nehalem" in cpu_name
            except (subprocess.TimeoutExpired, FileNotFoundError):
                logger.warning("Could not detect CPU features on Windows")
        elif platform.system() == "Darwin":  # macOS
            try:
                import subprocess
                result = subprocess.run(
                    ["sysctl", "-n", "machdep.cpu.features"], 
                    capture_output=True, 
                    text=True, 
                    timeout=5
                )
                if result.returncode == 0:
                    features = result.stdout.lower()
                    cpu_info["avx2"] = "avx2" in features
                    cpu_info["avx512"] = "avx512" in features
                    cpu_info["sse4"] = "sse4" in features
            except (subprocess.TimeoutExpired, FileNotFoundError):
                logger.warning("Could not detect CPU features on macOS")
                
    except Exception as e:
        logger.warning(f"Error detecting CPU features: {e}")
    
    return cpu_info


def optimize_cpu_scheduling(cpu_info: Dict[str, any], use_performance_cores: bool = True) -> bool:
    """
    Optimize CPU scheduling by pinning processes to specific cores.
    
    Args:
        cpu_info: CPU information dictionary from detect_cpu_features()
        use_performance_cores: Whether to use performance cores (first half)
        
    Returns:
        True if optimization was successful, False otherwise
    """
    try:
        num_cores = cpu_info["cores"]
        if num_cores is None or num_cores <= 0:
            logger.warning("Invalid number of CPU cores")
            return False
        
        if use_performance_cores:
            # Use first half of cores (typically performance cores on hybrid CPUs)
            target_cores = list(range(num_cores // 2))
        else:
            # Use second half of cores (typically efficiency cores on hybrid CPUs)
            target_cores = list(range(num_cores // 2, num_cores))
        
        # Set CPU affinity
        os.sched_setaffinity(0, target_cores)
        logger.info(f"Set CPU affinity to cores: {target_cores}")
        return True
        
    except (OSError, AttributeError) as e:
        logger.warning(f"Could not set CPU affinity: {e}")
        return False


def get_memory_info() -> Dict[str, int]:
    """
    Get system memory information.
    
    Returns:
        Dictionary containing memory information in bytes:
        - total: Total system memory
        - available: Available memory
        - used: Used memory
        - free: Free memory
    """
    try:
        memory = psutil.virtual_memory()
        return {
            "total": memory.total,
            "available": memory.available,
            "used": memory.used,
            "free": memory.free
        }
    except Exception as e:
        logger.warning(f"Error getting memory info: {e}")
        return {
            "total": 0,
            "available": 0,
            "used": 0,
            "free": 0
        }


def estimate_model_memory_usage(
    model_name: str,
    quant_config: Optional[str] = None,
    sequence_length: int = 2048
) -> int:
    """
    Estimate memory usage for a model.
    
    Args:
        model_name: HuggingFace model name
        quant_config: Quantization configuration
        sequence_length: Maximum sequence length
        
    Returns:
        Estimated memory usage in bytes
    """
    # Rough estimates based on model size and configuration
    model_sizes = {
        "tinyllama": 1.1e9,  # 1.1B parameters
        "pythia-70m": 70e6,  # 70M parameters
        "pythia-160m": 160e6,  # 160M parameters
        "gemma-2b": 2e9,  # 2B parameters
        "llama-3.2-1b": 1e9,  # 1B parameters
        "llama-3.2-3b": 3e9,  # 3B parameters
    }
    
    # Find matching model size
    param_count = 1e9  # Default to 1B parameters
    for key, size in model_sizes.items():
        if key in model_name.lower():
            param_count = size
            break
    
    # Estimate memory per parameter based on data type and quantization
    if quant_config == "int8":
        bytes_per_param = 1
    elif quant_config == "int4":
        bytes_per_param = 0.5
    elif quant_config == "int2":
        bytes_per_param = 0.25
    else:  # float32
        bytes_per_param = 4
    
    # Base model memory
    model_memory = param_count * bytes_per_param
    
    # Add KV cache memory (rough estimate)
    kv_cache_memory = param_count * 2 * sequence_length * 4  # 2 * seq_len * 4 bytes per param
    
    # Add overhead (activations, gradients, etc.)
    overhead = model_memory * 0.5
    
    total_memory = model_memory + kv_cache_memory + overhead
    return int(total_memory)


def check_system_requirements(
    model_name: str,
    quant_config: Optional[str] = None,
    sequence_length: int = 2048
) -> Tuple[bool, str]:
    """
    Check if the system meets requirements for running a model.
    
    Args:
        model_name: HuggingFace model name
        quant_config: Quantization configuration
        sequence_length: Maximum sequence length
        
    Returns:
        Tuple of (meets_requirements, message)
    """
    memory_info = get_memory_info()
    cpu_info = detect_cpu_features()
    
    # Check available memory
    required_memory = estimate_model_memory_usage(model_name, quant_config, sequence_length)
    available_memory = memory_info["available"]
    
    if available_memory < required_memory:
        return False, f"Insufficient memory. Required: {required_memory / 1e9:.1f}GB, Available: {available_memory / 1e9:.1f}GB"
    
    # Check CPU cores
    if cpu_info["cores"] < 2:
        return False, "Insufficient CPU cores. At least 2 cores recommended."
    
    # Check for recommended features
    warnings = []
    if not cpu_info["avx2"]:
        warnings.append("AVX2 not detected - performance may be suboptimal")
    
    if cpu_info["cores"] < 4:
        warnings.append("Less than 4 CPU cores - performance may be limited")
    
    if warnings:
        return True, "System meets minimum requirements. Warnings: " + "; ".join(warnings)
    
    return True, "System meets all requirements for optimal performance"


def quantize_kv_cache(
    kv_cache: Dict[str, torch.Tensor], 
    quant_level: str = "int8", 
    chunk_size: int = 1024
) -> Dict[str, torch.Tensor]:
    """
    Quantize key-value cache for memory efficiency.
    
    Args:
        kv_cache: Dictionary containing key-value cache tensors
        quant_level: Quantization level ("int8", "int4", "int2")
        chunk_size: Chunk size for chunk-based quantization
        
    Returns:
        Quantized key-value cache
    """
    if quant_level not in ["int8", "int4", "int2"]:
        logger.warning(f"Unsupported quantization level: {quant_level}")
        return kv_cache
    
    quantized = {}
    
    try:
        for key, value in kv_cache.items():
            if quant_level == "int8":
                # Simple int8 quantization
                if value.numel() > chunk_size:
                    chunks = torch.split(value, chunk_size, dim=0)
                    quantized_chunks = []
                    for chunk in chunks:
                        # Min-max quantization to int8
                        chunk_min = chunk.min()
                        chunk_max = chunk.max()
                        scale = (chunk_max - chunk_min) / 255.0
                        zero_point = -chunk_min / scale
                        quantized_chunk = torch.round(chunk / scale + zero_point).clamp(0, 255).to(torch.uint8)
                        quantized_chunks.append(quantized_chunk)
                    quantized[key] = torch.cat(quantized_chunks)
                else:
                    # Direct quantization for small tensors
                    value_min = value.min()
                    value_max = value.max()
                    scale = (value_max - value_min) / 255.0
                    zero_point = -value_min / scale
                    quantized[key] = torch.round(value / scale + zero_point).clamp(0, 255).to(torch.uint8)
            else:
                # For int4 and int2, use more sophisticated quantization
                quantized[key] = value  # Placeholder - implement proper quantization
        
        logger.info(f"Quantized KV cache using {quant_level} quantization")
        
    except Exception as e:
        logger.error(f"Error quantizing KV cache: {e}")
        return kv_cache
    
    return quantized


def dequantize_kv_cache(
    quantized_kv_cache: Dict[str, torch.Tensor],
    scale: float = 1.0,
    zero_point: float = 0.0
) -> Dict[str, torch.Tensor]:
    """
    Dequantize key-value cache back to float32.
    
    Args:
        quantized_kv_cache: Quantized key-value cache
        scale: Quantization scale factor
        zero_point: Quantization zero point
        
    Returns:
        Dequantized key-value cache
    """
    dequantized = {}
    
    try:
        for key, value in quantized_kv_cache.items():
            if value.dtype == torch.uint8:
                # Convert back to float32
                dequantized[key] = (value.float() - zero_point) * scale
            else:
                dequantized[key] = value
        
        logger.info("Dequantized KV cache")
        
    except Exception as e:
        logger.error(f"Error dequantizing KV cache: {e}")
        return quantized_kv_cache
    
    return dequantized
