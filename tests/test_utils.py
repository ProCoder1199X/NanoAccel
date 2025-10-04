"""
Tests for utility functions.
"""

import pytest
import torch
from unittest.mock import patch, mock_open
import platform
import os

from nanoaccel.utils import (
    detect_cpu_features,
    optimize_cpu_scheduling,
    get_memory_info,
    estimate_model_memory_usage,
    check_system_requirements,
    quantize_kv_cache,
    dequantize_kv_cache
)


class TestDetectCPFFeatures:
    """Test CPU feature detection."""
    
    def test_detect_cpu_features_basic(self):
        """Test basic CPU feature detection."""
        cpu_info = detect_cpu_features()
        
        assert isinstance(cpu_info, dict)
        assert "cores" in cpu_info
        assert "avx2" in cpu_info
        assert "avx512" in cpu_info
        assert "sse4" in cpu_info
        assert isinstance(cpu_info["cores"], int)
        assert cpu_info["cores"] > 0
        assert isinstance(cpu_info["avx2"], bool)
        assert isinstance(cpu_info["avx512"], bool)
        assert isinstance(cpu_info["sse4"], bool)
    
    @patch('platform.system')
    @patch('builtins.open', mock_open(read_data="flags : avx2 avx512 sse4"))
    def test_detect_cpu_features_linux(self, mock_platform):
        """Test CPU feature detection on Linux."""
        mock_platform.return_value = "Linux"
        
        cpu_info = detect_cpu_features()
        
        assert cpu_info["avx2"] is True
        assert cpu_info["avx512"] is True
        assert cpu_info["sse4"] is True
    
    @patch('platform.system')
    def test_detect_cpu_features_windows(self, mock_platform):
        """Test CPU feature detection on Windows."""
        mock_platform.return_value = "Windows"
        
        with patch('subprocess.run') as mock_run:
            mock_run.return_value.returncode = 0
            mock_run.return_value.stdout = "Intel(R) Core(TM) i5-8400 CPU @ 2.80GHz"
            
            cpu_info = detect_cpu_features()
            
            assert isinstance(cpu_info, dict)
            assert "cores" in cpu_info


class TestOptimizeCPUScheduling:
    """Test CPU scheduling optimization."""
    
    @patch('os.sched_setaffinity')
    def test_optimize_cpu_scheduling_success(self, mock_setaffinity):
        """Test successful CPU scheduling optimization."""
        cpu_info = {"cores": 8}
        
        result = optimize_cpu_scheduling(cpu_info, use_performance_cores=True)
        
        assert result is True
        mock_setaffinity.assert_called_once_with(0, [0, 1, 2, 3])
    
    @patch('os.sched_setaffinity')
    def test_optimize_cpu_scheduling_efficiency_cores(self, mock_setaffinity):
        """Test CPU scheduling with efficiency cores."""
        cpu_info = {"cores": 8}
        
        result = optimize_cpu_scheduling(cpu_info, use_performance_cores=False)
        
        assert result is True
        mock_setaffinity.assert_called_once_with(0, [4, 5, 6, 7])
    
    @patch('os.sched_setaffinity')
    def test_optimize_cpu_scheduling_failure(self, mock_setaffinity):
        """Test CPU scheduling optimization failure."""
        mock_setaffinity.side_effect = OSError("Permission denied")
        cpu_info = {"cores": 8}
        
        result = optimize_cpu_scheduling(cpu_info)
        
        assert result is False


class TestGetMemoryInfo:
    """Test memory information retrieval."""
    
    @patch('psutil.virtual_memory')
    def test_get_memory_info_success(self, mock_virtual_memory):
        """Test successful memory info retrieval."""
        mock_memory = type('MockMemory', (), {
            'total': 16 * 1024**3,  # 16GB
            'available': 8 * 1024**3,  # 8GB
            'used': 6 * 1024**3,  # 6GB
            'free': 2 * 1024**3  # 2GB
        })()
        mock_virtual_memory.return_value = mock_memory
        
        memory_info = get_memory_info()
        
        assert memory_info["total"] == 16 * 1024**3
        assert memory_info["available"] == 8 * 1024**3
        assert memory_info["used"] == 6 * 1024**3
        assert memory_info["free"] == 2 * 1024**3
    
    @patch('psutil.virtual_memory')
    def test_get_memory_info_failure(self, mock_virtual_memory):
        """Test memory info retrieval failure."""
        mock_virtual_memory.side_effect = Exception("Access denied")
        
        memory_info = get_memory_info()
        
        assert memory_info["total"] == 0
        assert memory_info["available"] == 0
        assert memory_info["used"] == 0
        assert memory_info["free"] == 0


class TestEstimateModelMemoryUsage:
    """Test model memory usage estimation."""
    
    def test_estimate_memory_usage_tinyllama(self):
        """Test memory usage estimation for TinyLlama."""
        memory_usage = estimate_model_memory_usage("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        
        assert memory_usage > 0
        # Should be reasonable for 1.1B parameters
        assert memory_usage < 10 * 1024**3  # Less than 10GB
    
    def test_estimate_memory_usage_quantized(self):
        """Test memory usage estimation with quantization."""
        memory_usage_int8 = estimate_model_memory_usage(
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0", 
            quant_config="int8"
        )
        memory_usage_fp32 = estimate_model_memory_usage(
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0", 
            quant_config=None
        )
        
        assert memory_usage_int8 < memory_usage_fp32
    
    def test_estimate_memory_usage_sequence_length(self):
        """Test memory usage estimation with different sequence lengths."""
        memory_usage_short = estimate_model_memory_usage(
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            sequence_length=512
        )
        memory_usage_long = estimate_model_memory_usage(
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            sequence_length=2048
        )
        
        assert memory_usage_long > memory_usage_short


class TestCheckSystemRequirements:
    """Test system requirements checking."""
    
    @patch('nanoaccel.utils.get_memory_info')
    @patch('nanoaccel.utils.detect_cpu_features')
    def test_check_system_requirements_sufficient(self, mock_cpu_info, mock_memory_info):
        """Test system requirements check with sufficient resources."""
        mock_cpu_info.return_value = {"cores": 8, "avx2": True}
        mock_memory_info.return_value = {
            "available": 16 * 1024**3  # 16GB available
        }
        
        meets_requirements, message = check_system_requirements("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        
        assert meets_requirements is True
        assert "meets all requirements" in message or "meets minimum requirements" in message
    
    @patch('nanoaccel.utils.get_memory_info')
    @patch('nanoaccel.utils.detect_cpu_features')
    def test_check_system_requirements_insufficient_memory(self, mock_cpu_info, mock_memory_info):
        """Test system requirements check with insufficient memory."""
        mock_cpu_info.return_value = {"cores": 8, "avx2": True}
        mock_memory_info.return_value = {
            "available": 1 * 1024**3  # 1GB available
        }
        
        meets_requirements, message = check_system_requirements("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        
        assert meets_requirements is False
        assert "Insufficient memory" in message
    
    @patch('nanoaccel.utils.get_memory_info')
    @patch('nanoaccel.utils.detect_cpu_features')
    def test_check_system_requirements_insufficient_cores(self, mock_cpu_info, mock_memory_info):
        """Test system requirements check with insufficient CPU cores."""
        mock_cpu_info.return_value = {"cores": 1, "avx2": True}
        mock_memory_info.return_value = {
            "available": 16 * 1024**3  # 16GB available
        }
        
        meets_requirements, message = check_system_requirements("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        
        assert meets_requirements is False
        assert "Insufficient CPU cores" in message


class TestQuantizeKVCache:
    """Test KV cache quantization."""
    
    def test_quantize_kv_cache_int8(self):
        """Test INT8 quantization of KV cache."""
        kv_cache = {
            "key": torch.randn(100, 512),
            "value": torch.randn(100, 512)
        }
        
        quantized = quantize_kv_cache(kv_cache, "int8")
        
        assert "key" in quantized
        assert "value" in quantized
        assert quantized["key"].dtype == torch.uint8
        assert quantized["value"].dtype == torch.uint8
    
    def test_quantize_kv_cache_unsupported_level(self):
        """Test quantization with unsupported level."""
        kv_cache = {"key": torch.randn(10, 10)}
        
        quantized = quantize_kv_cache(kv_cache, "unsupported")
        
        # Should return original cache unchanged
        assert quantized == kv_cache
    
    def test_quantize_kv_cache_large_tensor(self):
        """Test quantization of large tensors with chunking."""
        kv_cache = {"key": torch.randn(2048, 512)}
        
        quantized = quantize_kv_cache(kv_cache, "int8", chunk_size=1024)
        
        assert "key" in quantized
        assert quantized["key"].dtype == torch.uint8


class TestDequantizeKVCache:
    """Test KV cache dequantization."""
    
    def test_dequantize_kv_cache(self):
        """Test dequantization of KV cache."""
        quantized_kv_cache = {
            "key": torch.randint(0, 255, (10, 10), dtype=torch.uint8),
            "value": torch.randint(0, 255, (10, 10), dtype=torch.uint8)
        }
        
        dequantized = dequantize_kv_cache(quantized_kv_cache, scale=0.1, zero_point=128)
        
        assert "key" in dequantized
        assert "value" in dequantized
        assert dequantized["key"].dtype == torch.float32
        assert dequantized["value"].dtype == torch.float32
    
    def test_dequantize_kv_cache_mixed_types(self):
        """Test dequantization with mixed tensor types."""
        quantized_kv_cache = {
            "key": torch.randint(0, 255, (10, 10), dtype=torch.uint8),
            "value": torch.randn(10, 10)  # Already float32
        }
        
        dequantized = dequantize_kv_cache(quantized_kv_cache)
        
        assert dequantized["key"].dtype == torch.float32
        assert dequantized["value"].dtype == torch.float32
