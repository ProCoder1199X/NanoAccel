"""
Tests for quantization functionality.
"""

import pytest
import torch
from unittest.mock import patch, MagicMock

from nanoaccel.quantization import QuantizationConfig


class TestQuantizationConfig:
    """Test QuantizationConfig class."""
    
    def test_init_default(self):
        """Test default initialization."""
        config = QuantizationConfig()
        
        assert config.enabled is False
        assert config.quant_type == "int8"
        assert config.compute_dtype == torch.float32
        assert config.chunk_size == 1024
    
    def test_init_custom(self):
        """Test custom initialization."""
        config = QuantizationConfig(
            enabled=True,
            quant_type="int4",
            compute_dtype=torch.bfloat16,
            chunk_size=2048
        )
        
        assert config.enabled is True
        assert config.quant_type == "int4"
        assert config.compute_dtype == torch.bfloat16
        assert config.chunk_size == 2048
    
    def test_init_invalid_quant_type(self):
        """Test initialization with invalid quantization type."""
        with pytest.raises(ValueError, match="Unsupported quantization type"):
            QuantizationConfig(enabled=True, quant_type="invalid")
    
    def test_get_bitsandbytes_config_int4(self):
        """Test BitsAndBytesConfig for INT4 quantization."""
        config = QuantizationConfig(enabled=True, quant_type="int4")
        
        bitsandbytes_config = config.get_bitsandbytes_config(torch.float32)
        
        assert bitsandbytes_config.load_in_4bit is True
        assert bitsandbytes_config.bnb_4bit_quant_type == "nf4"
        assert bitsandbytes_config.bnb_4bit_compute_dtype == torch.float32
    
    def test_get_bitsandbytes_config_int2(self):
        """Test BitsAndBytesConfig for INT2 quantization (approximated as INT4)."""
        config = QuantizationConfig(enabled=True, quant_type="int2")
        
        bitsandbytes_config = config.get_bitsandbytes_config(torch.bfloat16)
        
        assert bitsandbytes_config.load_in_4bit is True
        assert bitsandbytes_config.bnb_4bit_compute_dtype == torch.bfloat16
    
    def test_get_bitsandbytes_config_disabled(self):
        """Test BitsAndBytesConfig when quantization is disabled."""
        config = QuantizationConfig(enabled=False)
        
        with pytest.raises(RuntimeError, match="Quantization is not enabled"):
            config.get_bitsandbytes_config(torch.float32)
    
    def test_get_bitsandbytes_config_unsupported_type(self):
        """Test BitsAndBytesConfig with unsupported quantization type."""
        config = QuantizationConfig(enabled=True, quant_type="int8")
        
        with pytest.raises(ValueError, match="BitsAndBytesConfig not supported"):
            config.get_bitsandbytes_config(torch.float32)
    
    def test_get_torch_quantization_config_int8(self):
        """Test PyTorch quantization configuration for INT8."""
        config = QuantizationConfig(enabled=True, quant_type="int8")
        
        torch_config = config.get_torch_quantization_config()
        
        assert torch_config["dtype"] == torch.qint8
        assert torch_config["scheme"] == "dynamic"
    
    def test_get_torch_quantization_config_disabled(self):
        """Test PyTorch quantization configuration when disabled."""
        config = QuantizationConfig(enabled=False)
        
        with pytest.raises(RuntimeError, match="Quantization is not enabled"):
            config.get_torch_quantization_config()
    
    def test_get_torch_quantization_config_unsupported_type(self):
        """Test PyTorch quantization configuration with unsupported type."""
        config = QuantizationConfig(enabled=True, quant_type="int4")
        
        with pytest.raises(ValueError, match="Torch quantization not supported"):
            config.get_torch_quantization_config()
    
    def test_quantize_kv_cache_disabled(self):
        """Test KV cache quantization when disabled."""
        config = QuantizationConfig(enabled=False)
        kv_cache = {"key": torch.randn(10, 10)}
        
        result = config.quantize_kv_cache(kv_cache)
        
        assert result == kv_cache  # Should return unchanged
    
    def test_quantize_kv_cache_int8(self):
        """Test INT8 KV cache quantization."""
        config = QuantizationConfig(enabled=True, quant_type="int8")
        kv_cache = {
            "key": torch.randn(100, 512),
            "value": torch.randn(100, 512)
        }
        
        quantized = config.quantize_kv_cache(kv_cache)
        
        assert "key" in quantized
        assert "value" in quantized
        
        # Check that quantization metadata is present
        assert "data" in quantized["key"]
        assert "scale" in quantized["key"]
        assert "zero_point" in quantized["key"]
        assert "min" in quantized["key"]
        assert "max" in quantized["key"]
    
    def test_quantize_kv_cache_int4(self):
        """Test INT4 KV cache quantization."""
        config = QuantizationConfig(enabled=True, quant_type="int4")
        kv_cache = {"key": torch.randn(100, 512)}
        
        quantized = config.quantize_kv_cache(kv_cache)
        
        assert "key" in quantized
        assert "data" in quantized["key"]
        assert "scale" in quantized["key"]
        assert "zero_point" in quantized["key"]
    
    def test_quantize_kv_cache_int2(self):
        """Test INT2 KV cache quantization."""
        config = QuantizationConfig(enabled=True, quant_type="int2")
        kv_cache = {"key": torch.randn(100, 512)}
        
        quantized = config.quantize_kv_cache(kv_cache)
        
        assert "key" in quantized
        assert "data" in quantized["key"]
        assert "scale" in quantized["key"]
        assert "zero_point" in quantized["key"]
    
    def test_quantize_kv_cache_chunked(self):
        """Test chunked KV cache quantization."""
        config = QuantizationConfig(enabled=True, quant_type="int8", chunk_size=50)
        kv_cache = {"key": torch.randn(200, 512)}  # Large tensor
        
        quantized = config.quantize_kv_cache(kv_cache)
        
        assert "key" in quantized
        assert isinstance(quantized["key"], torch.Tensor)
        assert quantized["key"].dtype == torch.uint8
    
    def test_dequantize_tensor(self):
        """Test tensor dequantization."""
        config = QuantizationConfig()
        
        # Create quantized data
        original_tensor = torch.randn(10, 10)
        quantized_data = config._quantize_to_4bit(original_tensor)
        
        dequantized = config.dequantize_tensor(quantized_data)
        
        assert dequantized.dtype == torch.float32
        assert dequantized.shape == original_tensor.shape
        # Note: Due to quantization, values won't be exactly equal
        assert torch.allclose(dequantized, original_tensor, atol=0.1)
    
    def test_dequantize_tensor_already_float(self):
        """Test dequantization of already float tensor."""
        config = QuantizationConfig()
        float_tensor = torch.randn(10, 10)
        
        result = config.dequantize_tensor(float_tensor)
        
        assert result is float_tensor  # Should return unchanged
    
    def test_dequantize_tensor_invalid_format(self):
        """Test dequantization with invalid format."""
        config = QuantizationConfig()
        invalid_data = {"invalid": "data"}
        
        with pytest.raises(ValueError, match="Invalid quantized data format"):
            config.dequantize_tensor(invalid_data)
    
    def test_quantize_to_4bit(self):
        """Test 4-bit quantization."""
        config = QuantizationConfig()
        tensor = torch.randn(10, 10)
        
        quantized = config._quantize_to_4bit(tensor)
        
        assert isinstance(quantized, dict)
        assert "data" in quantized
        assert "scale" in quantized
        assert "zero_point" in quantized
        assert "min" in quantized
        assert "max" in quantized
        
        assert quantized["data"].dtype == torch.uint8
        assert quantized["data"].shape == tensor.shape
        assert torch.all(quantized["data"] >= 0)
        assert torch.all(quantized["data"] <= 15)  # 4-bit range
    
    def test_quantize_to_2bit(self):
        """Test 2-bit quantization."""
        config = QuantizationConfig()
        tensor = torch.randn(10, 10)
        
        quantized = config._quantize_to_2bit(tensor)
        
        assert isinstance(quantized, dict)
        assert "data" in quantized
        assert "scale" in quantized
        assert "zero_point" in quantized
        assert "min" in quantized
        assert "max" in quantized
        
        assert quantized["data"].dtype == torch.uint8
        assert quantized["data"].shape == tensor.shape
        assert torch.all(quantized["data"] >= 0)
        assert torch.all(quantized["data"] <= 3)  # 2-bit range
