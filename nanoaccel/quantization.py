"""
Quantization utilities for NanoAccel.
"""

from typing import Optional, Dict, Any
import torch
from transformers import BitsAndBytesConfig


class QuantizationConfig:
    """
    Configuration class for model quantization.
    
    Supports various quantization schemes optimized for CPU inference.
    """
    
    def __init__(
        self,
        enabled: bool = False,
        quant_type: str = "int8",
        compute_dtype: torch.dtype = torch.float32,
        chunk_size: int = 1024
    ):
        """
        Initialize quantization configuration.
        
        Args:
            enabled: Whether to enable quantization
            quant_type: Type of quantization ("int2", "int4", "int8")
            compute_dtype: Data type for computations
            chunk_size: Chunk size for chunk-based quantization
        """
        self.enabled = enabled
        self.quant_type = quant_type
        self.compute_dtype = compute_dtype
        self.chunk_size = chunk_size
        
        # Validate quantization type
        if enabled and quant_type not in ["int2", "int4", "int8"]:
            raise ValueError(f"Unsupported quantization type: {quant_type}")
    
    def get_bitsandbytes_config(self, dtype: torch.dtype) -> BitsAndBytesConfig:
        """
        Get BitsAndBytesConfig for the specified quantization settings.
        
        Args:
            dtype: Computation data type
            
        Returns:
            BitsAndBytesConfig instance
        """
        if not self.enabled:
            raise RuntimeError("Quantization is not enabled")
        
        if self.quant_type in ["int2", "int4"]:
            # Use 4-bit quantization as approximation for 2-bit
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_use_double_quant=True
            )
        else:
            raise ValueError(f"BitsAndBytesConfig not supported for {self.quant_type}")
    
    def get_torch_quantization_config(self) -> Dict[str, Any]:
        """
        Get PyTorch quantization configuration.
        
        Returns:
            Dictionary with quantization configuration
        """
        if not self.enabled:
            raise RuntimeError("Quantization is not enabled")
        
        if self.quant_type == "int8":
            return {
                "dtype": torch.qint8,
                "scheme": "dynamic"
            }
        else:
            raise ValueError(f"Torch quantization not supported for {self.quant_type}")
    
    def quantize_kv_cache(
        self, 
        kv_cache: Dict[str, torch.Tensor], 
        quant_level: Optional[str] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Quantize key-value cache for memory efficiency.
        
        Args:
            kv_cache: Dictionary containing key-value cache tensors
            quant_level: Override quantization level
            
        Returns:
            Quantized key-value cache
        """
        if not self.enabled:
            return kv_cache
        
        quant_level = quant_level or self.quant_type
        
        if quant_level == "int8":
            return self._quantize_int8_kv_cache(kv_cache)
        elif quant_level in ["int4", "int2"]:
            # For lower bit quantization, we use chunk-based approach
            return self._quantize_chunked_kv_cache(kv_cache, quant_level)
        else:
            return kv_cache
    
    def _quantize_int8_kv_cache(self, kv_cache: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Quantize KV cache to int8."""
        quantized = {}
        for key, value in kv_cache.items():
            if value.dtype == torch.float32:
                # Simple min-max quantization to int8
                value_min = value.min()
                value_max = value.max()
                scale = (value_max - value_min) / 255.0
                zero_point = -value_min / scale
                
                quantized_value = torch.round(value / scale + zero_point).clamp(0, 255).to(torch.uint8)
                quantized[key] = {
                    'data': quantized_value,
                    'scale': scale,
                    'zero_point': zero_point,
                    'min': value_min,
                    'max': value_max
                }
            else:
                quantized[key] = value
        
        return quantized
    
    def _quantize_chunked_kv_cache(
        self, 
        kv_cache: Dict[str, torch.Tensor], 
        quant_level: str
    ) -> Dict[str, torch.Tensor]:
        """Quantize KV cache using chunk-based approach."""
        quantized = {}
        for key, value in kv_cache.items():
            if value.numel() > self.chunk_size:
                # Chunk-based quantization
                chunks = torch.split(value, self.chunk_size, dim=0)
                quantized_chunks = []
                
                for chunk in chunks:
                    if quant_level == "int4":
                        # 4-bit quantization
                        quantized_chunk = self._quantize_to_4bit(chunk)
                    else:  # int2
                        # 2-bit quantization
                        quantized_chunk = self._quantize_to_2bit(chunk)
                    
                    quantized_chunks.append(quantized_chunk)
                
                quantized[key] = torch.cat(quantized_chunks)
            else:
                # Small tensor, quantize directly
                if quant_level == "int4":
                    quantized[key] = self._quantize_to_4bit(value)
                else:
                    quantized[key] = self._quantize_to_2bit(value)
        
        return quantized
    
    def _quantize_to_4bit(self, tensor: torch.Tensor) -> torch.Tensor:
        """Quantize tensor to 4-bit representation."""
        # Simple linear quantization to 4-bit
        value_min = tensor.min()
        value_max = tensor.max()
        scale = (value_max - value_min) / 15.0  # 4-bit = 16 values (0-15)
        zero_point = -value_min / scale
        
        quantized = torch.round(tensor / scale + zero_point).clamp(0, 15).to(torch.uint8)
        
        return {
            'data': quantized,
            'scale': scale,
            'zero_point': zero_point,
            'min': value_min,
            'max': value_max
        }
    
    def _quantize_to_2bit(self, tensor: torch.Tensor) -> torch.Tensor:
        """Quantize tensor to 2-bit representation."""
        # Simple linear quantization to 2-bit
        value_min = tensor.min()
        value_max = tensor.max()
        scale = (value_max - value_min) / 3.0  # 2-bit = 4 values (0-3)
        zero_point = -value_min / scale
        
        quantized = torch.round(tensor / scale + zero_point).clamp(0, 3).to(torch.uint8)
        
        return {
            'data': quantized,
            'scale': scale,
            'zero_point': zero_point,
            'min': value_min,
            'max': value_max
        }
    
    def dequantize_tensor(self, quantized_data: Dict[str, Any]) -> torch.Tensor:
        """
        Dequantize a quantized tensor back to float32.
        
        Args:
            quantized_data: Dictionary containing quantized tensor data
            
        Returns:
            Dequantized tensor in float32
        """
        if isinstance(quantized_data, torch.Tensor):
            return quantized_data
        
        if 'data' in quantized_data:
            data = quantized_data['data'].float()
            scale = quantized_data['scale']
            zero_point = quantized_data['zero_point']
            
            return (data - zero_point) * scale
        else:
            raise ValueError("Invalid quantized data format")
