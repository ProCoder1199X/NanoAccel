"""
Advanced KV Cache Management with compression and eviction strategies.
"""

import time
import logging
from typing import Dict, Optional, Tuple, List
from collections import OrderedDict
from dataclasses import dataclass
import torch
import zlib
import pickle

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Entry in the KV cache."""
    key: str
    value: torch.Tensor
    timestamp: float
    access_count: int
    compressed: bool = False
    compressed_data: Optional[bytes] = None
    original_size: int = 0


class KVCacheManager:
    """
    Advanced KV Cache Manager with compression and intelligent eviction.
    
    Features:
    - LRU/LFU eviction strategies
    - Compression for cold cache entries
    - Automatic memory management
    - Cache hit/miss tracking
    - Prefetching support
    """
    
    def __init__(
        self,
        max_size_mb: int = 512,
        compression_enabled: bool = True,
        compression_threshold_kb: int = 100,
        eviction_strategy: str = "lru",  # lru, lfu, or hybrid
        compression_level: int = 6
    ):
        """
        Initialize KV Cache Manager.
        
        Args:
            max_size_mb: Maximum cache size in MB
            compression_enabled: Enable compression for cold entries
            compression_threshold_kb: Compress entries larger than this
            eviction_strategy: Cache eviction strategy
            compression_level: Compression level (1-9)
        """
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.compression_enabled = compression_enabled
        self.compression_threshold_bytes = compression_threshold_kb * 1024
        self.eviction_strategy = eviction_strategy
        self.compression_level = compression_level
        
        # Cache storage
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.current_size_bytes = 0
        
        # Statistics
        self.stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "compressions": 0,
            "decompressions": 0,
            "total_size_bytes": 0
        }
    
    def put(self, key: str, value: torch.Tensor) -> bool:
        """
        Add or update cache entry.
        
        Args:
            key: Cache key
            value: Tensor to cache
            
        Returns:
            True if successfully cached
        """
        try:
            # Calculate entry size
            entry_size = value.element_size() * value.nelement()
            
            # Check if we need to evict entries
            while self.current_size_bytes + entry_size > self.max_size_bytes and self.cache:
                self._evict_one()
            
            # Create cache entry
            entry = CacheEntry(
                key=key,
                value=value.clone(),
                timestamp=time.time(),
                access_count=1,
                original_size=entry_size
            )
            
            # Add to cache
            if key in self.cache:
                old_entry = self.cache[key]
                self.current_size_bytes -= old_entry.original_size
            
            self.cache[key] = entry
            self.cache.move_to_end(key)
            self.current_size_bytes += entry_size
            self.stats["total_size_bytes"] = self.current_size_bytes
            
            return True
            
        except Exception as e:
            logger.error(f"Error adding to cache: {e}")
            return False
    
    def get(self, key: str) -> Optional[torch.Tensor]:
        """
        Get cached tensor.
        
        Args:
            key: Cache key
            
        Returns:
            Cached tensor or None if not found
        """
        if key not in self.cache:
            self.stats["misses"] += 1
            return None
        
        self.stats["hits"] += 1
        entry = self.cache[key]
        
        # Update access info
        entry.access_count += 1
        entry.timestamp = time.time()
        self.cache.move_to_end(key)
        
        # Decompress if
