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
        
        # Decompress if needed
        if entry.compressed:
            self._decompress_entry(entry)
        
        return entry.value.clone()
    
    def _decompress_entry(self, entry: CacheEntry):
        """Decompress a cache entry."""
        if not entry.compressed or not entry.compressed_data:
            return
        
        try:
            decompressed = zlib.decompress(entry.compressed_data)
            tensor_data = pickle.loads(decompressed)
            entry.value = tensor_data
            entry.compressed = False
            entry.compressed_data = None
            self.stats["decompressions"] += 1
        except Exception as e:
            logger.error(f"Error decompressing cache entry: {e}")
    
    def _compress_entry(self, entry: CacheEntry):
        """Compress a cache entry to save memory."""
        if entry.compressed or entry.original_size < self.compression_threshold_bytes:
            return
        
        try:
            tensor_bytes = pickle.dumps(entry.value)
            compressed = zlib.compress(tensor_bytes, level=self.compression_level)
            
            # Only keep compression if it saves space
            if len(compressed) < entry.original_size * 0.8:
                entry.compressed_data = compressed
                entry.compressed = True
                entry.value = None
                self.stats["compressions"] += 1
        except Exception as e:
            logger.error(f"Error compressing cache entry: {e}")
    
    def _evict_one(self):
        """Evict one entry based on strategy."""
        if not self.cache:
            return
        
        if self.eviction_strategy == "lru":
            # Remove least recently used
            key, entry = self.cache.popitem(last=False)
        elif self.eviction_strategy == "lfu":
            # Remove least frequently used
            min_key = min(self.cache.keys(), 
                         key=lambda k: self.cache[k].access_count)
            entry = self.cache.pop(min_key)
        else:  # hybrid
            # Hybrid: Consider both frequency and recency
            now = time.time()
            min_key = min(self.cache.keys(),
                         key=lambda k: (self.cache[k].access_count / 
                                      (now - self.cache[k].timestamp + 1)))
            entry = self.cache.pop(min_key)
        
        self.current_size_bytes -= entry.original_size
        self.stats["evictions"] += 1
    
    def compress_cold_entries(self, age_threshold_seconds: float = 60.0):
        """Compress entries that haven't been accessed recently."""
        if not self.compression_enabled:
            return
        
        now = time.time()
        for entry in self.cache.values():
            if not entry.compressed and (now - entry.timestamp) > age_threshold_seconds:
                self._compress_entry(entry)
    
    def prefetch(self, keys: List[str]):
        """Prefetch and decompress entries for faster access."""
        for key in keys:
            if key in self.cache:
                entry = self.cache[key]
                if entry.compressed:
                    self._decompress_entry(entry)
    
    def clear(self):
        """Clear all cache entries."""
        self.cache.clear()
        self.current_size_bytes = 0
        self.stats["total_size_bytes"] = 0
    
    def get_stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        stats = self.stats.copy()
        stats["cache_size"] = len(self.cache)
        stats["hit_rate"] = (
            self.stats["hits"] / (self.stats["hits"] + self.stats["misses"])
            if (self.stats["hits"] + self.stats["misses"]) > 0 else 0
        )
        return stats
    
    def optimize(self):
        """Run optimization on cache (compression, cleanup)."""
        self.compress_cold_entries()
        
        # Remove empty or corrupted entries
        to_remove = []
        for key, entry in self.cache.items():
            if entry.value is None and not entry.compressed:
                to_remove.append(key)
        
        for key in to_remove:
            self.cache.pop(key)
