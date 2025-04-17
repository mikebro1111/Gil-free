import psutil
import numpy as np
from typing import Optional
from functools import wraps
import time
from contextlib import contextmanager

def get_available_memory() -> int:
    """Returns available memory in bytes"""
    return psutil.virtual_memory().available

def estimate_array_memory(shape: tuple, dtype: np.dtype = np.float64) -> int:
    """Estimate memory usage of numpy array"""
    return np.prod(shape) * dtype.itemsize

class MemoryTracker:
    """Class for tracking memory usage"""
    
    def __init__(self):
        self.process = psutil.Process()
        self.start_memory = None
        
    def start(self):
        """Start tracking memory"""
        try:
            self.start_memory = self.process.memory_info().rss
        except (AttributeError, psutil.AccessDenied):
            # Fallback for platforms where RSS is not available
            self.start_memory = psutil.virtual_memory().used
        
    def end(self) -> int:
        """End tracking and return memory difference in bytes"""
        if self.start_memory is None:
            raise RuntimeError("Memory tracking not started")
        try:
            end_memory = self.process.memory_info().rss
        except (AttributeError, psutil.AccessDenied):
            end_memory = psutil.virtual_memory().used
        return end_memory - self.start_memory

    @contextmanager
    def track(self):
        """Context manager for tracking memory usage"""
        self.start()
        yield
        memory_used = self.end()
        return memory_used

def memory_tracker(func):
    """Decorator for tracking memory usage of a function"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        tracker = MemoryTracker()
        tracker.start()
        result = func(*args, **kwargs)
        memory_used = tracker.end()
        return result, memory_used
    return wrapper

def check_memory_limit(required_memory: int, 
                      available_fraction: float = 0.8,
                      raise_error: bool = True) -> bool:
    """
    Check if there's enough memory available
    
    Parameters:
    -----------
    required_memory : int
        Required memory in bytes
    available_fraction : float
        Maximum fraction of available memory to use
    raise_error : bool
        Whether to raise error if not enough memory
        
    Returns:
    --------
    bool
        True if enough memory available
    """
    available = get_available_memory() * available_fraction
    if required_memory > available:
        if raise_error:
            raise MemoryError(
                f"Not enough memory. Required: {required_memory/1e9:.2f}GB, "
                f"Available: {available/1e9:.2f}GB"
            )
        return False
    return True 