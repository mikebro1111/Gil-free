import numpy as np
import pytest
from src.algorithms import (
    PurePythonCalculator,
    NumpyCalculator,
    NumbaCalculator,
    MultiprocessingCalculator,
    MultithreadingCalculator
)
from src.utils import DataGenerator

@pytest.fixture
def test_data():
    """Generate test data"""
    return DataGenerator.generate_arrays(size=10000, seed=42)

@pytest.fixture
def implementations():
    """Get all implementations"""
    return [
        PurePythonCalculator(),
        NumpyCalculator(),
        NumbaCalculator(),
        MultiprocessingCalculator(),
        MultithreadingCalculator()
    ]

def test_implementations_same_result(implementations, test_data):
    """Test that all implementations give the same result"""
    a, b = test_data
    
    # Get result from NumPy implementation as reference
    reference = NumpyCalculator().calculate(a, b)
    
    for impl in implementations:
        result = impl.calculate(a, b)
        np.testing.assert_allclose(
            result, reference,
            rtol=1e-10, atol=1e-10,
            err_msg=f"Failed for {impl.__class__.__name__}"
        )

@pytest.mark.parametrize("size", [100, 1000, 10000])
def test_different_sizes(implementations, size):
    """Test implementations with different array sizes"""
    a, b = DataGenerator.generate_arrays(size=size, seed=42)
    
    for impl in implementations:
        result = impl.calculate(a, b)
        assert len(result) == size, f"Wrong size for {impl.__class__.__name__}"

def test_input_validation(implementations):
    """Test input validation"""
    a = np.random.random(100)
    b = np.random.random(200)
    
    for impl in implementations:
        with pytest.raises(ValueError):
            impl.calculate(a, b)

@pytest.mark.parametrize("impl_class", [
    NumbaCalculator,
    MultiprocessingCalculator,
    MultithreadingCalculator
])
def test_parallel_implementations(impl_class):
    """Test parallel implementations specifically"""
    size = 100000
    a, b = DataGenerator.generate_arrays(size=size, seed=42)
    
    impl = impl_class()
    result = impl.calculate(a, b)
    
    reference = NumpyCalculator().calculate(a, b)
    np.testing.assert_allclose(result, reference, rtol=1e-10, atol=1e-10)

def test_memory_efficiency(implementations):
    """Test memory usage"""
    from src.utils import MemoryTracker
    
    size = 1000000
    a, b = DataGenerator.generate_arrays(size=size, seed=42)
    
    for impl in implementations:
        tracker = MemoryTracker()
        with tracker.track() as memory:
            impl.calculate(a, b)
        
        # Check that memory usage is reasonable (less than 10x input size)
        expected_memory = a.nbytes + b.nbytes
        assert memory < expected_memory * 10, f"High memory usage in {impl.__class__.__name__}" 