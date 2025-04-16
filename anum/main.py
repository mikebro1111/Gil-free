import numpy as np
import time
from multiprocessing import Pool, cpu_count
from numba import jit
import threading
from concurrent.futures import ThreadPoolExecutor
import psutil

def get_available_memory():
    """Returns available memory in bytes"""
    return psutil.virtual_memory().available

def create_data(size):
    """Create test data arrays"""
    return np.random.random(size), np.random.random(size)

def pure_python(a, b, chunk_size):
    result = [0] * chunk_size
    for i in range(chunk_size):
        result[i] = a[i] + b[i] * 2 - a[i] % 3
    return result

def numpy_calc(a, b):
    return a + b * 2 - a % 3

@jit(nopython=True)
def numba_calc(a, b, chunk_size):
    result = np.zeros(chunk_size)
    for i in range(chunk_size):
        result[i] = a[i] + b[i] * 2 - a[i] % 3
    return result

def process_chunk(args):
    a_chunk, b_chunk = args
    return pure_python(a_chunk, b_chunk, len(a_chunk))

def thread_worker(a_chunk, b_chunk, result, start_idx):
    chunk_result = numba_calc(a_chunk, b_chunk, len(a_chunk))
    result[start_idx:start_idx + len(a_chunk)] = chunk_result

def benchmark(sizes):
    results = []
    
    for size in sizes:
        print(f"\nTesting with array size: {size:,}")
        a, b = create_data(size)
        
        # Pure Python
        start = time.perf_counter()
        _ = pure_python(a, b, size)
        python_time = time.perf_counter() - start
        
        # NumPy
        start = time.perf_counter()
        _ = numpy_calc(a, b)
        numpy_time = time.perf_counter() - start
        
        # Numba
        start = time.perf_counter()
        _ = numba_calc(a, b, size)
        numba_time = time.perf_counter() - start
        
        # Multiprocessing
        chunks = np.array_split(a, cpu_count()), np.array_split(b, cpu_count())
        start = time.perf_counter()
        with Pool() as pool:
            _ = pool.map(process_chunk, zip(*chunks))
        mp_time = time.perf_counter() - start
        
        # GIL-free Multithreading with Numba
        result = np.zeros(size)
        chunk_size = size // cpu_count()
        threads = []
        start = time.perf_counter()
        
        with ThreadPoolExecutor(max_workers=cpu_count()) as executor:
            for i in range(cpu_count()):
                start_idx = i * chunk_size
                end_idx = start_idx + chunk_size if i < cpu_count() - 1 else size
                executor.submit(
                    thread_worker,
                    a[start_idx:end_idx],
                    b[start_idx:end_idx],
                    result,
                    start_idx
                )
                
        mt_time = time.perf_counter() - start
        
        results.append({
            'size': size,
            'python': python_time,
            'numpy': numpy_time,
            'numba': numba_time,
            'multiprocessing': mp_time,
            'multithreading': mt_time
        })
        
        print(f"Python: {python_time:.4f}s")
        print(f"NumPy: {numpy_time:.4f}s")
        print(f"Numba: {numba_time:.4f}s")
        print(f"Multiprocessing: {mp_time:.4f}s")
        print(f"Multithreading: {mt_time:.4f}s")

if __name__ == '__main__':
    sizes = [
        1_000_000,
        5_000_000,
        10_000_000,
        20_000_000
    ]
    
    benchmark(sizes)