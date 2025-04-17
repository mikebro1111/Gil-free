import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import json
from typing import Dict, List
import time

from src.algorithms import (
    PurePythonCalculator,
    NumpyCalculator,
    NumbaCalculator,
    MultiprocessingCalculator,
    MultithreadingCalculator
)
from src.utils import DataGenerator
from .config import BenchmarkConfig

class BasicOperationsBenchmark:
    """Benchmark for basic numerical operations"""
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.results: List[Dict] = []
        self.implementations = {
            'pure_python': PurePythonCalculator(),
            'numpy': NumpyCalculator(),
            'numba': NumbaCalculator(),
            'multiprocessing': MultiprocessingCalculator(),
            'multithreading': MultithreadingCalculator()
        }
        
    def _warmup_implementations(self):
        """Warm up implementations that need compilation"""
        print("Warming up implementations...")
        a, b = DataGenerator.generate_arrays(1000, self.config.random_seed)
        
        for _ in range(self.config.warmup_runs):
            for impl in self.implementations.values():
                impl.warmup()
    
    def run_single_test(self, implementation_name: str, size: int) -> Dict:
        """Run single implementation test"""
        implementation = self.implementations[implementation_name]
        a, b = DataGenerator.generate_arrays(size, self.config.random_seed)
        
        times = []
        
        for i in range(self.config.n_runs):
            start_time = time.perf_counter()
            _ = implementation.calculate(a, b)
            end_time = time.perf_counter()
            times.append(end_time - start_time)
            
        return {
            'implementation': implementation_name,
            'size': size,
            'mean_time': np.mean(times),
            'std_time': np.std(times),
            'min_time': np.min(times),
            'max_time': np.max(times),
            'n_runs': self.config.n_runs
        }
    
    def run_benchmark(self):
        """Run full benchmark suite"""
        self._warmup_implementations()
        
        for size in self.config.basic_sizes:
            print(f"\nTesting array size: {size:,}")
            
            for impl_name in self.implementations:
                print(f"Running {impl_name}...")
                try:
                    result = self.run_single_test(impl_name, size)
                    self.results.append(result)
                except Exception as e:
                    print(f"Error in {impl_name}: {str(e)}")
    
    def save_results(self, output_dir: Path):
        """Save benchmark results"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        result_dir = output_dir / 'basic_ops' / timestamp
        result_dir.mkdir(parents=True, exist_ok=True)
        
        # Save configuration
        self.config.save(result_dir)
        
        # Save results
        df = pd.DataFrame(self.results)
        if 'csv' in self.config.save_formats:
            df.to_csv(result_dir / 'results.csv', index=False)
        if 'json' in self.config.save_formats:
            with open(result_dir / 'results.json', 'w') as f:
                json.dump(self.results, f, indent=2) 