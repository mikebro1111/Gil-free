import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import json
from typing import Dict, List
import time

from src.linear_regression import (
    NormalEquationRegression,
    GradientDescentRegression,
    ParallelGradientDescentRegression,
    SklearnWrapper
)
from src.utils import DataGenerator, MemoryTracker, compute_metrics
from .config import BenchmarkConfig

class LinearRegressionBenchmark:
    """Benchmark for linear regression implementations"""
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.results: List[Dict] = []
        self.implementations = {
            'normal_equation': NormalEquationRegression(),
            'gradient_descent': GradientDescentRegression(),
            'parallel_gd': ParallelGradientDescentRegression(),
            'sklearn': SklearnWrapper()
        }
    
    def _warmup_implementations(self):
        """Warm up implementations that need compilation"""
        print("Warming up implementations...")
        X_train, X_test, y_train, y_test = DataGenerator.generate_regression_data(
            n_samples=1000,
            n_features=10,
            random_state=self.config.random_seed
        )
        
        for _ in range(self.config.warmup_runs):
            for impl in self.implementations.values():
                impl.warmup()
    
    def run_single_test(self, implementation_name: str, 
                       n_samples: int, n_features: int) -> Dict:
        """Run single implementation test"""
        implementation = self.implementations[implementation_name]
        
        X_train, X_test, y_train, y_test = DataGenerator.generate_regression_data(
            n_samples=n_samples,
            n_features=n_features,
            random_state=self.config.random_seed
        )
        
        times_fit = []
        times_predict = []
        memory_usage = []
        scores = []
        
        for i in range(self.config.n_runs):
            tracker = MemoryTracker()
            
            with tracker.track() as memory:
                # Measure fit time
                start_time = time.perf_counter()
                implementation.fit(X_train, y_train)
                fit_time = time.perf_counter() - start_time
                
                # Measure predict time
                start_time = time.perf_counter()
                y_pred = implementation.predict(X_test)
                predict_time = time.perf_counter() - start_time
                
                # Calculate score
                score = implementation.score(X_test, y_test)
            
            times_fit.append(fit_time)
            times_predict.append(predict_time)
            memory_usage.append(memory)
            scores.append(score)
            
        return {
            'implementation': implementation_name,
            'n_samples': n_samples,
            'n_features': n_features,
            'mean_fit_time': np.mean(times_fit),
            'std_fit_time': np.std(times_fit),
            'mean_predict_time': np.mean(times_predict),
            'std_predict_time': np.std(times_predict),
            'mean_memory': np.mean(memory_usage),
            'std_memory': np.std(memory_usage),
            'mean_score': np.mean(scores),
            'std_score': np.std(scores),
            'n_runs': self.config.n_runs
        }
    
    def run_benchmark(self):
        """Run full benchmark suite"""
        self._warmup_implementations()
        
        for n_samples, n_features in self.config.regression_sizes:
            print(f"\nTesting size: {n_samples:,} samples, {n_features} features")
            
            for impl_name in self.implementations:
                print(f"Running {impl_name}...")
                try:
                    result = self.run_single_test(impl_name, n_samples, n_features)
                    self.results.append(result)
                except Exception as e:
                    print(f"Error in {impl_name}: {str(e)}")
    
    def save_results(self, output_dir: Path):
        """Save benchmark results"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        result_dir = output_dir / 'linear_regression' / timestamp
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