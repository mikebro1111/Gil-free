from pathlib import Path
from datetime import datetime
import pandas as pd
from benchmarks.basic_operations import BasicOperationsBenchmark
from benchmarks.linear_regression import LinearRegressionBenchmark
from benchmarks.config import BenchmarkConfig
from benchmarks.results_processor import ResultsProcessor

def setup_results_dir(base_dir: str, benchmark_type: str) -> Path:
    """Create and return results directory for current run"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path(base_dir) / benchmark_type / timestamp
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir

def run_all_benchmarks():
    """Run all benchmarks and save results"""
    config = BenchmarkConfig()
    
    # Basic Operations Benchmark
    print("Running Basic Operations Benchmark...")
    basic_ops = BasicOperationsBenchmark(config)
    basic_ops.run_benchmark()  # Run benchmark
    basic_results = pd.DataFrame(basic_ops.results)  # Convert results to DataFrame
    
    results_dir = setup_results_dir(config.output_dir, "basic_ops")
    basic_results.to_csv(results_dir / "results.csv", index=False)
    
    # Linear Regression Benchmark
    print("\nRunning Linear Regression Benchmark...")
    linear_reg = LinearRegressionBenchmark(config)
    linear_reg.run_benchmark()  # Run benchmark
    linear_results = pd.DataFrame(linear_reg.results)  # Convert results to DataFrame
    
    results_dir = setup_results_dir(config.output_dir, "linear_regression")
    linear_results.to_csv(results_dir / "results.csv", index=False)
    
    # Generate plots
    print("\nGenerating plots...")
    processor = ResultsProcessor(Path(config.output_dir))
    
    basic_ops_plot = processor.plot_basic_operations()
    basic_ops_plot.savefig(Path(config.output_dir) / "basic_ops_comparison.png")
    
    linear_reg_plot = processor.plot_linear_regression()
    linear_reg_plot.savefig(Path(config.output_dir) / "linear_regression_comparison.png")
    
    print("\nBenchmark results have been saved to:", config.output_dir)

if __name__ == "__main__":
    run_all_benchmarks() 