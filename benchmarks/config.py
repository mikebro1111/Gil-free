from dataclasses import dataclass
from typing import List, Tuple, Dict
import json
from pathlib import Path

@dataclass
class BenchmarkConfig:
    """Configuration for benchmarks"""
    
    # General settings
    n_runs: int = 10
    warmup_runs: int = 2
    memory_limit_fraction: float = 0.8
    random_seed: int = 42
    
    # Basic operations settings
    basic_sizes: List[int] = (
        1_000_000,
        5_000_000,
        10_000_000,
        20_000_000
    )
    
    # Linear regression settings
    regression_sizes: List[Tuple[int, int]] = (
        (1000, 5),
        (10000, 10),
        (100000, 20),
        (1000000, 50)
    )
    
    # Output settings
    output_dir: str = "results"
    save_formats: List[str] = ("csv", "json")
    
    def save(self, path: Path) -> None:
        """Save configuration to file"""
        with open(path / 'config.json', 'w') as f:
            json.dump(self.__dict__, f, indent=2)
    
    @classmethod
    def load(cls, path: Path) -> 'BenchmarkConfig':
        """Load configuration from file"""
        with open(path / 'config.json', 'r') as f:
            return cls(**json.load(f)) 