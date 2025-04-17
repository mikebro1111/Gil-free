from dataclasses import dataclass, field
from typing import List, Tuple, Dict
import json
from pathlib import Path

@dataclass
class BenchmarkConfig:
    """Configuration for benchmarks"""
    
    # General settings
    n_runs: int = 10
    warmup_runs: int = 2
    random_seed: int = 42
    
    # Basic operations settings
    basic_sizes: List[int] = field(default_factory=lambda: [
        10_000,
        100_000,
        500_000,
        1_000_000
    ])
    
    # Linear regression settings
    regression_sizes: List[Tuple[int, int]] = field(default_factory=lambda: [
        (1000, 5),
        (5000, 10),
        (10000, 15),
        (20000, 20)
    ])
    
    # Output settings
    output_dir: str = "results"
    save_formats: List[str] = field(default_factory=lambda: ["csv", "json"])
    
    def save(self, path: Path) -> None:
        """Save configuration to file"""
        with open(path / 'config.json', 'w') as f:
            json.dump(self.__dict__, f, indent=2)
    
    @classmethod
    def load(cls, path: Path) -> 'BenchmarkConfig':
        """Load configuration from file"""
        with open(path / 'config.json', 'r') as f:
            return cls(**json.load(f)) 