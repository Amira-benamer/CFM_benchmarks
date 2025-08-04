from dataclasses import dataclass
from typing import List, Optional

@dataclass
class BenchmarkConfig:
    """Configuration for standardized benchmarks."""

    # Test configuration
    model_name: str
    framework: str  # 'pytorch', 'tensorflow', 'jax'
    task_type: str  # 'training', 'inference', 'compilation'

    # Hardware configuration
    platform: str  # 'neuron', 'nvidia'
    instance_type: str
    device_count: int = 1

    # Benchmark parameters
    batch_sizes: List[int] = None
    sequence_lengths: List[int] = None
    num_runs: int = 5
    warmup_runs: int = 2

    # Data configuration
    synthetic_data: bool = True
    dataset_size: int = 1000
    prompts: Optional[List[str]] = None  # List of prompts for prompt-based benchmarking

    def __post_init__(self):
        """Set default values."""
        if self.batch_sizes is None:
            self.batch_sizes = [1, 4, 8, 16, 32]
        if self.sequence_lengths is None:
            self.sequence_lengths = [128, 256, 512]
