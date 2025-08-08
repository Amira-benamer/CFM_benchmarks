from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Optional
from benchmark_config import BenchmarkConfig

@dataclass
class BenchmarkResult:
    """Standardized benchmark result format."""

    # Test identification
    benchmark_id: str
    timestamp: datetime
    config: BenchmarkConfig

    # Performance metrics
    throughput: float  # samples/second, tokens/second
    latency_ms: float  # milliseconds
    memory_usage_gb: float  # peak memory in GB
    compilation_time_s: Optional[float] = None  # seconds

    # Statistical data
    throughput_std: float = 0.0
    latency_std: float = 0.0
    raw_measurements: List[Dict] = None

    # Cost analysis
    cost_per_sample: Optional[float] = None  # USD
    cost_per_hour: Optional[float] = None  # USD

    # Additional metadata
    notes: str = ""
    error_message: Optional[str] = None

    def __post_init__(self):
        """Initialize lists."""
        if self.raw_measurements is None:
            self.raw_measurements = []
