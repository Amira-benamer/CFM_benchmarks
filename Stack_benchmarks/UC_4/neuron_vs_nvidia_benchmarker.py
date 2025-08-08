import json
import logging
import statistics
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import psutil

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from benchmark_config import BenchmarkConfig
from benchmark_result import BenchmarkResult
from standard_model_suite import StandardModelSuite

try:
    import torch_neuronx
    import torch_xla.core.xla_model as xm
    NEURON_AVAILABLE = True
except ImportError:
    NEURON_AVAILABLE = False

logger = logging.getLogger(__name__)

class NeuronVsNvidiaBenchmarker:
    def __init__(self, results_dir: str = "./benchmark_results", platform: str = None):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.current_platform = platform if platform else self._detect_platform()
        self.pricing = {
            "inf2.xlarge": 0.37,
            "inf2.8xlarge": 2.97,
            "inf2.24xlarge": 8.90,
            "p4d.24xlarge": 32.77,
            "p3.2xlarge": 3.06,
            "g5.xlarge": 1.01,
            "g5.4xlarge": 4.03,
        }
        self.models = StandardModelSuite()
        logger.info(f"🔬 Neuron vs Nvidia Benchmarker initialized")
        logger.info(f"   Platform detected: {self.current_platform}")
        logger.info(f"   Results directory: {self.results_dir}")

    def _detect_platform(self) -> str:
        if NEURON_AVAILABLE:
            try:
                devices = xm.get_xla_supported_devices()
                if any("NEURON" in str(device) for device in devices):
                    return "neuron"
            except Exception:
                pass
        if TORCH_AVAILABLE and torch.cuda.is_available():
            return "nvidia"
        return "cpu"

    def run_comprehensive_benchmark(self, config: BenchmarkConfig) -> List[BenchmarkResult]:
        logger.info(f"🚀 Starting comprehensive benchmark: {config.model_name}")
        logger.info(f"   Platform: {config.platform}")
        logger.info(f"   Task: {config.task_type}")
        results = []
        model = self._get_model(config.model_name)
        if model is None:
            logger.error(f"Model {config.model_name} not found")
            return results
        for batch_size in config.batch_sizes:
            for seq_len in config.sequence_lengths:
                benchmark_config = BenchmarkConfig(
                    model_name=config.model_name,
                    framework=config.framework,
                    task_type=config.task_type,
                    platform=config.platform,
                    instance_type=config.instance_type,
                    batch_sizes=[batch_size],
                    sequence_lengths=[seq_len],
                    num_runs=config.num_runs,
                    warmup_runs=config.warmup_runs,
                )
                if config.task_type == "training":
                    result = self._benchmark_training(model, benchmark_config, batch_size, seq_len)
                elif config.task_type == "inference":
                    result = self._benchmark_inference(model, benchmark_config, batch_size, seq_len)
                elif config.task_type == "compilation":
                    result = self._benchmark_compilation(model, benchmark_config, batch_size, seq_len)
                else:
                    logger.warning(f"Unknown task type: {config.task_type}")
                    continue
                if result:
                    results.append(result)
        self._save_results(results, config)
        logger.info(f"✅ Benchmark completed: {len(results)} results")
        return results

    def _get_model(self, model_name: str) -> Optional[nn.Module]:
        model_registry = {
            "roberta-base": lambda: self.models.get_roberta_base(platform=self.current_platform),
            "zephyr": lambda: self.models.get_zephyr_model(platform=self.current_platform),
        }
        if model_name in model_registry:
            return model_registry[model_name]()
        logger.error(f"Unknown model: {model_name}")
        return None

    def _chunk_prompt(self, prompt, tokenizer, chunk_size=400):
        tokens = tokenizer.encode(prompt, add_special_tokens=False)
        return [tokens[i:i+chunk_size] for i in range(0, len(tokens), chunk_size)]

    def _batch_prompts(self, prompts, tokenizer, batch_size, chunk_size=None):
        batches = []
        if chunk_size:
            all_chunks = []
            for prompt in prompts:
                chunks = self._chunk_prompt(prompt, tokenizer, chunk_size)
                all_chunks.extend(chunks)
            for i in range(0, len(all_chunks), batch_size):
                batch = all_chunks[i:i+batch_size]
                batch = tokenizer.pad({'input_ids': batch}, return_tensors='pt')['input_ids']
                batches.append(batch)
        else:
            encodings = tokenizer(prompts, padding=True, return_tensors='pt')
            input_ids = encodings['input_ids']
            for i in range(0, len(input_ids), batch_size):
                batches.append(input_ids[i:i+batch_size])
        return batches

    def _benchmark_inference(
        self,
        model: nn.Module,
        config: BenchmarkConfig,
        batch_size: int,
        sequence_length: int,
     ) -> Optional[BenchmarkResult]:
        """Benchmark inference performance."""
        logger.info(
            f"🔍 Inference benchmark: batch={batch_size}, seq_len={sequence_length}"
        )

        if not TORCH_AVAILABLE and config.platform != "neuron":
            logger.error("PyTorch is not available on this system and is required for non-Neuron platforms.")
            return BenchmarkResult(
                benchmark_id=f"{config.model_name}-inference-{batch_size}-{sequence_length}",
                timestamp=datetime.now(),
                config=config,
                throughput=0.0,
                latency_ms=0.0,
                memory_usage_gb=0.0,
                error_message="PyTorch is not available on this system.",
            )

        try:
            tokenizer = None
            batches = None
            if config.prompts:
                if config.model_name == "zephyr":
                    from transformers import AutoTokenizer
                    model_name = "HuggingFaceH4/zephyr-7b-beta"
                    tokenizer = AutoTokenizer.from_pretrained(model_name)
                elif config.model_name == "roberta-base":
                    from transformers import RobertaTokenizer
                    model_name = "roberta-base"
                    tokenizer = RobertaTokenizer.from_pretrained(model_name)
                else:
                    from transformers import AutoTokenizer
                    model_name = config.model_name
                    tokenizer = AutoTokenizer.from_pretrained(model_name)
                # Prepare batches of prompts/chunks
                batches = self._batch_prompts(config.prompts, tokenizer, batch_size, chunk_size=sequence_length)
            # Prepare model and data
            if self.current_platform == "neuron" and NEURON_AVAILABLE:
                device = xm.xla_device()
                model = model.to(device)
            elif self.current_platform == "nvidia" and TORCH_AVAILABLE and torch.cuda.is_available():
                device = torch.device("cuda")
                model = model.to(device)
            elif TORCH_AVAILABLE:
                device = torch.device("cpu")
                model = model.to(device)
            else:
                logger.error("PyTorch is not available for this benchmark.")
                return BenchmarkResult(
                    benchmark_id=f"{config.model_name}-inference-{batch_size}-{sequence_length}",
                    timestamp=datetime.now(),
                    config=config,
                    throughput=0.0,
                    latency_ms=0.0,
                    memory_usage_gb=0.0,
                    error_message="PyTorch is not available on this system.",
                )
            model.eval()
            # Warmup runs
            with torch.no_grad():
                if config.prompts and tokenizer:
                    for _ in range(config.warmup_runs):
                        for batch in batches:
                            batch = batch.to(device)
                            if config.model_name == "roberta-base":
                                attention_mask = (batch != tokenizer.pad_token_id).long().to(device)
                                _ = model(input_ids=batch, attention_mask=attention_mask)
                            else:
                                _ = model(input_ids=batch)
                            if self.current_platform == "neuron":
                                xm.wait_device_ops()
                            elif TORCH_AVAILABLE and torch.cuda.is_available():
                                torch.cuda.synchronize()
                else:
                    input_data = torch.randint(0, 30000, (batch_size, sequence_length)).to(device)
                    for _ in range(config.warmup_runs):
                        _ = model(input_data)
                        if self.current_platform == "neuron":
                            xm.wait_device_ops()
                        elif TORCH_AVAILABLE and torch.cuda.is_available():
                            torch.cuda.synchronize()
                        
            # Benchmark runs
            throughputs = []
            latencies = []
            memory_usages = []
            with torch.no_grad():
                if config.prompts and tokenizer:
                    for batch in batches:
                        batch = batch.to(device)
                        # Memory baseline
                        if config.platform == "neuron":
                            memory_before = psutil.virtual_memory().used / (1024**3)
                        elif TORCH_AVAILABLE and torch.cuda.is_available():
                            torch.cuda.synchronize()
                            memory_before = torch.cuda.memory_allocated(device) / (1024**3)
                        else:
                            memory_before = 0.0
                        start_time = time.time()
                        if config.model_name == "roberta-base":
                            attention_mask = (batch != tokenizer.pad_token_id).long().to(device)
                            _ = model(input_ids=batch, attention_mask=attention_mask)
                        else:
                            _ = model(input_ids=batch)
                        if self.current_platform == "neuron":
                            xm.wait_device_ops()
                        elif TORCH_AVAILABLE and torch.cuda.is_available():
                            torch.cuda.synchronize()
                        end_time = time.time()
                        if config.platform == "neuron":
                            memory_after = psutil.virtual_memory().used / (1024**3)
                        elif TORCH_AVAILABLE and torch.cuda.is_available():
                            memory_after = torch.cuda.max_memory_allocated(device) / (1024**3)
                            torch.cuda.reset_peak_memory_stats(device)
                        else:
                            memory_after = 0.0
                        batch_time = end_time - start_time
                        throughput = batch.size(0) / batch_time
                        latency = batch_time * 1000
                        memory_usage = memory_after - memory_before
                        throughputs.append(throughput)
                        latencies.append(latency)
                        memory_usages.append(max(0, memory_usage))
                else:
                    input_data = torch.randint(0, 30000, (batch_size, sequence_length)).to(device)
                    for run in range(config.num_runs):
                        if config.platform == "neuron":
                            memory_before = psutil.virtual_memory().used / (1024**3)
                        elif TORCH_AVAILABLE and torch.cuda.is_available():
                            torch.cuda.synchronize()
                            memory_before = torch.cuda.memory_allocated(device) / (1024**3)
                        else:
                            memory_before = 0.0
                        start_time = time.time()
                        outputs = model(input_data)
                        if self.current_platform == "neuron":
                            xm.wait_device_ops()
                        elif TORCH_AVAILABLE and torch.cuda.is_available():
                            torch.cuda.synchronize()
                        end_time = time.time()
                        if config.platform == "neuron":
                            memory_after = psutil.virtual_memory().used / (1024**3)
                        elif TORCH_AVAILABLE and torch.cuda.is_available():
                            memory_after = torch.cuda.max_memory_allocated(device) / (1024**3)
                            torch.cuda.reset_peak_memory_stats(device)
                        else:
                            memory_after = 0.0
                        batch_time = end_time - start_time
                        throughput = batch_size / batch_time
                        latency = batch_time * 1000
                        memory_usage = memory_after - memory_before
                        throughputs.append(throughput)
                        latencies.append(latency)
                        memory_usages.append(max(0, memory_usage))
            # Calculate statistics
            avg_throughput = statistics.mean(throughputs) if throughputs else 0.0
            avg_latency = statistics.mean(latencies) if latencies else 0.0
            avg_memory = statistics.mean(memory_usages) if memory_usages else 0.0
            throughput_std = statistics.stdev(throughputs) if len(throughputs) > 1 else 0.0
            latency_std = statistics.stdev(latencies) if len(latencies) > 1 else 0.0
            cost_per_hour = self.pricing.get(config.instance_type, 0.0)
            cost_per_sample = ((cost_per_hour / 3600) / avg_throughput if avg_throughput > 0 else 0.0)
            return BenchmarkResult(
                benchmark_id=f"{config.model_name}-inference-{batch_size}-{sequence_length}",
                timestamp=datetime.now(),
                config=config,
                throughput=avg_throughput,
                latency_ms=avg_latency,
                memory_usage_gb=avg_memory,
                throughput_std=throughput_std,
                latency_std=latency_std,
                cost_per_sample=cost_per_sample,
                cost_per_hour=cost_per_hour,
                raw_measurements=[
                    {
                        "throughputs": throughputs,
                        "latencies": latencies,
                        "memory_usages": memory_usages,
                    }
                ],
            )
        except Exception as e:
            logger.error(f"Inference benchmark failed: {e}")
            return BenchmarkResult(
                benchmark_id=f"{config.model_name}-inference-{batch_size}-{sequence_length}",
                timestamp=datetime.now(),
                config=config,
                throughput=0.0,
                latency_ms=0.0,
                memory_usage_gb=0.0,
                error_message=str(e),
            )

    def _benchmark_compilation(
        self,
        model: nn.Module,
        config: BenchmarkConfig,
        batch_size: int,
        sequence_length: int,
     ) -> Optional[BenchmarkResult]:
        """Benchmark model compilation time."""
        logger.info(
            f"⚙️ Compilation benchmark: batch={batch_size}, seq_len={sequence_length}"
        )

        if config.platform != "neuron" or not NEURON_AVAILABLE:
            logger.warning("Compilation benchmark only available on Neuron platform")
            return None

        try:
            device = xm.xla_device()
            model = model.to(device)

            # Generate sample input for compilation
            if (
                "roberta" in config.model_name
                or "zephyr" in config.model_name
            ):
                sample_input = torch.randint(
                    0, 30000, (batch_size, sequence_length), device=device
                )
            else:
                raise ValueError("Unsupported model for this benchmark suite.")

            # Benchmark compilation
            compilation_times = []

            for run in range(config.num_runs):
                start_time = time.time()

                # Trace/compile the model
                compiled_model = torch_neuronx.trace(model, sample_input)

                end_time = time.time()
                compilation_time = end_time - start_time
                compilation_times.append(compilation_time)

                logger.info(f"   Compilation run {run + 1}: {compilation_time:.2f}s")

            avg_compilation_time = statistics.mean(compilation_times)
            compilation_std = (
                statistics.stdev(compilation_times)
                if len(compilation_times) > 1
                else 0.0
            )

            return BenchmarkResult(
                benchmark_id=f"{config.model_name}-compilation-{batch_size}-{sequence_length}",
                timestamp=datetime.now(),
                config=config,
                throughput=0.0,  # Not applicable for compilation
                latency_ms=0.0,  # Not applicable for compilation
                memory_usage_gb=0.0,  # Could be measured if needed
                compilation_time_s=avg_compilation_time,
                raw_measurements=[{"compilation_times": compilation_times}],
            )

        except Exception as e:
            logger.error(f"Compilation benchmark failed: {e}")
            return BenchmarkResult(
                benchmark_id=f"{config.model_name}-compilation-{batch_size}-{sequence_length}",
                timestamp=datetime.now(),
                config=config,
                throughput=0.0,
                latency_ms=0.0,
                memory_usage_gb=0.0,
                error_message=str(e),
            )

    def _save_results(self, results: List[BenchmarkResult], config: BenchmarkConfig):
        """Save benchmark results to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = (
            f"{config.model_name}_{config.platform}_{config.task_type}_{timestamp}.json"
        )
        filepath = self.results_dir / filename

        # Convert results to serializable format
        results_data = []
        for result in results:
            result_dict = {
                "benchmark_id": result.benchmark_id,
                "timestamp": result.timestamp.isoformat(),
                "config": {
                    "model_name": result.config.model_name,
                    "framework": result.config.framework,
                    "task_type": result.config.task_type,
                    "platform": result.config.platform,
                    "instance_type": result.config.instance_type,
                },
                "throughput": result.throughput,
                "latency_ms": result.latency_ms,
                "memory_usage_gb": result.memory_usage_gb,
                "compilation_time_s": result.compilation_time_s,
                "throughput_std": result.throughput_std,
                "latency_std": result.latency_std,
                "cost_per_sample": result.cost_per_sample,
                "cost_per_hour": result.cost_per_hour,
                "raw_measurements": result.raw_measurements,
                "notes": result.notes,
                "error_message": result.error_message,
            }
            results_data.append(result_dict)

        with open(filepath, "w") as f:
            json.dump(results_data, f, indent=2)

        logger.info(f"💾 Results saved to {filepath}")

    def generate_comparison_report(
        self,
        neuron_results: List[BenchmarkResult],
        nvidia_results: List[BenchmarkResult],
     ) -> str:
        """Generate comprehensive comparison report."""
        logger.info("📊 Generating Neuron vs Nvidia comparison report")

        report = []
        report.append("# AWS Neuron vs Nvidia GPU Performance Comparison")
        report.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Neuron Results: {len(neuron_results)} benchmarks")
        report.append(f"Nvidia Results: {len(nvidia_results)} benchmarks")

        # Create comparison tables
        comparison_data = []

        for neuron_result in neuron_results:
            # Find matching Nvidia result
            nvidia_result = None
            for nr in nvidia_results:
                if (
                    nr.config.model_name == neuron_result.config.model_name
                    and nr.config.task_type == neuron_result.config.task_type
                ):
                    nvidia_result = nr
                    break

            if nvidia_result:
                comparison_data.append(
                    {
                        "model": neuron_result.config.model_name,
                        "task": neuron_result.config.task_type,
                        "neuron_throughput": neuron_result.throughput,
                        "nvidia_throughput": nvidia_result.throughput,
                        "neuron_latency": neuron_result.latency_ms,
                        "nvidia_latency": nvidia_result.latency_ms,
                        "neuron_memory": neuron_result.memory_usage_gb,
                        "nvidia_memory": nvidia_result.memory_usage_gb,
                        "neuron_cost": neuron_result.cost_per_sample or 0,
                        "nvidia_cost": nvidia_result.cost_per_sample or 0,
                        "neuron_instance": neuron_result.config.instance_type,
                        "nvidia_instance": nvidia_result.config.instance_type,
                    }
                )

        if comparison_data:
            df = pd.DataFrame(comparison_data)

            report.append("\n## Performance Summary")
            report.append(f"```")
            report.append(df.to_string(index=False))
            report.append(f"```")

            # Calculate relative performance
            report.append("\n## Relative Performance (Neuron vs Nvidia)")

            for _, row in df.iterrows():
                model_task = f"{row['model']} - {row['task']}"

                if row["nvidia_throughput"] > 0:
                    throughput_ratio = (
                        row["neuron_throughput"] / row["nvidia_throughput"]
                    )
                    throughput_comparison = (
                        f"{throughput_ratio:.2f}x"
                        if throughput_ratio >= 1
                        else f"{1/throughput_ratio:.2f}x slower"
                    )
                else:
                    throughput_comparison = "N/A"

                if row["nvidia_latency"] > 0:
                    latency_ratio = row["neuron_latency"] / row["nvidia_latency"]
                    latency_comparison = (
                        f"{latency_ratio:.2f}x"
                        if latency_ratio >= 1
                        else f"{1/latency_ratio:.2f}x faster"
                    )
                else:
                    latency_comparison = "N/A"

                if row["nvidia_cost"] > 0:
                    cost_ratio = row["neuron_cost"] / row["nvidia_cost"]
                    cost_comparison = (
                        f"{cost_ratio:.2f}x"
                        if cost_ratio >= 1
                        else f"{1/cost_ratio:.2f}x cheaper"
                    )
                else:
                    cost_comparison = "N/A"

                report.append(f"\n**{model_task}**:")
                report.append(f"- Throughput: {throughput_comparison}")
                report.append(f"- Latency: {latency_comparison}")
                report.append(f"- Cost per sample: {cost_comparison}")
                report.append(
                    f"- Instances: {row['neuron_instance']} vs {row['nvidia_instance']}"
                )

        # Recommendations
        report.append("\n## Recommendations")
        report.append("Based on the benchmark results:")

        if comparison_data:
            avg_cost_savings = np.mean(
                [
                    (row["nvidia_cost"] - row["neuron_cost"]) / row["nvidia_cost"]
                    for row in comparison_data
                    if row["nvidia_cost"] > 0 and row["neuron_cost"] > 0
                ]
            )

            if avg_cost_savings > 0:
                report.append(
                    f"- Average cost savings with Neuron: {avg_cost_savings*100:.1f}%"
                )

            report.append("- Use Neuron for cost-sensitive workloads")
            report.append("- Use Nvidia for maximum raw performance requirements")
            report.append("- Consider Neuron compilation time in deployment planning")

        report_text = "\n".join(report)

        # Save report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = self.results_dir / f"neuron_vs_nvidia_comparison_{timestamp}.md"
        with open(report_path, "w") as f:
            f.write(report_text)

        logger.info(f"📋 Comparison report saved to {report_path}")
        return report_text
