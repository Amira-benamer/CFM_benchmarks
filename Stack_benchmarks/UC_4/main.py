from neuron_vs_nvidia_benchmarker import NeuronVsNvidiaBenchmarker
from benchmark_config import BenchmarkConfig

def main():
    print("🔬 AWS Neuron vs Nvidia Benchmarking Framework")
    print("=" * 60)

    benchmarker = NeuronVsNvidiaBenchmarker()

    config = BenchmarkConfig(
        model_name="roberta-base",
        framework="pytorch",
        task_type="inference",
        platform=benchmarker.current_platform,
        instance_type="inf2.xlarge"
        if benchmarker.current_platform == "neuron"
        else "g5.xlarge",
        batch_sizes=[1, 8],
        sequence_lengths=[128],
        num_runs=3,
    )

    print(f"\n🚀 Running example benchmark on {benchmarker.current_platform} platform")
    results = benchmarker.run_comprehensive_benchmark(config)

    if results:
        print(f"\n📊 Benchmark Results:")
        for result in results:
            print(f"   {result.benchmark_id}")
            print(f"   Throughput: {result.throughput:.2f} samples/sec")
            print(f"   Latency: {result.latency_ms:.2f} ms")
            print(f"   Memory: {result.memory_usage_gb:.2f} GB")
            if result.cost_per_sample:
                print(f"   Cost: ${result.cost_per_sample:.6f} per sample")
            print()

    print("🎯 For complete sister tutorial comparison, run:")
    print("   results = benchmarker.run_standardized_benchmark_suite()")
    print("\n✅ Benchmarking framework ready for systematic comparisons")

if __name__ == "__main__":
    main()
