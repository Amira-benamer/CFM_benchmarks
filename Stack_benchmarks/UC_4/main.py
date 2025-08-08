from benchmark_config import BenchmarkConfig
from neuron_vs_nvidia_benchmarker import NeuronVsNvidiaBenchmarker
import textwrap

# 1. Niveau Prompt (Single Prompt)
niveau_prompt = [
    " Expliquez la théorie de la relativité en termes simples."
]

# 2. Multiple Prompts (Batching)
batch_prompts = [
    " Décrivez le fonctionnement d'un moteur à combustion.",
    " Qu'est-ce que l'intelligence artificielle?",
    " Résumez l'histoire de la Révolution française.",
    " Expliquez le concept de blockchain."
]

# 3. Document Chunking (simulate a long document split into 300–400 token chunks)
long_text = (
    "L'intelligence artificielle (IA) est un domaine de l'informatique qui vise à créer des systèmes capables de "
    "réaliser des tâches qui nécessitent normalement l'intelligence humaine. Cela inclut l'apprentissage, le raisonnement, "
    "la résolution de problèmes, la perception et la compréhension du langage. Les progrès récents dans l'apprentissage "
    "automatique, en particulier l'apprentissage profond, ont permis des avancées spectaculaires dans des domaines tels que "
    "la reconnaissance d'images, la traduction automatique et les assistants vocaux. Cependant, l'IA soulève également des "
    "questions éthiques et sociétales importantes, notamment en ce qui concerne l'emploi, la vie privée et la prise de décision automatisée."
)
chunked_prompts = textwrap.wrap(long_text, 300)

# Platform will be set dynamically below
config_zephyr = None
config_roberta = None

import torch

def compile_models_to_inferentia(platform):
    import torch_neuronx
    from standard_model_suite import StandardModelSuite
    models = StandardModelSuite()
    # Compile Zephyr
    zephyr_model = models.get_zephyr_model(platform=platform)
    zephyr_model.eval()
    example_input_zephyr = torch.randint(0, 32000, (4, 400))  # batch=8, seq_len=400
    print("Compiling Zephyr to Inferentia...")
    zephyr_neuron = torch_neuronx.trace(
        zephyr_model,
        example_input_zephyr,
        compiler_args=["--model-type=transformer", "--auto-cast=all", "--enable-fast-loading-neuron", "--batching=8", "--optimize-for-inference", "--enable-mixed-precision", "--neuroncore-pipeline-cores=1"],
        compiler_timeout=300,
    )
    torch.save(zephyr_neuron, "zephyr_neuron.pt")
    print("Zephyr compiled and saved as zephyr_neuron.pt")
    # Compile RoBERTa
    roberta_model = models.get_roberta_base(platform=platform)
    roberta_model.eval()
    example_input_roberta = torch.randint(0, 32000, (4, 400))  # batch=8, seq_len=400
    print("Compiling RoBERTa to Inferentia...")
    roberta_neuron = torch_neuronx.trace(
        roberta_model,
        example_input_roberta,
        compiler_args=["--model-type=transformer", "--auto-cast=all", "--enable-fast-loading-neuron", "--batching=8", "--optimize-for-inference", "--enable-mixed-precision", "--neuroncore-pipeline-cores=1"],
        compiler_timeout=300,
    )
    torch.save(roberta_neuron, "roberta_neuron.pt")
    print("RoBERTa compiled and saved as roberta_neuron.pt")


def detect_platform():
    try:
        import torch
        if torch.cuda.is_available():
            return "nvidia"
    except ImportError:
        pass
    try:
        import torch_xla.core.xla_model as xm
        devices = xm.get_xla_supported_devices()
        if any("NEURON" in str(device) for device in devices):
            return "neuron"
    except ImportError:
        pass
    return "neuron"  # Default to neuron if no CUDA and torch_xla is present

if __name__ == "__main__":
    platform = detect_platform()
    # Set configs with correct platform
    config_zephyr = BenchmarkConfig(
        model_name="zephyr",
        framework="pytorch",
        task_type="inference",
        platform=platform,
        instance_type="inf2.8xlarge" if platform == "neuron" else "g5.2xlarge",
        batch_sizes=[1, 2, 4],
        sequence_lengths=[400],  # chunk size in tokens
        num_runs=3,
        prompts=batch_prompts  # Try niveau_prompt or chunked_prompts as well
    )
    config_roberta = BenchmarkConfig(
        model_name="roberta-base",
        framework="pytorch",
        task_type="inference",
        platform=platform,
        instance_type="inf2.8xlarge" if platform == "neuron" else "g5.2xlarge",
        batch_sizes=[ 1, 2, 4],
        sequence_lengths=[400],  # chunk size in tokens
        num_runs=3,
        prompts=batch_prompts  # Try niveau_prompt or chunked_prompts as well
    )
    if platform == "neuron":
        compile_models_to_inferentia(platform)
    benchmarker = NeuronVsNvidiaBenchmarker(platform=platform)
    print("\n--- Zephyr Benchmark ---")
    results_zephyr = benchmarker.run_comprehensive_benchmark(config_zephyr)
    for result in results_zephyr:
        print(result)
    print("\n--- RoBERTa Benchmark ---")
    results_roberta = benchmarker.run_comprehensive_benchmark(config_roberta)
    for result in results_roberta:
        print(result)
