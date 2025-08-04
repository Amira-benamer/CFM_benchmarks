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

# 4. Example BenchmarkConfig for Zephyr
config_zephyr = BenchmarkConfig(
    model_name="zephyr",
    framework="pytorch",
    task_type="inference",
    platform="neuron",
    instance_type="inf2.xlarge",
    batch_sizes=[1, 2, 4],
    sequence_lengths=[400],  # chunk size in tokens
    num_runs=3,
    prompts=batch_prompts  # Try niveau_prompt or chunked_prompts as well
)

# 5. Example BenchmarkConfig for RoBERTa
config_roberta = BenchmarkConfig(
    model_name="roberta-base",
    framework="pytorch",
    task_type="inference",
    platform="neuron",
    instance_type="inf2.xlarge",
    batch_sizes=[1, 2, 4],
    sequence_lengths=[400],  # chunk size in tokens
    num_runs=3,
    prompts=batch_prompts  # Try niveau_prompt or chunked_prompts as well
)

if __name__ == "__main__":
    benchmarker = NeuronVsNvidiaBenchmarker()
    print("\n--- Zephyr Benchmark ---")
    results_zephyr = benchmarker.run_comprehensive_benchmark(config_zephyr)
    for result in results_zephyr:
        print(result)
    print("\n--- RoBERTa Benchmark ---")
    results_roberta = benchmarker.run_comprehensive_benchmark(config_roberta)
    for result in results_roberta:
        print(result)
