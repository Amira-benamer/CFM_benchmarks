import torch
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    RobertaTokenizer, RobertaForSequenceClassification
)
import time
import json
from datetime import datetime

class LLM_HuggingFace:
    def __init__(self, model_name, model_path=None, hourly_rate=0.227):
        self.model_name = model_name
        self.model_path = model_path
        self.hourly_rate = hourly_rate
        self.model = None
        self.tokenizer = None
        self.request_count = 0
        self.total_latency = 0.0
        self.server_start_time = datetime.now()
        self.is_generation = "zephyr" in model_name.lower() or "causal" in model_name.lower()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.load_model()

    def load_model(self):
        print(f" Loading model: {self.model_name} ...")
        if self.is_generation:
            model_id = self.model_path or self.model_name
            self.tokenizer = AutoTokenizer.from_pretrained(model_id)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32
            ).to(self.device)
        else:
            model_id = self.model_path or self.model_name
            self.tokenizer = RobertaTokenizer.from_pretrained(model_id)
            self.model = RobertaForSequenceClassification.from_pretrained(model_id).to(self.device)
        self.model.eval()
        print(" Model loaded successfully!")

    def predict(self, texts, max_new_tokens=64):
        start_time = time.time()
        if not texts:
            raise ValueError("No input provided.")

        if self.is_generation:
            encoded = self.tokenizer(
                texts,
                truncation=True,
                padding=True,
                max_length=128,
                return_tensors='pt'
            )
            encoded = {k: v.to(self.device) for k, v in encoded.items()}
            with torch.no_grad():
                outputs = self.model.generate(
                    **encoded,
                    max_new_tokens=max_new_tokens,
                    do_sample=False
                )
                generations = [
                    self.tokenizer.decode(out, skip_special_tokens=True)[len(prompt):].strip()
                    for out, prompt in zip(outputs, texts)
                ]
            result = {'generations': generations}
        else:
            encoded = self.tokenizer(
                texts,
                truncation=True,
                padding=True,
                max_length=128,
                return_tensors='pt'
            )
            encoded = {k: v.to(self.device) for k, v in encoded.items()}
            with torch.no_grad():
                outputs = self.model(**encoded)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            result = {'predictions': predictions.tolist()}

        latency_ms = (time.time() - start_time) * 1000
        self.request_count += len(texts)
        self.total_latency += latency_ms
        avg_latency = self.total_latency / self.request_count if self.request_count > 0 else 0
        runtime_hours = (datetime.now() - self.server_start_time).total_seconds() / 3600
        total_cost = runtime_hours * self.hourly_rate
        cost_per_1k_requests = (total_cost / self.request_count) * 1000 if self.request_count > 0 else 0

        result.update({
            'latency_ms': round(latency_ms, 2),
            'batch_size': len(texts),
            'statistics': {
                'total_requests': self.request_count,
                'average_latency_ms': round(avg_latency, 2),
                'total_cost_usd': round(total_cost, 4),
                'cost_per_1k_requests': round(cost_per_1k_requests, 4),
                'requests_per_dollar': round(1000 / cost_per_1k_requests, 0) if cost_per_1k_requests > 0 else 0
            }
        })
        return result

if __name__ == '__main__':
    use_zephyr = False  # Set to True for Zephyr

    if use_zephyr:
        model_name = "HuggingFaceH4/zephyr-7b-beta"
        model_path = None
        prompts = [
            "What is the capital of France?",
            "Explain the theory of relativity in simple terms.",
            "Write a short poem about the ocean.",
            "How do you make pancakes?"
        ]
    else:
        model_name = "FacebookAI/roberta-base"
        model_path = "/home/ubuntu/results/roberta-finetuned"
        prompts = [
            "I love using this product!",
            "Worst support ever received.",
            "Could be better, but it works.",
            "Absolutely fantastic experience!"
        ]

    llm = LLM_HuggingFace(model_name, model_path)
    try:
        result = llm.predict(prompts)
        print(json.dumps(result, indent=2))
    except Exception as e:
        print(" Error:", str(e))