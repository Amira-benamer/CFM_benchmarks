import torch.nn as nn

class StandardModelSuite:
    """Collection of standardized models for fair comparison."""

    @staticmethod
    def get_roberta_base(sequence_length: int = 512) -> nn.Module:
        from transformers import RobertaConfig, RobertaModel

        config = RobertaConfig(
            max_position_embeddings=sequence_length,
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,
            vocab_size=50265,
            type_vocab_size=1,
            layer_norm_eps=1e-5,
        )
        return RobertaModel(config)

    @staticmethod
    def get_zephyr_model() -> nn.Module:
        from transformers import AutoModelForCausalLM
        model_name = "HuggingFaceH4/zephyr-7b-beta"
        model = AutoModelForCausalLM.from_pretrained(model_name)
        return model
