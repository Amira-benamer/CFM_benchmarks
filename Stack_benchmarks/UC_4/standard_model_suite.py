import torch.nn as nn

class StandardModelSuite:
    """Collection of standardized models for fair comparison."""

    @staticmethod
    def get_roberta_base(sequence_length: int = 512, pretrained: bool = False, platform: str = "cpu") -> nn.Module:
        from transformers import RobertaConfig, RobertaModel
        if pretrained:
            from transformers import AutoModel
            import torch
            print(torch.cuda.is_available())
            print(torch.cuda.device_count())
            #print(torch.cuda.get_device_name(0))
            if platform == "neuron":
                model = AutoModel.from_pretrained("roberta-base")
            else:
                model = AutoModel.from_pretrained(
                    "roberta-base",
                    device_map="cuda",
                    low_cpu_mem_usage=True
                )
            return model
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
    def get_zephyr_model(platform: str = "cpu") -> nn.Module:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
        print(torch.cuda.is_available())
        print(torch.cuda.device_count())
        print(torch.cuda.get_device_name(0))
        model_name = "TheBloke/zephyr-7B-alpha-AWQ"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        if platform == "nvidia" and torch.cuda.is_available():
            model = model.to("cuda")
        return model
