import os
import json
import logging
from transformers import T5ForConditionalGeneration

logger = logging.getLogger(__name__)

class PortfolioQueryT5Model(T5ForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        self.portfolio_query_config = {
            "domain": "portfolio_management",
            "task": "natural_language_to_json",
            "optimization_level": "production",
            "generation_strategy": "constrained_beam_search",
            "model_version": "1.0",
            "specialized_for": "financial_query_understanding"
        }

    @classmethod
    def from_pretrained(cls, model_name_or_path, **kwargs):
        model = super().from_pretrained(model_name_or_path, **kwargs)

        if not hasattr(model, 'portfolio_query_config'):
            model.portfolio_query_config = {
                "domain": "portfolio_management",
                "task": "natural_language_to_json",
                "optimization_level": "production",
                "generation_strategy": "constrained_beam_search",
                "model_version": "1.0",
                "specialized_for": "financial_query_understanding"
            }

        config_path = os.path.join(model_name_or_path, "portfolio_query_config.json")
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                saved_config = json.load(f)
                model.portfolio_query_config.update(saved_config)
                logger.info("Loaded saved portfolio query configuration")

        return model

    def generate_portfolio_query(self, input_ids, attention_mask=None, **generation_kwargs):
        default_kwargs = {
            "max_length": 256,
            "num_beams": 4,
            "early_stopping": True,
            "do_sample": False,
            "repetition_penalty": 1.2,
            "length_penalty": 1.0,
            "no_repeat_ngram_size": 2,
            "encoder_no_repeat_ngram_size": 2,
            "pad_token_id": self.config.pad_token_id,
            "eos_token_id": self.config.eos_token_id,
        }

        if generation_kwargs.get("do_sample", False):
            default_kwargs["temperature"] = generation_kwargs.get("temperature", 0.7)

        default_kwargs.update(generation_kwargs)

        return self.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **default_kwargs
        )

    def save_pretrained(self, save_directory, **kwargs):
        super().save_pretrained(save_directory, **kwargs)
        config_path = os.path.join(save_directory, "portfolio_query_config.json")
        with open(config_path, 'w') as f:
            json.dump(self.portfolio_query_config, f, indent=2)
        logger.info(f"Saved PortfolioQueryT5Model with config to {save_directory}")