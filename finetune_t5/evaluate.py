import os
import json
import torch
import logging
from transformers import T5Tokenizer
from config import OUTPUT_DIR
from utils import setup_model_and_tokenizer, validate_json_schema, compute_semantic_score
from portfolio_model import PortfolioQueryT5Model

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class PortfolioQueryEvaluator:
    def __init__(self, model_path: str = OUTPUT_DIR):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        try:
            self.model = PortfolioQueryT5Model.from_pretrained(model_path).to(self.device)
            self.tokenizer = T5Tokenizer.from_pretrained(model_path)
            logger.info("Loaded PortfolioQueryT5Model")
        except Exception as e:
            logger.warning(f"Fallback to default loader due to: {e}")
            self.model, self.tokenizer = setup_model_and_tokenizer(model_path)

    def generate_prediction(self, query: str, **generation_kwargs):
        inputs = self.tokenizer(query, return_tensors="pt").to(self.device)
        outputs = self.model.generate(**inputs, **generation_kwargs)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def evaluate_single_query(self, query: str, **generation_kwargs):
        prediction = self.generate_prediction(query, **generation_kwargs)
        is_valid, schema_score, parsed = validate_json_schema(prediction)  # Fix: unpack 3 values
        semantic_score = compute_semantic_score(query, parsed)

        return {
            "query": query,
            "prediction": prediction,
            "is_valid_json": is_valid,
            "parsed_output": parsed if is_valid else {},
            "semantic_score": semantic_score,
            "schema_score": schema_score, 
            "prediction_length": len(prediction),
            "word_count": len(prediction.split()),
            "generation_quality": "good" if is_valid and semantic_score > 0.7 else "poor"
        }

    def evaluate_test_set(self, queries, **generation_kwargs):
        results = []
        for query in queries:
            results.append(self.evaluate_single_query(query, **generation_kwargs))
        return results

def load_test_queries(path: str):
    with open(path, 'r') as f:
        return json.load(f)

def main():
    logger.info("Running PortfolioQueryEvaluator")
    evaluator = PortfolioQueryEvaluator()

    test_queries = load_test_queries("../data/test_data.json")
    generation_kwargs = {
        "num_beams": 4,
        "max_length": 256,
        "early_stopping": True
    }
    results = evaluator.evaluate_test_set(test_queries, **generation_kwargs)

    with open("evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Saved evaluation results to evaluation_results.json")

if __name__ == "__main__":
    main()
