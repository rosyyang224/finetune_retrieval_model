import json
import torch
import logging
from config import OUTPUT_DIR
from utils import setup_model_and_tokenizer, validate_json_schema, compute_semantic_score

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class PortfolioQueryEvaluator:
    def __init__(self, model_path: str = OUTPUT_DIR):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        try:
            self.model, self.tokenizer = setup_model_and_tokenizer(model_path)
            self.model = self.model.to(self.device)
            logger.info("Loaded Deepseek-coder-1.3b-instruct model")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def generate_prediction(self, query: str, **generation_kwargs):
        formatted_input = f"# Convert this query to JSON filter:\n# Query: {query}\n# JSON:\n"
        inputs = self.tokenizer(formatted_input, return_tensors="pt").to(self.device)

        default_kwargs = {
            "max_new_tokens": 256,
            "do_sample": False,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        default_kwargs.update(generation_kwargs)

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                **default_kwargs
            )
        generated_tokens = outputs[0][len(inputs["input_ids"][0]):]
        prediction = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

        if prediction.startswith('{'):
            brace_count = 0
            json_end = -1
            for i, char in enumerate(prediction):
                if char == '{': brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        json_end = i + 1
                        break
            if json_end > 0:
                prediction = prediction[:json_end]
        return prediction

    def evaluate_single_query(self, query: str, **generation_kwargs):
        try:
            prediction = self.generate_prediction(query, **generation_kwargs)
            is_valid, schema_score, parsed = validate_json_schema(prediction)
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
        except Exception as e:
            logger.error(f"Error evaluating query '{query}': {e}")
            return {
                "query": query,
                "prediction": "",
                "is_valid_json": False,
                "parsed_output": {},
                "semantic_score": 0.0,
                "schema_score": 0.0,
                "prediction_length": 0,
                "word_count": 0,
                "generation_quality": "error",
                "error": str(e)
            }

    def evaluate_test_set(self, queries, **generation_kwargs):
        results = []
        total_queries = len(queries)
        for i, query in enumerate(queries, 1):
            logger.info(f"Evaluating query {i}/{total_queries}: {query[:50]}...")
            result = self.evaluate_single_query(query, **generation_kwargs)
            results.append(result)
            if i % 10 == 0 or i == total_queries:
                valid = sum(1 for r in results if r["is_valid_json"])
                logger.info(f"Progress: {i}/{total_queries}, Valid JSON: {valid}/{i} ({valid/i*100:.1f}%)")
        return results

def load_test_queries(path: str):
    try:
        with open(path, 'r') as f:
            data = json.load(f)
            if isinstance(data, list):
                return data
            elif isinstance(data, dict) and "queries" in data:
                return data["queries"]
            else:
                logger.error(f"Unexpected format in {path}")
                return []
    except FileNotFoundError:
        logger.error(f"Test file {path} not found")
        return []
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in {path}: {e}")
        return []
