import logging
import json
from trainer import train_model
from evaluate import PortfolioQueryEvaluator, load_test_queries
from config import GENERATION_KWARGS

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def main():
    logger.info("Starting full pipeline: training + evaluation")
    final_model_path = train_model()
    logger.info("Starting evaluation after training...")

    evaluator = PortfolioQueryEvaluator(model_path=final_model_path)
    test_queries = load_test_queries("../data/test_data.json")
    results = evaluator.evaluate_test_set(test_queries, **GENERATION_KWARGS)

    logger.info(f"Finished evaluation with {len(results)} results")

    with open("evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Saved evaluation results to evaluation_results.json")

if __name__ == "__main__":
    main()
