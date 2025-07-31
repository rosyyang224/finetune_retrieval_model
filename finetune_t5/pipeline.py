import logging
from trainer import train_model
from evaluate import PortfolioQueryEvaluator, load_test_queries

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def main():
    # Step 1: Train the model
    logger.info("Starting full pipeline: training + evaluation")
    final_model_path = train_model()  # returns OUTPUT_DIR

    # Step 2: Evaluate the model
    logger.info("Starting evaluation after training...")
    evaluator = PortfolioQueryEvaluator(model_path=final_model_path)

    test_queries = load_test_queries("../data/test_data.json") 
    generation_kwargs = {
        "num_beams": 4,
        "max_length": 256,
        "early_stopping": True
    }

    results = evaluator.evaluate_test_set(test_queries, **generation_kwargs)
    logger.info(f"Finished evaluation with {len(results)} results")

    import json
    with open("evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Saved evaluation results to evaluation_results.json")

if __name__ == "__main__":
    main()
