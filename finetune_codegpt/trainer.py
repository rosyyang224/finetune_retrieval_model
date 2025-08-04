from transformers import (
    Trainer,           
    TrainingArguments, 
    DataCollatorForLanguageModeling,  
    AutoTokenizer,     
)
import logging
import os
import warnings
import wandb

from config import (
    MODEL_NAME, OUTPUT_DIR, get_optimized_training_args, OPTIMIZATION_CONFIG,
    create_model
)
from utils import (
    create_preprocess_function,  
    load_training_data,
    create_enhanced_compute_metrics_causal  
)

warnings.filterwarnings("ignore", category=UserWarning)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_model():
    logger.info("=" * 60)
    logger.info("PORTFOLIO QUERY CODEGPT FINE-TUNING PIPELINE")
    logger.info("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    wandb.init( 
        project="query-to-filter",
        name="codegpt_small_lr9e5_batch4",
        config={
            "model": MODEL_NAME,
            "model_class": "CodeGPT",
            "domain": "portfolio_management",
            "task": "natural_language_to_json",
            "data_size": 1150,
            "optimization_config": OPTIMIZATION_CONFIG
        }
    )

    # START FROM SCRATCH: Always create new model and tokenizer
    logger.info(f"Creating CodeGPT model from scratch: {MODEL_NAME}")
    model = create_model()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    latest_checkpoint = None

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")

    train_dataset, val_dataset = load_training_data(
        validation_split=OPTIMIZATION_CONFIG["use_validation_split"],
        validation_size=OPTIMIZATION_CONFIG["validation_size"]
    )

    if train_dataset is None:
        logger.error("Could not load training data. Exiting.")
        return OUTPUT_DIR

    logger.info(f"Training samples: {len(train_dataset)}")
    if val_dataset:
        logger.info(f"Validation samples: {len(val_dataset)}")

    preprocess_function = create_preprocess_function(tokenizer)
    logger.info("Preprocessing datasets...")

    # if OPTIMIZATION_CONFIG["use_length_grouping"]:
    #     train_dataset = train_dataset.map(add_length_column, batched=True)

    train_dataset = train_dataset.map(preprocess_function, batched=True, remove_columns=["input", "output"])
    eval_dataset = val_dataset.map(preprocess_function, batched=True, remove_columns=["input", "output"]) if val_dataset else None

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, 
        mlm=False,
        return_tensors="pt"
    )

    training_args_dict = get_optimized_training_args()
    training_args_dict.update({
        "report_to": ["wandb"],
        "logging_strategy": "steps",
        "logging_steps": 25,
        "save_strategy": "steps",
        "save_steps": 100,
        "eval_strategy": "steps",
        "eval_steps": 100,
        "save_total_limit": 3,
        "save_safetensors": True,
        "torch_compile": False,
        "include_inputs_for_metrics": True,
        "prediction_loss_only": False,
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
    })

    compute_metrics = None
    if eval_dataset is not None:
        compute_metrics = create_enhanced_compute_metrics_causal(tokenizer, None)
        logger.info("Evaluation dataset configured with enhanced metrics")
    else:
        logger.warning("No evaluation dataset - validation loss will not be available")

    args = TrainingArguments(**training_args_dict)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    logger.info("Starting training from scratch...")
    trainer.train()

    if eval_dataset:
        preds = trainer.predict(eval_dataset)
        final_metrics = {f"final_{k}": v for k, v in preds.metrics.items()}
        wandb.log(final_metrics)

    trainer.save_model()
    tokenizer.save_pretrained(OUTPUT_DIR)

    wandb.finish()
    return OUTPUT_DIR

if __name__ == "__main__":
    train_model()