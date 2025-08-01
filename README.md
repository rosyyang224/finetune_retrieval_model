# Query-to-Filter ML Fine-Tuning Pipelines

This repository provides a unified framework for fine-tuning and evaluating machine learning models that translate natural language portfolio queries into structured JSON filter instructions.  

Each model implementation is contained in its own folder with modular, self-contained code and configs.  
**Supported model types in this repo: Flan-T5, CodeGPT, DeepSeek**

---

### Model Pipeline Overviews

- **finetune_t5**: [Flan-T5-Small](https://huggingface.co/google/flan-t5-small)  
  Sequence-to-sequence transformer model. t5-small worked much better than t5-base, because my dataset only has 1k entries. 

- **finetune_codegpt**: [CodeGPT-Small](https://huggingface.co/microsoft/CodeGPT-small-py)  
  Language model specialized for code and structured data generation. Decent at generating correct JSON from user input, but needs to train on more data for correctness.

- **finetune_deepseek**: [Deepseek-Coder-1.3b-instruct](https://huggingface.co/deepseek-ai/deepseek-coder-1.3b-instruct)  
  Significantly bigger model, broke my memory lol. But honestly not bad for the few epochs it ran!

Each pipeline is fully independent. Run any by entering its folder and executing pipeline.py.

---

## Quick Start

1. **Clone the Repository**  
   Clone this repository and change into the project directory.

       git clone https://github.com/rosyyang224/finetune_retrieval_model.git
       cd finetune_retrieval_model

2. **Set Up Python Environment**  
   Create and activate a virtual environment, then install all dependencies.

       python3 -m venv finetune-env  
       source finetune-env/bin/activate  
       pip install --upgrade pip  
       pip install -r requirements.txt

3. **Initialize Weights & Biases (wandb)**  
   Login to Weights & Biases before running any pipeline:

       wandb login

   Then enter your API key from https://wandb.ai/authorize when prompted.

4. **Run a Model Pipeline**  
   Choose your desired model folder, change into it, and execute. For example:

       cd finetune_t5  
       python pipeline.py

   You can do the same for other models (e.g., finetune_codegpt, finetune_deepseek). Outputs and logs will be saved according to the pipeline's config.

5. **(Optional) Validate Data**  
   Before training, you can use the provided script to check data format:

       python validate_training_data.py

   This checks that your training/evaluation data in the `data/` folder is properly formatted.

---

## Configuring Model Runs

Each pipeline supports a wide range of configurable training arguments, both for the model/training setup and for wandb experiment tracking.  
You can adjust all core settings in `config.py`.

### Example Key Parameters

- `output_dir`: Directory to store checkpoints and logs.
- `logging_dir`: Directory to store training logs.
- `num_train_epochs`: Total number of training epochs (e.g., 30).
- `learning_rate`: Optimizer learning rate (e.g., 9e-5).
- `weight_decay`: Weight decay regularization (e.g., 0.01).
- `warmup_ratio`: Fraction of steps for LR warmup (e.g., 0.1).
- `lr_scheduler_type`: LR schedule type ("cosine", "linear", etc).
- `logging_steps`: Frequency of logging updates (in steps).
- `save_strategy`: When to save checkpoints (e.g., "steps").
- `save_steps`: Save checkpoint every N steps.
- `save_total_limit`: Max number of checkpoints to keep.
- `eval_strategy`: When to run evaluation ("epoch" or "steps").
- `eval_steps`: Evaluate every N steps.
- `optim`: Optimizer type (e.g., "adamw_torch_fused").
- `max_grad_norm`: Gradient clipping value.
- `per_device_train_batch_size`: Batch size per device for training.
- `per_device_eval_batch_size`: Batch size per device for evaluation.
- `gradient_accumulation_steps`: Gradients to accumulate before optimizer step.
- `group_by_length`: Whether to group batches by input length.
- `predict_with_generate`: For seq2seq models, use `generate()` during eval.
- `generation_max_length`: Max generation length for model outputs.
- `generation_num_beams`: Number of beams for beam search decoding.
- `bf16`, `fp16`, `tf32`: Enable mixed precision if supported by device.
- `gradient_checkpointing`: Enable gradient checkpointing to save memory.
- `torch_compile`: Enable torch 2.0+ model compilation for speed (if supported).
- `metric_for_best_model`: Metric used for model selection (e.g., "eval_loss").
- `load_best_model_at_end`: Whether to reload the best model after training.

#### Example: Device-aware config

Depending on your device and available GPU memory, training batch sizes and precision flags may be set dynamically. You can increase or lower settings depending on what your device can manage.

```python
if device.type == "cuda":
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    if gpu_memory >= 16:
        gpu_args = {
            "per_device_train_batch_size": 16,
            "per_device_eval_batch_size": 24,
            "gradient_accumulation_steps": 1,
        }
    elif gpu_memory >= 8:
        gpu_args = {
            "per_device_train_batch_size": 12,
            "per_device_eval_batch_size": 16,
            "gradient_accumulation_steps": 2,
        }
    else:
        gpu_args = {
            "per_device_train_batch_size": 8,
            "per_device_eval_batch_size": 8,
            "gradient_accumulation_steps": 3,
        }
elif device.type == "mps":
    gpu_args = {
        "per_device_train_batch_size": 4,
        "per_device_eval_batch_size": 8,
        "gradient_accumulation_steps": 4,
        "bf16": False,
        "fp16": False,
        "dataloader_num_workers": 0,
        "dataloader_pin_memory": False,
    }
else:
    gpu_args = {
        "per_device_train_batch_size": 4,
        "per_device_eval_batch_size": 4,
        "gradient_accumulation_steps": 8,
        "bf16": False,
        "fp16": False,
        "dataloader_num_workers": 0,
        "dataloader_pin_memory": False,
    }
```

#### Example: wandb Initialization in Code

In each pipeline, wandb is initialized as follows:

```python
wandb.init(
    project="query-to-filter",  # Do not change; keeps all runs organized under one project.
    name="small_cosine_beams2_smoothing0.0_lr0.9e4_batch4",  # Descriptive run name for easy comparison in the dashboard.
    config={
        "model": MODEL_NAME,                    # The model architecture or checkpoint name.
        "model_class": "PortfolioQueryT5Model", # The Python class or wrapper used.
        "domain": "portfolio_management",       # The problem or business domain.
        "task": "natural_language_to_json",     # Task type for clarity in tracking.
        "data_size": 1150,                      # Training set size (or any relevant stat).
        "optimization_config": OPTIMIZATION_CONFIG  # Hyperparameter/optimization config dict.
    }
)
```

- `project`: Keep fixed for all experiments in this repository.
- `name`: Customize for each run (e.g., to describe key hyperparameters, model variants, etc.).
- `config`: Free-form dictionary for tracking all relevant metadata about your run, model, data, and hyperparameters.

---

## Data Validation

A helper script is provided to check and validate the contents of the `data/` directory:

       python validate_training_data.py

This script will:
- Verify that training, validation, and test datasets exist and are readable.
- Print a summary of dataset statistics and any issues found so that you can add more data in undercovered areas.

Always run this after updating data.

---

## Experiment Tracking with wandb

- All runs, metrics, logs, and configs are automatically tracked on [Weights & Biases (wandb)](https://wandb.ai/).
- After running any pipeline, a direct dashboard link will be printed in your terminal for that run.
- You can view:
  - Training/eval loss curves, learning rate schedules, and generated outputs.
  - Run configs, hyperparameters, and full experiment metadata.
  - Side-by-side comparisons of multiple runs for model and hyperparameter selection.

You can further group, rename, or annotate your runs directly in the wandb web interface for better experiment management.
