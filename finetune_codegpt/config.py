import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys

MODEL_NAME = "microsoft/CodeGPT-small-py"
OUTPUT_DIR = "./codegpt_outputs"
PYTHON_VERSION_COMPATIBLE = sys.version_info < (3, 12)

def create_model():
    """Create CodeGPT model from base model"""
    try:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            use_safetensors=True
        )
    except:
        import warnings
        warnings.filterwarnings("ignore")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
    return model

def get_optimized_training_args():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    base_args = {
        "output_dir": OUTPUT_DIR,
        "logging_dir": f"{OUTPUT_DIR}/logs",
        "push_to_hub": False,
        "remove_unused_columns": True,
        "save_safetensors": True,
        "num_train_epochs": 30,
        "learning_rate": 9e-5,
        "weight_decay": 0.01,
        "warmup_ratio": 0.1,
        "lr_scheduler_type": "cosine",
        "logging_steps": 20,
        "save_strategy": "steps",
        "save_steps": 100,
        "save_total_limit": 5,
        "eval_strategy": "steps",
        "eval_steps": 100,
        "optim": "adamw_torch_fused" if device.type == "cuda" else "adamw_torch",
        "max_grad_norm": 0.5,
        "adam_beta1": 0.9,
        "adam_beta2": 0.999,
        "adam_epsilon": 1e-8,
        "dataloader_pin_memory": True,
        "dataloader_num_workers": 4,
        "label_smoothing_factor": 0.0,
        "bf16": device.type == "cuda" and torch.cuda.is_bf16_supported(),
        "fp16": device.type == "cuda" and not torch.cuda.is_bf16_supported(),
        "tf32": device.type == "cuda",
        "gradient_checkpointing": True,
        "torch_compile": PYTHON_VERSION_COMPATIBLE,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "load_best_model_at_end": True,
    }

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

    return {**base_args, **gpu_args}

OPTIMIZATION_CONFIG = {
    "use_validation_split": True,
    "validation_size": 0.1,
    "use_length_grouping": True,
    "compile_model": PYTHON_VERSION_COMPATIBLE,
}