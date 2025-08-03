"""
Shared utilities for model loading, data preprocessing, and validation
"""

import json
import re
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.model_selection import train_test_split
from datasets import Dataset
import logging

logger = logging.getLogger(__name__)

VALID_FIELDS = {
    "symbol", "cusip", "fxrate", "marketvalueinbccy", "totalmarketvalue",
    "assetclass", "uuid", "costpricesccy", "accrualsymbol", "totalcostinbccy",
    "maturitydate", "assettemplatetype", "marketyield", "accruedcashvaluesccy",
    "marketpricesccy", "marketplinsccy", "forexpl", "accounttype",
    "countryregion", "marketplinbccy", "marketplpercentinsccy", "accrualtype",
    "totalmarketvaluesccy", "totalmarketvalueccy", "ytm", "securitytype",
    "accrualsccy", "fxmarket", "sccy"
}

VALID_OPERATORS = {"eq", "ne", "lt", "lte", "gt", "gte", "in", "not_in"}
VALID_OPERATIONS = {"sum", "avg", "min", "max", "count", "median"}

# Model Loading
def setup_model_and_tokenizer(model_path: str, compile_model: bool = False):
    """Set up model and tokenizer with optimizations"""
    logger.info(f"Loading model from {model_path}")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    if compile_model and hasattr(torch, "compile"):
        try:
            model = torch.compile(model)
            logger.info("Model compiled with PyTorch 2.0")
        except Exception as e:
            logger.warning(f"Could not compile model: {e}")
    
    return model, tokenizer

def create_preprocess_function(tokenizer):
    """Create preprocessing function for causal language modeling"""
    def preprocess_function(examples):
        # Format as conversation/instruction following
        texts = []
        for inp, out in zip(examples['input'], examples['output']):
            text = f"# Convert this query to JSON filter:\n# Query: {inp}\n# JSON:\n{out}{tokenizer.eos_token}"
            texts.append(text)
        
        model_inputs = tokenizer(
            texts,
            max_length=512,
            truncation=True,
            padding="max_length",
            return_tensors=None
        )

        model_inputs["labels"] = model_inputs["input_ids"]

        
        return model_inputs
    
    return preprocess_function

def add_length_column(examples):
    """Add length column for grouping"""
    examples["length"] = [len(inp.split()) + len(out.split()) 
                         for inp, out in zip(examples["input"], examples["output"])]
    return examples

def load_training_data(file_path: str = "../data/training_data.jsonl", 
                      validation_split: bool = True, 
                      validation_size: float = 0.1) -> Tuple[Dataset, Optional[Dataset]]:
    """Load and prepare training data with optional validation split"""
    try:
        data = []
        logger.info("Loading training data...")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                    if 'input' in item and 'output' in item:
                        is_valid, error_msg = validate_training_example(item)
                        if is_valid:
                            data.append(item)
                        else:
                            logger.warning(f"Line {line_num}: {error_msg}")
                except json.JSONDecodeError:
                    logger.warning(f"Line {line_num}: Invalid JSON")
        
        logger.info(f"Loaded {len(data)} valid examples")
        
        if len(data) == 0:
            raise ValueError("No valid training examples found")
        
        if validation_split:
            train_data, val_data = train_test_split(
                data, 
                test_size=validation_size,
                random_state=42,
                shuffle=True
            )
            logger.info(f"Split: {len(train_data)} train, {len(val_data)} validation")
            return Dataset.from_list(train_data), Dataset.from_list(val_data)
        else:
            return Dataset.from_list(data), None
            
    except FileNotFoundError:
        logger.error(f"{file_path} not found!")
        return None, None

# CHANGED: New compute metrics for causal LM
def create_enhanced_compute_metrics_causal(tokenizer, rouge_metric=None):
    """Factory function to create compute_metrics for causal LM"""
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred.predictions, eval_pred.label_ids
        
        # Handle predictions properly
        if len(predictions.shape) == 3:
            predictions = np.argmax(predictions, axis=-1)
        
        # Get vocab size for validation
        vocab_size = len(tokenizer.get_vocab())
        
        # Clean predictions
        valid_token_mask = (predictions >= 0) & (predictions < vocab_size)
        predictions = np.where(valid_token_mask, predictions, tokenizer.pad_token_id)
        
        # For causal LM, we need to extract just the generated part
        # Find where the JSON starts in each sequence
        try:
            decoded_preds = []
            for pred_seq in predictions:
                # Decode the full sequence
                full_text = tokenizer.decode(pred_seq, skip_special_tokens=True)
                
                # Extract just the JSON part after "# JSON:\n"
                if "# JSON:\n" in full_text:
                    json_part = full_text.split("# JSON:\n", 1)[1]
                    # Remove any text after the first complete JSON
                    if json_part.strip():
                        # Try to find the end of the JSON
                        brace_count = 0
                        json_end = -1
                        for i, char in enumerate(json_part):
                            if char == '{':
                                brace_count += 1
                            elif char == '}':
                                brace_count -= 1
                                if brace_count == 0:
                                    json_end = i + 1
                                    break
                        if json_end > 0:
                            json_part = json_part[:json_end]
                    decoded_preds.append(json_part.strip())
                else:
                    decoded_preds.append(full_text.strip())
            
            # For labels, we need to extract the JSON part as well
            decoded_labels = []
            labels_clean = np.where(labels != -100, labels, tokenizer.pad_token_id)
            for label_seq in labels_clean:
                full_text = tokenizer.decode(label_seq, skip_special_tokens=True)
                if "# JSON:\n" in full_text:
                    json_part = full_text.split("# JSON:\n", 1)[1]
                    decoded_labels.append(json_part.strip())
                else:
                    decoded_labels.append(full_text.strip())
                
        except Exception as e:
            logger.error(f"Decoding failed: {e}")
            return {
                "valid_json_ratio": 0.0,
                "avg_schema_score": 0.0,
                "avg_semantic_score": 0.0,
                "decoding_error": 1.0
            }
        
        results = {}
        
        # Custom schema validation metrics
        valid_json_count = 0
        schema_scores = []
        semantic_scores = []
        
        for i, (pred, label) in enumerate(zip(decoded_preds, decoded_labels)):
            try:
                is_valid, schema_score, parsed = validate_json_schema(pred)
                if is_valid:
                    valid_json_count += 1
                    schema_scores.append(schema_score)
                    semantic_scores.append(schema_score)
                else:
                    schema_scores.append(0.0)
                    semantic_scores.append(0.0)
                    
            except Exception as e:
                logger.warning(f"Error processing example {i}: {e}")
                schema_scores.append(0.0)
                semantic_scores.append(0.0)
        
        results.update({
            "valid_json_ratio": valid_json_count / len(decoded_preds) if decoded_preds else 0.0,
            "avg_schema_score": np.mean(schema_scores) if schema_scores else 0.0,
            "avg_semantic_score": np.mean(semantic_scores) if semantic_scores else 0.0,
            "total_examples": len(decoded_preds),
        })
        
        return results
    
    return compute_metrics

# Keep the original for T5 compatibility
def create_enhanced_compute_metrics(tokenizer, rouge_metric=None):
    """Factory function to create compute_metrics with proper tokenizer closure"""
    def compute_metrics(eval_pred):
        # Handle different eval_pred formats - sometimes it has 3 values (predictions, labels, inputs)
        if hasattr(eval_pred, 'predictions') and hasattr(eval_pred, 'label_ids'):
            # NamedTuple format
            predictions = eval_pred.predictions
            labels = eval_pred.label_ids
        elif isinstance(eval_pred, (tuple, list)):
            if len(eval_pred) == 2:
                predictions, labels = eval_pred
            elif len(eval_pred) == 3:
                predictions, labels, inputs = eval_pred
            else:
                logger.error(f"Unexpected eval_pred format with {len(eval_pred)} elements")
                return {"error": f"Invalid eval_pred format: {len(eval_pred)} elements"}
        else:
            logger.error(f"Unknown eval_pred format: {type(eval_pred)}")
            return {"error": "Unknown eval_pred format"}
        
        # Handle predictions properly - they might be logits or token IDs
        if len(predictions.shape) == 3:
            # Predictions are logits, convert to token IDs
            predictions = np.argmax(predictions, axis=-1)
        
        # Ensure predictions are 2D
        if len(predictions.shape) != 2:
            logger.error(f"Unexpected prediction shape: {predictions.shape}")
            return {"error": "Invalid prediction shape"}
        
        # Get vocab size for validation
        vocab_size = len(tokenizer.get_vocab())
        
        # Clean predictions - replace any invalid token IDs with pad token
        valid_token_mask = (predictions >= 0) & (predictions < vocab_size)
        predictions = np.where(valid_token_mask, predictions, tokenizer.pad_token_id)
        
        # Clean labels - replace -100 with pad token for decoding
        labels_clean = np.where(labels != -100, labels, tokenizer.pad_token_id)
        
        try:
            # Decode predictions and labels
            decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
            decoded_labels = tokenizer.batch_decode(labels_clean, skip_special_tokens=True)
            
            # Validate that we got the expected number of decoded sequences
            if len(decoded_preds) != len(decoded_labels):
                logger.warning(f"Mismatch in decoded lengths: {len(decoded_preds)} vs {len(decoded_labels)}")
                min_len = min(len(decoded_preds), len(decoded_labels))
                decoded_preds = decoded_preds[:min_len]
                decoded_labels = decoded_labels[:min_len]
                
        except Exception as e:
            logger.error(f"Decoding failed: {e}")
            logger.error(f"Predictions shape: {predictions.shape}, Labels shape: {labels.shape}")
            logger.error(f"Vocab size: {vocab_size}")
            logger.error(f"Prediction range: {predictions.min()} to {predictions.max()}")
            
            # Return default metrics on failure
            return {
                "rouge1": 0.0,
                "rouge2": 0.0, 
                "rougeL": 0.0,
                "valid_json_ratio": 0.0,
                "avg_schema_score": 0.0,
                "avg_semantic_score": 0.0,
                "decoding_error": 1.0
            }
        
        results = {}
        
        # Standard ROUGE metrics (if available)
        if rouge_metric is not None:
            try:
                rouge_result = rouge_metric.compute(
                    predictions=decoded_preds,
                    references=decoded_labels,
                    use_stemmer=True
                )
                results.update({
                    "rouge1": rouge_result["rouge1"],
                    "rouge2": rouge_result["rouge2"], 
                    "rougeL": rouge_result["rougeL"],
                })
            except Exception as e:
                logger.warning(f"Could not compute ROUGE metrics: {e}")
                results.update({
                    "rouge1": 0.0,
                    "rouge2": 0.0, 
                    "rougeL": 0.0,
                })
        
        # Custom schema validation metrics
        valid_json_count = 0
        schema_scores = []
        semantic_scores = []
        
        for i, (pred, label) in enumerate(zip(decoded_preds, decoded_labels)):
            try:
                is_valid, schema_score, parsed = validate_json_schema(pred)
                if is_valid:
                    valid_json_count += 1
                    schema_scores.append(schema_score)
                    
                    # Try to compute semantic score if we have an input
                    # Note: We don't have access to input here, so using schema score as proxy
                    semantic_scores.append(schema_score)
                else:
                    schema_scores.append(0.0)
                    semantic_scores.append(0.0)
                    
            except Exception as e:
                logger.warning(f"Error processing example {i}: {e}")
                schema_scores.append(0.0)
                semantic_scores.append(0.0)
        
        results.update({
            "valid_json_ratio": valid_json_count / len(decoded_preds) if decoded_preds else 0.0,
            "avg_schema_score": np.mean(schema_scores) if schema_scores else 0.0,
            "avg_semantic_score": np.mean(semantic_scores) if semantic_scores else 0.0,
            "total_examples": len(decoded_preds),
        })
        
        return results
    
    return compute_metrics

def compute_semantic_score(input_text: str, parsed_output: Dict) -> float:
    """Compute semantic accuracy for portfolio queries"""
    query = input_text.lower()
    score = 0.0
    
    # Asset class detection (30% weight)
    asset_mappings = {
        ("bond", "bonds", "fixed income"): "Bond",
        ("stock", "stocks", "equity", "equities", "share", "shares"): "Equity", 
        ("etf", "etfs"): "ETF",
        ("cash",): "Cash"
    }
    
    for keywords, asset_class in asset_mappings.items():
        if any(kw in query for kw in keywords):
            filters = parsed_output.get("filters", [])
            for f in filters:
                if f.get("field") == "assetclass" and f.get("value") == asset_class:
                    score += 0.3
                    break
            break
    
    # Country/region detection (20% weight)
    region_mappings = {
        ("us", "usa", "american", "united states"): "USA",
        ("china", "chinese"): "China", 
        ("europe", "european"): "Europe",
        ("canada", "canadian"): "Canada",
        ("japan", "japanese"): "Japan",
        ("uk", "britain", "british"): "UK"
    }
    
    for keywords, region in region_mappings.items():
        if any(kw in query for kw in keywords):
            filters = parsed_output.get("filters", [])
            for f in filters:
                if f.get("field") == "countryregion" and region.lower() in f.get("value", "").lower():
                    score += 0.2
                    break
            break
    
    # Symbol detection (30% weight)
    symbols = ["apple", "tesla", "microsoft", "google", "amazon", "meta", "nvidia", "salesforce"]
    for symbol in symbols:
        if symbol in query:
            filters = parsed_output.get("filters", [])
            for f in filters:
                if (f.get("field") == "symbol" and 
                    symbol.lower() in f.get("value", "").lower()):
                    score += 0.3
                    break
            break
    
    # Sorting logic (10% weight)
    if any(term in query for term in ["top", "best", "highest", "largest"]):
        sort_list = parsed_output.get("sort", [])
        if any(s.get("order") == "desc" for s in sort_list):
            score += 0.1
    
    if any(term in query for term in ["worst", "smallest", "lowest", "bottom"]):
        sort_list = parsed_output.get("sort", [])
        if any(s.get("order") == "asc" for s in sort_list):
            score += 0.1
    
    # Limit detection (10% weight)
    numbers = re.findall(r'\b(\d+)\b', query)
    if numbers and "limit" in parsed_output:
        try:
            expected = int(numbers[0])
            if parsed_output["limit"] == expected:
                score += 0.1
        except:
            pass
    
    return min(score, 1.0)

def validate_training_example(example: Dict) -> Tuple[bool, str]:
    """Validate a single training example"""
    try:
        if 'input' not in example or 'output' not in example:
            return False, "Missing 'input' or 'output' key"
        
        is_valid, _, _ = validate_json_schema(example['output'])
        if not is_valid:
            return False, "Output is not valid JSON schema"
        
        return True, "Valid"
        
    except Exception as e:
        return False, f"Validation error: {e}"

def validate_json_schema(output_str: str) -> Tuple[bool, float, Dict]:
    """Validate JSON against portfolio query schema with better error handling"""
    try:
        # Clean output string more aggressively
        output_str = output_str.strip()
        
        # Remove common prefix/suffix artifacts from model generation
        if output_str.startswith("Transform this") or output_str.startswith("Query:"):
            # Find the first { and start from there
            start_idx = output_str.find('{')
            if start_idx != -1:
                output_str = output_str[start_idx:]
        
        # Ensure it starts and ends with braces
        if not output_str.startswith('{'):
            output_str = '{' + output_str
        if not output_str.endswith('}'):
            output_str = output_str + '}'
        
        # Try to fix common JSON issues
        output_str = output_str.replace("'", '"')  # Single quotes to double quotes
        output_str = re.sub(r',\s*}', '}', output_str)  # Remove trailing commas
        output_str = re.sub(r',\s*]', ']', output_str)  # Remove trailing commas in arrays
            
        parsed = json.loads(output_str)
        if not isinstance(parsed, dict):
            return False, 0.0, {}
            
        score = 0.0
        max_score = 0.0
        
        # Check valid top-level keys (20% of score)
        valid_top_keys = {"filters", "sort", "limit", "aggregate"}
        if set(parsed.keys()).issubset(valid_top_keys):
            score += 0.2
        max_score += 0.2
        
        # Validate filters (40% of score)
        if "filters" in parsed:
            max_score += 0.4
            filters = parsed["filters"]
            if isinstance(filters, list):
                score += 0.1
                filter_score = 0.0
                for f in filters:
                    if isinstance(f, dict) and all(k in f for k in ["field", "op", "value"]):
                        filter_score += 0.1
                        if f["field"] in VALID_FIELDS and f["op"] in VALID_OPERATORS:
                            filter_score += 0.1
                score += min(filter_score, 0.3)
        
        # Validate sort (20% of score)
        if "sort" in parsed:
            max_score += 0.2
            sort_obj = parsed["sort"]
            if isinstance(sort_obj, list):
                for s in sort_obj:
                    if isinstance(s, dict) and "field" in s and "order" in s:
                        if s["field"] in VALID_FIELDS and s["order"] in ["asc", "desc"]:
                            score += 0.2
                            break
        
        # Validate aggregate (20% of score)
        if "aggregate" in parsed:
            max_score += 0.2
            agg = parsed["aggregate"]
            if isinstance(agg, dict):
                if ("operation" in agg and "field" in agg and 
                    agg["operation"] in VALID_OPERATIONS and agg["field"] in VALID_FIELDS):
                    score += 0.1
                    if "group_by" in agg and agg["group_by"] in VALID_FIELDS:
                        score += 0.1
                    else:
                        score += 0.1
        
        return True, score / max_score if max_score > 0 else 0.0, parsed
        
    except json.JSONDecodeError as e:
        logger.debug(f"JSON decode error: {e} for string: {output_str[:100]}...")
        return False, 0.0, {}
    except Exception as e:
        logger.debug(f"Validation error: {e}")
        return False, 0.0, {}