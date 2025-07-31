import json
from datasets import Dataset
import logging

def load_json_dataset(jsonl_path):
    """
    Load JSONL dataset with better error handling and validation
    """
    data = []
    skipped_lines = 0
    
    try:
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                    
                try:
                    item = json.loads(line)
                    
                    # Validate required fields
                    if not isinstance(item, dict):
                        logging.warning(f"Line {line_num}: Item is not a dictionary")
                        skipped_lines += 1
                        continue
                        
                    if 'input' not in item or 'output' not in item:
                        logging.warning(f"Line {line_num}: Missing 'input' or 'output' field")
                        skipped_lines += 1
                        continue
                    
                    # Validate output is valid JSON
                    try:
                        json.loads(item['output'])
                    except json.JSONDecodeError:
                        logging.warning(f"Line {line_num}: Output field contains invalid JSON")
                        skipped_lines += 1
                        continue
                    
                    data.append(item)
                    
                except json.JSONDecodeError as e:
                    logging.warning(f"Line {line_num}: JSON decode error - {e}")
                    skipped_lines += 1
                    continue
                    
    except FileNotFoundError:
        raise FileNotFoundError(f"Training data file not found: {jsonl_path}")
    
    if skipped_lines > 0:
        logging.info(f"Loaded {len(data)} valid samples, skipped {skipped_lines} invalid lines")
    else:
        logging.info(f"Successfully loaded {len(data)} samples")
    
    if len(data) == 0:
        raise ValueError("No valid training samples found")
    
    return data

def validate_dataset_format(data):
    """
    Additional validation to ensure dataset quality
    """
    issues = []
    
    for i, item in enumerate(data):
        # Check input length
        if len(item['input'].split()) < 2:
            issues.append(f"Sample {i}: Input too short")
        
        # Check output JSON structure
        try:
            output_dict = json.loads(item['output'])
            if not isinstance(output_dict, dict):
                issues.append(f"Sample {i}: Output is not a JSON object")
        except:
            issues.append(f"Sample {i}: Invalid JSON in output")
    
    return issues