from transformers import T5ForConditionalGeneration, T5Tokenizer
import json

# === MODEL LOADING ===
model_name = "google/flan-t5-base"  # or flan-t5-small
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)
model.eval()

# === TEST QUERIES ===
test_queries = [
    "show me US equities over 10k",
    "list chinese bonds with negative return",
    "what are my european ETFs",
    "what are my top gainers",
    "show me bonds under 5000",
    "what are my US cash positions with return above 2",
    "show me all equities below 3000",
    "list ETFs with return below 1",
    "find Canadian stocks with return greater than 5%",
    "show me my Tesla shares"
]

# === PROMPT TEMPLATE ===
prompt_prefix = """
You are an assistant that converts natural language queries into structured JSON filters.

The JSON must follow this schema:
{
  "filters": [{ "field": str, "op": str, "value": any }],
  "sort": { "field": str, "order": "asc" | "desc" },
  "limit": int
}

Only include valid fields: "symbol", "assetclass", "countryregion", "return", "marketvalueinbccy".
Only use valid ops: "eq", "ne", "gt", "gte", "lt", "lte", "in", "not_in".

Examples:

Query: show me my top 5 gainers  
Output: {"filters":[{"field":"return","op":"gt","value":0}],"sort":{"field":"return","order":"desc"},"limit":5}

Query: Apple positions  
Output: {"filters":[{"field":"symbol","op":"eq","value":"Apple"}]}

Query: bonds in my portfolio  
Output: {"filters":[{"field":"assetclass","op":"eq","value":"Bond"}]}

"""

# === SCHEMA VALIDATION LOGIC ===

VALID_KEYS = {
    "symbol": str,
    "assetclass": str,
    "countryregion": str,
    "min_return": float,
    "max_return": float,
    "min_marketvalueinbccy": float,
    "max_marketvalueinbccy": float,
}

def validate_output(output_str: str):
    try:
        output_str = output_str.strip()
        if not output_str.startswith("{"):
            output_str = "{" + output_str
        if not output_str.endswith("}"):
            output_str += "}"
        output_dict = json.loads(output_str.replace("'", "\""))

        invalid_keys = []
        wrong_types = []

        for key, value in output_dict.items():
            if key not in VALID_KEYS:
                invalid_keys.append(key)
            else:
                expected_type = VALID_KEYS[key]
                if expected_type == float and not isinstance(value, (int, float)):
                    try:
                        float(value)  # allow floatable strings
                    except:
                        wrong_types.append((key, value))
                elif expected_type == str and not isinstance(value, str):
                    wrong_types.append((key, value))

        return {
            "valid": len(invalid_keys) == 0 and len(wrong_types) == 0,
            "invalid_keys": invalid_keys,
            "wrong_types": wrong_types
        }
    except Exception as e:
        return {"valid": False, "error": str(e)}

# === INFERENCE + VALIDATION LOOP ===

for i, query in enumerate(test_queries):
    print(f"\n=== Test {i+1}: {query}")

    full_prompt = prompt_prefix + f"Query: {query}\nOutput:"
    inputs = tokenizer(full_prompt, return_tensors="pt", truncation=True, padding=True)

    outputs = model.generate(
        **inputs,
        max_length=128,
        num_beams=4,
        early_stopping=True,
        repetition_penalty=1.2,
        do_sample=False
    )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("Model Output:", decoded)

    validation = validate_output(decoded)
    if validation["valid"]:
        print("Output is valid.")
    else:
        print("Output failed validation.")
        if "invalid_keys" in validation and validation["invalid_keys"]:
            print("Invalid keys:", validation["invalid_keys"])
        if "wrong_types" in validation and validation["wrong_types"]:
            print("Type mismatches:", validation["wrong_types"])
        if "error" in validation:
            print("JSON error:", validation["error"])
