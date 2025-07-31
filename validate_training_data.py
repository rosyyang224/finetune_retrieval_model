import json
from collections import defaultdict

'''
{
  "filters": [
    {
      "field": "string",
      "op": "eq" | "ne" | "lt" | "lte" | "gt" | "gte" | "in" | "not_in",
      "value": "string" | number | array
    }
  ],
  "sort": [
    {
      "field": "string",
      "order": "asc" | "desc"
    }
  ],
  "limit": number,
  "aggregate": {
    "operation": "sum" | "avg" | "min" | "max" | "count" | "median",
    "field": "string",
    "group_by": "string" (optional)
  }
}
'''

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

def validate_example(entry: dict, idx: int) -> str:
    VALID_TOP_KEYS = {"filters", "sort", "limit", "aggregate"}

    try:
        input_text = entry.get("input", "").strip()
        output_text = entry.get("output", "").strip()

        parsed = json.loads(output_text)
        if not isinstance(parsed, dict):
            return f"Line {idx}: Output is not a JSON object"

        unknown_keys = set(parsed.keys()) - VALID_TOP_KEYS
        if unknown_keys:
            return f"Line {idx}: Unknown top-level keys: {unknown_keys}"

        # Validate filters
        if "filters" in parsed:
            filters = parsed["filters"]
            if not isinstance(filters, list):
                return f"Line {idx}: 'filters' must be a list"

            for i, f in enumerate(filters):
                if not isinstance(f, dict):
                    return f"Line {idx}, filter {i}: Must be a dictionary"

                missing = [k for k in ["field", "op", "value"] if k not in f]
                if missing:
                    return f"Line {idx}, filter {i}: Missing keys: {missing}"

                if f["field"] not in VALID_FIELDS:
                    return f"Line {idx}, filter {i}: Invalid field '{f['field']}'"

                if f["op"] not in VALID_OPERATORS:
                    return f"Line {idx}, filter {i}: Invalid operator '{f['op']}'"

                # Validate value types for array operators
                if f["op"] in {"in", "not_in"}:
                    if not isinstance(f["value"], list):
                        return f"Line {idx}, filter {i}: Operator '{f['op']}' requires array value"
                elif f["op"] in {"eq", "ne", "lt", "lte", "gt", "gte"}:
                    if isinstance(f["value"], list):
                        return f"Line {idx}, filter {i}: Operator '{f['op']}' cannot use array value"

        # Validate sort
        if "sort" in parsed:
            sort = parsed["sort"]
            if isinstance(sort, dict):  # support single dict for backward compatibility
                sort = [sort]
            if not isinstance(sort, list):
                return f"Line {idx}: 'sort' must be a list"
            
            for i, s in enumerate(sort):
                if not isinstance(s, dict):
                    return f"Line {idx}, sort {i}: Must be a dictionary"
                if "field" not in s or "order" not in s:
                    return f"Line {idx}, sort {i}: Missing 'field' or 'order'"
                if s["field"] not in VALID_FIELDS:
                    return f"Line {idx}, sort {i}: Invalid field '{s['field']}'"
                if s["order"] not in {"asc", "desc"}:
                    return f"Line {idx}, sort {i}: Invalid order '{s['order']}'"

        # Validate limit
        if "limit" in parsed:
            if not isinstance(parsed["limit"], int) or parsed["limit"] <= 0:
                return f"Line {idx}: 'limit' must be a positive integer"

        # Validate aggregate
        if "aggregate" in parsed:
            agg = parsed["aggregate"]
            if not isinstance(agg, dict):
                return f"Line {idx}: 'aggregate' must be a dictionary"
            
            # Check required fields
            if "operation" not in agg:
                return f"Line {idx}: 'aggregate' missing required 'operation'"
            if "field" not in agg:
                return f"Line {idx}: 'aggregate' missing required 'field'"
            
            # Validate operation
            if agg["operation"] not in VALID_OPERATIONS:
                return f"Line {idx}: Invalid aggregate operation '{agg['operation']}'"
            
            # Validate field
            if agg["field"] not in VALID_FIELDS:
                return f"Line {idx}: Invalid aggregate field '{agg['field']}'"
            
            # Validate optional group_by
            if "group_by" in agg:
                if not isinstance(agg["group_by"], str):
                    return f"Line {idx}: 'group_by' must be a string"
                if agg["group_by"] not in VALID_FIELDS:
                    return f"Line {idx}: Invalid group_by field '{agg['group_by']}'"
            
            # Check for unexpected keys in aggregate
            valid_agg_keys = {"operation", "field", "group_by"}
            unknown_agg_keys = set(agg.keys()) - valid_agg_keys
            if unknown_agg_keys:
                return f"Line {idx}: Unknown aggregate keys: {unknown_agg_keys}"

    except json.JSONDecodeError as e:
        return f"Line {idx}: JSON decode error: {e}"
    except Exception as e:
        return f"Line {idx}: Unexpected error: {e}"

    return ""  # No error


def validate_file(filename: str):
    with open(filename, "r", encoding="utf-8") as f:
        lines = f.readlines()

    errors = []
    field_coverage = defaultdict(int)
    operator_coverage = defaultdict(int)
    top_key_coverage = defaultdict(int)
    operation_coverage = defaultdict(int)
    group_by_coverage = defaultdict(int)
    sort_field_coverage = defaultdict(int)
    aggregate_field_coverage = defaultdict(int)

    for idx, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError as e:
            errors.append(f"Line {idx + 1}: JSON decode error: {e}")
            continue

        error = validate_example(data, idx + 1)
        if error:
            errors.append(error)
            continue

        try:
            parsed = json.loads(data["output"])
            
            # Track top-level keys
            for key in parsed:
                top_key_coverage[key] += 1
            
            # Track filter usage
            for f in parsed.get("filters", []):
                field_coverage[f["field"]] += 1
                operator_coverage[f["op"]] += 1
            
            # Track sort usage
            if "sort" in parsed:
                sort_list = parsed["sort"]
                if isinstance(sort_list, dict):
                    sort_list = [sort_list]
                for s in sort_list:
                    sort_field_coverage[s["field"]] += 1
            
            # Track aggregate usage
            if "aggregate" in parsed:
                agg = parsed["aggregate"]
                operation_coverage[agg["operation"]] += 1
                aggregate_field_coverage[agg["field"]] += 1
                if "group_by" in agg:
                    group_by_coverage[agg["group_by"]] += 1
                    
        except Exception as e:
            errors.append(f"Line {idx + 1}: Error processing coverage: {e}")
            continue

    # Print results
    if errors:
        print("Errors found in training data:")
        for e in errors:
            print("-", e)
        print(f"\nTotal errors: {len(errors)}")
    else:
        print("All training examples are valid!")

    print(f"\nTotal examples processed: {len([l for l in lines if l.strip()])}")

    print("\n=== Top-Level Key Coverage ===")
    for key, count in sorted(top_key_coverage.items(), key=lambda x: x[1], reverse=True):
        print(f"{key}: {count} uses")

    print("\n=== Filter Field Usage Coverage ===")
    for field, count in sorted(field_coverage.items(), key=lambda x: x[1], reverse=True):
        print(f"{field}: {count} uses")

    print("\n=== Operator Usage Coverage ===")
    for op, count in sorted(operator_coverage.items(), key=lambda x: x[1], reverse=True):
        print(f"{op}: {count} uses")

    print("\n=== Sort Field Usage Coverage ===")
    for field, count in sorted(sort_field_coverage.items(), key=lambda x: x[1], reverse=True):
        print(f"{field}: {count} uses")

    print("\n=== Aggregate Operation Coverage ===")
    for op, count in sorted(operation_coverage.items(), key=lambda x: x[1], reverse=True):
        print(f"{op}: {count} uses")

    print("\n=== Aggregate Field Coverage ===")
    for field, count in sorted(aggregate_field_coverage.items(), key=lambda x: x[1], reverse=True):
        print(f"{field}: {count} uses")

    print("\n=== Group By Field Coverage ===")
    for field, count in sorted(group_by_coverage.items(), key=lambda x: x[1], reverse=True):
        print(f"{field}: {count} uses")

    # Show unused fields
    used_fields = set(field_coverage.keys()) | set(sort_field_coverage.keys()) | set(aggregate_field_coverage.keys()) | set(group_by_coverage.keys())
    unused_fields = VALID_FIELDS - used_fields
    if unused_fields:
        print(f"\n=== Unused Fields ({len(unused_fields)}) ===")
        for field in sorted(unused_fields):
            print(f"- {field}")

    # Show unused operators
    unused_operators = VALID_OPERATORS - set(operator_coverage.keys())
    if unused_operators:
        print(f"\n=== Unused Operators ({len(unused_operators)}) ===")
        for op in sorted(unused_operators):
            print(f"- {op}")

    # Show unused operations
    unused_operations = VALID_OPERATIONS - set(operation_coverage.keys())
    if unused_operations:
        print(f"\n=== Unused Aggregate Operations ({len(unused_operations)}) ===")
        for op in sorted(unused_operations):
            print(f"- {op}")


if __name__ == "__main__":
    validate_file("data/training_data.jsonl")