def _compute_cost(input_tokens, output_tokens, model_cfg):
    """
    Compute detection cost in dollars using model's per-1M token prices.
    """
    input_millions = (input_tokens or 0) / 1_000_000.0
    output_millions = (output_tokens or 0) / 1_000_000.0
    input_costs = model_cfg.get("input_costs", 1.925)
    output_costs = model_cfg.get("output_costs", 15.40)
    return (input_millions * input_costs) + (output_millions * output_costs)

def compute_cost(input_tokens, output_tokens, model_dict):
    """
    Compute cost in dollars using model's per-1M token prices.
    """
    return _compute_cost(input_tokens, output_tokens, model_dict)

# Flagship Model GPT 5.2 (Only Model)

GPT_5_2 = {
    "model_name": "gpt-5.2-2025-12-11-us-data-zone",
    "api_version": "2024-10-21",
    "input_costs": 1.925,
    "output_costs": 15.40,
    "endpoint_env": "GPT_5_2_ENDPOINT",
    "key_env": "GPT_5_2_API_KEY",
}

MODEL_REGISTRY = {
    # Full deployment names
    "gpt-5.2-2025-12-11-us-data-zone": GPT_5_2,
    # Standard short names
    "gpt-5-2": GPT_5_2,
    # Fallback mappings to direct all old requests to GPT_5_2
    "gpt-4.1-2025-04-14-us-data-zone": GPT_5_2,
    "gpt-4.1-mini-2025-04-14-us-data-zone": GPT_5_2,
    "gpt-5-2025-08-07-us-data-zone": GPT_5_2,
    "gpt-5-mini-2025-08-07-us-data-zone": GPT_5_2,
    "gpt-5.1-2025-11-13-us-data-zone": GPT_5_2,
    "gpt-35-turbo": GPT_5_2,
    "gpt-4-1": GPT_5_2,
    "gpt-4-1-mini": GPT_5_2,
    "gpt-5": GPT_5_2,
    "gpt-5-mini": GPT_5_2,
    "gpt-5-1": GPT_5_2,
    "gpt-3.5-turbo": GPT_5_2,
}
