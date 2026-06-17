import json
import os
from collections import defaultdict

# In-memory metrics store:
# metrics[company][file][stage] = {
#     "time_seconds": float,
#     "input_tokens": int,
#     "output_tokens": int,
#     "cost": float
# }

_metrics = defaultdict(lambda: defaultdict(dict))


def record_stage(
    company: str,
    filename: str,
    stage: str,
    *,
    time_seconds: float | None = None,
    input_tokens: int | None = None,
    output_tokens: int | None = None,
    cost: float | None = None,
):
    """
    Update metrics for a given (company, file, stage).
    Fields are additive where it makes sense (tokens, time, cost).
    """
    rec = _metrics[company][filename].setdefault(
        stage,
        {"time_seconds": 0.0, "input_tokens": 0, "output_tokens": 0, "cost": 0.0},
    )

    if time_seconds is not None:
        rec["time_seconds"] += float(time_seconds)
    if input_tokens is not None:
        rec["input_tokens"] += int(input_tokens)
    if output_tokens is not None:
        rec["output_tokens"] += int(output_tokens)
    if cost is not None:
        rec["cost"] += float(cost)


def recompute_totals():
    """
    For each (company, file), compute a 'TOTAL' aggregate over all stages,
    and for each company compute a 'COMPANY_TOTAL' over all files.
    """
    company_totals = {}

    for company, files in _metrics.items():
        c_time = c_in = c_out = c_cost = 0.0

        for filename, stages in files.items():
            total_time = total_in = total_out = total_cost = 0.0

            for stage_name, vals in stages.items():
                if stage_name == "TOTAL":
                    continue

                total_time += float(vals.get("time_seconds", 0.0))
                total_in += float(vals.get("input_tokens", 0))
                total_out += float(vals.get("output_tokens", 0))
                total_cost += float(vals.get("cost", 0.0))

            stages["TOTAL"] = {
                "time_seconds": total_time,
                "input_tokens": int(total_in),
                "output_tokens": int(total_out),
                "cost": total_cost,
            }

            c_time += total_time
            c_in += total_in
            c_out += total_out
            c_cost += total_cost

        company_totals[company] = {
            "time_seconds": c_time,
            "input_tokens": int(c_in),
            "output_tokens": int(c_out),
            "cost": c_cost,
        }

    for company, totals in company_totals.items():
        _metrics[company]["COMPANY_TOTAL"] = totals


def save_metrics_json(output_path: str):
    """
    Recompute per-file totals and write metrics JSON to disk.
    """
    recompute_totals()

    data = {
        company: {
            filename: dict(stages)
            for filename, stages in files.items()
        }
        for company, files in _metrics.items()
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
