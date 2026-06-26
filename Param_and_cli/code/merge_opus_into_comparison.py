#!/usr/bin/env python3
"""
Merge Claude Opus 4.8's scored results with Claude Sonnet 4.6 and the existing
Ollama model results into a single comparison table / xlsx, in the same shape as
resultats_full*.xlsx. Extends merge_claude_into_comparison.py to add the opus dir.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SOURCE_DIRS = [
    ROOT / "Param_and_cli" / "results" / "list models",
    ROOT / "Param_and_cli" / "results" / "claude",            # claude-sonnet-4.6
    ROOT / "Param_and_cli" / "results" / "claude-opus-4.8",   # claude-opus-4.8
]
OUT_XLSX = ROOT / "Param_and_cli" / "results" / "claude-opus-4.8" / "resultats_full_with_opus.xlsx"

CATEGORIES = ["beginners", "expert vocab", "missingparam", "normal vocab",
              "french", "portuguese", "spanish", "short", "long"]


def main():
    all_data = []

    for query in CATEGORIES:
        for source_dir in SOURCE_DIRS:
            f = source_dir / f"results_{query}.json"
            if not f.exists():
                continue
            model_results = json.loads(f.read_text())
            for result in model_results:
                model = result.get("model", "Unknown")
                latency = result.get("latency_stats", {}) or {}
                metrics = {
                    "total_tests": int(result.get("total_tests")),
                    "avg_tool_selection": np.round(result.get("avg_tool_selection_accuracy"), 3),
                    "combined_perfect": int(result.get("combined_perfect")),
                    "param_extraction_perfect": int(result.get("parameter_extraction_perfect")),
                    "avg_param_accuracy": np.round(result.get("avg_parameter_accuracy"), 3),
                    "latency_mean": np.round(latency.get("mean_s"), 3) if latency.get("mean_s") else "N/A",
                }
                for metric_name, value in metrics.items():
                    all_data.append({"json_file": query, "metric": metric_name, "model": model, "value": value})

    df_raw = pd.DataFrame(all_data)
    df = df_raw.pivot_table(index=["json_file", "metric"], columns="model", values="value", aggfunc="first")

    final_rows = []
    for json_name in CATEGORIES:
        if json_name not in df.index.get_level_values(0):
            continue
        group = df.loc[json_name]
        group.index = pd.MultiIndex.from_product([[json_name], group.index])
        final_rows.append(group)
        empty_index = pd.MultiIndex.from_tuples([(" ", " ")], names=["json_file", "metric"])
        final_rows.append(pd.DataFrame(index=empty_index, columns=df.columns))

    df_final = pd.concat(final_rows)
    df_final.to_excel(OUT_XLSX)
    print(f"Wrote {OUT_XLSX}\n")

    # Console summary: overall avg per model across all categories
    overall = df_raw[df_raw["metric"].isin(["avg_tool_selection", "avg_param_accuracy"])]
    pivot_overall = overall.pivot_table(index="model", columns="metric", values="value", aggfunc="mean")
    pivot_overall = pivot_overall.sort_values("avg_param_accuracy", ascending=False)
    print("Overall average across the 9 query categories (per model):")
    print(pivot_overall.round(3).to_string())


if __name__ == "__main__":
    main()
