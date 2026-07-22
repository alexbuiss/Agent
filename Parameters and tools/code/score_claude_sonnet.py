#!/usr/bin/env python3
"""
Score Claude Sonnet 4.6's own tool-selection + parameter-extraction predictions
against the existing benchmark query files, using the exact same conversion
and comparison logic as compare_all_models.py so the numbers are comparable
to the other (Ollama) models already in Param_and_cli/results/.

Inputs:
  - Param_and_cli/queries/<file>.json                      (ground truth)
  - Param_and_cli/results/claude/raw_predictions/<key>.json (Claude's raw extraction, index-aligned)

Outputs:
  - Param_and_cli/results/claude/results_<file>.json   (same schema as "list models"/"med models" results)
  - Param_and_cli/results/claude/per_query_<file>.csv   (per-query tool+param predicted vs expected)
  - Param_and_cli/results/claude/summary.csv            (one row per category + overall)
"""

import json
import csv
import statistics
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent))
from parameter_extraction_improved import ImprovedParameterExtractor

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "manifest.yaml"
QUERIES_DIR = ROOT / "Param_and_cli" / "queries"
PRED_DIR = ROOT / "Param_and_cli" / "results" / "claude" / "raw_predictions"
OUT_DIR = ROOT / "Param_and_cli" / "results" / "claude"

MODEL_NAME = "claude-sonnet-4.6"

CATEGORIES = [
    ("short", "short"),
    ("long", "long"),
    ("normal vocab", "normal_vocab"),
    ("beginners", "beginners"),
    ("expert vocab", "expert_vocab"),
    ("missingparam", "missingparam"),
    ("french", "french"),
    ("spanish", "spanish"),
    ("portuguese", "portuguese"),
]


def compare_parameters(extracted, expected):
    """Identical logic to compare_all_models.ComprehensiveModelComparator.compare_parameters"""
    errors = []
    correct_count = 0
    total_expected = len(expected)

    for param_name, expected_value in expected.items():
        if param_name not in extracted:
            errors.append(f"Missing: {param_name}")
            continue
        extracted_value = extracted[param_name]
        if _values_match(extracted_value, expected_value):
            correct_count += 1
        else:
            errors.append(f"'{param_name}': expected {repr(expected_value)[:30]}, got {repr(extracted_value)[:30]}")

    for param_name in extracted:
        if param_name not in expected:
            errors.append(f"Extra: {param_name}")

    accuracy = correct_count / total_expected if total_expected > 0 else 0.0
    is_perfect = len(errors) == 0
    return is_perfect, accuracy, errors


def _values_match(extracted, expected) -> bool:
    if extracted == expected:
        return True
    try:
        if isinstance(expected, str) and isinstance(extracted, str):
            return expected.strip() == extracted.strip()
        if isinstance(expected, (list, tuple)) and isinstance(extracted, (list, tuple)):
            return set(str(x) for x in expected) == set(str(x) for x in extracted)
        if isinstance(expected, bool):
            if isinstance(extracted, bool):
                return extracted == expected
            if isinstance(extracted, str):
                return str(extracted).lower() == str(expected).lower()
        if isinstance(expected, (int, float)):
            return float(extracted) == float(expected)
        return str(extracted).lower() == str(expected).lower()
    except (ValueError, TypeError):
        return False


def score_category(query_file, pred_key, manifest, extractor):
    queries = json.loads((QUERIES_DIR / f"{query_file}.json").read_text())
    preds = json.loads((PRED_DIR / f"{pred_key}.json").read_text())
    assert len(queries) == len(preds), f"{query_file}: {len(queries)} queries vs {len(preds)} predictions"

    scripts = {s["name"]: s for s in manifest.get("scripts", [])}

    results = {
        "model": MODEL_NAME,
        "total_tests": len(queries),
        "tool_selection_correct": 0,
        "tool_selection_total": 0,
        "parameter_extraction_perfect": 0,
        "parameter_extraction_total": 0,
        "combined_perfect": 0,
        "latencies": [],
        "test_details": [],
        "errors": [],
    }

    tool_accs, param_accs, combined_accs = [], [], []
    per_query_rows = []

    for i, (case, pred) in enumerate(zip(queries, preds), 1):
        query = case.get("query", "")
        expected_tool = case.get("expected_tool", "")
        expected_params = case.get("expected_params", {})

        predicted_tool = pred.get("tool", "")
        raw_params = pred.get("raw_params", {})

        tool_correct = predicted_tool == expected_tool
        results["tool_selection_total"] += 1
        if tool_correct:
            results["tool_selection_correct"] += 1

        converted_params = {}
        param_accuracy = 0.0
        is_perfect = False
        param_errors = []

        if tool_correct:
            tool_spec = scripts.get(expected_tool)
            if tool_spec:
                converted_params, conv_errors = extractor.convert_types(raw_params, tool_spec)
                is_perfect, param_accuracy, param_errors = compare_parameters(converted_params, expected_params)
                results["parameter_extraction_total"] += 1
                if is_perfect:
                    results["parameter_extraction_perfect"] += 1
                    results["combined_perfect"] += 1

        tool_accs.append(1.0 if tool_correct else 0.0)
        param_accs.append(param_accuracy)
        combined_accs.append(1.0 if is_perfect else param_accuracy)

        results["test_details"].append({
            "index": i,
            "query": query[:80],
            "expected_tool": expected_tool,
            "selected_tool": predicted_tool,
            "tool_correct": tool_correct,
            "param_accuracy": param_accuracy,
            "is_perfect": is_perfect,
        })

        per_query_rows.append({
            "category": query_file,
            "index": i,
            "query": query,
            "expected_tool": expected_tool,
            "predicted_tool": predicted_tool,
            "tool_correct": tool_correct,
            "expected_params": json.dumps(expected_params, ensure_ascii=False),
            "predicted_params_raw": json.dumps(raw_params, ensure_ascii=False),
            "predicted_params_converted": json.dumps(converted_params, ensure_ascii=False, default=str),
            "param_accuracy": round(param_accuracy, 3),
            "params_perfect": is_perfect,
            "param_errors": "; ".join(param_errors),
        })

    results["avg_tool_selection_accuracy"] = sum(tool_accs) / len(tool_accs) if tool_accs else 0.0
    results["avg_parameter_accuracy"] = sum(param_accs) / len(param_accs) if param_accs else 0.0
    results["avg_combined_accuracy"] = sum(combined_accs) / len(combined_accs) if combined_accs else 0.0
    results["latency_stats"] = {"mean_s": 0.0, "median_s": 0.0, "min_s": 0.0, "max_s": 0.0}

    return results, per_query_rows


def tool_selection_f1(all_rows):
    """Macro precision/recall/F1 over tool classes, using all categories combined."""
    classes = sorted(set(r["expected_tool"] for r in all_rows) | set(r["predicted_tool"] for r in all_rows if r["predicted_tool"]))
    per_class = {}
    for c in classes:
        tp = sum(1 for r in all_rows if r["predicted_tool"] == c and r["expected_tool"] == c)
        fp = sum(1 for r in all_rows if r["predicted_tool"] == c and r["expected_tool"] != c)
        fn = sum(1 for r in all_rows if r["predicted_tool"] != c and r["expected_tool"] == c)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        per_class[c] = {"precision": precision, "recall": recall, "f1": f1, "support": tp + fn}

    supported = [c for c in per_class if per_class[c]["support"] > 0]
    macro_p = statistics.mean(per_class[c]["precision"] for c in supported) if supported else 0.0
    macro_r = statistics.mean(per_class[c]["recall"] for c in supported) if supported else 0.0
    macro_f1 = statistics.mean(per_class[c]["f1"] for c in supported) if supported else 0.0

    return {"macro_precision": macro_p, "macro_recall": macro_r, "macro_f1": macro_f1, "per_class": per_class}


def main():
    manifest = yaml.safe_load(MANIFEST.read_text())
    extractor = ImprovedParameterExtractor(str(MANIFEST))

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_rows = []
    summary_rows = []

    for query_file, pred_key in CATEGORIES:
        results, rows = score_category(query_file, pred_key, manifest, extractor)
        all_rows.extend(rows)

        out_json = OUT_DIR / f"results_{query_file}.json"
        out_json.write_text(json.dumps([results], indent=2, ensure_ascii=False))

        csv_path = OUT_DIR / f"per_query_{pred_key}.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

        summary_rows.append({
            "category": query_file,
            "total_tests": results["total_tests"],
            "avg_tool_selection_accuracy": round(results["avg_tool_selection_accuracy"], 3),
            "avg_parameter_accuracy": round(results["avg_parameter_accuracy"], 3),
            "avg_combined_accuracy": round(results["avg_combined_accuracy"], 3),
            "combined_perfect": results["combined_perfect"],
            "parameter_extraction_perfect": results["parameter_extraction_perfect"],
        })

        print(f"{query_file:>14} | tool_sel={results['avg_tool_selection_accuracy']*100:5.1f}% | "
              f"param_acc={results['avg_parameter_accuracy']*100:5.1f}% | "
              f"combined={results['avg_combined_accuracy']*100:5.1f}% | "
              f"perfect={results['combined_perfect']}/{results['total_tests']}")

    # Overall combined CSV with all queries
    all_csv_path = OUT_DIR / "per_query_ALL.csv"
    with open(all_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)

    # Overall summary stats (micro-average across all queries)
    n = len(all_rows)
    tool_acc = sum(1 for r in all_rows if r["tool_correct"]) / n
    param_acc = sum(r["param_accuracy"] for r in all_rows) / n
    combined_perfect = sum(1 for r in all_rows if r["params_perfect"]) / n

    f1_stats = tool_selection_f1(all_rows)

    summary_rows.append({
        "category": "OVERALL",
        "total_tests": n,
        "avg_tool_selection_accuracy": round(tool_acc, 3),
        "avg_parameter_accuracy": round(param_acc, 3),
        "avg_combined_accuracy": round(param_acc, 3),
        "combined_perfect": sum(1 for r in all_rows if r["params_perfect"]),
        "parameter_extraction_perfect": sum(1 for r in all_rows if r["params_perfect"]),
    })

    summary_csv = OUT_DIR / "summary.csv"
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    f1_path = OUT_DIR / "tool_selection_f1.json"
    f1_path.write_text(json.dumps(f1_stats, indent=2))

    print("\n" + "=" * 90)
    print(f"OVERALL ({n} queries): tool_sel={tool_acc*100:.1f}% | param_acc={param_acc*100:.1f}% | "
          f"perfect={combined_perfect*100:.1f}%")
    print(f"Tool-selection macro F1={f1_stats['macro_f1']:.3f} "
          f"(precision={f1_stats['macro_precision']:.3f}, recall={f1_stats['macro_recall']:.3f})")
    print(f"\nWrote: {summary_csv}, {all_csv_path}, {f1_path}")
    print(f"Per-category results: {OUT_DIR}/results_<category>.json, per_query_<category>.csv")


if __name__ == "__main__":
    main()
