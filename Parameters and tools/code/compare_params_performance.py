#!/usr/bin/env python3
"""
Advanced Performance Comparison Tool
Compares LLM models on CLI parameter extraction accuracy.
The model is considered accurate ONLY if it extracts ALL parameters correctly.
"""

import json
import subprocess
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple
import statistics
from datetime import datetime

class AdvancedPerformanceComparison:
    def __init__(self, queries_file):
        self.queries_file = queries_file
        self.results = {}
        self.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Load test queries with expected parameters
        with open(queries_file, "r") as f:
            self.test_cases = json.load(f)
        
        print(f"✓ Loaded {len(self.test_cases)} test cases with parameters")
    
    def extract_params_from_output(self, output_json: Dict) -> Dict:
        """Extract parameters from agent_router output."""
        params = {}
        
        # Try different possible keys where parameters might be stored
        if "extracted_parameters" in output_json:
            params = output_json["extracted_parameters"]
        elif "parameters" in output_json:
            params = output_json["parameters"]
        elif "params" in output_json:
            params = output_json["params"]
        
        return params
    
    def compare_parameters(self, extracted: Dict, expected: Dict) -> Tuple[bool, float, List[str]]:
        """
        Compare extracted parameters with expected ones.
        
        Returns:
            - is_match (bool): True only if ALL parameters match
            - confidence (float): Percentage of correct parameters (0-1)
            - mismatches (List[str]): List of mismatched parameters
        """
        mismatches = []
        matches = 0
        
        # Check each expected parameter
        for param_name, expected_value in expected.items():
            if param_name not in extracted:
                mismatches.append(f"MISSING: {param_name}")
            else:
                extracted_value = extracted[param_name]
                # Normalize for comparison
                expected_str = str(expected_value).strip().lower()
                extracted_str = str(extracted_value).strip().lower()
                
                if expected_str == extracted_str:
                    matches += 1
                else:
                    mismatches.append(f"WRONG: {param_name} (expected: {expected_value}, got: {extracted_value})")
        
        # Check for extra parameters (not expected)
        for param_name in extracted:
            if param_name not in expected:
                mismatches.append(f"EXTRA: {param_name}")
        
        total_params = len(expected)
        confidence = matches / total_params if total_params > 0 else 0
        is_match = len(mismatches) == 0  # ALL parameters must match
        
        return is_match, confidence, mismatches
    
    def test_configuration(self, model: str, label: str = None, dry_run: bool = True) -> Dict:
        """Test router with specific model."""
        
        if label is None:
            label = model
        
        print(f"\n{'='*70}")
        print(f"Testing: {label}")
        print(f"{'='*70}")
        
        # Set environment
        env = os.environ.copy()
        env["ROUTER_MODEL"] = model
        
        config_results = {
            "model": model,
            "label": label,
            "timestamp": self.timestamp,
            "total_queries": len(self.test_cases),
            "results": [],
            "latencies": [],
            "tool_accuracies": [],
            "param_accuracies": [],
            "perfect_matches": 0,  # All params correct
            "partial_matches": 0,  # Some params correct
            "full_misses": 0,      # Tool wrong or no params extracted
            "errors": 0,
            "param_accuracy_scores": []  # Confidence for each query
        }
        
        for i, test_case in enumerate(self.test_cases):
            query = test_case["query"]
            expected_tool = test_case["expected_tool"]
            expected_params = test_case["expected_params"]
            
            progress = f"[{i+1}/{len(self.test_cases)}]"
            print(f"{progress} Testing: {query[:60]}...", end=" ", flush=True)
            
            try:
                # Run agent_router
                cmd = ["python", "Param_and_cli/code/agent_router.py", query, "--benchmark", "--dry-run"]
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=600,
                    env=env
                )
                
                if result.returncode not in [0]:
                    print(f"✗ ERROR (exit code {result.returncode})")
                    config_results["errors"] += 1
                    config_results["results"].append({
                        "query": query,
                        "status": "error",
                        "error": result.stderr[:200]
                    })
                    continue
                
                # Parse output
                try:
                    output = json.loads(result.stdout)
                except json.JSONDecodeError:
                    print(f"✗ JSON PARSE ERROR")
                    config_results["errors"] += 1
                    config_results["results"].append({
                        "query": query,
                        "status": "parse_error",
                        "stdout": result.stdout[:200]
                    })
                    continue
                
                # Check tool selection
                selected_tool = output.get("tool", None)
                tool_correct = selected_tool == expected_tool
                
                # Extract and compare parameters
                extracted_params = self.extract_params_from_output(output)
                params_match, param_confidence, param_mismatches = self.compare_parameters(
                    extracted_params, expected_params
                )
                
                # Record result
                query_result = {
                    "query": query[:80],
                    "expected_tool": expected_tool,
                    "selected_tool": selected_tool,
                    "tool_correct": tool_correct,
                    "expected_params": expected_params,
                    "extracted_params": extracted_params,
                    "params_match": params_match,
                    "param_confidence": param_confidence,
                    "param_mismatches": param_mismatches,
                    "processing_time": output.get("processing_time", 0)
                }
                
                config_results["results"].append(query_result)
                
                # Update statistics
                if output.get("processing_time"):
                    config_results["latencies"].append(output["processing_time"])
                
                config_results["param_accuracy_scores"].append(param_confidence)
                
                if tool_correct and params_match:
                    # PERFECT: tool correct AND all params correct
                    print("✓✓ PERFECT")
                    config_results["perfect_matches"] += 1
                elif tool_correct and param_confidence > 0:
                    # PARTIAL: tool correct but some params wrong
                    print(f"✓~ PARTIAL ({param_confidence*100:.0f}%)")
                    config_results["partial_matches"] += 1
                else:
                    # FULL MISS: tool wrong or no params
                    print("✗✗ MISS")
                    config_results["full_misses"] += 1
                
            except subprocess.TimeoutExpired:
                print("✗ TIMEOUT")
                config_results["errors"] += 1
                config_results["results"].append({
                    "query": query,
                    "status": "timeout"
                })
            except Exception as e:
                print(f"✗ EXCEPTION: {str(e)[:30]}")
                config_results["errors"] += 1
                config_results["results"].append({
                    "query": query,
                    "status": "exception",
                    "error": str(e)[:100]
                })
        
        # Calculate aggregate statistics
        config_results["statistics"] = self._calculate_statistics(config_results)
        
        return config_results
    
    def _calculate_statistics(self, config_results: Dict) -> Dict:
        """Calculate aggregate statistics from results."""
        stats = {
            "total_queries": len(self.test_cases),
            "perfect_accuracy": f"{(config_results['perfect_matches'] / len(self.test_cases) * 100):.1f}%",
            "partial_accuracy": f"{(config_results['partial_matches'] / len(self.test_cases) * 100):.1f}%",
            "miss_rate": f"{(config_results['full_misses'] / len(self.test_cases) * 100):.1f}%",
            "error_rate": f"{(config_results['errors'] / len(self.test_cases) * 100):.1f}%",
            "avg_param_confidence": f"{(sum(config_results['param_accuracy_scores']) / len(config_results['param_accuracy_scores']) * 100):.1f}%" if config_results['param_accuracy_scores'] else "N/A"
        }
        
        if config_results["latencies"]:
            stats["latency_ms"] = {
                "min": f"{min(config_results['latencies'])*1000:.0f}",
                "max": f"{max(config_results['latencies'])*1000:.0f}",
                "mean": f"{statistics.mean(config_results['latencies'])*1000:.0f}",
                "median": f"{statistics.median(config_results['latencies'])*1000:.0f}"
            }
        
        return stats
    
    def run_comparison(self, models: List[Tuple[str, str]]) -> Dict:
        """
        Run comparison across multiple models.
        
        Args:
            models: List of (model_name, label) tuples
        """
        comparison_results = {
            "comparison_timestamp": self.timestamp,
            "queries_file": self.queries_file,
            "total_queries_tested": len(self.test_cases),
            "models": {}
        }
        
        for model_name, label in models:
            config_results = self.test_configuration(model_name, label)
            comparison_results["models"][label] = config_results
        
        return comparison_results
    
    def save_results(self, results: Dict, output_file: str = None) -> str:
        """Save results to JSON file."""
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"Param_and_cli/results/performance_comparison.json"
        
        output_path = Path(output_file)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"\n✓ Results saved to: {output_file}")
        return output_file
    
    def print_comparison_table(self, results: Dict):
        """Print formatted comparison table."""
        print(f"\n{'='*100}")
        print("COMPARISON SUMMARY")
        print(f"{'='*100}\n")
        
        # Prepare table data
        headers = ["Model", "Perfect", "Partial", "Misses", "Errors", "Avg Confidence", "Avg Latency (ms)"]
        rows = []
        
        for label, config in results["models"].items():
            stats = config["statistics"]
            rows.append([
                label,
                f"{config['perfect_matches']}/{len(self.test_cases)} ({stats['perfect_accuracy']})",
                f"{config['partial_matches']}/{len(self.test_cases)} ({stats['partial_accuracy']})",
                f"{config['full_misses']}/{len(self.test_cases)} ({stats['miss_rate']})",
                f"{config['errors']}/{len(self.test_cases)} ({stats['error_rate']})",
                stats["avg_param_confidence"],
                stats.get("latency_ms", {}).get("mean", "N/A")
            ])
        
        # Print table
        col_widths = [max(len(h), max(len(str(row[i])) for row in rows)) for i, h in enumerate(headers)]
        
        # Header
        header_line = " | ".join(h.ljust(w) for h, w in zip(headers, col_widths))
        print(header_line)
        print("-" * len(header_line))
        
        # Rows
        for row in rows:
            print(" | ".join(str(v).ljust(w) for v, w in zip(row, col_widths)))
        
        print(f"\n{'='*100}\n")
    
    def print_detailed_analysis(self, results: Dict):
        """Print detailed analysis per model."""
        print(f"\n{'='*100}")
        print("DETAILED ANALYSIS")
        print(f"{'='*100}\n")
        
        for label, config in results["models"].items():
            print(f"\n📊 {label}")
            print("-" * 80)
            
            stats = config["statistics"]
            print(f"  Perfect matches (all params correct): {config['perfect_matches']}/{len(self.test_cases)} ({stats['perfect_accuracy']})")
            print(f"  Partial matches (some params correct): {config['partial_matches']}/{len(self.test_cases)} ({stats['partial_accuracy']})")
            print(f"  Full misses (tool or params wrong): {config['full_misses']}/{len(self.test_cases)} ({stats['miss_rate']})")
            print(f"  Errors (timeouts/exceptions): {config['errors']}/{len(self.test_cases)} ({stats['error_rate']})")
            print(f"  Average parameter confidence: {stats['avg_param_confidence']}")
            
            if "latency_ms" in stats:
                latency = stats["latency_ms"]
                print(f"  Response times:")
                print(f"    - Min: {latency['min']}ms")
                print(f"    - Max: {latency['max']}ms")
                print(f"    - Mean: {latency['mean']}ms")
                print(f"    - Median: {latency['median']}ms")
            
            # Show some error examples
            errors_and_misses = [
                r for r in config["results"]
                if not r.get("params_match", False)
            ]
            
            if errors_and_misses and len(errors_and_misses) <= 5:
                print(f"\n  Sample failures ({len(errors_and_misses)} total):")
                for i, result in enumerate(errors_and_misses[:3]):
                    print(f"    {i+1}. {result.get('query', 'Unknown')[:60]}")
                    if "param_mismatches" in result:
                        for mismatch in result["param_mismatches"][:2]:
                            print(f"       - {mismatch}")


def main():
    """Main entry point."""
    import sys
    
    # Configuration
    models_to_test = [
        ("mistral-nemo","mistral-nemo"),
        ("qwen3","qwen3"),
        ("gemma3:12b","gemma3:12b"),
        ("gpt-oss:120b","GPT-OSS"),
        ("phi4","phi4"),
        ("deepseek-r1:70b","deepseek-r1:70b"),
        ("llama3.2","llama3.2"),
        ("nemotron-mini","nemotron-mini")
    ]
    
    print("="*100)
    print("ADVANCED PERFORMANCE COMPARISON - CLI PARAMETER EXTRACTION")
    print("="*100)
    print("\nTest criteria: Model must extract ALL parameters correctly to be considered accurate")
    print(f"Test cases: 50 realistic CLI scenarios with parameter specifications\n")
    
    # Run comparison
    comparator = AdvancedPerformanceComparison("Param_and_cli/queries/missingparam.json")
    results = comparator.run_comparison(models_to_test)
    
    # Save results
    output_file = comparator.save_results(results)
    
    # Print summaries
    comparator.print_comparison_table(results)
    comparator.print_detailed_analysis(results)
    
    print("\n✓ Comparison complete!")
    print(f"✓ Results saved to: {output_file}")


if __name__ == "__main__":
    main()
