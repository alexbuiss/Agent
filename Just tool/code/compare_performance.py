#!/usr/bin/env python3
"""
Performance comparison tool for different LLM models and configurations.
Compares latency, accuracy, and resource usage across settings.
"""

import json
import subprocess
import os
import time
from pathlib import Path
from typing import Dict, List
import statistics

class PerformanceComparison:
    def __init__(self, queries_file="just_cli/queries/benchmark_queries.json"):
        self.queries_file = queries_file
        self.results = {}
        
        # Load test queries
        with open(queries_file, "r") as f:
            self.queries = json.load(f)
    
    def test_configuration(self, model: str, backend: str, threshold: str = "0.40", label: str = None) -> Dict:
        """Test router with specific configuration."""
        
        if label is None:
            label = f"{model}/{backend}"
        
        print(f"\nTesting: {label}")
        print("-" * 60)
        
        # Set environment
        env = os.environ.copy()
        env["ROUTER_MODEL"] = model
        env["ROUTER_BACKEND"] = backend
        env["ROUTER_THRESHOLD"] = threshold
        
        config_results = {
            "model": model,
            "backend": backend,
            "threshold": threshold,
            "label": label,
            "latencies": [],
            "confidences": [],
            "matches": 0,
            "errors": 0
        }
        
        for i, (query, expected_tool) in enumerate(self.queries):
            print(f"  [{i+1}/{len(self.queries)}] {query[:50]}...", end=" ", flush=True)
            
            try:
                result = subprocess.run(
                    ["python", "just_cli/code/router.py", query, "--benchmark", "--dry-run"],
                    capture_output=True,
                    text=True,
                    timeout=120,
                    env=env
                )
                
                if result.returncode not in [0, 2]:
                    print("✗ (error)")
                    config_results["errors"] += 1
                    continue
                
                output = json.loads(result.stdout)
                
                if output.get("chosen") == expected_tool:
                    config_results["matches"] += 1
                    print("✓", end="")
                else:
                    print("✗", end="")
                
                # Collect metrics
                if "metrics" in output:
                    total_time = output["metrics"].get("total_time", 0)
                    if total_time > 0:
                        config_results["latencies"].append(total_time)
                
                confidence = output.get("confidence", 0)
                config_results["confidences"].append(confidence)
                
                print(f" ({total_time:.2f}s)")
                
            except subprocess.TimeoutExpired:
                print("✗ (timeout)")
                config_results["errors"] += 1
            except Exception as e:
                print(f"✗ (exception: {str(e)[:30]})")
                config_results["errors"] += 1
        
        # Calculate statistics
        if config_results["latencies"]:
            config_results["latency_stats"] = {
                "min": min(config_results["latencies"]),
                "max": max(config_results["latencies"]),
                "mean": statistics.mean(config_results["latencies"]),
                "median": statistics.median(config_results["latencies"]),
            }
        
        if config_results["confidences"]:
            config_results["confidence_stats"] = {
                "mean": statistics.mean(config_results["confidences"]),
                "median": statistics.median(config_results["confidences"]),
            }
        
        config_results["accuracy"] = config_results["matches"] / len(self.queries)
        
        return config_results
    
    def run_comparison_suite(self) -> None:
        """Run predefined comparison suite."""
        
        configs = [
            ("gemma", "ollama", "0.40", "Gemma (7B, optimized)")
        ]
        
        print("=" * 60)
        print("LLM MODEL AND CONFIGURATION COMPARISON")
        print("=" * 60)
        
        for model, backend, threshold, label in configs:
            try:
                config_results = self.test_configuration(model, backend, threshold, label)
                self.results[label] = config_results
            except Exception as e:
                print(f"\nError testing {label}: {e}")
    
    def print_comparison_summary(self) -> None:
        """Print comparison summary table."""
        
        if not self.results:
            print("No results to compare")
            return
        
        print("\n" + "=" * 100)
        print("COMPARISON SUMMARY")
        print("=" * 100)
        
        # Header
        print(f"{'Configuration':<25} {'Accuracy':<12} {'Mean Latency':<15} {'Confidence':<15} {'Errors':<10}")
        print("-" * 100)
        
        # Sort by accuracy (descending) then by latency (ascending)
        sorted_results = sorted(
            self.results.items(),
            key=lambda x: (-x[1].get("accuracy", 0), x[1].get("latency_stats", {}).get("mean", float('inf')))
        )
        
        for label, result in sorted_results:
            accuracy = f"{result.get('accuracy', 0)*100:.1f}%"
            
            latency_stats = result.get("latency_stats", {})
            if latency_stats:
                latency = f"{latency_stats['mean']:.2f}s (±{latency_stats['max']-latency_stats['min']:.2f})"
            else:
                latency = "N/A"
            
            conf_stats = result.get("confidence_stats", {})
            if conf_stats:
                confidence = f"{conf_stats['mean']:.3f}"
            else:
                confidence = "N/A"
            
            errors = result.get("errors", 0)
            
            print(f"{label:<25} {accuracy:<12} {latency:<15} {confidence:<15} {errors:<10}")
        
        print("=" * 100)
    
    def print_detailed_results(self) -> None:
        """Print detailed results for best performer."""
        
        if not self.results:
            return
        
        # Find best by accuracy
        best = max(self.results.items(), key=lambda x: x[1].get("accuracy", 0))
        label, result = best
        
        print(f"\n✓ BEST PERFORMER: {label}")
        print("=" * 60)
        print(f"Accuracy:        {result.get('accuracy', 0)*100:.1f}%")
        print(f"Matches:         {result.get('matches', 0)}/{len(self.queries)}")
        print(f"Errors:          {result.get('errors', 0)}")
        
        latency_stats = result.get("latency_stats", {})
        if latency_stats:
            print(f"\nLatency:")
            print(f"  Mean:    {latency_stats['mean']:.3f}s")
            print(f"  Median:  {latency_stats['median']:.3f}s")
            print(f"  Range:   {latency_stats['min']:.3f}s - {latency_stats['max']:.3f}s")
        
        conf_stats = result.get("confidence_stats", {})
        if conf_stats:
            print(f"\nConfidence:")
            print(f"  Mean:    {conf_stats['mean']:.3f}")
            print(f"  Median:  {conf_stats['median']:.3f}")
    
    def save_results(self, filename="just_cli/performance_comparison3.json") -> None:
        """Save comparison results to JSON."""
        
        output = {
            "comparison_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "queries_tested": len(self.queries),
            "configurations": self.results
        }
        
        with open(filename, "w") as f:
            json.dump(output, f, indent=2)
        
        print(f"\nResults saved to {filename}")


if __name__ == "__main__":
    import sys
    
    # Allow custom queries file
    queries_file = sys.argv[1] if len(sys.argv) > 1 else "just_cli/queries/benchmark_queries.json"
    
    if not Path(queries_file).exists():
        print(f"Error: Queries file '{queries_file}' not found")
        sys.exit(1)
    
    comparison = PerformanceComparison(queries_file)
    comparison.run_comparison_suite()
    comparison.print_comparison_summary()
    comparison.print_detailed_results()
    comparison.save_results()
